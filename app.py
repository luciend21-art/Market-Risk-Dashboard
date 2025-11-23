# app.py  — Market Risk Dashboard (clean build with EMA + Canary + Tsunami fixes)

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
)

PRICE_WINDOW_DAYS = 90          # zoom window for SPY/QQQ charts
TSUNAMI_WINDOW_DAYS = 120       # lookback window for "active" tsunami
EMA_SHORT = 21
EMA_LONG = 200
VIX_SD_WINDOW = 20

TODAY = datetime.today().date()


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------

def fetch_close_series(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download OHLC data for `symbol` and return a DataFrame with a single
    'Close' column, indexed by DateTime.
    """
    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No 'Close' data for symbol {symbol}")

    df = data[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def load_all_data():
    """
    Load several years of data for SPY, QQQ, VIX, VVIX.
    We keep full history for indicator calculations.
    """
    end = datetime.combine(TODAY + timedelta(days=1), datetime.min.time())
    start = end - relativedelta(years=3)

    spy = fetch_close_series("SPY", start, end)
    qqq = fetch_close_series("QQQ", start, end)
    vix = fetch_close_series("^VIX", start, end)     # S&P 500 Volatility Index
    vvix = fetch_close_series("^VVIX", start, end)   # VVIX – VIX of VIX

    return spy, qqq, vix, vvix


# ---------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 21-day and 200-day EMAs on the FULL history of df['Close'].
    """
    df = df.copy()
    close = df["Close"]
    df["ema_21"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_200"] = close.ewm(span=EMA_LONG, adjust=False).mean()
    return df


def build_vix_features(vix_full: pd.DataFrame, vvix_full: pd.DataFrame) -> pd.DataFrame:
    """
    Combine VIX and VVIX Close series and compute 20-day rolling stdev
    for each. This single DataFrame is then reused by tsunami_status()
    and the VIX chart.
    """
    df = pd.DataFrame(index=vix_full.index)
    df["VIX"] = vix_full["Close"]
    df["VVIX"] = vvix_full["Close"].reindex(df.index)

    df["vix_sd_20"] = df["VIX"].rolling(VIX_SD_WINDOW).std()
    df["vvix_sd_20"] = df["VVIX"].rolling(VIX_SD_WINDOW).std()

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------
# Canary logic (SPY 1-year high)
# ---------------------------------------------------------------------

def canary_status(spy_full: pd.DataFrame):
    """
    Evaluate how far SPY is from its 1-year high and translate that into:
    - emoji (flashlight color)
    - headline
    - detail string
    """
    if spy_full.empty:
        return "⚪", "No Canary reading", "No SPY data available."

    # Compute over the last ~1 year of data
    last_close = float(spy_full["Close"].iloc[-1])
    one_year_ago = spy_full.index.max() - timedelta(days=365)
    window = spy_full[spy_full.index >= one_year_ago]

    if window.empty:
        return "⚪", "No Canary reading", "Insufficient history for 1-year high."

    high_1y = float(window["Close"].max())

    if high_1y <= 0:
        return "⚪", "No Canary reading", "Invalid 1-year high for SPY."

    pct_off = (last_close / high_1y - 1.0) * 100.0  # negative when below high

    # Interpret the pullback
    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
    else:
        emoji = "🔴"
        headline = "Deep pullback (>10% from 1-year high)"

    detail = f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
    return emoji, headline, detail


# ---------------------------------------------------------------------
# Tsunami logic (VIX & VVIX compression)
# ---------------------------------------------------------------------

def tsunami_status(vix_features: pd.DataFrame):
    """
    Simple Tsunami 'compression' definition:
    - compute distribution of 20-day stdevs for VIX & VVIX
    - compression when both are in the lowest 25% of history
    - if the last such event is within TSUNAMI_WINDOW_DAYS, treat as active
    """
    hist = vix_features.dropna(subset=["vix_sd_20", "vvix_sd_20"])

    if hist.empty:
        return "⚪", "No Tsunami reading", "Not enough VIX / VVIX history."

    vix_q = float(hist["vix_sd_20"].quantile(0.25))
    vvix_q = float(hist["vvix_sd_20"].quantile(0.25))

    compressed = hist[
        (hist["vix_sd_20"] <= vix_q) &
        (hist["vvix_sd_20"] <= vvix_q)
    ]

    if compressed.empty:
        return "🟢", "No Tsunami in window", "No VIX/VVIX compression signals in history."

    last_date = compressed.index[-1].date()
    days_since = (TODAY - last_date).days

    if days_since <= TSUNAMI_WINDOW_DAYS:
        emoji = "🟡"
        headline = "Tsunami compression active"
    else:
        emoji = "🟢"
        headline = "No Tsunami in window"

    detail = f"Last Tsunami compression signal on {last_date.isoformat()} ({days_since} days ago)."
    return emoji, headline, detail


# ---------------------------------------------------------------------
# Market snapshot helpers
# ---------------------------------------------------------------------

def price_and_off_high(df_full: pd.DataFrame, lookback_days: int = 365):
    """
    Return (last_close, pct_off_1yr_high) for df_full['Close'] over lookback_days.
    """
    if df_full.empty:
        return math.nan, math.nan

    last_close = float(df_full["Close"].iloc[-1])

    end = df_full.index.max()
    start = end - timedelta(days=lookback_days)
    window = df_full[df_full.index >= start]

    if window.empty:
        return last_close, math.nan

    high = float(window["Close"].max())
    if high <= 0:
        return last_close, math.nan

    pct_off = (last_close / high - 1.0) * 100.0
    return last_close, pct_off


# ---------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------

def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str) -> go.Figure:
    """
    SPY / QQQ price chart with 21- and 200-day EMAs.
    Uses full-history EMAs but only plots the last PRICE_WINDOW_DAYS.
    """
    df_full = add_emas(df_full)

    if len(df_full) > PRICE_WINDOW_DAYS:
        recent = df_full.iloc[-PRICE_WINDOW_DAYS:]
    else:
        recent = df_full.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent["Close"],
        name=price_label,
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent["ema_21"],
        name="21-day EMA",
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent["ema_200"],
        name="200-day EMA",
        mode="lines",
    ))

    # Auto-fit Y with a bit of padding
    ymin = float(recent["Close"].min())
    ymax = float(recent["Close"].max())
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0

    fig.update_yaxes(range=[ymin - pad, ymax + pad])
    fig.update_layout(
        title=title,
        margin=dict(l=30, r=30, t=40, b=30),
        height=350,
        showlegend=True,
    )
    return fig


def build_vix_chart(vix_features: pd.DataFrame, timeframe: str = "Daily") -> go.Figure:
    """
    VIX & Tsunami chart:
    - VIX level
    - 20-day stdev of VIX & VVIX
    - marks compression points as diamonds
    timeframe: "Daily" or "Weekly"
    """
    df = vix_features.copy()

    # Restrict to roughly 1 year for clarity
    start_cut = df.index.max() - timedelta(days=365)
    df = df[df.index >= start_cut]

    if timeframe == "Weekly":
        # Resample to weekly (Friday) closes
        df = df.resample("W-FRI").last()
        df["vix_sd_20"] = df["VIX"].rolling(20).std()
        df["vvix_sd_20"] = df["VVIX"].rolling(20).std()
        df.dropna(inplace=True)

    # Recompute compression points for the plotted range
    hist = df.dropna(subset=["vix_sd_20", "vvix_sd_20"])
    vix_q = float(hist["vix_sd_20"].quantile(0.25))
    vvix_q = float(hist["vvix_sd_20"].quantile(0.25))
    compressed = hist[
        (hist["vix_sd_20"] <= vix_q) &
        (hist["vvix_sd_20"] <= vvix_q)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["VIX"],
        name="VIX level",
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["vix_sd_20"],
        name="VIX 20-day stdev",
        mode="lines",
        line=dict(dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["vvix_sd_20"],
        name="VVIX 20-day stdev",
        mode="lines",
        line=dict(dash="dot"),
    ))

    # Mark compression signals as diamonds on the VIX level
    if not compressed.empty:
        fig.add_trace(go.Scatter(
            x=compressed.index,
            y=compressed["VIX"],
            mode="markers",
            name="Tsunami compression",
            marker=dict(symbol="diamond", size=10),
        ))

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        margin=dict(l=30, r=30, t=40, b=30),
        height=380,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    try:
        spy_full, qqq_full, vix_full, vvix_full = load_all_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Pre-compute VIX feature DataFrame
    vix_features = build_vix_features(vix_full, vvix_full)

    # ----- Status panels -----
    can_emoji, can_headline, can_detail = canary_status(spy_full)
    tsu_emoji, tsu_headline, tsu_detail = tsunami_status(vix_features)

    spy_price, spy_off = price_and_off_high(spy_full)
    qqq_price, qqq_off = price_and_off_high(qqq_full)
    vix_last = float(vix_full["Close"].iloc[-1]) if not vix_full.empty else math.nan

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Canary Status")
        st.markdown(f"{can_emoji} **{can_headline}**")
        st.write(can_detail)

    with col2:
        st.subheader("Tsunami Status")
        st.markdown(f"{tsu_emoji} **{tsu_headline}**")
        st.write(tsu_detail)

    with col3:
        st.subheader("Market Snapshot")
        if not math.isnan(spy_price):
            st.write(f"SPY: {spy_price:.2f}")
            if not math.isnan(spy_off):
                st.write(f"Off 1-year high: {spy_off:.1f}%")
        if not math.isnan(qqq_price):
            st.write(f"QQQ: {qqq_price:.2f}")
            if not math.isnan(qqq_off):
                st.write(f"QQQ off 1-year high: {qqq_off:.1f}%")
        if not math.isnan(vix_last):
            st.write(f"VIX: {vix_last:.2f}")

    st.markdown("---")

    # ----- SPY & QQQ price charts -----
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
    qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price")

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")

    # ----- VIX & Tsunami chart -----
    timeframe = st.radio("VIX timeframe", options=["Daily", "Weekly"], horizontal=True)
    vix_fig = build_vix_chart(vix_features, timeframe=timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)

    st.caption("Lines: VIX level, VIX 20-day stdev, VVIX 20-day stdev. "
               "Diamonds mark Tsunami compression signals.")


if __name__ == "__main__":
    main()
