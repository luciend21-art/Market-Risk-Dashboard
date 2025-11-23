import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -----------------------------------------------------
# Config
# -----------------------------------------------------
st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
)

TODAY = datetime.today()
YEARS_OF_HISTORY = 5          # how much to pull for EMAs
CANARY_LOOKBACK_DAYS = 252    # ~1 year
TSUNAMI_LOOKBACK_DAYS = 120   # for status text
PRICE_WINDOW_DAYS = 90        # window for SPY/QQQ charts


# -----------------------------------------------------
# Data loading (EMAs on full history, then slice)
# -----------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    start = TODAY - timedelta(days=365 * YEARS_OF_HISTORY)
    end = TODAY

    def fetch(symbol: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df = df[["Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    spy_full = fetch("SPY")
    qqq_full = fetch("QQQ")
    vix_full = fetch("^VIX")
    # VVIX index (Chicago Board Options Exchange VVIX)
    vvix_full = fetch("^VVIX")

    # EMAs on full history (accuracy fix)
    for df in (spy_full, qqq_full):
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return spy_full, qqq_full, vix_full, vvix_full


# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def pct_off_high(closes: pd.Series, lookback_days: int = CANARY_LOOKBACK_DAYS) -> float:
    closes = closes.dropna()
    if closes.empty:
        return np.nan
    window = closes.iloc[-lookback_days:]
    high = float(window.max())
    last = float(closes.iloc[-1])
    return (last / high - 1.0) * 100.0


# ------------------ Canary status --------------------
def canary_status(spy_full: pd.DataFrame):
    closes = spy_full["Close"].dropna()
    if len(closes) < CANARY_LOOKBACK_DAYS:
        return (
            "⚪",
            "No Canary reading",
            "Insufficient data for 1-year high comparison.",
        )

    last_close = float(closes.iloc[-1])
    recent = closes.iloc[-CANARY_LOOKBACK_DAYS:]
    high_1y = float(recent.max())

    pct_off = (last_close / high_1y - 1.0) * 100.0

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
    else:
        emoji = "🔴"
        headline = "Deep pullback (>10% from 1-year high)"

    detail = f"SPY is {pct_off:.1f}% below its 1-year high."
    return emoji, headline, detail


# ------------------ Tsunami status -------------------
def build_vix_df(vix_full: pd.DataFrame, vvix_full: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=vix_full.index)
    df["VIX"] = vix_full["Close"]

    # align VVIX onto same index
    vvix = vvix_full["Close"].reindex(df.index).ffill()
    df["VVIX"] = vvix

    df["VIX_sd20"] = df["VIX"].rolling(20).std()
    df["VVIX_sd20"] = df["VVIX"].rolling(20).std()

    # "Compression" when both vol-of-vol measures are in their lower regime
    vix_thresh = df["VIX_sd20"].rolling(252, min_periods=60).quantile(0.3)
    vvix_thresh = df["VVIX_sd20"].rolling(252, min_periods=60).quantile(0.3)

    df["Tsunami"] = (df["VIX_sd20"] < vix_thresh) & (df["VVIX_sd20"] < vvix_thresh)
    return df


def tsunami_status(vix_df: pd.DataFrame):
    recent = vix_df.tail(TSUNAMI_LOOKBACK_DAYS)
    recent_signals = recent[recent["Tsunami"]]

    if recent_signals.empty:
        return (
            "🟢",
            "No Tsunami in window",
            "No Tsunami compression signal in the last "
            f"{TSUNAMI_LOOKBACK_DAYS} days.",
        )

    last_date = recent_signals.index[-1].date().isoformat()
    return (
        "🟡",
        "Tsunami compression active",
        f"Last Tsunami compression signal on {last_date}.",
    )


# ------------------ Market snapshot ------------------
def market_snapshot(spy_full: pd.DataFrame,
                    qqq_full: pd.DataFrame,
                    vix_full: pd.DataFrame):
    spy_closes = spy_full["Close"].dropna()
    qqq_closes = qqq_full["Close"].dropna()
    vix_closes = vix_full["Close"].dropna()

    data = {}

    if not spy_closes.empty:
        data["spy_price"] = float(spy_closes.iloc[-1])
        data["spy_off_high"] = pct_off_high(spy_closes)
    else:
        data["spy_price"] = np.nan
        data["spy_off_high"] = np.nan

    if not qqq_closes.empty:
        data["qqq_price"] = float(qqq_closes.iloc[-1])
        data["qqq_off_high"] = pct_off_high(qqq_closes)
    else:
        data["qqq_price"] = np.nan
        data["qqq_off_high"] = np.nan

    if not vix_closes.empty:
        data["vix_level"] = float(vix_closes.iloc[-1])
    else:
        data["vix_level"] = np.nan

    return data


# -----------------------------------------------------
# Chart builders (Plotly, legend + auto-fit)
# -----------------------------------------------------
def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str):
    recent_start = TODAY - timedelta(days=PRICE_WINDOW_DAYS)
    recent = df_full[df_full.index >= recent_start].dropna(subset=["Close"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            mode="lines",
            name=f"{price_label} Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["ema_21"],
            mode="lines",
            name="21-day EMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["ema_200"],
            mode="lines",
            name="200-day EMA",
        )
    )

    # Auto-fit + small padding
    ymin = float(recent["Close"].min())
    ymax = float(recent["Close"].max())
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    return fig


def build_vix_chart(vix_df_full: pd.DataFrame,
                    vvix_df_full: pd.DataFrame,
                    timeframe: str):
    # choose daily or weekly series
    if timeframe == "Weekly":
        vix = vix_df_full["Close"].resample("W-FRI").last()
        vvix = vvix_df_full["Close"].resample("W-FRI").last()
    else:
        vix = vix_df_full["Close"]
        vvix = vvix_df_full["Close"]

    df = pd.DataFrame({"VIX": vix})
    df["VVIX"] = vvix.reindex(df.index).ffill()

    df["VIX_sd20"] = df["VIX"].rolling(20).std()
    df["VVIX_sd20"] = df["VVIX"].rolling(20).std()

    # reuse compression rule
    vix_thresh = df["VIX_sd20"].rolling(252, min_periods=60).quantile(0.3)
    vvix_thresh = df["VVIX_sd20"].rolling(252, min_periods=60).quantile(0.3)
    df["Tsunami"] = (df["VIX_sd20"] < vix_thresh) & (df["VVIX_sd20"] < vvix_thresh)

    recent_start = TODAY - timedelta(days=365)
    recent = df[df.index >= recent_start]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["VIX"],
            mode="lines",
            name="VIX",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["VIX_sd20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["VVIX_sd20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    # Tsunami markers
    tsunami_points = recent[recent["Tsunami"]]
    if not tsunami_points.empty:
        fig.add_trace(
            go.Scatter(
                x=tsunami_points.index,
                y=tsunami_points["VIX"],
                mode="markers",
                name="Tsunami compression",
                marker=dict(symbol="diamond", size=10),
            )
        )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
    )
    return fig, df  # return full df so status can use daily version


# -----------------------------------------------------
# Main app
# -----------------------------------------------------
def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    try:
        spy_full, qqq_full, vix_full, vvix_full = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Status panels
    can_emoji, can_headline, can_detail = canary_status(spy_full)
    vix_df_full = build_vix_df(vix_full, vvix_full)
    tsu_emoji, tsu_headline, tsu_detail = tsunami_status(vix_df_full)
    snap = market_snapshot(spy_full, qqq_full, vix_full)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Canary Status")
        st.write(f"{can_emoji} {can_headline}")
        st.caption(can_detail)

    with col2:
        st.subheader("Tsunami Status")
        st.write(f"{tsu_emoji} {tsu_headline}")
        st.caption(tsu_detail)

    with col3:
        st.subheader("Market Snapshot")
        st.write(f"SPY: {snap['spy_price']:.2f}")
        st.write(f"Off 52-week high: {snap['spy_off_high']:.1f}%")
        st.write(f"QQQ: {snap['qqq_price']:.2f}")
        st.write(f"Off 52-week high: {snap['qqq_off_high']:.1f}%")
        st.write(f"VIX: {snap['vix_level']:.2f}")

    st.markdown("---")

    # SPY & QQQ charts
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY")
    qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ")

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown(
        "Blue = price, Yellow = 21-day EMA, Green = 200-day EMA."
    )

    st.markdown("---")

    # VIX / Tsunami chart
    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)
    vix_fig, _ = build_vix_chart(vix_full, vvix_full, timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)
    st.caption(
        "Lines: VIX level, VIX 20-day stdev, VVIX 20-day stdev. "
        "Diamonds mark Tsunami compression signals."
    )


if __name__ == "__main__":
    main()

