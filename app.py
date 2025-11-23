import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go


# -----------------------------
# Data loading & preparation
# -----------------------------

@st.cache_data(show_spinner=True)
def load_data():
    """
    Load several years of history for SPY, QQQ, VIX, VVIX.
    Compute EMAs on the *full* history, then return full DataFrames.
    """
    end = datetime.today()
    # 5 years is plenty for 200-day / 40-week EMAs and 1-year highs
    start = end - timedelta(days=365 * 5)

    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "VIX": "^VIX",
        "VVIX": "^VVIX",
    }

    dfs = {}
    for key, symbol in tickers.items():
        df = yf.download(symbol, start=start, end=end, progress=False)
        # Use Close only; drop rows without a close
        df = df[["Close"]].dropna()
        dfs[key] = df

    spy_full = dfs["SPY"]
    qqq_full = dfs["QQQ"]
    vix_full = dfs["VIX"]
    vvix_full = dfs["VVIX"]

    # --- EMAs on full history (Close, not Adj Close) ---
    for df in (spy_full, qqq_full):
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # --- 20-day stdev for VIX / VVIX (for Tsunami watch) ---
    vix_full["sd_20"] = vix_full["Close"].rolling(20).std()
    vvix_full["sd_20"] = vvix_full["Close"].rolling(20).std()

    return spy_full, qqq_full, vix_full, vvix_full


# -----------------------------
# Status helpers
# -----------------------------

def canary_status(spy_df: pd.DataFrame):
    """Simple 1-year-high based Canary status for the summary panel."""
    closes = spy_df["Close"].dropna()
    last_close = closes.iloc[-1]
    one_year = closes.tail(252)
    high_1y = one_year.max()
    pct_off = (last_close / high_1y - 1.0) * 100

    if pct_off >= -5:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
        detail = f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
    elif pct_off >= -10:
        emoji = "🟡"
        headline = "Moderate 5–10% pullback"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. "
            "Canary is watching – consider tightening risk."
        )
    else:
        emoji = "🔴"
        headline = "Deep 10%+ correction"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. "
            "Treat as a confirmed risk-off backdrop."
        )

    return emoji, headline, detail, pct_off


def tsunami_status(vix_df: pd.DataFrame, vvix_df: pd.DataFrame, lookback_days: int = 120):
    """
    Very simple Tsunami-compression status:
    We look at the sum of 20-day stdevs of VIX + VVIX and
    flag compressions when that combo sits in the lower decile of its 1-year range.
    """
    df = pd.DataFrame(index=vix_df.index)
    df["VIX"] = vix_df["Close"]
    df["vix_sd"] = vix_df["sd_20"]
    df["vvix_sd"] = vvix_df["sd_20"].reindex(df.index).interpolate()

    combo = df["vix_sd"] + df["vvix_sd"]
    # rolling 1-year 10th percentile threshold
    thresh = combo.rolling(252, min_periods=100).quantile(0.10)
    signal = combo < thresh

    recent = signal.tail(lookback_days)
    last_signal_idx = recent[recent].index.max() if recent.any() else None

    if last_signal_idx is None:
        emoji = "🟢"
        headline = "No Tsunami in window"
        detail = f"No VIX/VVIX compression signal in the last {lookback_days} days."
    else:
        days_ago = (df.index.max() - last_signal_idx).days
        emoji = "🟡"
        headline = "Tsunami compression watch"
        detail = (
            f"Last compression signal {days_ago} days ago "
            f"on {last_signal_idx.date()}."
        )

    return emoji, headline, detail, last_signal_idx


def market_snapshot(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, vix_df: pd.DataFrame):
    """Small dict with prices and % off 52-week high for SPY/QQQ and current VIX."""
    snap = {}
    for label, df in [("SPY", spy_df), ("QQQ", qqq_df)]:
        closes = df["Close"].dropna()
        last = closes.iloc[-1]
        high_52w = closes.tail(252).max()
        pct_off = (last / high_52w - 1.0) * 100

        snap[f"{label}_price"] = last
        snap[f"{label}_off_high"] = pct_off

    vix_last = vix_df["Close"].dropna().iloc[-1]
    snap["VIX"] = vix_last
    return snap


# -----------------------------
# Charts
# -----------------------------

def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str):
    """
    Build a Plotly line chart for the last ~3 months:
    - Blue: price
    - Yellow: 21-day EMA
    - Green: 200-day EMA
    EMAs are assumed to already be computed on the full history.
    """
    recent = df_full.tail(60).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            mode="lines",
            name="Price",
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

    # Auto-fit Y with a little padding based on price range
    ymin = recent["Close"].min()
    ymax = recent["Close"].max()
    pad = (ymax - ymin) * 0.05
    fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_label,
        showlegend=True,
        margin=dict(l=40, r=10, t=40, b=40),
        height=350,
    )

    return fig


def build_vix_chart(vix_df: pd.DataFrame, vvix_df: pd.DataFrame, timeframe: str = "Daily"):
    """
    Plot:
    - VIX level
    - VIX 20-day stdev (dotted)
    - VVIX 20-day stdev (dotted)
    """
    df = pd.DataFrame(index=vix_df.index)
    df["VIX"] = vix_df["Close"]
    df["vix_sd"] = vix_df["sd_20"]
    df["vvix_sd"] = vvix_df["sd_20"].reindex(df.index).interpolate()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX"],
            mode="lines",
            name="VIX level",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vix_sd"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vvix_sd"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    fig.update_layout(
        title=f"VIX & Volatility Tsunami Watch ({timeframe})",
        xaxis_title="Date",
        yaxis_title="VIX / Volatility",
        showlegend=True,
        margin=dict(l=40, r=10, t=40, b=40),
        height=350,
    )

    return fig


# -----------------------------
# Main app
# -----------------------------

def main():
    st.set_page_config(page_title="Market Risk Dashboard", layout="wide")
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # --- Load data (with full-history EMAs) ---
    try:
        spy_full, qqq_full, vix_full, vvix_full = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # --- Status panels ---
    can_emoji, can_headline, can_detail, _ = canary_status(spy_full)
    tsu_emoji, tsu_headline, tsu_detail, _ = tsunami_status(vix_full, vvix_full)
    snapshot = market_snapshot(spy_full, qqq_full, vix_full)

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
        st.write(f"SPY: {snapshot['SPY_price']:.2f}")
        st.write(f"Off 52-week high: {snapshot['SPY_off_high']:.1f}%")
        st.write(f"QQQ: {snapshot['QQQ_price']:.2f}")
        st.write(f"Off 52-week high: {snapshot['QQQ_off_high']:.1f}%")
        st.write(f"VIX: {snapshot['VIX']:.2f}")

    st.markdown("---")
    st.subheader("Index Trends (last ~3 months)")

    # --- Price charts with full-history EMAs but 3-month window ---
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
    qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(spy_fig, use_container_width=True)
    with c2:
        st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Volatility Tsunami Watch")

    vix_fig = build_vix_chart(vix_full, vvix_full, timeframe="Daily")
    st.plotly_chart(vix_fig, use_container_width=True)


if __name__ == "__main__":
    main()

