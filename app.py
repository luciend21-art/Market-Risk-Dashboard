import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config("Market Risk Dashboard", layout="wide")


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(ttl=60 * 60)
def load_data():
    """
    Download 5 years of history for SPY, QQQ, VIX and VVIX.
    We compute EMAs and vol stats on the full history and only
    slice the most recent data for plotting.
    """
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=365 * 5)

    def fetch(symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        out = df[["Close"]].copy()
        out.index = pd.to_datetime(out.index)
        return out

    spy = fetch("SPY")
    qqq = fetch("QQQ")
    vix = fetch("^VIX")
    vvix = fetch("^VVIX")

    # EMAs on full history
    for df in (spy, qqq):
        if not df.empty:
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return spy, qqq, vix, vvix


# -----------------------------
# Canary logic
# -----------------------------
def canary_status(df: pd.DataFrame):
    """High-level Canary regime based on SPY 1-year drawdown."""
    closes = df["Close"].astype(float)

    # 1-year rolling high (approx 252 trading days)
    high_1y_series = closes.rolling(window=252, min_periods=1).max()

    # Use only the most recent values for the status
    last_close = float(closes.iloc[-1])
    last_high_1y = float(high_1y_series.iloc[-1])

    # Guard against weird data (all NaNs, etc.)
    if pd.isna(last_high_1y) or last_high_1y == 0:
        return "⚪", "No Canary reading", "Insufficient data for 1-year high.", np.nan

    pct_off = (last_close / last_high_1y - 1.0) * 100.0  # negative = below high

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
        detail = f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. Stay alert and tighten risk."
        )
    else:
        emoji = "🔴"
        headline = "Deep pullback (>10% from 1-year high)"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. Treat this as a high-risk zone."
        )

    return emoji, headline, detail, pct_off

# -----------------------------
# Tsunami logic
# -----------------------------
def build_vix_df(vix: pd.DataFrame, vvix: pd.DataFrame) -> pd.DataFrame:
    """
    Build combined VIX/VVIX dataframe with 20-day stdevs and Tsunami signals.
    Tsunami signal = combo stdev (VIX_sd20 + VVIX_sd20) above its rolling 90th percentile.
    """
    if vix.empty:
        return pd.DataFrame()

    df = pd.DataFrame(index=vix.index)
    df["VIX"] = vix["Close"]

    if vvix is not None and not vvix.empty and "Close" in vvix.columns:
        df["VVIX"] = vvix["Close"].reindex(df.index)
    else:
        df["VVIX"] = np.nan

    df["VIX_SD20"] = df["VIX"].rolling(20).std()
    df["VVIX_SD20"] = df["VVIX"].rolling(20).std()
    df["Combo_SD"] = df["VIX_SD20"] + df["VVIX_SD20"]

    # Rolling 90th percentile as dynamic threshold (about 3 years of lookback)
    df["Combo_Thresh"] = df["Combo_SD"].rolling(252 * 3, min_periods=100).quantile(0.9)
    df["Tsunami"] = (df["Combo_SD"] > df["Combo_Thresh"]) & df["Combo_Thresh"].notna()

    return df


def tsunami_status(vix_df: pd.DataFrame, lookback_days: int = 120):
    """
    Returns (emoji, headline, detail) based on last Tsunami signal
    in a given lookback window.
    """
    if vix_df.empty or "Tsunami" not in vix_df.columns:
        return "⚪", "No data", "No VIX data available."

    recent = vix_df.tail(lookback_days)
    signals = recent[recent["Tsunami"]]

    if signals.empty:
        return "🟢", "No Tsunami in window", f"No Tsunami signal in the last {lookback_days} days."

    last_date = signals.index[-1].date()
    return (
        "🔴",
        "Tsunami compression in window",
        f"Last Tsunami compression signal: {last_date.isoformat()} "
        f"(within the last {lookback_days} days).",
    )


# -----------------------------
# Chart builders
# -----------------------------
def build_price_chart(df_full: pd.DataFrame, title: str, symbol: str) -> go.Figure:
    """
    Plotly price chart: last ~3 months of data, with 21- and 200-day EMAs,
    and a horizontal line at the recent low.
    """
    if df_full.empty:
        return go.Figure()

    days_window = 65  # roughly 3 months of trading days
    df = df_full.tail(days_window).copy()

    recent_low = df["Close"].min()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=f"{symbol} Price",
        )
    )
    if "EMA21" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA21"],
                mode="lines",
                name="21-day EMA",
            )
        )
    if "EMA200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA200"],
                mode="lines",
                name="200-day EMA",
            )
        )

    # Horizontal line at most recent low in the window
    fig.add_hline(
        y=recent_low,
        line_dash="dot",
        line_color="gray",
        annotation_text="Recent low",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        height=320,
        yaxis_title=f"{symbol} Price",
    )
    return fig


def build_vix_chart(vix_df: pd.DataFrame, timeframe: str = "Daily") -> go.Figure:
    """
    Plotly chart of VIX, VIX 20-day stdev and VVIX 20-day stdev with Tsunami markers.
    """
    if vix_df.empty:
        return go.Figure()

    if timeframe == "Weekly":
        df = vix_df.resample("W").last()
    else:
        df = vix_df.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX"],
            mode="lines",
            name="VIX",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX_SD20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VVIX_SD20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    # Tsunami markers
    sig_idx = df.index[df["Tsunami"]]
    sig_vals = df.loc[df["Tsunami"], "VIX"]
    fig.add_trace(
        go.Scatter(
            x=sig_idx,
            y=sig_vals,
            mode="markers",
            name="Tsunami signal",
            marker=dict(symbol="diamond", size=9),
        )
    )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        height=360,
        yaxis_title="VIX",
    )

    return fig


# -----------------------------
# Helper for market snapshot
# -----------------------------
def pct_off_high(series: pd.Series, lookback_days: int = 252) -> float:
    if series is None or series.empty:
        return np.nan
    s = series.tail(lookback_days)
    if s.empty:
        return np.nan
    high = s.max()
    last = s.iloc[-1]
    return (last / high - 1.0) * 100.0


# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    spy_full, qqq_full, vix_full, vvix_full = load_data()

    # ----- Status panels -----
    can_emoji, can_headline, can_detail, _ = canary_status(spy_full)
    vix_df = build_vix_df(vix_full, vvix_full)
    tsu_emoji, tsu_headline, tsu_detail = tsunami_status(vix_df)

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

        if not spy_full.empty:
            spy_last = spy_full["Close"].iloc[-1]
            spy_off = pct_off_high(spy_full["Close"])
            st.write(f"SPY: {spy_last:,.2f}")
            st.write(f"Off 52-week high: {spy_off:.1f}%")

        if not qqq_full.empty:
            qqq_last = qqq_full["Close"].iloc[-1]
            qqq_off = pct_off_high(qqq_full["Close"])
            st.write(f"QQQ: {qqq_last:,.2f}")
            st.write(f"QQQ off 52-week high: {qqq_off:.1f}%")

        if not vix_full.empty:
            vix_last = vix_full["Close"].iloc[-1]
            st.write(f"VIX: {vix_last:.2f}")

    st.markdown("---")

    # ----- SPY & QQQ price charts -----
    st.subheader("SPY & QQQ with 5% Canary Context (3-month daily)")

    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY")
    qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ")

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    # ----- VIX / Tsunami chart -----
    st.markdown("---")
    st.subheader("Volatility Tsunami Watch")

    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True, key="vix_tf")

    vix_fig = build_vix_chart(vix_df, timeframe=timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)

    st.caption("Lines: VIX level, VIX 20-day stdev, VVIX 20-day stdev. "
               "Diamond markers show Tsunami compression signals.")


if __name__ == "__main__":
    main()
