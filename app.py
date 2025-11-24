import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
TODAY = datetime.today().date()
PRICE_WINDOW_DAYS = 90       # window for price charts (~3 months)
ONE_YEAR_DAYS = 252          # trading days
TSUNAMI_LOOKBACK_DAYS = 120  # last N days for status text

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """
    Load several years of history so EMAs & VIX stats are accurate.
    """
    end = datetime.combine(TODAY + timedelta(days=1), datetime.min.time())
    start = end - timedelta(days=365 * 3)  # ~3 years

    def fetch(symbol: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            return df

        # Prefer 'Close', fall back to 'Adj Close' if needed
        price_col = "Close" if "Close" in df.columns else "Adj Close"
        df = df[[price_col]].rename(columns={price_col: "Close"})
        df.index = pd.to_datetime(df.index)
        return df

    spy = fetch("SPY")
    qqq = fetch("QQQ")
    vix = fetch("^VIX")
    vvix = fetch("^VVIX")

    return spy, qqq, vix, vvix


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EMAs on the FULL history. We only slice later for plotting.
    """
    if df.empty:
        return df

    out = df.copy()
    out["ema_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["ema_200"] = out["Close"].ewm(span=200, adjust=False).mean()
    return out


# -------------------------------------------------------------------
# Canary status
# -------------------------------------------------------------------
def canary_status(df_full: pd.DataFrame):
    """
    Simple 5% Canary: where is price vs 1-year high?
    Used only for the status panel (no markers on chart yet).
    """
    closes = df_full["Close"].dropna()
    if len(closes) < ONE_YEAR_DAYS:
        return (
            "🟢",
            "No Canary reading",
            "Insufficient data for a 1-year lookback.",
        )

    last_close = closes.iloc[-1]
    high_1y = closes.tail(ONE_YEAR_DAYS).max()

    if pd.isna(high_1y) or high_1y == 0:
        return (
            "🟢",
            "No Canary reading",
            "Unable to compute 1-year high (data issue).",
        )

    pct_off = (last_close / high_1y - 1.0) * 100.0

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
        )
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. Caution is warranted."
        )
    else:
        emoji = "🔴"
        headline = "Deep pullback (>10% from 1-year high)"
        detail = (
            f"SPY is {pct_off:.1f}% below its 1-year high. Risk-off conditions."
        )

    return emoji, headline, detail


# -------------------------------------------------------------------
# Tsunami status & VIX data
# -------------------------------------------------------------------
def build_vix_dataframe(vix_full: pd.DataFrame,
                        vvix_full: pd.DataFrame,
                        timeframe: str = "Daily") -> pd.DataFrame:
    """
    Create a VIX/VVIX dataframe with 20-day stdevs and a Tsunami flag.
    timeframe = 'Daily' or 'Weekly' (weekly = Friday close).
    """
    if vix_full.empty or vvix_full.empty:
        return pd.DataFrame()

    base = pd.concat(
        {
            "VIX": vix_full["Close"],
            "VVIX": vvix_full["Close"],
        },
        axis=1,
    ).dropna()

    if base.empty:
        return base

    if timeframe == "Weekly":
        # Resample to Friday closes
        base = base.resample("W-FRI").last().dropna()

    # 20-day rolling stdevs
    base["vix_stdev_20"] = base["VIX"].rolling(window=20, min_periods=10).std()
    base["vvix_stdev_20"] = base["VVIX"].rolling(window=20, min_periods=10).std()

    # Compression = both stdevs relatively low vs. recent history
    lookback = TSUNAMI_LOOKBACK_DAYS if timeframe == "Daily" else 26
    recent = base.tail(lookback)

    if recent["vix_stdev_20"].notna().sum() == 0 or recent["vvix_stdev_20"].notna().sum() == 0:
        base["tsunami"] = False
        return base

    vix_thresh = recent["vix_stdev_20"].quantile(0.3)
    vvix_thresh = recent["vvix_stdev_20"].quantile(0.3)

    base["tsunami"] = (
        (base["vix_stdev_20"] < vix_thresh) &
        (base["vvix_stdev_20"] < vvix_thresh)
    )

    return base


def tsunami_status(vix_full: pd.DataFrame, vvix_full: pd.DataFrame):
    """
    Summarize Tsunami state over the last TSUNAMI_LOOKBACK_DAYS.
    """
    df = build_vix_dataframe(vix_full, vvix_full, timeframe="Daily")

    if df.empty or not df["tsunami"].any():
        return (
            "🟢",
            "No Tsunami in window",
            "No Tsunami signal in the last 120 days.",
            None,
        )

    window = df.tail(TSUNAMI_LOOKBACK_DAYS)
    signals = window[window["tsunami"]]

    if signals.empty:
        return (
            "🟢",
            "No Tsunami in window",
            "No Tsunami signal in the last 120 days.",
            None,
        )

    last_signal_date = signals.index[-1].date()
    days_ago = (TODAY - last_signal_date).days

    emoji = "🟡"
    headline = "Tsunami compression active"
    detail = (
        f"Last Tsunami compression signal on {last_signal_date} "
        f"({days_ago} days ago)."
    )
    return emoji, headline, detail, last_signal_date


def build_vix_chart(df: pd.DataFrame) -> go.Figure:
    """
    Plot VIX level, stdev lines, and Tsunami compression markers.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="VIX",
            showlegend=True,
            height=380,
        )
        return fig

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
            y=df["vix_stdev_20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vvix_stdev_20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    tsunami_points = df[df["tsunami"]]
    if not tsunami_points.empty:
        fig.add_trace(
            go.Scatter(
                x=tsunami_points.index,
                y=tsunami_points["VIX"],
                mode="markers",
                name="Tsunami compression",
                marker=dict(symbol="diamond", size=8),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="VIX",
        showlegend=True,
        margin=dict(l=40, r=20, t=20, b=40),
        height=380,
    )
    return fig


# -------------------------------------------------------------------
# Market snapshot
# -------------------------------------------------------------------
def pct_off_high(closes: pd.Series, window: int = ONE_YEAR_DAYS) -> float:
    """
    % off the max over the last `window` points.
    """
    closes = closes.dropna()
    if len(closes) < window:
        return np.nan
    recent = closes.tail(window)
    high = recent.max()
    if high == 0 or pd.isna(high):
        return np.nan
    last = recent.iloc[-1]
    return (last / high - 1.0) * 100.0


def market_snapshot(spy_full: pd.DataFrame,
                    qqq_full: pd.DataFrame,
                    vix_full: pd.DataFrame):
    """
    Build lines of text for the Market Snapshot panel.
    """
    spy_last = spy_full["Close"].iloc[-1] if not spy_full.empty else np.nan
    qqq_last = qqq_full["Close"].iloc[-1] if not qqq_full.empty else np.nan
    vix_last = vix_full["Close"].iloc[-1] if not vix_full.empty else np.nan

    spy_off = pct_off_high(spy_full["Close"])
    qqq_off = pct_off_high(qqq_full["Close"])

    lines = []
    if not np.isnan(spy_last):
        lines.append(f"SPY: {spy_last:.2f}")
        if not np.isnan(spy_off):
            lines.append(f"Off 1-year high: {spy_off:.1f}%")
    if not np.isnan(qqq_last):
        lines.append(f"QQQ: {qqq_last:.2f}")
        if not np.isnan(qqq_off):
            lines.append(f"QQQ off 1-year high: {qqq_off:.1f}%")
    if not np.isnan(vix_last):
        lines.append(f"VIX: {vix_last:.2f}")

    return lines


# -------------------------------------------------------------------
# PRICE CHARTS (SPY / QQQ) – FIXED
# -------------------------------------------------------------------
def build_price_chart(df_full: pd.DataFrame,
                      title: str,
                      price_label: str) -> go.Figure:
    """
    Plot the last ~PRICE_WINDOW_DAYS with Close, 21-EMA, 200-EMA.
    EMAs are assumed to have been computed on the full history already.
    """
    recent_start = TODAY - timedelta(days=PRICE_WINDOW_DAYS)
    recent = df_full[df_full.index.date >= recent_start].dropna(
        subset=["Close"]
    )

    fig = go.Figure()

    if recent.empty:
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
            height=280,
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            name=price_label,
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["ema_21"],
            name="21-day EMA",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["ema_200"],
            name="200-day EMA",
            mode="lines",
        )
    )

    # Auto-fit Y with a little padding
    ymin = recent["Close"].min()
    ymax = recent["Close"].max()
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
    )
    return fig


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Market Risk Dashboard", layout="wide")

    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # Load data & compute EMAs
    spy_full, qqq_full, vix_full, vvix_full = load_data()
    spy_full = add_emas(spy_full)
    qqq_full = add_emas(qqq_full)

    # ---------------- Status panels ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Canary Status")
        can_emoji, can_headline, can_detail = canary_status(spy_full)
        st.markdown(f"{can_emoji} {can_headline}")
        st.caption(can_detail)

    with col2:
        st.subheader("Tsunami Status")
        tsu_emoji, tsu_headline, tsu_detail, _ = tsunami_status(vix_full, vvix_full)
        st.markdown(f"{tsu_emoji} {tsu_headline}")
        st.caption(tsu_detail)

    with col3:
        st.subheader("Market Snapshot")
        snapshot_lines = market_snapshot(spy_full, qqq_full, vix_full)
        for line in snapshot_lines:
            st.markdown(line)

    st.markdown("---")

    # ---------------- SPY / QQQ price charts ----------------
    st.subheader("SPY with 5% Canary Signals")
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
    st.plotly_chart(spy_fig, use_container_width=True)

    st.subheader("QQQ (NASDAQ) with 5% Canary Signals")
    qqq_fig = build_price_chart(
        qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price"
    )
    st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")

    # ---------------- VIX / Tsunami chart ----------------
    st.subheader("VIX & Volatility Tsunami Watch")
    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)

    vix_df = build_vix_dataframe(vix_full, vvix_full, timeframe=timeframe)
    vix_fig = build_vix_chart(vix_df)
    st.plotly_chart(vix_fig, use_container_width=True)
    st.caption(
        "Lines: VIX level, VIX 20-day stdev, VVIX 20-day stdev. "
        "Diamonds mark Tsunami compression signals."
    )


if __name__ == "__main__":
    main()

