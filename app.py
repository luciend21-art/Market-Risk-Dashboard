import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go

# --------------------------------------------------
# Config / constants
# --------------------------------------------------

TODAY: date = date.today()
YEARS_HISTORY = 5          # how far back we pull for EMAs
PRICE_WINDOW_DAYS = 90     # zoom window for SPY / QQQ charts
TSUNAMI_LOOKBACK_DAYS = 120
TSUNAMI_ROLL_DAYS = 20


# --------------------------------------------------
# Data loading helpers
# --------------------------------------------------

def _fetch_history(symbol: str, years: int = YEARS_HISTORY, with_emas: bool = False) -> pd.DataFrame:
    """
    Download several years of history for a symbol and (optionally) compute EMAs
    on the full history using the Close price.
    """
    end = TODAY + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        return pd.DataFrame(columns=["Close"])

    # Keep just Close to keep things tight and predictable
    df = df[["Close"]].copy()

    if with_emas:
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load all price series needed for the dashboard."""
    spy_full = _fetch_history("SPY", with_emas=True)
    qqq_full = _fetch_history("QQQ", with_emas=True)
    vix_full = _fetch_history("^VIX")
    vvix_full = _fetch_history("^VVIX")
    return spy_full, qqq_full, vix_full, vvix_full


# --------------------------------------------------
# Canary, Tsunami, Snapshot
# --------------------------------------------------

def _one_year_window(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    end = series.index.max()
    start = end - timedelta(days=365)
    return series[series.index >= start]


def canary_status(spy_df: pd.DataFrame):
    """
    Canary based on how far SPY is from its 1-year high.
    Returns: emoji, headline, detail_text
    """
    if spy_df.empty or "Close" not in spy_df.columns:
        return "⚪", "No Canary reading", "Insufficient SPY data for 1-year high window."

    closes = spy_df["Close"].dropna()
    if closes.empty:
        return "⚪", "No Canary reading", "Insufficient SPY data for 1-year high window."

    last_close = float(closes.iloc[-1])
    window = _one_year_window(closes)
    if window.empty:
        return "⚪", "No Canary reading", "Insufficient SPY data for 1-year high window."

    high_1y = float(window.max())
    pct_off = (last_close / high_1y - 1.0) * 100.0

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
    else:
        emoji = "🔴"
        headline = "Deep correction (>10% off 1-year high)"

    detail = f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
    return emoji, headline, detail


def tsunami_status(vix_df: pd.DataFrame, vvix_df: pd.DataFrame):
    """
    Tsunami compression based on 20-day stdev of VIX + VVIX over a lookback window.
    Returns: emoji, headline, detail, last_signal_date (or None)
    """
    if vix_df.empty or vvix_df.empty:
        return "⚪", "No Tsunami reading", "Insufficient volatility data.", None

    df = pd.DataFrame(
        {
            "VIX": vix_df["Close"],
            "VVIX": vvix_df["Close"],
        }
    ).dropna()

    if df.empty:
        return "⚪", "No Tsunami reading", "Insufficient volatility data.", None

    df["vix_sd"] = df["VIX"].rolling(TSUNAMI_ROLL_DAYS).std()
    df["vvix_sd"] = df["VVIX"].rolling(TSUNAMI_ROLL_DAYS).std()
    df["combo"] = df["vix_sd"] + df["vvix_sd"]

    lookback_start = df.index.max() - timedelta(days=TSUNAMI_LOOKBACK_DAYS)
    window_df = df[df.index >= lookback_start].copy()

    if window_df["combo"].dropna().empty:
        return "⚪", "No Tsunami reading", "Not enough data for Tsunami window.", None

    threshold = window_df["combo"].quantile(0.9)
    window_df["tsunami"] = window_df["combo"] >= threshold

    if not bool(window_df["tsunami"].any()):
        return "🟢", "No Tsunami in window", "No Tsunami signal in the last 120 days.", None

    last_signal_idx = window_df.index[window_df["tsunami"]].max()
    last_signal_date = last_signal_idx.date()

    # Is the most recent reading still in Tsunami mode?
    active = bool(window_df["tsunami"].iloc[-1])

    if active:
        emoji = "🟡"
        headline = "Tsunami compression active"
    else:
        emoji = "🟡"
        headline = "Tsunami compression cooling"

    detail = f"Last Tsunami compression signal on {last_signal_date:%Y-%m-%d}."
    return emoji, headline, detail, last_signal_date


def market_snapshot(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, vix_df: pd.DataFrame):
    """Return simple snapshot numbers for the top-right panel."""
    def _last_and_off(series: pd.Series):
        series = series.dropna()
        if series.empty:
            return np.nan, np.nan
        last = float(series.iloc[-1])
        window = _one_year_window(series)
        high_1y = float(window.max()) if not window.empty else np.nan
        pct_off = (last / high_1y - 1.0) * 100.0 if high_1y == high_1y else np.nan
        return last, pct_off

    spy_last, spy_off = _last_and_off(spy_df["Close"])
    qqq_last, qqq_off = _last_and_off(qqq_df["Close"])

    vix_last = float(vix_df["Close"].dropna().iloc[-1]) if not vix_df.empty else np.nan

    return {
        "spy_last": spy_last,
        "spy_off": spy_off,
        "qqq_last": qqq_last,
        "qqq_off": qqq_off,
        "vix_last": vix_last,
    }


# --------------------------------------------------
# Charts
# --------------------------------------------------

def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str) -> go.Figure:
    """SPY / QQQ price + 21 / 200 EMAs, zoomed to recent window."""
    fig = go.Figure()

    if df_full.empty or "Close" not in df_full.columns:
        fig.update_layout(title=title)
        return fig

    recent_start = TODAY - timedelta(days=PRICE_WINDOW_DAYS)
    recent = df_full[df_full.index >= recent_start].dropna()

    if recent.empty:
        fig.update_layout(title=title)
        return fig

    # Price
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            name=price_label,
            mode="lines",
        )
    )

    # EMAs (only if present)
    if "ema_21" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["ema_21"],
                name="21-day EMA",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

    if "ema_200" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["ema_200"],
                name="200-day EMA",
                mode="lines",
                line=dict(dash="dot"),
            )
        )

    # Auto-fit with a little padding
    ymin = float(recent["Close"].min())
    ymax = float(recent["Close"].max())
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_label,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
    )

    return fig


def build_vix_chart(vix_full: pd.DataFrame, vvix_full: pd.DataFrame, timeframe: str = "Daily") -> go.Figure:
    """VIX level + 20-day stdevs for VIX and VVIX."""
    fig = go.Figure()

    if vix_full.empty or vvix_full.empty:
        fig.update_layout(title="VIX & Volatility Tsunami Watch")
        return fig

    df = pd.DataFrame(
        {
            "VIX": vix_full["Close"],
            "VVIX": vvix_full["Close"],
        }
    ).dropna()

    if df.empty:
        fig.update_layout(title="VIX & Volatility Tsunami Watch")
        return fig

    df["vix_sd"] = df["VIX"].rolling(TSUNAMI_ROLL_DAYS).std()
    df["vvix_sd"] = df["VVIX"].rolling(TSUNAMI_ROLL_DAYS).std()

    if timeframe == "Weekly":
        df = df.resample("W-FRI").last().dropna(how="all")

    fig.add_trace(go.Scatter(x=df.index, y=df["VIX"], name="VIX level", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vix_sd"],
            name="VIX 20-day stdev",
            mode="lines",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vvix_sd"],
            name="VVIX 20-day stdev",
            mode="lines",
            line=dict(dash="dot"),
        )
    )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        xaxis_title="Date",
        yaxis_title="VIX, VVIX, 20-day SD",
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
    )

    return fig


# --------------------------------------------------
# Main app
# --------------------------------------------------

def main():
    st.set_page_config(page_title="Market Risk Dashboard", layout="wide")
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    try:
        spy_full, qqq_full, vix_full, vvix_full = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Status panels
    can_emoji, can_headline, can_detail = canary_status(spy_full)
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
        st.markdown(f"**SPY: {snapshot['spy_last']:.2f}**")
        st.markdown(f"Off 52-week high: {snapshot['spy_off']:.1f}%")
        st.markdown(f"**QQQ: {snapshot['qqq_last']:.2f}**")
        st.markdown(f"QQQ off 52-week high: {snapshot['qqq_off']:.1f}%")
        st.markdown(f"**VIX: {snapshot['vix_last']:.2f}**")

    st.markdown("---")

    # SPY / QQQ charts
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
    qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price")

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")

    # VIX / Tsunami chart
    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)
    vix_fig = build_vix_chart(vix_full, vvix_full, timeframe=timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)


if __name__ == "__main__":
    main()
