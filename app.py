import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

YEARS_HISTORY = 5          # how much data to pull
PRICE_WINDOW_DAYS = 90     # how many calendar days to show on SPY/QQQ charts
ONE_YEAR_DAYS = 252        # trading days for "1-year high" approximation


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=True)
def load_history():
    """
    Load SPY, QQQ, VIX, VVIX with several years of history.
    Compute EMAs and Tsunami metrics on the *full* history.
    """
    end = datetime.today()
    start = end - timedelta(days=365 * YEARS_HISTORY)

    def fetch_price(symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame(columns=["Close", "ema_21", "ema_200"])

        # Ensure we only keep the Close column
        df = df[["Close"]].copy()
        df = df.dropna(subset=["Close"])

        # Full-history EMAs
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        return df

    spy_full = fetch_price("SPY")
    qqq_full = fetch_price("QQQ")

    # VIX & VVIX for Tsunami logic
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    vvix_raw = yf.download("^VVIX", start=start, end=end, auto_adjust=False, progress=False)

    vix_full = pd.DataFrame(index=vix_raw.index)
    vix_full["VIX"] = vix_raw["Close"].astype(float)
    vix_full = vix_full.dropna(subset=["VIX"])

    vvix_full = pd.DataFrame(index=vvix_raw.index)
    vvix_full["VVIX"] = vvix_raw["Close"].astype(float)
    vvix_full = vvix_full.dropna(subset=["VVIX"])

    # Align VVIX to VIX index
    vvix_full = vvix_full.reindex(vix_full.index)

    # 20-day rolling stdevs
    vix_full["vix_sd20"] = vix_full["VIX"].rolling(20, min_periods=10).std()
    vix_full["vvix_sd20"] = vvix_full["VVIX"].rolling(20, min_periods=10).std()

    # Tsunami compression score: low combined volatility of VIX & VVIX
    combo = vix_full["vix_sd20"] + vix_full["vvix_sd20"]
    # Use a long lookback for "normal" level
    roll_mean = combo.rolling(252, min_periods=40).mean()
    roll_std = combo.rolling(252, min_periods=40).std()
    zscore = (combo - roll_mean) / roll_std

    # Compression when z-score is meaningfully below normal
    vix_full["tsunami"] = zscore < -1.0

    return spy_full, qqq_full, vix_full


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def pct_off_high(closes: pd.Series, days: int = ONE_YEAR_DAYS) -> float | None:
    closes = closes.dropna()
    if closes.empty:
        return None

    window = closes.tail(days)
    if window.empty:
        return None

    high = float(window.max())
    last = float(closes.iloc[-1])
    if high == 0:
        return None

    return (last / high - 1.0) * 100.0


def canary_status(spy_df: pd.DataFrame):
    """
    Canary panel:
    - Uses SPY's % off 1-year high
    - Avoids ambiguous Series truth-tests
    """
    closes = spy_df.get("Close", pd.Series(dtype=float)).dropna()

    if closes.empty or len(closes) < 10:
        return "⚪", "No Canary reading", "Insufficient data for SPY."

    pct = pct_off_high(closes, ONE_YEAR_DAYS)
    if pct is None:
        return "⚪", "No Canary reading", "Insufficient data for SPY."

    if pct >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
    elif pct >= -10.0:
        emoji = "🟡"
        headline = "Standard pullback (5–10% from 1-year high)"
    else:
        emoji = "🔴"
        headline = "Deep pullback (>10% from 1-year high)"

    detail = f"SPY is {pct:.1f}% below its 1-year high. No Canary warning."
    return emoji, headline, detail


def tsunami_status(vix_df: pd.DataFrame):
    """
    Tsunami panel based on current 'tsunami' flag series.
    """
    if vix_df.empty or "tsunami" not in vix_df.columns:
        return "⚪", "No Tsunami reading", "Insufficient VIX / VVIX data."

    tsunami_series = vix_df["tsunami"].dropna()
    if tsunami_series.empty:
        return "🟢", "No Tsunami in window", "No recent Tsunami compression signals."

    tsunami_dates = tsunami_series[tsunami_series].index
    if len(tsunami_dates) == 0:
        return "🟢", "No Tsunami in window", "No recent Tsunami compression signals."

    last_date = tsunami_dates[-1]
    days_ago = (datetime.today().date() - last_date.date()).days

    emoji = "🟡"
    headline = "Tsunami compression active"
    detail = f"Last Tsunami compression signal on {last_date.date()} ({days_ago} days ago)."
    return emoji, headline, detail


def market_snapshot(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, vix_df: pd.DataFrame) -> dict:
    out: dict[str, str] = {}

    # SPY
    if not spy_df.empty:
        last_spy = float(spy_df["Close"].iloc[-1])
        spy_off = pct_off_high(spy_df["Close"])
        out["SPY_price"] = f"{last_spy:,.2f}"
        out["SPY_off"] = f"{spy_off:.1f}%" if spy_off is not None else "n/a"
    else:
        out["SPY_price"] = "n/a"
        out["SPY_off"] = "n/a"

    # QQQ
    if not qqq_df.empty:
        last_qqq = float(qqq_df["Close"].iloc[-1])
        qqq_off = pct_off_high(qqq_df["Close"])
        out["QQQ_price"] = f"{last_qqq:,.2f}"
        out["QQQ_off"] = f"{qqq_off:.1f}%" if qqq_off is not None else "n/a"
    else:
        out["QQQ_price"] = "n/a"
        out["QQQ_off"] = "n/a"

    # VIX
    if not vix_df.empty:
        last_vix = float(vix_df["VIX"].iloc[-1])
        out["VIX_level"] = f"{last_vix:.2f}"
    else:
        out["VIX_level"] = "n/a"

    return out


def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str) -> go.Figure:
    """
    Build SPY/QQQ price chart using full-history EMAs but only
    plotting the most recent PRICE_WINDOW_DAYS.
    """
    fig = go.Figure()

    if df_full.empty:
        fig.update_layout(title=title)
        return fig

    last_date = df_full.index.max()
    recent_start = last_date - pd.Timedelta(days=PRICE_WINDOW_DAYS)
    recent = df_full[df_full.index >= recent_start].copy()

    if recent.empty:
        recent = df_full.tail(60).copy()

    # Lines
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            mode="lines",
            name=price_label,
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

    # Auto-fit Y with a touch of padding
    stack = recent[["Close", "ema_21", "ema_200"]].dropna(how="all")
    if not stack.empty:
        ymin = float(stack.min().min())
        ymax = float(stack.max().max())
        pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_layout(
        title=title,
        showlegend=True,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def build_vix_chart(vix_full: pd.DataFrame, timeframe: str = "Daily") -> go.Figure:
    """
    VIX / VVIX Tsunami chart with daily/weekly toggle.
    """
    if vix_full.empty:
        return go.Figure()

    if timeframe == "Weekly":
        # Weekly (Friday) view
        idx = vix_full.index
        df = pd.DataFrame(
            index=vix_full["VIX"].resample("W-FRI").last().index
        )
        df["VIX"] = vix_full["VIX"].resample("W-FRI").last()
        df["vix_sd20"] = vix_full["vix_sd20"].resample("W-FRI").last()
        df["vvix_sd20"] = vix_full["vvix_sd20"].resample("W-FRI").last()
        df["tsunami"] = vix_full["tsunami"].resample("W-FRI").max().astype(bool)
    else:
        df = vix_full.copy()

    df = df.dropna(subset=["VIX"])

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
            y=df["vix_sd20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["vvix_sd20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    # Tsunami compression markers
    if "tsunami" in df.columns:
        mask = df["tsunami"].fillna(False)
        comp_dates = df.index[mask]
        if len(comp_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=comp_dates,
                    y=df.loc[comp_dates, "VIX"],
                    mode="markers",
                    name="Tsunami compression",
                    marker=dict(symbol="diamond", size=8),
                )
            )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        showlegend=True,
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Market Risk Dashboard",
        layout="wide",
    )

    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # ----- Data -----
    try:
        spy_full, qqq_full, vix_full = load_history()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # ----- Top status panels -----
    can_col, tsu_col, snap_col = st.columns(3)

    # Canary
    with can_col:
        st.subheader("Canary Status")
        can_emoji, can_headline, can_detail = canary_status(spy_full)
        st.markdown(f"{can_emoji} **{can_headline}**")
        st.markdown(can_detail)

    # Tsunami
    with tsu_col:
        st.subheader("Tsunami Status")
        tsu_emoji, tsu_headline, tsu_detail = tsunami_status(vix_full)
        st.markdown(f"{tsu_emoji} **{tsu_headline}**")
        st.markdown(tsu_detail)

    # Market snapshot
    with snap_col:
        st.subheader("Market Snapshot")
        snap = market_snapshot(spy_full, qqq_full, vix_full)
        st.markdown(f"**SPY:** {snap['SPY_price']}")
        st.markdown(f"Off 1-year high: {snap['SPY_off']}")
        st.markdown("---")
        st.markdown(f"**QQQ:** {snap['QQQ_price']}")
        st.markdown(f"Off 1-year high: {snap['QQQ_off']}")
        st.markdown("---")
        st.markdown(f"**VIX:** {snap['VIX_level']}")

    st.markdown("---")

    # ----- SPY & QQQ charts -----
    st.subheader("SPY & QQQ with 5% Canary Signals (3-month view)")
    col_spy, col_qqq = st.columns(2)

    with col_spy:
        spy_fig = build_price_chart(spy_full, "SPY with EMAs", "SPY Price")
        st.plotly_chart(spy_fig, use_container_width=True)

    with col_qqq:
        qqq_fig = build_price_chart(qqq_full, "QQQ (NASDAQ) with EMAs", "QQQ Price")
        st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")

    # ----- VIX Tsunami chart -----
    st.subheader("VIX & Volatility Tsunami Watch")

    tf_col1, _ = st.columns([1, 5])
    with tf_col1:
        timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)

    vix_fig = build_vix_chart(vix_full, timeframe=timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)

    st.caption(
        "Lines: VIX level, VIX 20-day stdev, VVIX 20-day stdev. "
        "Diamonds mark Tsunami compression signals."
    )


if __name__ == "__main__":
    main()
