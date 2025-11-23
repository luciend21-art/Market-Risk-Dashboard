import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """
    Download ~3 years of daily history for SPY, QQQ, VIX and VVIX.
    Keep the full history for EMAs and volatility calculations,
    but we'll only plot a recent window.
    """
    end = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=365 * 3)

    def fetch(symbol: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
        return df

    spy = fetch("SPY")
    qqq = fetch("QQQ")
    vix = fetch("^VIX")
    vvix = fetch("AVVIX")  # VVIX index on Yahoo

    return spy, qqq, vix, vvix


# ---------------------------------------------------------
# Helper calculations
# ---------------------------------------------------------
def pct_off_high(series: pd.Series, lookback_days: int = 365) -> float:
    """Percent off the highest close in the last `lookback_days` calendar days."""
    series = series.dropna()
    if series.empty:
        return np.nan

    last_date = series.index[-1]
    cutoff = last_date - pd.Timedelta(days=lookback_days)
    window = series[series.index >= cutoff]
    if window.empty:
        window = series

    high = float(window.max())
    last = float(series.iloc[-1])
    if high == 0:
        return np.nan
    return (last / high - 1.0) * 100.0


def canary_status(spy_df: pd.DataFrame):
    """
    Simple Canary 'flashlight' based on SPY % off 1-year high.
    We can plug in the full 5% Canary logic later, but this gives
    a stable, interpretable regime signal now.
    """
    closes = spy_df["Close"].dropna()
    if closes.empty or len(closes) < 60:
        return (
            "⚪",
            "No Canary reading",
            "Insufficient history for a 1-year lookback.",
            np.nan,
        )

    pct_off = pct_off_high(closes, lookback_days=365)

    if np.isnan(pct_off):
        return (
            "⚪",
            "No Canary reading",
            "Unable to compute 1-year high.",
            pct_off,
        )

    if pct_off >= -5:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
    elif pct_off >= -10:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% below 1-year high)"
    else:
        emoji = "🔴"
        headline = "Deep correction (>10% below 1-year high)"

    detail = f"SPY is {pct_off:.1f}% below its 1-year high."
    return emoji, headline, detail, pct_off


def build_price_chart(df_full: pd.DataFrame, title: str, price_label: str) -> alt.Chart:
    """
    Build a 3-month price chart with 21-day and 200-day EMAs.
    df_full is the full daily history; we compute EMAs on full
    history, then slice the most recent window for plotting.
    """
    if df_full.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "Price": []})).mark_line()

    df = df_full.copy()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    days_window = 90
    df_plot = df.tail(days_window).reset_index().rename(columns={"Date": "Date"})

    base = alt.Chart(df_plot).encode(x=alt.X("Date:T", title="Date"))

    price_line = base.mark_line().encode(
        y=alt.Y("Close:Q", title=price_label),
        color=alt.value("#1f77b4"),
        tooltip=["Date:T", "Close:Q", "EMA_21:Q", "EMA_200:Q"],
    )

    ema21_line = base.mark_line(strokeDash=[5, 3]).encode(
        y="EMA_21:Q",
        color=alt.value("#ffbf00"),
    )

    ema200_line = base.mark_line(strokeDash=[2, 2]).encode(
        y="EMA_200:Q",
        color=alt.value("#2ca02c"),
    )

    chart = (
        alt.layer(price_line, ema21_line, ema200_line)
        .resolve_scale(y="shared")
        .properties(title=title, height=260)
    )

    chart = chart.configure_legend(orient="bottom")

    return chart


def build_vix_df(vix_df: pd.DataFrame, vvix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align VIX and VVIX, compute 20-day rolling standard deviations.
    Returns a single DataFrame with VIX, VIX_SD20, VVIX_SD20.
    """
    if vix_df.empty or vvix_df.empty:
        return pd.DataFrame(columns=["VIX", "VIX_SD20", "VVIX_SD20"])

    idx = vix_df.index.union(vvix_df.index)
    df = pd.DataFrame(index=idx)
    df["VIX"] = vix_df["Close"].reindex(idx).interpolate()
    df["VVIX"] = vvix_df["Close"].reindex(idx).interpolate()

    df["VIX_SD20"] = df["VIX"].rolling(20).std()
    df["VVIX_SD20"] = df["VVIX"].rolling(20).std()

    return df.dropna(subset=["VIX"])


def tsunami_status(vix_df: pd.DataFrame, vvix_df: pd.DataFrame, window_days: int = 120):
    """
    Simple Tsunami compression signal:
    - Compute 20-day stdev of VIX & VVIX
    - Flag 'compression' when both stdevs are in the lower quartile
      of their trailing 1-year distribution.
    - Look for the last such signal in the recent `window_days`.
    """
    df = build_vix_df(vix_df, vvix_df)
    if df.empty:
        return (
            "⚪",
            "No Tsunami reading",
            "Insufficient VIX / VVIX history.",
            None,
        )

    # Baseline over last 1 year of data
    if len(df) < 252:
        baseline = df
    else:
        baseline = df.tail(252)

    vix_thresh = baseline["VIX_SD20"].quantile(0.25)
    vvix_thresh = baseline["VVIX_SD20"].quantile(0.25)

    cond = (df["VIX_SD20"] < vix_thresh) & (df["VVIX_SD20"] < vvix_thresh)

    recent_cutoff = df.index[-1] - pd.Timedelta(days=window_days)
    recent_cond = cond & (df.index >= recent_cutoff)

    if recent_cond.any():
        last_idx = recent_cond[recent_cond].index[-1]
        days_ago = (df.index[-1] - last_idx).days

        if days_ago <= 60:
            emoji = "🔴"
        else:
            emoji = "🟡"

        headline = "Tsunami compression signal detected"
        detail = f"Last compression signal on {last_idx.date().isoformat()} ({days_ago} days ago)."
        return emoji, headline, detail, last_idx

    emoji = "🟢"
    headline = "No Tsunami in window"
    detail = f"No compression signal in the last {window_days} days."
    return emoji, headline, detail, None


def build_vix_chart(vix_df: pd.DataFrame, vvix_df: pd.DataFrame, timeframe: str) -> alt.Chart:
    """
    VIX chart with 20-day (or 20-week) stdevs for VIX & VVIX.
    timeframe: "Daily" or "Weekly"
    """
    df = build_vix_df(vix_df, vvix_df)
    if df.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "VIX": []})).mark_line()

    if timeframe == "Weekly":
        # Resample to weekly (Friday close)
        df_w = pd.DataFrame()
        df_w["VIX"] = df["VIX"].resample("W-FRI").last()
        df_w["VIX_SD20"] = df["VIX_SD20"].resample("W-FRI").last()
        df_w["VVIX_SD20"] = df["VVIX_SD20"].resample("W-FRI").last()
        df_plot = df_w.dropna().reset_index().rename(columns={"Date": "Date"})
        title_suffix = " (Weekly)"
    else:
        # Daily – show ~1.5 years for context
        df_plot = df.tail(380).reset_index().rename(columns={"Date": "Date"})
        title_suffix = " (Daily)"

    base = alt.Chart(df_plot).encode(x=alt.X("Date:T", title="Date"))

    vix_line = base.mark_line(color="#1f77b4").encode(
        y=alt.Y("VIX:Q", title="VIX / 20-period stdev")
    )

    vix_sd = base.mark_line(strokeDash=[4, 2], color="#2ca02c").encode(
        y="VIX_SD20:Q"
    )

    vvix_sd = base.mark_line(strokeDash=[2, 2], color="#ff7f0e").encode(
        y="VVIX_SD20:Q"
    )

    chart = (
        alt.layer(vix_line, vix_sd, vvix_sd)
        .resolve_scale(y="shared")
        .properties(
            title="VIX & Volatility Tsunami Watch" + title_suffix,
            height=260,
        )
    )

    chart = chart.configure_legend(orient="bottom")

    return chart


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # ---- Load data ----
    try:
        spy_full, qqq_full, vix_full, vvix_full = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # ---- Status panels ----
    can_emoji, can_headline, can_detail, spy_off = canary_status(spy_full)
    tsu_emoji, tsu_headline, tsu_detail, tsu_last = tsunami_status(vix_full, vvix_full)

    # Snapshot metrics
    spy_price = float(spy_full["Close"].iloc[-1]) if not spy_full.empty else np.nan
    qqq_price = float(qqq_full["Close"].iloc[-1]) if not qqq_full.empty else np.nan
    spy_off_high = pct_off_high(spy_full["Close"], lookback_days=365)
    qqq_off_high = pct_off_high(qqq_full["Close"], lookback_days=365)
    vix_last = float(vix_full["Close"].iloc[-1]) if not vix_full.empty else np.nan

    # Simple "breadth proxy": index above its 50-day EMA or not
    def breadth_proxy(df: pd.DataFrame) -> str:
        if df.empty:
            return "n/a"
        close = df["Close"]
        ema50 = close.ewm(span=50, adjust=False).mean()
        signal = float(close.iloc[-1]) > float(ema50.iloc[-1])
        return "100% (price above 50-EMA)" if signal else "0% (price below 50-EMA)"

    spy_breadth = breadth_proxy(spy_full)
    qqq_breadth = breadth_proxy(qqq_full)

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
        if not np.isnan(spy_price):
            st.markdown(f"**SPY:** {spy_price:.2f}")
            if not np.isnan(spy_off_high):
                st.markdown(f"Off 1-year high: {spy_off_high:.1f}%")
        else:
            st.write("SPY: n/a")

        if not np.isnan(qqq_price):
            st.markdown(f"**QQQ:** {qqq_price:.2f}")
            if not np.isnan(qqq_off_high):
                st.markdown(f"Off 1-year high: {qqq_off_high:.1f}%")
        else:
            st.write("QQQ: n/a")

        if not np.isnan(vix_last):
            st.markdown(f"**VIX:** {vix_last:.2f}")
        else:
            st.write("VIX: n/a")

        st.markdown("---")
        st.markdown("**Breadth proxy (above 50-EMA):**")
        st.write(f"SPY: {spy_breadth}")
        st.write(f"QQQ: {qqq_breadth}")

    st.markdown("---")

    # ---- SPY & QQQ price charts ----
    st.markdown(
        f"### SPY with 5% Canary Signals &nbsp;&nbsp;{can_emoji}",
        unsafe_allow_html=True,
    )
    spy_chart = build_price_chart(spy_full, "", "SPY Price")
    st.altair_chart(spy_chart, use_container_width=True)
    st.caption("Lines: Blue = price, Gold = 21-day EMA, Green = 200-day EMA.")

    st.markdown(
        "### QQQ (NASDAQ) with 5% Canary Signals",
        unsafe_allow_html=True,
    )
    qqq_chart = build_price_chart(qqq_full, "", "QQQ Price")
    st.altair_chart(qqq_chart, use_container_width=True)
    st.caption("Lines: Blue = price, Gold = 21-day EMA, Green = 200-day EMA.")

    st.markdown("---")

    # ---- VIX & Tsunami chart ----
    st.subheader("VIX & Volatility Tsunami Watch")

    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)

    vix_chart = build_vix_chart(vix_full, vvix_full, timeframe=timeframe)
    st.altair_chart(vix_chart, use_container_width=True)
    st.caption(
        "Lines: VIX level, VIX 20-period stdev, VVIX 20-period stdev. "
        "Compression in both stdevs is used for the Tsunami signal."
    )


if __name__ == "__main__":
    main()
