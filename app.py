import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_ohlc(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download OHLC data from yfinance and return as a DataFrame with a Date column."""
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df

    df = df.reset_index()  # make Date a column
    # Standardise column names we care about
    # yfinance already gives "Date", "Open", "High", "Low", "Close", ...
    return df


@st.cache_data(ttl=3600)
def load_data():
    """Load full-history data for SPY, QQQ, VIX, VVIX and compute EMAs & volatility stats."""
    end = datetime.today()
    start_full = end - timedelta(days=365 * 5)  # 5 years of history for EMAs

    spy = fetch_ohlc("SPY", start_full, end)
    qqq = fetch_ohlc("QQQ", start_full, end)
    vix = fetch_ohlc("^VIX", start_full, end)
    vvix = fetch_ohlc("^VVIX", start_full, end)

    # Compute EMAs for SPY & QQQ on the full history
    for df in (spy, qqq):
        if not df.empty:
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # For VIX / VVIX build a combined frame for later stats
    if not vix.empty and not vvix.empty:
        vix_full = (
            vix[["Date", "Close"]]
            .rename(columns={"Close": "VIX"})
            .merge(
                vvix[["Date", "Close"]].rename(columns={"Close": "VVIX"}),
                on="Date",
                how="inner",
            )
            .sort_values("Date")
        )
        vix_full["VIX_SD20"] = vix_full["VIX"].rolling(20).std()
        vix_full["VVIX_SD20"] = vix_full["VVIX"].rolling(20).std()
    else:
        vix_full = pd.DataFrame(columns=["Date", "VIX", "VVIX", "VIX_SD20", "VVIX_SD20"])

    return spy, qqq, vix_full


# ---------------------------------------------------------
# Indicator logic
# ---------------------------------------------------------

def canary_status(df: pd.DataFrame):
    """
    Canary status for an index based on distance from 1-year high.

    Returns: emoji, headline, detail, color
    """
    if df.empty:
        return "⚪", "No Canary reading", "No data available.", "gray"

    closes = df["Close"].dropna()
    if len(closes) < 252:
        return "⚪", "No Canary reading", "Insufficient data for 1-year high.", "gray"

    last_close = float(closes.iloc[-1])
    high_1y = float(closes.iloc[-252:].max())
    pct_off = (last_close / high_1y - 1.0) * 100.0  # negative means below high

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
        detail = f"Index is {pct_off:.1f}% below its 1-year high."
        color = "green"
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Moderate pullback (5–10% from 1-year high)"
        detail = f"Index is {pct_off:.1f}% below its 1-year high."
        color = "gold"
    else:
        emoji = "🔴"
        headline = "Deep correction (>10% from 1-year high)"
        detail = f"Index is {pct_off:.1f}% below its 1-year high."
        color = "red"

    return emoji, headline, detail, color


def tsunami_status(vix_df: pd.DataFrame, lookback_days: int = 120):
    """
    Simple 'volatility tsunami' compression watch:

    - Compute 20-day stdev of VIX and VVIX.
    - Define compression as both stdevs being in the lowest 25% of their
      historical ranges.
    - Look back 'lookback_days' calendar days for the last compression signal.

    Returns: emoji, headline, detail, color
    """
    if vix_df.empty or vix_df["VIX_SD20"].isna().all():
        return "⚪", "No Tsunami reading", "No volatility data available.", "gray"

    df = vix_df.dropna(subset=["VIX_SD20", "VVIX_SD20"]).copy()
    if df.empty:
        return "⚪", "No Tsunami reading", "No volatility data available.", "gray"

    vix_thresh = df["VIX_SD20"].quantile(0.25)
    vvix_thresh = df["VVIX_SD20"].quantile(0.25)

    df["compression"] = (df["VIX_SD20"] <= vix_thresh) & (df["VVIX_SD20"] <= vvix_thresh)

    # Restrict to recent window
    max_date = df["Date"].max()
    window_start = max_date - timedelta(days=lookback_days)
    recent = df[df["Date"] >= window_start]

    hits = recent[recent["compression"]]
    if hits.empty:
        emoji = "🟢"
        headline = "No Tsunami in window"
        detail = f"No volatility compression signals in the last {lookback_days} days."
        color = "green"
    else:
        last_hit_date = pd.to_datetime(hits["Date"].iloc[-1]).date()
        emoji = "🟡"
        headline = "Tsunami compression watch"
        detail = f"Last compression signal on {last_hit_date.isoformat()}."
        color = "gold"

    return emoji, headline, detail, color


def market_snapshot(spy: pd.DataFrame, qqq: pd.DataFrame):
    """Return a dict of simple snapshot numbers for SPY / QQQ / VIX-proxy."""
    snapshot = {}

    def off_high(df: pd.DataFrame):
        closes = df["Close"].dropna()
        if len(closes) < 252:
            return float("nan")
        last_close = float(closes.iloc[-1])
        high_1y = float(closes.iloc[-252:].max())
        return (last_close / high_1y - 1.0) * 100.0

    if not spy.empty:
        snapshot["SPY_price"] = float(spy["Close"].iloc[-1])
        snapshot["SPY_off_high"] = off_high(spy)
    else:
        snapshot["SPY_price"] = float("nan")
        snapshot["SPY_off_high"] = float("nan")

    if not qqq.empty:
        snapshot["QQQ_price"] = float(qqq["Close"].iloc[-1])
        snapshot["QQQ_off_high"] = off_high(qqq)
    else:
        snapshot["QQQ_price"] = float("nan")
        snapshot["QQQ_off_high"] = float("nan")

    return snapshot


# ---------------------------------------------------------
# Chart builders
# ---------------------------------------------------------

def build_price_chart(df: pd.DataFrame, title: str, price_label: str):
    """Altair chart of last ~3 months with Price, 21-EMA and 200-EMA."""
    if df.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "value": [], "Line": []}))

    max_date = df["Date"].max()
    start_window = max_date - timedelta(days=120)  # ~3–4 calendar months
    recent = df[df["Date"] >= start_window].copy()

    plot_df = recent[["Date", "Close", "EMA21", "EMA200"]].melt(
        "Date", value_vars=["Close", "EMA21", "EMA200"], var_name="Line", value_name="Price"
    )

    line_order = ["Close", "EMA21", "EMA200"]

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title=price_label),
            color=alt.Color(
                "Line:N",
                sort=line_order,
                title="",
                scale=alt.Scale(
                    domain=line_order,
                    range=["#1f77b4", "#ff7f0e", "#2ca02c"],  # blue, orange, green
                ),
            ),
        )
        .properties(title=title, height=260)
        .configure_legend(orient="bottom")
    )

    return chart


def build_vix_chart(vix_df: pd.DataFrame):
    """Altair chart of VIX level + 20-day stdevs of VIX and VVIX over full window."""
    if vix_df.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "value": [], "Series": []}))

    plot_df = vix_df[["Date", "VIX", "VIX_SD20", "VVIX_SD20"]].melt(
        "Date",
        value_vars=["VIX", "VIX_SD20", "VVIX_SD20"],
        var_name="Series",
        value_name="Value",
    )

    order = ["VIX", "VIX_SD20", "VVIX_SD20"]

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="VIX / Volatility"),
            color=alt.Color(
                "Series:N",
                sort=order,
                title="",
                scale=alt.Scale(
                    domain=order,
                    range=["#1f77b4", "#ff7f0e", "#2ca02c"],  # blue, orange, green
                ),
            ),
        )
        .properties(title="VIX & Volatility Tsunami Watch", height=260)
        .configure_legend(orient="bottom")
    )

    return chart


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------

def flashlight(color: str) -> str:
    """Return a colored circle 'flashlight' HTML snippet."""
    return f"<span style='font-size:28px;color:{color}'>●</span>"


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # Load data
    spy_full, qqq_full, vix_full = load_data()

    # ---- Status panels ----
    can_emoji, can_headline, can_detail, can_color = canary_status(spy_full)
    tsu_emoji, tsu_headline, tsu_detail, tsu_color = tsunami_status(vix_full)
    snap = market_snapshot(spy_full, qqq_full)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Canary Status")
        st.markdown(
            f"{flashlight(can_color)} {can_emoji} {can_headline}",
            unsafe_allow_html=True,
        )
        st.caption(can_detail)

    with col2:
        st.subheader("Tsunami Status")
        st.markdown(
            f"{flashlight(tsu_color)} {tsu_emoji} {tsu_headline}",
            unsafe_allow_html=True,
        )
        st.caption(tsu_detail)

    with col3:
        st.subheader("Market Snapshot")
        if not np.isnan(snap["SPY_price"]):
            st.markdown(f"**SPY:** {snap['SPY_price']:.2f}")
            st.markdown(
                f"Off 1-year high: {snap['SPY_off_high']:.1f}%"
                if not np.isnan(snap["SPY_off_high"])
                else "Off 1-year high: n/a"
            )
        else:
            st.markdown("SPY data unavailable.")

        st.markdown("---")
        if not np.isnan(snap["QQQ_price"]):
            st.markdown(f"**QQQ:** {snap['QQQ_price']:.2f}")
            st.markdown(
                f"Off 1-year high: {snap['QQQ_off_high']:.1f}%"
                if not np.isnan(snap["QQQ_off_high"])
                else "Off 1-year high: n/a"
            )
        else:
            st.markdown("QQQ data unavailable.")

    st.markdown("---")

    # ---- SPY & QQQ price panels (3-month window) ----
    st.subheader("SPY & NASDAQ with 5% Canary Framework")

    c1, c2 = st.columns(2)

    with c1:
        spy_chart = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
        st.altair_chart(spy_chart, use_container_width=True)
        st.caption("Blue: Price, Orange: 21-day EMA, Green: 200-day EMA.")

    with c2:
        qqq_chart = build_price_chart(qqq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price")
        st.altair_chart(qqq_chart, use_container_width=True)
        st.caption("Blue: Price, Orange: 21-day EMA, Green: 200-day EMA.")

    st.markdown("---")

    # ---- VIX / Tsunami panel ----
    st.subheader("Volatility Tsunami Watch")

    # Timeframe toggle (Daily / Weekly)
    timeframe = st.radio("VIX timeframe", ["Daily", "Weekly"], horizontal=True)

    if timeframe == "Daily":
        vix_tf = vix_full.copy()
    else:
        # Weekly resample to Friday closes
        if vix_full.empty:
            vix_tf = vix_full.copy()
        else:
            temp = vix_full.set_index("Date")
            weekly = (
                temp[["VIX", "VVIX"]]
                .resample("W-FRI")
                .last()
                .dropna()
                .reset_index()
            )
            weekly["VIX_SD20"] = weekly["VIX"].rolling(20).std()
            weekly["VVIX_SD20"] = weekly["VVIX"].rolling(20).std()
            vix_tf = weekly

    vix_chart = build_vix_chart(vix_tf)
    st.altair_chart(vix_chart, use_container_width=True)
    st.caption("Lines: VIX level, 20-day stdev of VIX, 20-day stdev of VVIX.")

if __name__ == "__main__":
    main()
