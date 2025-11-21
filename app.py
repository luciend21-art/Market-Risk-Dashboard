import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import altair as alt

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Lucien Market Risk Dashboard",
    layout="wide"
)

st.title("Lucien Market Risk Dashboard")
st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data

end_date = dt.date.today()
default_start = end_date - dt.timedelta(days=365 * 2)

start_date = st.sidebar.date_input("Start Date", default_start)

tickers = ["SPY", "^VIX", "^VVIX", "TLT", "BTC-USD", "GLD"]
prices = load_prices(tickers, start_date, end_date).dropna(how="all")

spy = prices["SPY"].dropna()
vix = prices["^VIX"].dropna()
vvix = prices["^VVIX"].dropna()
tlt = prices["TLT"].dropna()
btc = prices["BTC-USD"].dropna()
gld = prices["GLD"].dropna()

# -----------------------------
# 5% Canary Logic
# -----------------------------
def compute_canary(spy_close,
                   fast_bars=15,
                   confirm_window=42,
                   lookback_days=252,
                   long_ma_len=200):

    df = pd.DataFrame({"close": spy_close})
    df["long_ma"] = df["close"].rolling(long_ma_len).mean()
    df["roll_high"] = df["close"].rolling(lookback_days).max()
    df["drawdown"] = df["close"] / df["roll_high"] - 1.0

    # Bars since last 52-week high
    bars_since_peak = []
    count = 0
    for price, rh in zip(df["close"], df["roll_high"]):
        if price == rh:
            count = 0
        else:
            count += 1
        bars_since_peak.append(count)
    df["bars_since_peak"] = bars_since_peak

    # First cross below -5%
    threshold = -0.05
    df["five_pct_cross"] = (df["drawdown"] <= threshold) & (
        df["drawdown"].shift(1) > threshold
    )

    df["slow_canary"] = False
    df["fast_canary"] = False
    df["confirmed_canary"] = False

    active_fast = False
    bars_since_event = 0
    below_long_count = 0

    for i in range(len(df)):
        if df["five_pct_cross"].iloc[i]:
            bars_since_peak_i = df["bars_since_peak"].iloc[i]
            if bars_since_peak_i <= fast_bars:
                df.iat[i, df.columns.get_loc("fast_canary")] = True
                active_fast = True
                bars_since_event = 0
                below_long_count = 0
            else:
                df.iat[i, df.columns.get_loc("slow_canary")] = True
                active_fast = False
                bars_since_event = 0
                below_long_count = 0
            continue

        if active_fast:
            bars_since_event += 1
            if df["close"].iloc[i] < df["long_ma"].iloc[i]:
                below_long_count += 1
            else:
                below_long_count = 0

            if below_long_count >= 2 and bars_since_event <= confirm_window:
                df.iat[i, df.columns.get_loc("confirmed_canary")] = True
                active_fast = False
            elif bars_since_event > confirm_window:
                active_fast = False

    return df


canary_df = compute_canary(spy)

# Latest Canary state
last_row = canary_df.dropna().iloc[-1]
if last_row["confirmed_canary"]:
    canary_state = "CONFIRMED CANARY (Red)"
elif last_row["fast_canary"]:
    canary_state = "Fast 5% Canary (Yellow)"
elif last_row["slow_canary"]:
    canary_state = "Slow 5% Canary (Buy-the-Dip)"
else:
    canary_state = "No active Canary signal"

# -----------------------------
# Volatility Tsunami Logic
# -----------------------------
def compute_tsunami(vix_close,
                    vvix_close,
                    sd_len=20,
                    th_vix=0.86,
                    th_vvix=3.16,
                    min_gap=10):

    df = pd.DataFrame({"vix": vix_close, "vvix": vvix_close}).dropna()
    df["vix_sd"] = df["vix"].rolling(sd_len).std()
    df["vvix_sd"] = df["vvix"].rolling(sd_len).std()

    df["vix_below"] = df["vix_sd"] <= th_vix
    df["vvix_below"] = df["vvix_sd"] <= th_vvix

    df["vix_prev_min"] = df["vix_sd"].shift(1).rolling(min_gap).min()
    df["vvix_prev_min"] = df["vvix_sd"].shift(1).rolling(min_gap).min()

    df["vix_signal"] = df["vix_below"] & (df["vix_prev_min"] > th_vix)
    df["vvix_signal"] = df["vvix_below"] & (df["vvix_prev_min"] > th_vvix)

    df["tsunami"] = df["vix_signal"] & df["vvix_signal"]
    return df


tsu_df = compute_tsunami(vix, vvix)

if len(tsu_df.dropna()) > 0 and tsu_df["tsunami"].any():
    last_tsu_date = tsu_df[tsu_df["tsunami"]].index[-1].date()
    tsunami_state = f"Tsunami WARNING (last: {last_tsu_date})"
else:
    tsunami_state = "No active Tsunami signal"

# -----------------------------
# Top Summary Row
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Canary Status")
    st.write(canary_state)

with col2:
    st.subheader("Tsunami Status")
    st.write(tsunami_state)

with col3:
    st.subheader("Market Snapshot")
    recent = spy.iloc[-1]
    hi_252 = spy.rolling(252).max().iloc[-1]
    dd = (recent / hi_252 - 1.0) * 100
    st.write(f"SPY: {recent:.2f}")
    st.write(f"Off 52-week high: {dd:.1f}%")
    st.write(f"VIX: {vix.iloc[-1]:.2f}")

st.markdown("---")

# -----------------------------
# S&P 500 + Canary Chart
# -----------------------------
st.subheader("S&P 500 with 5% Canary Signals")

plot_df = pd.DataFrame({"Date": spy.index, "SPY": spy.values})
plot_df = plot_df.set_index("Date").join(
    canary_df[["slow_canary", "fast_canary", "confirmed_canary"]]
)

# Ensure numeric and clean
plot_df["SPY"] = pd.to_numeric(plot_df["SPY"], errors="coerce")
plot_df = plot_df.dropna(subset=["SPY"])

base = alt.Chart(plot_df.reset_index()).properties(height=350)

price_line = base.mark_line(color="#4CC9F0", strokeWidth=2).encode(
    x="Date:T",
    y=alt.Y("SPY:Q", title="SPY Price")
)

slow_pts = base.mark_point(color="green", size=70).encode(
    x="Date:T",
    y="SPY:Q",
).transform_filter("datum.slow_canary == true")

fast_pts = base.mark_point(color="orange", size=90, shape="triangle-up").encode(
    x="Date:T",
    y="SPY:Q",
).transform_filter("datum.fast_canary == true")

conf_pts = base.mark_point(color="red", size=120, shape="diamond").encode(
    x="Date:T",
    y="SPY:Q",
).transform_filter("datum.confirmed_canary == true")

st.altair_chart(price_line + slow_pts + fast_pts + conf_pts,
                use_container_width=True)

# -----------------------------
# VIX + Tsunami Chart
# -----------------------------
st.subheader("VIX & Volatility Tsunami Watch")

vix_plot_df = pd.DataFrame({"Date": vix.index, "VIX": vix.values})
vix_plot_df = vix_plot_df.set_index("Date").join(
    tsu_df[["vix_sd", "vvix_sd", "vix_signal", "vvix_signal", "tsunami"]]
)

vix_plot_df["VIX"] = pd.to_numeric(vix_plot_df["VIX"], errors="coerce")
vix_plot_df["vix_sd"] = pd.to_numeric(vix_plot_df["vix_sd"], errors="coerce")
vix_plot_df["vvix_sd"] = pd.to_numeric(vix_plot_df["vvix_sd"], errors="coerce")
vix_plot_df = vix_plot_df.dropna(subset=["VIX"])

base_vix = alt.Chart(vix_plot_df.reset_index()).properties(height=350)

# Main VIX line (use bright color, not white)
vix_line = base_vix.mark_line(color="#4CC9F0", strokeWidth=2).encode(
    x="Date:T",
    y=alt.Y("VIX:Q", title="VIX")
)

# Optional: show VIX and VVIX 20d stdev for context
vix_sd_line = base_vix.mark_line(color="#F9C74F", strokeDash=[4, 4]).encode(
    x="Date:T",
    y=alt.Y("vix_sd:Q", title="VIX / SD"),
)

vvix_sd_line = base_vix.mark_line(color="#90BE6D", strokeDash=[2, 4]).encode(
    x="Date:T",
    y="vvix_sd:Q",
)

# Tsunami points on the VIX line
tsu_pts = base_vix.mark_point(color="red", size=120, shape="diamond").encode(
    x="Date:T",
    y="VIX:Q",
).transform_filter("datum.tsunami == true")

st.altair_chart(vix_line + vix_sd_line + vvix_sd_line + tsu_pts,
                use_container_width=True)

st.caption(
    "Blue: VIX. Yellow/Green: 20-day stdev of VIX/VVIX. "
    "Red diamonds: Volatility Tsunami compression signals."
)

# -----------------------------
# SPY / TLT Ratio
# -----------------------------
st.subheader("SPY vs TLT (Risk-On / Risk-Off)")

ratio_df = (spy / tlt).dropna()
ratio_df = pd.DataFrame({"Date": ratio_df.index, "SPY_TLT": ratio_df.values})

ratio_chart = alt.Chart(ratio_df).properties(height=250).mark_line().encode(
    x="Date:T",
    y=alt.Y("SPY_TLT:Q", title="SPY / TLT")
)

st.altair_chart(ratio_chart, use_container_width=True)

# -----------------------------
# BTC / Gold Ratio
# -----------------------------
st.subheader("Bitcoin vs Gold")

btc_gold = (btc / gld).dropna()
btc_gold_df = pd.DataFrame({"Date": btc_gold.index, "BTC_Gold": btc_gold.values})

btc_gold_chart = alt.Chart(btc_gold_df).properties(height=250).mark_line().encode(
    x="Date:T",
    y=alt.Y("BTC_Gold:Q", title="BTC / Gold")
)

st.altair_chart(btc_gold_chart, use_container_width=True)

st.markdown("---")
st.caption("v1 – Core risk framework. Future: breadth, sector regimes, and AI-optimized trend overlays.")
