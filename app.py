import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import altair as alt

# -----------------------------
# Config
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
    data = yf.download(tickers, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    return data

end_date = dt.date.today()
start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=365*2))

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
def compute_canary(spy_close, fast_bars=15, confirm_window=42, lookback_days=252, long_ma_len=200):
    df = pd.DataFrame({"close": spy_close})
    df["long_ma"] = df["close"].rolling(long_ma_len).mean()
    df["roll_high"] = df["close"].rolling(lookback_days).max()
    df["drawdown"] = df["close"] / df["roll_high"] - 1.0

    # Track bars since last 52-week high
    bars_since_peak = []
    last_peak = None
    count = 0
    for price, rh in zip(df["close"], df["roll_high"]):
        if price == rh:
            count = 0
        else:
            count += 1
        bars_since_peak.append(count)
    df["bars_since_peak"] = bars_since_peak

    # Detect first cross below -5%
    threshold = -0.05
    df["five_pct_cross"] = (df["drawdown"] <= threshold) & (df["drawdown"].shift(1) > threshold)

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
                df.at[df.index[i], "fast_canary"] = True
                active_fast = True
                bars_since_event = 0
                below_long_count = 0
            else:
                df.at[df.index[i], "slow_canary"] = True
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
                df.at[df.index[i], "confirmed_canary"] = True
                active_fast = False
            elif bars_since_event > confirm_window:
                active_fast = False

    return df

canary_df = compute_canary(spy)

# Determine latest canary state
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
def compute_tsunami(vix_close, vvix_close, sd_len=20, th_vix=0.86, th_vvix=3.16, min_gap=10):
    df = pd.DataFrame({"vix": vix_close, "vvix": vvix_close}).dropna()
    df["vix_sd"] = df["vix"].rolling(sd_len).std()
    df["vvix_sd"] = df["vvix"].rolling(sd_len).std()

    df["vix_below"] = df["vix_sd"] <= th_vix
    df["vvix_below"] = df["vvix_sd"] <= th_vvix

    # De-cluster: require previous min_gap days to be above threshold
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
    st.write(f"Off 52w high: {dd:.1f}%")
    st.write(f"VIX: {vix.iloc[-1]:.2f}")

st.markdown("---")

# -----------------------------
# S&P 500 + Canary Chart
# -----------------------------
st.subheader("S&P 500 with 5% Canary Signals")

plot_df = pd.DataFrame({
    "Date": spy.index,
    "SPY": spy.values,
})
plot_df = plot_df.set_index("Date").join(canary_df[["slow_canary", "fast_canary", "confirmed_canary"]])

base = alt.Chart(plot_df.reset_index()).encode(x="Date:T")

price_line = base.mark_line(color="white").encode(y="SPY:Q")

slow_pts = base.mark_point(color="green", size=60).encode(
    y="SPY:Q",
    tooltip=["Date:T", "SPY:Q"]
).transform_filter("datum.slow_canary == true")

fast_pts = base.mark_point(color="orange", size=60, shape="triangle-up").encode(
    y="SPY:Q",
    tooltip=["Date:T", "SPY:Q"]
).transform_filter("datum.fast_canary == true")

conf_pts = base.mark_point(color="red", size=80, shape="diamond").encode(
    y="SPY:Q",
    tooltip=["Date:T", "SPY:Q"]
).transform_filter("datum.confirmed_canary == true")

st.altair_chart(price_line + slow_pts + fast_pts + conf_pts, use_container_width=True)

# -----------------------------
# VIX + Tsunami Chart
# -----------------------------
st.subheader("VIX & Volatility Tsunami Watch")

vix_plot_df = pd.DataFrame({
    "Date": vix.index,
    "VIX": vix.values
}).set_index("Date").join(tsu_df[["vix_sd", "vvix_sd", "vix_signal", "vvix_signal", "tsunami"]])

base_vix = alt.Chart(vix_plot_df.reset_index()).encode(x="Date:T")
vix_line = base_vix.mark_line(color="white").encode(y="VIX:Q")

vix_sd_line = base_vix.mark_line(color="cyan").encode(y="vix_sd:Q")
vvix_sd_line = base_vix.mark_line(color="yellow").encode(y="vvix_sd:Q")

tsu_pts = base_vix.mark_point(color="red", size=80, shape="diamond").encode(
    y="VIX:Q",
    tooltip=["Date:T", "VIX:Q"]
).transform_filter("datum.tsunami == true")

st.altair_chart(vix_line + tsu_pts, use_container_width=True)
st.caption("White: VIX. Red diamonds: Tsunami compression signals (VIX + VVIX SD).")

# -----------------------------
# SPY / TLT Ratio
# -----------------------------
st.subheader("SPY vs TLT (Risk-On / Risk-Off)")

ratio_df = (spy / tlt).dropna()
ratio_chart = alt.Chart(
    pd.DataFrame({"Date": ratio_df.index, "SPY_TLT": ratio_df.values})
).mark_line().encode(x="Date:T", y="SPY_TLT:Q")

st.altair_chart(ratio_chart, use_container_width=True)

# -----------------------------
# BTC / Gold Ratio
# -----------------------------
st.subheader("Bitcoin vs Gold")

btc_gold = (btc / gld).dropna()
btc_gold_chart = alt.Chart(
    pd.DataFrame({"Date": btc_gold.index, "BTC_Gold": btc_gold.values})
).mark_line().encode(x="Date:T", y="BTC_Gold:Q")

st.altair_chart(btc_gold_chart, use_container_width=True)

st.markdown("---")
st.caption("v1 – core risk framework. Next iterations: breadth, sector regimes, AI trend overlays.")
