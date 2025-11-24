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
def load_closes(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    closes = data["Close"]
    return closes

end_date = dt.date.today()
# same default window as before; you can change to 3–5 yrs if you like
default_start = end_date - dt.timedelta(days=365 * 2)

start_date = st.sidebar.date_input("Start Date", default_start)

tickers = ["SPY", "QQQ", "^VIX", "^VVIX", "TLT", "BTC-USD", "GLD"]
closes = load_closes(tickers, start_date, end_date).dropna(how="all")

spy = closes["SPY"].dropna()
qqq = closes["QQQ"].dropna()
vix = closes["^VIX"].dropna()
vvix = closes["^VVIX"].dropna()
tlt = closes["TLT"].dropna()
btc = closes["BTC-USD"].dropna()
gld = closes["GLD"].dropna()

# -----------------------------
# Helper functions
# -----------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def off_high(series, lookback=252):
    roll_high = series.rolling(lookback).max()
    return (series / roll_high - 1.0) * 100

def pct_above_ema50_proxy(close_series):
    """Simple v1 breadth proxy: 100% if close > 50-EMA, else 0%."""
    ema50 = ema(close_series, 50)
    above = close_series.iloc[-1] > ema50.iloc[-1]
    return 100.0 if above else 0.0

# -----------------------------
# 5% Canary Logic (SPY & QQQ)
# -----------------------------
def compute_canary(close_series,
                   fast_bars=15,
                   confirm_window=42,
                   lookback_days=252,
                   long_ma_len=200):

    df = pd.DataFrame({"close": close_series})
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

spy_canary = compute_canary(spy)
qqq_canary = compute_canary(qqq)

def latest_canary_state(canary_df):
    last_row = canary_df.dropna().iloc[-1]
    if last_row["confirmed_canary"]:
        return "CONFIRMED CANARY (Red)", "red"
    elif last_row["fast_canary"]:
        return "Fast 5% Canary (Yellow)", "orange"
    elif last_row["slow_canary"]:
        return "Slow 5% Canary (Buy-the-Dip)", "green"
    else:
        return "No active Canary signal", "gray"

spy_state_text, spy_state_color = latest_canary_state(spy_canary)

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

last_tsu_date = None
if len(tsu_df.dropna()) > 0 and tsu_df["tsunami"].any():
    last_tsu_date = tsu_df[tsu_df["tsunami"]].index[-1].date()
    tsunami_state_text = f"Tsunami WARNING (last: {last_tsu_date})"
    tsunami_active = True
else:
    tsunami_state_text = "No active Tsunami signal"
    tsunami_active = False

# -----------------------------
# Market Snapshot + Trend/Breadth
# -----------------------------
def trend_label(close_series):
    ema_short = ema(close_series, 21)
    ema_long = ema(close_series, 50)
    ema_200 = ema(close_series, 200)

    c = close_series.iloc[-1]
    s = ema_short.iloc[-1]
    l = ema_long.iloc[-1]
    e200 = ema_200.iloc[-1]

    if c > e200 and s > l > e200:
        return "Strong Uptrend"
    elif c > e200:
        return "Uptrend / Neutral"
    else:
        return "Downtrend"

spy_off = off_high(spy).iloc[-1]
qqq_off = off_high(qqq).iloc[-1]

spy_breadth = pct_above_ema50_proxy(spy)
qqq_breadth = pct_above_ema50_proxy(qqq)

spy_trend = trend_label(spy)
qqq_trend = trend_label(qqq)

# -----------------------------
# Recommendation Engine
# -----------------------------
def recommendation(spy_state, tsunami_on, spy_trend, spy_breadth):
    if "CONFIRMED" in spy_state:
        if tsunami_on:
            regime = "Purple / Red – Confirmed Canary + Tsunami"
            msg = (
                "Defensive posture. 0–25% equity exposure, "
                "hedged if possible. Focus on capital preservation."
            )
        else:
            regime = "Red – Confirmed Canary"
            msg = (
                "Defensive. 0–40% equity exposure. Prioritize risk reduction, "
                "avoid new longs except very selective."
            )
    elif "Fast 5%" in spy_state:
        regime = "Orange – Fast 5% Canary"
        msg = (
            "Caution. 40–70% equity exposure. Tighten stops, "
            "consider partial hedges or profit-taking into strength."
        )
    elif "Slow 5%" in spy_state:
        if spy_breadth >= 50 and "Uptrend" in spy_trend:
            regime = "Yellow-Green – Buy-the-Dip"
            msg = (
                "Constructive environment. 80–100% equity exposure OK "
                "for your playbook. Look for high-quality entries."
            )
        else:
            regime = "Yellow – Mixed"
            msg = (
                "Selective risk. 60–80% exposure. Focus on leading names "
                "and manage risk carefully."
            )
    else:
        if tsunami_on:
            regime = "Purple – Volatility Tsunami Watch"
            msg = (
                "Volatility compression warning. Maintain exposure but "
                "plan hedges and know your exit rules."
            )
        elif "Strong Uptrend" in spy_trend and spy_breadth >= 50:
            regime = "Green – Bullish Regime"
            msg = (
                "Favorable trend & breadth. 90–100% exposure fits the playbook. "
                "Press advantages but respect ATR-based profit-taking."
            )
        elif "Downtrend" in spy_trend:
            regime = "Orange-Red – Weak Trend"
            msg = (
                "Trend is weak. 0–50% exposure. Focus on defense, cash, or hedges "
                "until conditions improve."
            )
        else:
            regime = "Yellow – Neutral / Transition"
            msg = (
                "Transition regime. 50–80% exposure. Be selective with new risk, "
                "wait for clearer trend & breadth."
            )

    return regime, msg

regime_label, regime_msg = recommendation(
    spy_state_text, tsunami_active, spy_trend, spy_breadth
)

def regime_light_from_label(label: str) -> str:
    """Return a colored dot/emoji based on the current regime label."""
    l = label.lower()
    if "purple" in l:
        return "🟣"
    if "red" in l:
        return "🔴"
    if "orange" in l:
        return "🟠"
    if "yellow" in l:
        return "🟡"
    if "green" in l:
        return "🟢"
    return "⚪"

# -----------------------------
# Top Summary Row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Canary Status")
    st.write(spy_state_text)

with col2:
    st.subheader("Tsunami Status")
    st.write(tsunami_state_text)

with col3:
    st.subheader("Market Snapshot")
    st.write(f"SPY: {spy.iloc[-1]:.2f}")
    st.write(f"Off 52-week high: {spy_off:.1f}%")
    st.write(f"QQQ: {qqq.iloc[-1]:.2f}")
    st.write(f"QQQ off 52-week high: {qqq_off:.1f}%")
    st.write(f"VIX: {vix.iloc[-1]:.2f}")

with col4:
    st.subheader("Breadth (Proxy)")
    st.write(f"SPY above 50-EMA: {spy_breadth:.0f}%")
    st.write(f"QQQ above 50-EMA: {qqq_breadth:.0f}%")
    st.write(f"SPY Trend: {spy_trend}")
    st.write(f"QQQ Trend: {qqq_trend}")

st.markdown("---")

# -----------------------------
# Recommendation Panel
# -----------------------------
st.subheader("Regime & Positioning Guidance")
st.markdown(f"**Regime:** {regime_label}")
st.write(regime_msg)

st.markdown("---")

# -----------------------------
# Regime series helper
# -----------------------------
def regime_series(canary_df, tsunami_df=None):
    df = pd.DataFrame(index=canary_df.index)
    df["regime"] = "Normal"

    df.loc[canary_df["slow_canary"], "regime"] = "Slow"
    df.loc[canary_df["fast_canary"], "regime"] = "Fast"
    df.loc[canary_df["confirmed_canary"], "regime"] = "Confirmed"

    if tsunami_df is not None:
        common_idx = df.index.intersection(tsunami_df.index)
        tsu_mask = tsunami_df.loc[common_idx, "tsunami"]
        df.loc[common_idx[tsu_mask], "regime"] = "Tsunami"

    return df

spy_regime = regime_series(spy_canary, tsu_df)

# -----------------------------
# SPY & QQQ Charts (Daily / Weekly)
# -----------------------------
tf_choice = st.radio(
    "Price timeframe for SPY & QQQ charts",
    ["Daily", "Weekly"],
    horizontal=True,
    key="tf_prices"
)

price_scale_choice = st.radio(
    "Price y-axis scale",
    ["Linear", "Log"],
    horizontal=True,
    key="price_scale"
)

# NOTE: new argument cutoff_date is used to slice AFTER EMAs are computed
def make_price_chart(close_series, canary_df, label, scale_choice, cutoff_date=None):
    """
    Build price chart with inline legend and auto-fit y-scale w/ padding.

    EMAs are computed on the FULL `close_series`, then the result is
    optionally sliced to `cutoff_date` for display so EMA math stays accurate.
    """
    df = pd.DataFrame({"close": close_series})
    df["Date"] = df.index

    # Choose EMA lengths based on timeframe selection
    if tf_choice == "Daily":
        df["ema_short"] = ema(df["close"], 21)
        df["ema_long"] = ema(df["close"], 200)
        short_label = "21-day EMA"
        long_label = "200-day EMA"
    else:  # Weekly
        df["ema_short"] = ema(df["close"], 10)
        df["ema_long"] = ema(df["close"], 40)
        short_label = "10-week EMA"
        long_label = "40-week EMA"

    merged = df.set_index("Date").join(
        canary_df[["slow_canary", "fast_canary", "confirmed_canary"]]
    )
    merged = merged.reset_index().rename(columns={"index": "Date"})

    # 🔍 Slice for plotting *after* EMAs are computed
    if cutoff_date is not None:
        merged = merged[merged["Date"] >= cutoff_date]

    # ---- Legend: melt to long form so Altair can create a color legend ----
    line_df = merged.melt(
        id_vars=["Date", "slow_canary", "fast_canary", "confirmed_canary"],
        value_vars=["close", "ema_short", "ema_long"],
        var_name="line",
        value_name="value",
    )

    line_df["line"] = line_df["line"].map({
        "close": "Price",
        "ema_short": short_label,
        "ema_long": long_label,
    })

    # ---- Auto-fit y-scale with small padding ----
    y_min = line_df["value"].min()
    y_max = line_df["value"].max()
    pad = (y_max - y_min) * 0.05 if pd.notnull(y_max) else 0.0

    scale_type = "log" if scale_choice == "Log" else "linear"

    if pad > 0:
        y_domain = [float(y_min - pad), float(y_max + pad)]
        y_scale = alt.Scale(zero=False, type=scale_type, domain=y_domain)
    else:
        y_scale = alt.Scale(zero=False, type=scale_type)

    base_lines = alt.Chart(line_df).properties(height=350)

    color_scale = alt.Scale(
        domain=["Price", short_label, long_label],
        range=["#4CC9F0", "#F9C74F", "#90BE6D"],
    )

    lines = base_lines.mark_line(strokeWidth=2).encode(
        x="Date:T",
        y=alt.Y("value:Q", title=f"{label} Price", scale=y_scale),
        color=alt.Color("line:N", scale=color_scale, title=None),
        tooltip=[
            "Date:T",
            "line:N",
            alt.Tooltip("value:Q", format=".2f"),
        ],
    )

    # Canary points share the same y-scale
    base_pts = alt.Chart(merged).encode(
        x="Date:T",
        y=alt.Y("close:Q", scale=y_scale),
    )

    slow_pts = base_pts.mark_point(color="green", size=60).transform_filter(
        "datum.slow_canary == true"
    )
    fast_pts = base_pts.mark_point(color="orange", size=80, shape="triangle-up").transform_filter(
        "datum.fast_canary == true"
    )
    conf_pts = base_pts.mark_point(color="red", size=100, shape="diamond").transform_filter(
        "datum.confirmed_canary == true"
    )

    chart = lines + slow_pts + fast_pts + conf_pts
    return chart

# Resample for weekly if needed (no slicing yet)
if tf_choice == "Weekly":
    spy_tf = spy.resample("W-FRI").last()
    qqq_tf = qqq.resample("W-FRI").last()

    spy_can_tf = compute_canary(spy_tf)
    qqq_can_tf = compute_canary(qqq_tf)

    spy_regime_tf = regime_series(spy_can_tf, tsu_df)
else:
    spy_tf = spy
    qqq_tf = qqq
    spy_can_tf = spy_canary
    qqq_can_tf = qqq_canary
    spy_regime_tf = spy_regime

# 🔍 Define lookback window for display (EMAs already use full history)
lookback_days = 90
cutoff = pd.Timestamp(end_date - dt.timedelta(days=lookback_days))

# Regime light
light = regime_light_from_label(regime_label)

# SPY chart
spy_col1, spy_col2 = st.columns([4, 1])
with spy_col1:
    st.subheader("SPY with 5% Canary Signals")
with spy_col2:
    st.markdown(f"### {light}")

st.altair_chart(
    make_price_chart(spy_tf, spy_can_tf, "SPY", price_scale_choice, cutoff),
    use_container_width=True
)

if tf_choice == "Daily":
    st.caption("SPY: Blue = price, Yellow = 21-day EMA, Green = 200-day EMA.")
else:
    st.caption("SPY: Blue = price, Yellow = 10-week EMA, Green = 40-week EMA.")

# QQQ chart
qqq_regime = regime_series(qqq_can_tf, tsu_df)

qqq_col1, qqq_col2 = st.columns([4, 1])
with qqq_col1:
    st.subheader("QQQ (NASDAQ) with 5% Canary Signals")
with qqq_col2:
    st.markdown(f"### {light}")

st.altair_chart(
    make_price_chart(qqq_tf, qqq_can_tf, "QQQ", price_scale_choice, cutoff),
    use_container_width=True
)

if tf_choice == "Daily":
    st.caption("QQQ: Blue = price, Yellow = 21-day EMA, Green = 200-day EMA.")
else:
    st.caption("QQQ: Blue = price, Yellow = 10-week EMA, Green = 40-week EMA.")

st.markdown("---")

# -----------------------------
# VIX & Volatility Tsunami Watch (Daily / Weekly)
# -----------------------------
vix_tf_choice = st.radio(
    "VIX timeframe",
    ["Daily", "Weekly"],
    horizontal=True,
    key="tf_vix"
)

if vix_tf_choice == "Weekly":
    vix_tf = vix.resample("W-FRI").last()
    vvix_tf = vvix.resample("W-FRI").last()
    tsu_tf = compute_tsunami(vix_tf, vvix_tf)
else:
    vix_tf = vix
    vvix_tf = vvix
    tsu_tf = tsu_df

vix_col1, vix_col2 = st.columns([4, 1])
with vix_col1:
    st.subheader("VIX & Volatility Tsunami Watch")
with vix_col2:
    # Flashlight style indicator for Tsunami status
    if tsunami_active:
        light_vix = "🟣"
        label_vix = f"Last Tsunami: {last_tsu_date}" if last_tsu_date else "Tsunami active"
    else:
        light_vix = "🟢"
        label_vix = "No Tsunami in window"
    st.markdown(f"### {light_vix}")
    st.caption(label_vix)

vix_plot_df = pd.DataFrame({"Date": vix_tf.index, "VIX": vix_tf.values})
tsu_join = tsu_tf.reindex(vix_plot_df["Date"]).reset_index(drop=True)
vix_plot_df["vix_sd"] = tsu_join["vix_sd"].values
vix_plot_df["vvix_sd"] = tsu_join["vvix_sd"].values
vix_plot_df["tsunami"] = tsu_join["tsunami"].fillna(False).values

vix_plot_df["VIX"] = pd.to_numeric(vix_plot_df["VIX"], errors="coerce")
vix_plot_df["vix_sd"] = pd.to_numeric(vix_plot_df["vix_sd"], errors="coerce")
vix_plot_df["vvix_sd"] = pd.to_numeric(vix_plot_df["vvix_sd"], errors="coerce")
vix_plot_df = vix_plot_df.dropna(subset=["VIX"])

base_vix = alt.Chart(vix_plot_df).properties(height=350)

vix_line = base_vix.mark_line(color="#4CC9F0", strokeWidth=2).encode(
    x="Date:T",
    y=alt.Y("VIX:Q", title="VIX", scale=alt.Scale(zero=False))
)

vix_sd_line = base_vix.mark_line(color="#F9C74F", strokeDash=[4, 4]).encode(
    x="Date:T",
    y="vix_sd:Q"
)

vvix_sd_line = base_vix.mark_line(color="#90BE6D", strokeDash=[2, 4]).encode(
    x="Date:T",
    y="vvix_sd:Q"
)

tsu_pts = base_vix.mark_point(color="red", size=120, shape="diamond").encode(
    x="Date:T",
    y="VIX:Q",
).transform_filter("datum.tsunami == true")

st.altair_chart(
    vix_line + vix_sd_line + vvix_sd_line + tsu_pts,
    use_container_width=True
)

st.caption(
    "Blue = VIX. Yellow = 20-day stdev of VIX. Green = 20-day stdev of VVIX. "
    "Red diamonds = Volatility Tsunami compression signals."
)

st.markdown("---")
st.caption("v3 – Trend, Canary, Tsunami & Cross-Asset snapshot (with zoomed y-axis and log option).")

if __name__ == "__main__":
    main()
