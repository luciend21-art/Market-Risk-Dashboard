import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LOOKBACK_DAYS_PRICE = 65           # ~3 months for SPY/QQQ plots
CANARY_LOOKBACK_DAYS = 90          # regime flashlight window
TSUNAMI_LOOKBACK_DAYS = 90         # ~4 months for Tsunami flashlight
VIX_SD_WINDOW = 20                 # 20-day SD for VIX & VVIX
EMA_HISTORY_YEARS = 5              # how far back we fetch for EMAs

# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------


def load_price_history(symbol: str, years: int = EMA_HISTORY_YEARS):
    """Download full history and compute EMAs. Return (daily_df, weekly_df)."""
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")

    close = df["Close"]

    # Daily EMAs on full history
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # Weekly resample
    weekly = df.resample("W-FRI").last()
    wclose = weekly["Close"]
    weekly["ema_10w"] = wclose.ewm(span=10, adjust=False).mean()
    weekly["ema_40w"] = wclose.ewm(span=40, adjust=False).mean()

    return df, weekly


def load_vix_vvix(years: int = EMA_HISTORY_YEARS):
    """Load VIX and VVIX and compute 20-day SDs and Tsunami compression flag."""
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    vix = yf.download("^VIX", start=start, end=end, progress=False)
    vvix = yf.download("^VVIX", start=start, end=end, progress=False)

    vix = vix.rename(columns={"Close": "VIX"})
    vvix = vvix.rename(columns={"Close": "VVIX"})

    df = pd.DataFrame(index=vix.index.union(vvix.index))
    df["VIX"] = vix["VIX"]
    df["VVIX"] = vvix["VVIX"]
    df = df.ffill().dropna()

    df["VIX_SD"] = df["VIX"].rolling(VIX_SD_WINDOW).std()
    df["VVIX_SD"] = df["VVIX"].rolling(VIX_SD_WINDOW).std()

    # Percentile-based compression: both SDs in lowest 20% of history
    vix_thresh = df["VIX_SD"].quantile(0.2)
    vvix_thresh = df["VVIX_SD"].quantile(0.2)
    df["tsunami_flag"] = (df["VIX_SD"] <= vix_thresh) & (
        df["VVIX_SD"] <= vvix_thresh
    )

    return df


# ---------------------------------------------------------------------
# 5% Canary logic (simplified Python analog of your Pine script)
# ---------------------------------------------------------------------


@dataclass
class CanaryStatus:
    regime: str              # "Normal", "Caution", "Confirmed"
    regime_color: str        # emoji / color name
    last_fast_date: dt.date | None
    last_slow_date: dt.date | None
    last_confirmed_date: dt.date | None


def detect_canary(df: pd.DataFrame, fast_bars: int = 10) -> CanaryStatus:
    """
    Simple 5% Canary:
    - Use 1-year rolling high as reference
    - When price falls 5% from that high, label Fast vs Slow based on bars from high.
    - If price later spends >=2 closes below 200-EMA within lookback window, mark Confirmed.
    """
    close = df["Close"]
    high_1y = close.rolling(252, min_periods=20).max()
    drawdown = close / high_1y - 1.0

    # Locations where we hit -5% from rolling high
    breach = drawdown <= -0.05
    events = []
    last_high_idx = None

    for i in range(len(df)):
        if close[i] >= high_1y[i] * 0.999:  # near new high
            last_high_idx = i
        if breach[i] and last_high_idx is not None:
            bars_from_high = i - last_high_idx
            events.append((df.index[i], bars_from_high))

    last_fast_date = None
    last_slow_date = None

    if events:
        for dt_i, bars_from_high in events:
            if bars_from_high <= fast_bars:
                last_fast_date = dt_i.date()
            else:
                last_slow_date = dt_i.date()

    # Confirmation: two closes below 200 EMA after a fast 5% drop
    below_200 = close < df["ema_200"]
    confirm = below_200 & below_200.shift(1, fill_value=False)
    last_confirmed_date = (
        df.index[confirm].max().date() if confirm.any() else None
    )

    # Regime for flashlight: keep as in previous versions
    today_cutoff = df.index[-1] - pd.Timedelta(days=CANARY_LOOKBACK_DAYS)

    def recent(date):
        return date is not None and pd.Timestamp(date) >= today_cutoff

    if recent(last_confirmed_date):
        regime = "Confirmed Canary"
        color = "red"
    elif recent(last_fast_date) or recent(last_slow_date):
        regime = "Caution"
        color = "yellow"
    else:
        regime = "Normal"
        color = "green"

    return CanaryStatus(
        regime=regime,
        regime_color=color,
        last_fast_date=last_fast_date,
        last_slow_date=last_slow_date,
        last_confirmed_date=last_confirmed_date,
    )


# ---------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------


def plot_spy_qqq(spy_df: pd.DataFrame, qqq_df: pd.DataFrame):
    """Daily SPY & QQQ with EMAs, legends, and 3-month zoom."""
    spy_recent = spy_df.tail(LOOKBACK_DAYS_PRICE)
    qqq_recent = qqq_df.tail(LOOKBACK_DAYS_PRICE)

    # ----- SPY
    fig_spy = go.Figure()
    fig_spy.add_trace(
        go.Scatter(
            x=spy_recent.index,
            y=spy_recent["Close"],
            name="SPY Price",
            mode="lines",
            line=dict(color="#00B5E2"),
        )
    )
    fig_spy.add_trace(
        go.Scatter(
            x=spy_recent.index,
            y=spy_recent["ema_21"],
            name="21-day EMA",
            mode="lines",
            line=dict(color="#F4B000"),
        )
    )
    fig_spy.add_trace(
        go.Scatter(
            x=spy_recent.index,
            y=spy_recent["ema_200"],
            name="200-day EMA",
            mode="lines",
            line=dict(color="#00A651"),
        )
    )

    fig_spy.update_layout(
        title="SPY with 5% Canary Signals",
        xaxis_title="Date",
        yaxis_title="SPY Price",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    # ----- QQQ
    fig_qqq = go.Figure()
    fig_qqq.add_trace(
        go.Scatter(
            x=qqq_recent.index,
            y=qqq_recent["Close"],
            name="QQQ Price",
            mode="lines",
            line=dict(color="#00B5E2"),
        )
    )
    fig_qqq.add_trace(
        go.Scatter(
            x=qqq_recent.index,
            y=qqq_recent["ema_21"],
            name="21-day EMA",
            mode="lines",
            line=dict(color="#F4B000"),
        )
    )
    fig_qqq.add_trace(
        go.Scatter(
            x=qqq_recent.index,
            y=qqq_recent["ema_200"],
            name="200-day EMA",
            mode="lines",
            line=dict(color="#00A651"),
        )
    )

    fig_qqq.update_layout(
        title="QQQ (NASDAQ) with 5% Canary Signals",
        xaxis_title="Date",
        yaxis_title="QQQ Price",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig_spy, fig_qqq


def plot_vix_tsunami(vix_df: pd.DataFrame, timeframe: str = "Daily"):
    """VIX + VIX_SD + VVIX_SD with Tsunami flashlight and legend."""
    df = vix_df.copy()
    if timeframe == "Weekly":
        df = df.resample("W-FRI").last()
        df["VIX_SD"] = df["VIX"].rolling(VIX_SD_WINDOW).std()
        df["VVIX_SD"] = df["VVIX"].rolling(VIX_SD_WINDOW).std()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX"],
            name="VIX",
            mode="lines",
            line=dict(color="#00B5E2"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX_SD"],
            name="VIX 20-day SD",
            mode="lines",
            line=dict(color="#7FBA00", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VVIX_SD"],
            name="VVIX 20-day SD",
            mode="lines",
            line=dict(color="#F4B000", dash="dot"),
        )
    )

    # Tsunami flashlight in last TSUNAMI_LOOKBACK_DAYS
    cutoff = df.index[-1] - pd.Timedelta(days=TSUNAMI_LOOKBACK_DAYS)
    recent = df[df.index >= cutoff]
    recent_tsunamis = recent[recent["tsunami_flag"]]

    if not recent_tsunamis.empty:
        last_ts = recent_tsunamis.index.max()
        emoji = "🟡"
        text = f"Tsunami compression on {last_ts.date()}"
    else:
        emoji = "🟢"
        text = "No Tsunami in window"

    fig.add_annotation(
        x=df.index[-1],
        y=df["VIX"].max(),
        xref="x",
        yref="y",
        text=f"{emoji} {text}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
    )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        xaxis_title="Date",
        yaxis_title="VIX",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Market Risk Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # Load data
    spy_daily, spy_weekly = load_price_history("SPY")
    qqq_daily, qqq_weekly = load_price_history("QQQ")
    vix_df = load_vix_vvix()

    # Canary status based on daily SPY
    spy_canary = detect_canary(spy_daily)
    qqq_canary = detect_canary(qqq_daily)

    # Top status row
    col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.5, 1.5])

    with col_a:
        st.subheader("Canary Status")
        if spy_canary.regime_color == "red":
            emoji = "🔴"
        elif spy_canary.regime_color == "yellow":
            emoji = "🟡"
        else:
            emoji = "🟢"
        st.write(f"{emoji} SPY: {spy_canary.regime}")
        if spy_canary.last_confirmed_date:
            st.write(f"Last confirmed: {spy_canary.last_confirmed_date}")
        elif spy_canary.last_fast_date or spy_canary.last_slow_date:
            st.write(
                f"Last Canary: "
                f"{spy_canary.last_fast_date or spy_canary.last_slow_date}"
            )
        else:
            st.write("No recent 5% Canary event")

    with col_b:
        st.subheader("Tsunami Status")
        cutoff = vix_df.index[-1] - pd.Timedelta(days=TSUNAMI_LOOKBACK_DAYS)
        recent_tsunamis = vix_df[
            (vix_df.index >= cutoff) & (vix_df["tsunami_flag"])
        ]

        if not recent_tsunamis.empty:
            emoji = "🟡"
            last_ts = recent_tsunamis.index.max().date()
            st.write(f"{emoji} Compression: {last_ts}")
        else:
            emoji = "🟢"
            st.write(f"{emoji} No Tsunami in last {TSUNAMI_LOOKBACK_DAYS} days")

    with col_c:
        st.subheader("Market Snapshot")

        spy_px = spy_daily["Close"].iloc[-1]
        spy_hi = spy_daily["Close"].rolling(252).max().iloc[-1]
        spy_off = (spy_px / spy_hi - 1.0) * 100

        qqq_px = qqq_daily["Close"].iloc[-1]
        qqq_hi = qqq_daily["Close"].rolling(252).max().iloc[-1]
        qqq_off = (qqq_px / qqq_hi - 1.0) * 100

        vix_last = vix_df["VIX"].iloc[-1]

        st.write(f"SPY: {spy_px:.2f}  |  Off 52-week high: {spy_off:.1f}%")
        st.write(f"QQQ: {qqq_px:.2f}  |  Off 52-week high: {qqq_off:.1f}%")
        st.write(f"VIX: {vix_last:.2f}")

    with col_d:
        st.subheader("Breadth (Proxy)")
        # Very rough proxy: price vs its own 50-EMA
        spy_above = float(spy_daily["Close"].iloc[-1] > spy_daily["ema_50"].iloc[-1])
        qqq_above = float(qqq_daily["Close"].iloc[-1] > qqq_daily["ema_50"].iloc[-1])
        st.write(f"SPY above 50-EMA: {spy_above * 100:.0f}%")
        st.write(f"QQQ above 50-EMA: {qqq_above * 100:.0f}%")
        st.write("SPY Trend: Strong Uptrend" if spy_above else "SPY Trend: Under 50-EMA")
        st.write("QQQ Trend: Strong Uptrend" if qqq_above else "QQQ Trend: Under 50-EMA")

    # ------------------------------------------------------------------
    # SPY / QQQ price panels
    # ------------------------------------------------------------------
    fig_spy, fig_qqq = plot_spy_qqq(spy_daily, qqq_daily)

    st.plotly_chart(fig_spy, use_container_width=True)
    st.caption("Legend: Blue = Price, Yellow = 21-day EMA, Green = 200-day EMA.")

    st.plotly_chart(fig_qqq, use_container_width=True)
    st.caption("Legend: Blue = Price, Yellow = 21-day EMA, Green = 200-day EMA.")

    # ------------------------------------------------------------------
    # VIX & Volatility Tsunami
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("VIX & Volatility Tsunami Watch")

    tf = st.radio("VIX timeframe", options=["Daily", "Weekly"], horizontal=True)
    fig_vix = plot_vix_tsunami(vix_df, timeframe=tf)
    st.plotly_chart(fig_vix, use_container_width=True)


if __name__ == "__main__":
    main()
