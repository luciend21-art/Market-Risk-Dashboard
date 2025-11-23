import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


# ----------------------------
# Data loading & basic helpers
# ----------------------------

@st.cache_data(ttl=3600)
def load_data():
    """Download SPY, QQQ, VIX, VVIX with enough history for EMAs and signals."""
    end = dt.date.today()
    start = end - dt.timedelta(days=800)  # a bit over 3 years

    spy = yf.download("SPY", start=start, end=end, progress=False)
    qqq = yf.download("QQQ", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    vvix = yf.download("^VVIX", start=start, end=end, progress=False)

    # Just in case, drop any rows without Close data
    spy = spy.dropna(subset=["Close"])
    qqq = qqq.dropna(subset=["Close"])
    vix = vix.dropna(subset=["Close"])
    vvix = vvix.dropna(subset=["Close"])

    return spy, qqq, vix, vvix


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def last_close(df: pd.DataFrame) -> float:
    """Return latest closing price as a float."""
    return float(df["Close"].iloc[-1])


def pct_off_high(series: pd.Series, lookback_days: int = 252) -> float:
    """
    Percent off the highest close in the last `lookback_days`.
    Negative value means below the high.
    """
    recent = series.tail(lookback_days)
    high = float(recent.max())
    last = float(recent.iloc[-1])
    return (last / high - 1.0) * 100.0


# ----------------------------
# 5% Canary detection
# ----------------------------

def detect_canary(
    df: pd.DataFrame,
    lookback_days: int = 252,
    fast_bars: int = 10,
    confirm_bars: int = 60,
    long_ema_len: int = 200,
):
    """
    Very lightweight approximation of your 5% Canary logic.

    - When drawdown from 1-year high first reaches -5%, mark:
        * 'fast' if it happened within `fast_bars` after the high
        * otherwise 'slow'
    - If a fast drop is followed (within `confirm_bars`) by a close below
      the long EMA, mark 'confirmed'.
    """
    close = df["Close"]
    high_1y = close.rolling(window=lookback_days, min_periods=1).max()
    long_ema = ema(close, long_ema_len)

    index = close.index
    slow = pd.Series(False, index=index)
    fast = pd.Series(False, index=index)
    confirmed = pd.Series(False, index=index)

    last_high_idx = None
    last_fast_idx = None

    for i, (idx, c) in enumerate(close.items()):
        # Track "local" 1y high index
        if high_1y.iloc[i] == 0 or np.isnan(high_1y.iloc[i]):
            continue

        if c >= high_1y.iloc[i] * 0.999:
            last_high_idx = i

        # Drawdown from 1-year high
        drop_pct = (c / high_1y.iloc[i] - 1.0) * 100.0

        # First time reaching -5% after a high
        if drop_pct <= -5.0 and last_high_idx is not None:
            bars_from_high = i - last_high_idx
            # Only trigger once per leg down
            if not (slow.iloc[i] or fast.iloc[i]):
                if bars_from_high <= fast_bars:
                    fast.iloc[i] = True
                    last_fast_idx = i
                else:
                    slow.iloc[i] = True

        # Confirmation: close below long EMA within confirm window after fast drop
        if last_fast_idx is not None and (i - last_fast_idx) <= confirm_bars:
            if c < long_ema.iloc[i]:
                confirmed.iloc[i] = True
                # Only need the first confirmation
                last_fast_idx = None

    flags_df = pd.DataFrame(
        {"slow": slow, "fast": fast, "confirmed": confirmed}
    )

    # Determine the most recent and most severe event for regime label
    last_idx = None
    last_type = None

    for t in ["confirmed", "fast", "slow"]:
        s = flags_df[t]
        if s.any():
            idx_last = s[s].index[-1]
            if last_idx is None or idx_last > last_idx:
                last_idx = idx_last
                last_type = t

    if last_type == "confirmed":
        regime = "Confirmed Canary"
    elif last_type == "fast":
        regime = "Fast 5% drop"
    elif last_type == "slow":
        regime = "Slow 5% (Buy-the-Dip)"
    else:
        regime = "No active Canary signal"

    return {
        "flags": flags_df,
        "regime": regime,
        "last_type": last_type,
        "last_date": last_idx,
    }


def canary_panel_text(canary_info: dict, df_full: pd.DataFrame):
    """Return emoji + text for the Canary status panel."""
    last_type = canary_info["last_type"]

    if last_type == "confirmed":
        return "🔴", "Confirmed Canary – high risk, defense first."

    if last_type == "fast":
        return "🟠", "Fast 5% drop – early warning. Tighten risk and consider hedges."

    if last_type == "slow":
        return "🟢", "Slow 5% (Buy-the-Dip) signal – constructive, but manage size."

    # No explicit canary – describe the drawdown from 1-year high
    close = df_full["Close"]
    high_1y = close.rolling(252, min_periods=1).max()
    dd = (close.iloc[-1] / high_1y.iloc[-1] - 1.0) * 100.0

    if dd > -5.0:
        return "🟢", f"Shallow pullback ({dd:.1f}% off 1-year high)."
    else:
        return "🟡", f"No Canary trigger; {dd:.1f}% off 1-year high."


# ----------------------------
# Volatility Tsunami (daily)
# ----------------------------

def compute_tsunami_daily(vix_df: pd.DataFrame, vvix_df: pd.DataFrame) -> pd.DataFrame:
    """Build a daily dataframe with VIX/VVIX 20-day stdevs and a compression signal."""
    vix = vix_df["Close"]
    vvix = vvix_df["Close"]

    vix_ret = vix.pct_change()
    vvix_ret = vvix.pct_change()

    vix_sd20 = vix_ret.rolling(20).std() * 100.0
    vvix_sd20 = vvix_ret.rolling(20).std() * 100.0

    # Compression thresholds based on low decile over the past ~1y
    lookback = 252
    th_vix = vix_sd20.rolling(lookback, min_periods=60).quantile(0.10)
    th_vvix = vvix_sd20.rolling(lookback, min_periods=60).quantile(0.10)

    signal = (vix_sd20 < th_vix) & (vvix_sd20 < th_vvix)

    df = pd.DataFrame(
        {
            "VIX": vix,
            "VIX_SD20": vix_sd20,
            "VVIX_SD20": vvix_sd20,
            "signal": signal,
        }
    )

    return df


def tsunami_status(tsunami_df: pd.DataFrame, window_days: int = 120) -> dict:
    """Return whether a Tsunami compression has fired in the recent window."""
    recent = tsunami_df.tail(window_days)
    if recent["signal"].any():
        idx = recent[recent["signal"]].index[-1]
        return {"has_signal": True, "last_date": idx}
    return {"has_signal": False, "last_date": None}


def build_vix_chart(tsunami_df: pd.DataFrame, timeframe: str) -> go.Figure:
    """Plot VIX, its 20-day stdev, and VVIX 20-day stdev, with Tsunami diamonds."""
    df = tsunami_df.copy()

    if timeframe == "Weekly":
        df = df.resample("W-FRI").last()

    df_recent = df.tail(365)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["VIX"],
            mode="lines",
            name="VIX",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["VIX_SD20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["VVIX_SD20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot"),
        )
    )

    sig_mask = df_recent["signal"].fillna(False)
    if sig_mask.any():
        xs = df_recent.index[sig_mask]
        ys = df_recent.loc[sig_mask, "VIX"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="Tsunami compression",
                marker=dict(symbol="diamond", size=10),
            )
        )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="VIX")

    return fig


# ----------------------------
# Price chart building (SPY/QQQ)
# ----------------------------

def build_price_chart(
    df_full: pd.DataFrame,
    title: str,
    price_name: str,
    canary_info: dict,
    short_len: int = 21,
    long_len: int = 200,
    days_window: int = 65,
) -> go.Figure:
    """
    Build a 3-month price chart with:
    - Close
    - short EMA (21-day)
    - long EMA (200-day)
    - markers for slow / fast / confirmed Canary events
    """
    df = df_full.copy()
    df["EMA_short"] = ema(df["Close"], short_len)
    df["EMA_long"] = ema(df["Close"], long_len)

    df_recent = df.tail(days_window)
    flags_full = canary_info["flags"]
    flags_recent = flags_full.loc[df_recent.index]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["Close"],
            mode="lines",
            name=price_name,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["EMA_short"],
            mode="lines",
            name=f"{short_len}-day EMA",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["EMA_long"],
            mode="lines",
            name=f"{long_len}-day EMA",
        )
    )

    # Canary markers
    marker_styles = {
        "slow": dict(symbol="circle", color="green"),
        "fast": dict(symbol="triangle-up", color="orange"),
        "confirmed": dict(symbol="diamond", color="red"),
    }
    labels = {
        "slow": "Slow 5% (Buy-the-Dip)",
        "fast": "Fast 5% drop",
        "confirmed": "Confirmed Canary",
    }

    for key in ["slow", "fast", "confirmed"]:
        mask = flags_recent[key].fillna(False)
        if mask.any():
            xs = flags_recent.index[mask]
            ys = df_recent.loc[xs, "Close"]
            style = marker_styles[key]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=labels[key],
                    marker=dict(
                        symbol=style["symbol"],
                        size=10,
                        color=style["color"],
                        line=dict(width=1, color="black"),
                    ),
                )
            )

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=price_name)

    return fig


# ----------------------------
# Main app
# ----------------------------

def main():
    st.set_page_config(
        page_title="Market Risk Dashboard",
        layout="wide",
    )

    st.title("Market Risk Dashboard")
    st.markdown("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # Load data
    spy_full, qqq_full, vix_full, vvix_full = load_data()

    # Signals
    spy_canary = detect_canary(spy_full)
    qqq_canary = detect_canary(qqq_full)

    tsunami_df = compute_tsunami_daily(vix_full, vvix_full)
    tsu_info = tsunami_status(tsunami_df, window_days=120)

    # ---------------- Top summary row ----------------
    col1, col2, col3 = st.columns(3)

    # Canary Status
    with col1:
        st.subheader("Canary Status")
        emoji, text = canary_panel_text(spy_canary, spy_full)
        st.markdown(f"{emoji} {text}")

    # Tsunami Status
    with col2:
        st.subheader("Tsunami Status")
        if tsu_info["has_signal"]:
            date_str = tsu_info["last_date"].strftime("%Y-%m-d")
            st.markdown(
                f"🟡 Tsunami compression signal on **{date_str}** "
                f"(within last 120 days)."
            )
        else:
            st.markdown("🟢 No Tsunami in window")
            st.caption("No Tsunami signal in the last 120 days.")

    # Market Snapshot
    with col3:
        st.subheader("Market Snapshot")

        spy_last = last_close(spy_full)
        qqq_last = last_close(qqq_full)
        spy_off = pct_off_high(spy_full["Close"])
        qqq_off = pct_off_high(qqq_full["Close"])
        vix_last = float(vix_full["Close"].iloc[-1])

        st.write(f"SPY: {spy_last:.2f}")
        st.write(f"Off 52-week high: {spy_off:.1f}%")

        st.write(f"QQQ: {qqq_last:.2f}")
        st.write(f"Off 52-week high: {qqq_off:.1f}%")

        st.write(f"VIX: {vix_last:.2f}")

    st.markdown("---")

    # ---------------- SPY & QQQ charts ----------------
    spy_fig = build_price_chart(
        spy_full,
        "SPY with 5% Canary Signals",
        "SPY Price",
        spy_canary,
    )
    qqq_fig = build_price_chart(
        qqq_full,
        "QQQ (NASDAQ) with 5% Canary Signals",
        "QQQ Price",
        qqq_canary,
    )

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    st.markdown("---")

    # ---------------- VIX & Tsunami chart ----------------
    st.markdown("### VIX & Volatility Tsunami Watch")
    timeframe = st.radio("VIX timeframe", ("Daily", "Weekly"), horizontal=True)

    vix_fig = build_vix_chart(tsunami_df, timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)
    st.caption(
        "Lines: VIX, VIX 20-day stdev, VVIX 20-day stdev. "
        "Diamonds mark Tsunami compression signals."
    )


if __name__ == "__main__":
    main()
