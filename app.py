import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_price_history(symbol: str, years: int = 5) -> pd.DataFrame:
    """Download long history for a symbol and compute EMAs."""
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * years + 30)

    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    if df.empty:
        return df

    df.index = pd.to_datetime(df.index)
    close = df["Close"]

    # Daily EMAs
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    return df


def canary_regime_from_series(df: pd.DataFrame) -> dict:
    """
    Simple 5% Canary-style detector using daily close and 200d EMA.

    Returns dict with:
      status_text, regime (none/slow/fast/confirmed),
      drop_pct, bars_since_high, last_high_date
    """
    out = {
        "status_text": "No active Canary signal",
        "regime": "none",
        "drop_pct": np.nan,
        "bars_since_high": None,
        "last_high_date": None,
    }

    if df.empty or "Close" not in df.columns:
        return out

    close = df["Close"].dropna()
    if len(close) < 252:
        return out

    ema200 = df["EMA200"].reindex(close.index).iloc[-1]

    # 1-year rolling high of close
    high_1y = close.rolling(window=252, min_periods=100).max()
    last_close = close.iloc[-1]
    last_high = high_1y.iloc[-1]

    if pd.isna(last_high):
        return out

    drop_pct = (last_close / last_high - 1.0) * 100.0
    out["drop_pct"] = drop_pct

    # Most recent close within 0.1% of that 1-year high
    near_high_mask = close >= 0.999 * last_high
    if near_high_mask.any():
        last_high_pos = np.where(near_high_mask.values)[0][-1]
        bars_since_high = len(close) - 1 - last_high_pos
        last_high_date = close.index[last_high_pos]
    else:
        bars_since_high = None
        last_high_date = None

    out["bars_since_high"] = bars_since_high
    out["last_high_date"] = last_high_date

    # Classify regime
    if drop_pct <= -10 and last_close < ema200:
        out["status_text"] = "Confirmed Canary (≥10% drop, below 200-day EMA)"
        out["regime"] = "confirmed"
    elif drop_pct <= -5:
        if bars_since_high is not None and bars_since_high <= 5:
            out["status_text"] = "Fast 5% Canary drop (within 5 bars of high)"
            out["regime"] = "fast"
        else:
            out["status_text"] = "Slow 5% (Buy-the-Dip opportunity)"
            out["regime"] = "slow"
    else:
        out["status_text"] = "No active Canary signal"
        out["regime"] = "none"

    return out


def build_price_chart(
    df_full: pd.DataFrame,
    title: str,
    days_window: int = 90,
    price_name: str = "Price",
) -> tuple[go.Figure, dict]:
    """
    Build a daily price chart for last `days_window` days with 21 & 200 EMAs
    and a 'flashlight' marker for the canary regime.
    """
    if df_full.empty:
        return go.Figure(), {"regime": "none", "status_text": "No data"}

    # 3-month window for plotting
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days_window)
    df_recent = df_full[df_full.index >= cutoff].copy()
    if df_recent.empty:
        df_recent = df_full.tail(days_window)

    # Canary regime based on full history
    canary_info = canary_regime_from_series(df_full)

    # Colors for the flashlight
    regime_color_map = {
        "none": "#2ecc71",       # green
        "slow": "#2ecc71",       # green
        "fast": "#f1c40f",       # yellow
        "confirmed": "#e74c3c",  # red
    }
    flash_color = regime_color_map.get(canary_info["regime"], "#95a5a6")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["Close"],
            mode="lines",
            name=price_name,
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["EMA21"],
            mode="lines",
            name="21-day EMA",
            line=dict(width=1.5, color="#f1c40f"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["EMA200"],
            mode="lines",
            name="200-day EMA",
            line=dict(width=1.5, color="#27ae60"),
        )
    )

    # Flashlight marker at latest bar
    latest_x = df_recent.index[-1]
    latest_y = df_recent["Close"].iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[latest_x],
            y=[latest_y],
            mode="markers",
            marker=dict(size=24, color=flash_color, symbol="circle"),
            name="Canary Regime",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_name,
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig, canary_info


def load_vix_vvix(years: int = 5) -> pd.DataFrame:
    """Load VIX and VVIX and compute 20-day rolling standard deviations."""
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * years + 30)

    vix = yf.download("^VIX", start=start, end=end, interval="1d", progress=False)
    vvix = yf.download("^VVIX", start=start, end=end, interval="1d", progress=False)

    if vix.empty:
        return pd.DataFrame()

    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    if not vvix.empty:
        vvix = vvix[["Close"]].rename(columns={"Close": "VVIX"})
        df = vix.join(vvix, how="inner")
    else:
        df = vix.copy()
        df["VVIX"] = np.nan

    df["VIX_SD20"] = df["VIX"].rolling(window=20, min_periods=10).std()
    df["VVIX_SD20"] = df["VVIX"].rolling(window=20, min_periods=10).std()

    return df.dropna()


def detect_tsunami(df: pd.DataFrame, lookback_days: int = 80) -> dict:
    """
    Simple volatility tsunami 'compression' detector based on low combined
    20-day std of VIX + VVIX.

    Active if most recent compression is within `lookback_days`.
    """
    out = {
        "active": False,
        "last_signal_date": None,
        "status_text": "No Tsunami in window",
    }

    if df.empty or "VIX_SD20" not in df.columns:
        return out

    df = df.copy()
    df["ComboSD"] = df["VIX_SD20"] + df["VVIX_SD20"]

    combo = df["ComboSD"].dropna()
    if combo.empty:
        return out

    threshold = np.percentile(combo, 25)  # lower quartile = compression
    compress_mask = df["ComboSD"] <= threshold
    signal_dates = df.index[compress_mask]

    if len(signal_dates) == 0:
        return out

    last_signal = signal_dates[-1]
    out["last_signal_date"] = last_signal

    if df.index[-1] - last_signal <= pd.Timedelta(days=lookback_days):
        out["active"] = True
        out["status_text"] = f"Tsunami watch since {last_signal.date()}"
    else:
        out["status_text"] = "No Tsunami in window"

    return out


def build_vix_chart(df: pd.DataFrame, tsunami_info: dict, timeframe: str) -> go.Figure:
    """Build VIX + VVIX SD chart with tsunami flashlight."""
    if df.empty:
        return go.Figure()

    if timeframe == "Weekly":
        df_plot = df.resample("W-FRI").last()
    else:
        df_plot = df.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["VIX"],
            mode="lines",
            name="VIX",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["VIX_SD20"],
            mode="lines",
            name="VIX 20-day SD",
            line=dict(width=1.5, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["VVIX_SD20"],
            mode="lines",
            name="VVIX 20-day SD",
            line=dict(width=1.5, dash="dash"),
        )
    )

    # Tsunami flashlight
    flash_color = "#2ecc71" if not tsunami_info.get("active") else "#f39c12"
    y_flash = df_plot["VIX"].iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[df_plot.index[-1]],
            y=[y_flash],
            mode="markers",
            marker=dict(size=24, color=flash_color, symbol="circle"),
            name="Tsunami Status",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="VIX & Volatility Tsunami Watch",
        xaxis_title="Date",
        yaxis_title="VIX / Volatility",
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def percent_off_high(close: pd.Series, days: int = 252) -> float:
    """% off the highest close in the last `days` calendar days."""
    if close.empty:
        return np.nan
    recent = close[close.index >= (close.index[-1] - pd.Timedelta(days=days))]
    if recent.empty:
        recent = close
    high = recent.max()
    return (close.iloc[-1] / high - 1.0) * 100.0


def short_trend_label(close: float, ema50: float, ema200: float) -> str:
    """Simple trend label using 50 and 200 EMAs."""
    if np.isnan(ema50) or np.isnan(ema200):
        return "Trend: n/a"

    if close > ema50 > ema200:
        return "Strong Uptrend"
    if close > ema50 and ema50 <= ema200:
        return "Uptrend"
    if close >= ema200 and close <= ema50:
        return "Choppy / Range"
    if close < ema200 < ema50:
        return "Downtrend"
    if close < ema200 and ema50 < ema200:
        return "Bearish"
    return "Mixed"


# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="Market Risk Dashboard", layout="wide")
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # ---- Load data ----
    with st.spinner("Loading market data..."):
        spy_full = load_price_history("SPY", years=5)
        qqq_full = load_price_history("QQQ", years=5)
        vix_df = load_vix_vvix(years=5)

    # ---- Build SPY & QQQ charts + canary info ----
    spy_fig, spy_canary = build_price_chart(spy_full, "SPY with 5% Canary Signals", price_name="SPY Price")
    qqq_fig, qqq_canary = build_price_chart(qq_full, "QQQ (NASDAQ) with 5% Canary Signals", price_name="QQQ Price")

    # ---- Tsunami info + chart ----
    tsunami_info = detect_tsunami(vix_df, lookback_days=80)
    vix_tf = st.radio("VIX timeframe", ["Daily", "Weekly"], index=0, horizontal=True)
    vix_fig = build_vix_chart(vix_df, tsunami_info, timeframe=vix_tf)

    # ---- Summary panel ----
    col1, col2, col3, col4 = st.columns(4)

    # Canary status (based on SPY)
    with col1:
        st.subheader("Canary Status")
        st.write(spy_canary["status_text"])
        if not np.isnan(spy_canary.get("drop_pct", np.nan)):
            st.write(f"Current drawdown from 1-year high: {spy_canary['drop_pct']:.1f}%")

    # Tsunami status
    with col2:
        st.subheader("Tsunami Status")
        st.write(tsunami_info["status_text"])

    # Market snapshot
    with col3:
        st.subheader("Market Snapshot")

        if not spy_full.empty:
            spy_close = spy_full["Close"]
            spy_price = spy_close.iloc[-1]
            spy_off = percent_off_high(spy_close, days=252)
            st.write(f"SPY: {spy_price:,.2f}")
            st.write(f"SPY off 52-week high: {spy_off:.1f}%")

        if not qqq_full.empty:
            qqq_close = qqq_full["Close"]
            qqq_price = qqq_close.iloc[-1]
            qqq_off = percent_off_high(qqq_close, days=252)
            st.write(f"QQQ: {qqq_price:,.2f}")
            st.write(f"QQQ off 52-week high: {qqq_off:.1f}%")

        if not vix_df.empty:
            st.write(f"VIX: {vix_df['VIX'].iloc[-1]:.2f}")

    # Breadth proxy / trend
    with col4:
        st.subheader("Breadth (Proxy)")

        if not spy_full.empty:
            spy_close = spy_full["Close"]
            spy_ema50 = spy_full["EMA50"]
            recent_mask = spy_close.index >= (spy_close.index[-1] - pd.Timedelta(days=60))
            breadth_spy = (spy_close[recent_mask] > spy_ema50[recent_mask]).mean() * 100.0
            st.write(f"SPY > 50-EMA (last 60d): {breadth_spy:.0f}%")
            st.write("SPY Trend: " + short_trend_label(spy_close.iloc[-1], spy_ema50.iloc[-1], spy_full["EMA200"].iloc[-1]))

        if not qqq_full.empty:
            qqq_close = qqq_full["Close"]
            qqq_ema50 = qqq_full["EMA50"]
            recent_mask_q = qqq_close.index >= (qqq_close.index[-1] - pd.Timedelta(days=60))
            breadth_qqq = (qqq_close[recent_mask_q] > qqq_ema50[recent_mask_q]).mean() * 100.0
            st.write(f"QQQ > 50-EMA (last 60d): {breadth_qqq:.0f}%")
            st.write("QQQ Trend: " + short_trend_label(qqq_close.iloc[-1], qqq_ema50.iloc[-1], qqq_full["EMA200"].iloc[-1]))

    st.markdown("---")

    # ---- SPY & QQQ charts ----
    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qqq_fig, use_container_width=True)

    # Small legend note
    st.caption("SPY/QQQ charts — Blue: price, Yellow: 21-day EMA, Green: 200-day EMA. Flashlight color reflects Canary regime.")

    st.markdown("---")

    # ---- VIX / Tsunami chart ----
    st.plotly_chart(vix_fig, use_container_width=True)
    st.caption("VIX chart — Solid: VIX, Dotted: VIX 20-day SD, Dashed: VVIX 20-day SD. Flashlight shows current Tsunami watch status.")

    st.markdown("Last updated: " + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
