# app.py  – Market Risk Dashboard (5% Canary + Volatility Tsunami)

import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

START_DATE = "2015-01-01"
LOOKBACK_DAYS_PRICE = 90          # 3-month zoom for SPY/QQQ
TSUNAMI_LOOKBACK_DAYS = 120       # window to scan for Tsunami signals

st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def load_price_history(symbol: str, start: str = START_DATE) -> pd.DataFrame:
    """Download full daily close series and compute EMAs on full history."""
    df = yf.download(symbol, start=start, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


def slice_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return only the last `days` calendar days for plotting."""
    last_date = df.index.max()
    cutoff = last_date - pd.Timedelta(days=days)
    return df[df.index >= cutoff].copy()


def load_vix_vvix(start: str = START_DATE) -> pd.DataFrame:
    """Load VIX and VVIX and compute 20-day stdevs & z-scores."""
    vix = yf.download("^VIX", start=start, progress=False)[["Close"]].rename(
        columns={"Close": "VIX"}
    )
    vvix = yf.download("^VVIX", start=start, progress=False)[["Close"]].rename(
        columns={"Close": "VVIX"}
    )

    df = vix.join(vvix, how="inner")
    df.index = pd.to_datetime(df.index)

    # 20-day rolling stdev
    df["VIX_SD20"] = df["VIX"].rolling(20).std()
    df["VVIX_SD20"] = df["VVIX"].rolling(20).std()

    # 1-year rolling baseline for z-scores
    base_win = 252
    df["VIX_SD20_Z"] = (
        df["VIX_SD20"] - df["VIX_SD20"].rolling(base_win).mean()
    ) / df["VIX_SD20"].rolling(base_win).std()
    df["VVIX_SD20_Z"] = (
        df["VVIX_SD20"] - df["VVIX_SD20"].rolling(base_win).mean()
    ) / df["VVIX_SD20"].rolling(base_win).std()

    # Tsunami rule (you can refine thresholds later)
    df["Tsunami"] = (df["VIX_SD20_Z"] > 1.0) & (df["VVIX_SD20_Z"] > 1.0)

    return df


# ------------------------------------------------------------------
# Canary regime logic (SPY / QQQ)
# ------------------------------------------------------------------


def canary_regime_from_close(
    close: pd.Series, ema_long: pd.Series, *, dd_threshold: float = -5.0
) -> dict:
    """
    Approximate 5% Canary regime based on:
      - Drawdown from 1-year high
      - Speed of the drop (bars from peak)
      - Position vs long EMA (for 'confirmed')
    Returns dict with 'label' and 'color' (for flashlight).
    """
    out = {"label": "No active Canary signal", "color": "🟢"}

    if len(close) < 260:
        return out

    close = close.dropna()
    ema_long = ema_long.reindex(close.index).dropna()

    # 1-year rolling high on full series
    high_1y = close.rolling(252).max()

    last_close = float(close.iloc[-1])
    last_high = float(high_1y.iloc[-1])

    if np.isnan(last_high):
        return out

    drop_pct = (last_close / last_high - 1.0) * 100.0

    # Bars from that 1-year high
    idx_last = close.index[-1]
    # choose the *last* time we made that 1-year high
    high_mask = high_1y == last_high
    high_idx = high_1y[high_mask].index[-1]
    bars_from_high = np.sum(close.index >= high_idx)

    fast_window = 21  # ~1 trading month

    if drop_pct <= dd_threshold:
        # 5% off the highs
        if bars_from_high <= fast_window:
            # Fast 5% drop
            if last_close < float(ema_long.iloc[-1]):
                out["label"] = "Confirmed 5% Canary (fast drop, below long EMA)"
                out["color"] = "🔴"
            else:
                out["label"] = "Fast 5% Canary (watch risk, hedge)"
                out["color"] = "🟡"
        else:
            # Slow 5% pullback
            out["label"] = "Slow 5% Canary (buy-the-dip setup)"
            out["color"] = "🟢"
    else:
        # No 5% canary, but could still be shallow pullback
        if drop_pct <= -3:
            out["label"] = "Shallow pullback (<5% from 1-year high)"
            out["color"] = "🟢"
        else:
            out["label"] = "Near highs (no Canary)"
            out["color"] = "🟢"

    return out


# ------------------------------------------------------------------
# Chart builders
# ------------------------------------------------------------------


def build_price_chart(
    df_full: pd.DataFrame,
    price_name: str,
    symbol: str,
    lookback_days: int = LOOKBACK_DAYS_PRICE,
) -> tuple[go.Figure, dict]:
    """
    Build SPY/QQQ daily price chart with 21 & 200 EMAs.
    Returns (fig, canary_info).
    """
    # Canary regime uses **full history**
    canary_info = canary_regime_from_close(
        df_full["Close"], df_full["EMA200"], dd_threshold=-5.0
    )

    # Plot only recent slice for readability
    df_plot = slice_recent(df_full, lookback_days)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Close"],
            mode="lines",
            name=price_name,
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["EMA21"],
            mode="lines",
            name="21-day EMA",
            line=dict(width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["EMA200"],
            mode="lines",
            name="200-day EMA",
            line=dict(width=1.5, dash="dot"),
        )
    )

    # Last swing-low (3-month low)
    recent_low = df_plot["Close"].min()
    fig.add_trace(
        go.Scatter(
            x=[df_plot.index.min(), df_plot.index.max()],
            y=[recent_low, recent_low],
            mode="lines",
            name="Recent swing-low",
            line=dict(width=1, dash="dot"),
            showlegend=True,
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
        xaxis_title="Date",
        yaxis_title=f"{symbol} Price",
    )

    subtitle = (
        "Blue: price, Orange: 21-day EMA, Green: 200-day EMA. "
        "Dotted line: most recent swing-low."
    )

    return fig, {"subtitle": subtitle, **canary_info}


def build_vix_chart(
    vix_df: pd.DataFrame, timeframe: str = "Daily"
) -> tuple[go.Figure, dict]:
    """
    Build VIX & Volatility Tsunami chart and return (fig, tsunami_info).
    """
    df = vix_df.copy()

    if timeframe == "Weekly":
        # Resample to weekly (Friday close)
        df = (
            df.resample("W-FRI")
            .agg(
                {
                    "VIX": "last",
                    "VVIX": "last",
                    "VIX_SD20": "last",
                    "VVIX_SD20": "last",
                    "VIX_SD20_Z": "last",
                    "VVIX_SD20_Z": "last",
                    "Tsunami": "max",
                }
            )
            .dropna(how="all")
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX"],
            mode="lines",
            name="VIX",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VIX_SD20"],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VVIX_SD20"],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(width=1.5, dash="dot"),
        )
    )

    # Mark Tsunami signals with red diamonds on the VIX line
    tsu_points = df[df["Tsunami"]]
    if not tsu_points.empty:
        fig.add_trace(
            go.Scatter(
                x=tsu_points.index,
                y=tsu_points["VIX"],
                mode="markers",
                name="Tsunami signal",
                marker=dict(symbol="diamond", size=9),
            )
        )

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="VIX / Vol stats",
    )

    # Tsunami flashlight info (look-back window)
    cutoff = df.index.max() - pd.Timedelta(days=TSUNAMI_LOOKBACK_DAYS)
    recent = df[df.index >= cutoff]
    recent_signals = recent[recent["Tsunami"]]

    if recent_signals.empty:
        info = {
            "label": "No Tsunami in window",
            "color": "🟢",
            "detail": f"No Tsunami signal in the last {TSUNAMI_LOOKBACK_DAYS} days.",
        }
    else:
        last_date = recent_signals.index[-1].date()
        info = {
            "label": "Tsunami Warning",
            "color": "🟠",
            "detail": f"Last Tsunami signal on {last_date}.",
        }

    return fig, info


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------


def main():
    st.title("Market Risk Dashboard")

    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    # ------------------------------------------------------------------
    # Load data (cached by Streamlit)
    # ------------------------------------------------------------------

    @st.cache_data(show_spinner=True)
    def load_all():
        spy = load_price_history("SPY", START_DATE)
        qqq = load_price_history("QQQ", START_DATE)
        vix_df = load_vix_vvix(START_DATE)
        return spy, qqq, vix_df

    spy_full, qqq_full, vix_full = load_all()

    # ------------------------------------------------------------------
    # Canary / Tsunami / Snapshot summary
    # ------------------------------------------------------------------

    col_canary, col_tsunami, col_snapshot, col_breadth = st.columns(4)

    # Canary info from SPY
    spy_canary_info = canary_regime_from_close(
        spy_full["Close"], spy_full["EMA200"], dd_threshold=-5.0
    )

    with col_canary:
        st.subheader("Canary Status")
        st.markdown(f"{spy_canary_info['color']}  **{spy_canary_info['label']}**")

    # Tsunami info
    _, tsunami_info = build_vix_chart(vix_full, timeframe="Daily")

    with col_tsunami:
        st.subheader("Tsunami Status")
        st.markdown(f"{tsunami_info['color']}  **{tsunami_info['label']}**")
        st.caption(tsunami_info["detail"])

    # Market snapshot (SPY, QQQ, % off high)
    with col_snapshot:
        st.subheader("Market Snapshot")

        def pct_off_high(series: pd.Series) -> float:
            recent_high = series.rolling(252).max().iloc[-1]
            return (series.iloc[-1] / recent_high - 1.0) * 100.0

        spy_off = pct_off_high(spy_full["Close"])
        qqq_off = pct_off_high(qqq_full["Close"])

        st.write(f"SPY: {spy_full['Close'].iloc[-1]:.2f}")
        st.write(f"Off 52-week high: {spy_off:.1f}%")
        st.write(f"QQQ: {qqq_full['Close'].iloc[-1]:.2f}")
        st.write(f"QQQ off 52-week high: {qqq_off:.1f}%")

    # Breadth proxy via % above 50-EMA (approx using 50-day EMA via EWM)
    with col_breadth:
        st.subheader("Breadth (Proxy)")
        for name, df in [("SPY", spy_full), ("QQQ", qqq_full)]:
            ema50 = df["Close"].ewm(span=50, adjust=False).mean()
            above = (df["Close"].iloc[-1] > ema50.iloc[-1])
            status = "Above 50-EMA" if above else "Below 50-EMA"
            dot = "🟢" if above else "🔴"
            st.write(f"{dot} {name}: {status}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # SPY & QQQ price panels
    # ------------------------------------------------------------------

    st.subheader("SPY with 5% Canary Signals")
    spy_fig, spy_canary_chart_info = build_price_chart(
        spy_full, "SPY Price", "SPY", lookback_days=LOOKBACK_DAYS_PRICE
    )
    st.plotly_chart(spy_fig, use_container_width=True)
    st.caption(spy_canary_chart_info["subtitle"])

    st.subheader("QQQ (NASDAQ) with 5% Canary Signals")
    qqq_fig, _ = build_price_chart(
        qqq_full, "QQQ Price", "QQQ", lookback_days=LOOKBACK_DAYS_PRICE
    )
    st.plotly_chart(qqq_fig, use_container_width=True)
    st.caption("Blue: price, Orange: 21-day EMA, Green: 200-day EMA.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # VIX & Tsunami panel
    # ------------------------------------------------------------------

    st.subheader("VIX & Volatility Tsunami Watch")

    timeframe = st.radio(
        "VIX timeframe", ["Daily", "Weekly"], horizontal=True, index=0
    )

    vix_fig, _ = build_vix_chart(vix_full, timeframe=timeframe)
    st.plotly_chart(vix_fig, use_container_width=True)
    st.caption(
        "Lines: VIX, VIX 20-day stdev, VVIX 20-day stdev. "
        "Diamonds mark Tsunami compression signals."
    )


if __name__ == "__main__":
    main()
