import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

HISTORY_YEARS = 5          # how much history to download for EMAs, highs, etc.
PLOT_DAYS = 90             # how much history to show on the SPY / QQQ charts
EMA_FAST = 21              # daily fast EMA
EMA_SLOW = 200             # daily slow EMA
VIX_SD_WINDOW = 20         # window for VIX / VVIX stdev
TSUNAMI_LOOKBACK = 120     # days for "tsunami in window?" check

st.set_page_config(page_title="Market Risk Dashboard", layout="wide")


# ------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------

def fetch_close_history(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download full history for a single symbol, return a DataFrame with:
    index = Date, column = 'Close'
    """
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    # Ensure we have a simple Date index and a single Close column
    df = df[['Close']].copy()
    df.index.name = "Date"
    return df


def add_daily_emas(df: pd.DataFrame,
                   fast_span: int = EMA_FAST,
                   slow_span: int = EMA_SLOW) -> pd.DataFrame:
    """
    Add daily EMAs to a full-history price DataFrame (df['Close'] must exist).
    """
    closes = df['Close']
    df['ema_fast'] = closes.ewm(span=fast_span, adjust=False).mean()
    df['ema_slow'] = closes.ewm(span=slow_span, adjust=False).mean()
    return df


@st.cache_data(show_spinner=True)
def load_data():
    """
    Load several years of daily history for SPY, QQQ, VIX, and VVIX.
    Compute EMAs on the full SPY / QQQ history, and return full DataFrames.

    We *only* slice to the last PLOT_DAYS later for plotting, so EMAs and
    1-year highs are based on the full history.
    """
    end = datetime.today()
    start_hist = end - timedelta(days=365 * HISTORY_YEARS)

    spy_full = fetch_close_history("SPY", start_hist, end)
    qqq_full = fetch_close_history("QQQ", start_hist, end)
    vix_full = fetch_close_history("^VIX", start_hist, end)
    vvix_full = fetch_close_history("^VVIX", start_hist, end)

    # EMAs on full history (Close, not Adj Close)
    spy_full = add_daily_emas(spy_full)
    qqq_full = add_daily_emas(qq_full)

    return spy_full, qqq_full, vix_full, vvix_full


# ------------------------------------------------------------
# Risk status helpers
# ------------------------------------------------------------

def canary_status(spy_full: pd.DataFrame):
    """
    Simple 1-year drawdown based Canary status.
    Uses full SPY history so the 1-year high is accurate.
    """
    closes = spy_full['Close']
    last_close = float(closes.iloc[-1])

    # Use up to 252 trading days for "1-year" high, but handle shorter histories safely
    window = min(len(closes), 252)
    high_1y = float(closes.tail(window).max())

    pct_off = (last_close / high_1y - 1.0) * 100.0

    if pct_off >= -5.0:
        emoji = "🟢"
        headline = "Shallow pullback (<5% from 1-year high)"
        detail = f"SPY is {pct_off:.1f}% below its 1-year high. No Canary warning."
    elif pct_off >= -10.0:
        emoji = "🟡"
        headline = "Medium drawdown (5–10% off high)"
        detail = f"SPY is {pct_off:.1f}% below its 1-year high. Stay alert for Canary / Tsunami signals."
    else:
        emoji = "🔴"
        headline = "Deep drawdown (>10% off high)"
        detail = f"SPY is {pct_off:.1f}% below its 1-year high. Risk is elevated — respect risk controls."

    return emoji, headline, detail, pct_off


def build_vix_df(vix_full: pd.DataFrame, vvix_full: pd.DataFrame) -> pd.DataFrame:
    """
    Align VIX and VVIX, and compute 20-day stdevs.
    """
    df = pd.DataFrame(index=vix_full.index.union(vvix_full.index))
    df['VIX'] = vix_full['Close']
    df['VVIX'] = vvix_full['Close']
    df['VIX_SD20'] = df['VIX'].rolling(VIX_SD_WINDOW).std()
    df['VVIX_SD20'] = df['VVIX'].rolling(VIX_SD_WINDOW).std()
    return df


def tsunami_status(vix_df: pd.DataFrame):
    """
    Very simple Tsunami status based on the last TSUNAMI_LOOKBACK days of VIX/VVIX stdevs.
    (This keeps the same spirit / window as our previous working version.)
    """
    recent = vix_df.dropna().tail(TSUNAMI_LOOKBACK)
    if recent.empty:
        return "⚪", "No Tsunami reading", "Insufficient data for Tsunami analysis."

    last = recent.iloc[-1]

    # A crude compression check: both stdevs below their recent medians
    vix_sd_med = recent['VIX_SD20'].median()
    vvix_sd_med = recent['VVIX_SD20'].median()

    compressed = (last['VIX_SD20'] < vix_sd_med) and (last['VVIX_SD20'] < vvix_sd_med)

    if compressed:
        emoji = "🟡"
        headline = "Tsunami watch (volatility compression)"
        detail = "VIX & VVIX 20-day stdevs are compressed vs recent medians. A future volatility spike is more likely."
    else:
        emoji = "🟢"
        headline = "No Tsunami in window"
        detail = "No significant VIX/VVIX compression in the last few months."

    return emoji, headline, detail


# ------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------

def build_price_chart(df_full: pd.DataFrame, title: str, price_name: str):
    """
    Build a Plotly line chart for SPY or QQQ:
      - Uses full-history EMAs in df_full (ema_fast / ema_slow)
      - Slices to the last PLOT_DAYS for display
      - Auto-fits Y with a small padding
      - Includes a legend for Price / 21-day EMA / 200-day EMA
    """
    recent = df_full.tail(PLOT_DAYS).copy()

    fig = go.Figure()

    # Price
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent['Close'],
            mode="lines",
            name=price_name
        )
    )

    # 21-day EMA
    if 'ema_fast' in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent['ema_fast'],
                mode="lines",
                name="21-day EMA"
            )
        )

    # 200-day EMA
    if 'ema_slow' in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent['ema_slow'],
                mode="lines",
                name="200-day EMA"
            )
        )

    # Auto-fit Y with a bit of padding
    ymin = recent['Close'].min()
    ymax = recent['Close'].max()
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0

    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
        xaxis_title="Date",
        yaxis_title=price_name
    )
    fig.update_yaxes(range=[ymin - pad, ymax + pad])

    return fig


def build_vix_chart(vix_df: pd.DataFrame, timeframe_label: str = "Daily"):
    """
    Build the VIX & Tsunami chart.
    """
    recent = vix_df.tail(365).copy()  # show ~1 year

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent['VIX'],
            mode="lines",
            name="VIX"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent['VIX_SD20'],
            mode="lines",
            name="VIX 20-day stdev",
            line=dict(dash="dot")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent['VVIX_SD20'],
            mode="lines",
            name="VVIX 20-day stdev",
            line=dict(dash="dot")
        )
    )

    fig.update_layout(
        title=f"VIX & Volatility Tsunami Watch ({timeframe_label})",
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
        xaxis_title="Date",
        yaxis_title="VIX / stdev"
    )

    return fig


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------

def main():
    st.title("Market Risk Dashboard")
    st.caption("5% Canary • Volatility Tsunami • Cross-Asset Regimes")

    try:
        spy_full, qqq_full, vix_full, vvix_full = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # ----- Status panels -----
    can_emoji, can_headline, can_detail, pct_off = canary_status(spy_full)
    vix_df = build_vix_df(vix_full, vvix_full)
    tsu_emoji, tsu_headline, tsu_detail = tsunami_status(vix_df)

    spy_price = float(spy_full['Close'].iloc[-1])
    qqq_price = float(qq_full['Close'].iloc[-1])

    # One-year off-highs for snapshot (based on full history)
    def pct_off_high(df: pd.DataFrame) -> float:
        closes = df['Close']
        last = float(closes.iloc[-1])
        window = min(len(closes), 252)
        high_1y = float(closes.tail(window).max())
        return (last / high_1y - 1.0) * 100.0

    spy_off = pct_off_high(spy_full)
    qqq_off = pct_off_high(qq_full)

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
        st.markdown(f"**SPY:** {spy_price:.2f}")
        st.markdown(f"Off 1-year high: {spy_off:.1f}%")
        st.markdown(f"**QQQ:** {qqq_price:.2f}")
        st.markdown(f"Off 1-year high: {qqq_off:.1f}%")

    st.markdown("---")

    # ----- SPY / QQQ charts -----
    spy_fig = build_price_chart(spy_full, "SPY with 5% Canary Signals", "SPY Price")
    qqq_fig = build_price_chart(qq_full, "QQQ (NASDAQ) with 5% Canary Signals", "QQQ Price")

    st.plotly_chart(spy_fig, use_container_width=True)
    st.plotly_chart(qq_fig, use_container_width=True)

    st.markdown("---")

    # ----- VIX & Tsunami chart -----
    vix_fig = build_vix_chart(vix_df, timeframe_label="Daily")
    st.plotly_chart(vix_fig, use_container_width=True)


if __name__ == "__main__":
    main()

