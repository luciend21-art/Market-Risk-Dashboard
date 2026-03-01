# app.py
# Streamlit Relative Rotation Graph (RRG) dashboard
# Includes:
# - Predefined universes + Manage Universe (CSV/XLSX upload) with Groups
# - "First group wins" de-dupe (no duplicates if ticker appears twice)
# - Daily/Weekly toggle affecting chart + tables (Daily=1y default, Weekly=3y default)
# - Cleaner chart: tail thinning option to reduce clutter, head marker visible
# - Snapshot tables with color legend per ticker (matches plot colors)

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ----------------------------
# Predefined universes
# ----------------------------

SPDR_SECTORS: Dict[str, str] = {
    "XLB Materials": "XLB",
    "XLC Communication Services": "XLC",
    "XLE Energy": "XLE",
    "XLF Financials": "XLF",
    "XLI Industrials": "XLI",
    "XLK Technology": "XLK",
    "XLP Consumer Staples": "XLP",
    "XLRE Real Estate": "XLRE",
    "XLU Utilities": "XLU",
    "XLV Health Care": "XLV",
    "XLY Consumer Discretionary": "XLY",
}

THEMES: Dict[str, str] = {
    "Semiconductors (SOXX)": "SOXX",
    "Semiconductors 2 (SMH)": "SMH",
    "Cybersecurity (HACK)": "HACK",
    "Cybersecurity 2 (CIBR)": "CIBR",
    "Cloud (CLOU)": "CLOU",
    "Cloud 2 (SKYY)": "SKYY",
    "Software (IGV)": "IGV",
    "Defense & Aerospace (ITA)": "ITA",
    "Defense & Aerospace 2 (XAR)": "XAR",
    "European Defense (EUAD)": "EUAD",
    "Clean Energy (ICLN)": "ICLN",
    "Solar (TAN)": "TAN",
    "Fintech / Innovation (ARKF)": "ARKF",
    "Infrastructure (PAVE)": "PAVE",
    "Digital Infrastructure (DTCR)": "DTCR",
    "Digital Infrastructure 2 (TCAI)": "TCAI",
    "Bitcoin Mining / HPC 2 (WGMI)": "WGMI",
    "Home Construction (ITB)": "ITB",
    "Natural Gas (BOIL)": "BOIL",
    "Robotics (ROBO)": "ROBO",
    "Robotics 2 (BOTZ)": "BOTZ",
    "Nuclear (NLR)": "NLR",
    "Nuclear 2 (NUKZ)": "NUKZ",
    "Biotech (ARKG)": "ARKG",
    "Biotech 2 (BIB)": "BIB",
    "Pharmaceutical (PPH)": "PPH",
    "Drone 2 (ARKQ)": "ARKQ",
    "Brokerage 2 (IAI)": "IAI",
    "Retail Shopping (XRT)": "XRT",
    "Space (UFO)": "UFO",
    "Regional Banking (KRE)": "KRE",
    "Banking (KBE)": "KBE",
    "Airlines (JETS)": "JETS",
    "Rare Earth (REMX)": "REMX",
    "Quantum (QTUM)": "QTUM",
    "Cannabis (MSOS)": "MSOS",
}

COMMODITIES: Dict[str, str] = {
    "Gold Miners (RING)": "RING",
    "Gold (IAU)": "IAU",
    "Silver Miners (SIL)": "SIL",
    "Silver (SLV)": "SLV",
    "Copper Miners (COPX)": "COPX",
    "Crude Oil (USO)": "USO",
    "Bitcoin proxy (BTC-USD)": "BTC-USD",
    "Ethereum proxy (ETH-USD)": "ETH-USD",
    "Solana proxy (SOL-USD)": "SOL-USD",
}

COUNTRIES: Dict[str, str] = {
    "All World ex-US (VEU)": "VEU",
    "Emerging Mkts ex-China (EMXC)": "EMXC",
    "Brazil (EWZ)": "EWZ",
    "China (MCHI)": "MCHI",
    "China Internet (KWEB)": "KWEB",
    "Germany (EWG)": "EWG",
    "Canada (EWC)": "EWC",
    "Singapore (EWS)": "EWS",
}

UNIVERSES: Dict[str, Dict[str, str]] = {
    "SPDR Sectors": SPDR_SECTORS,
    "Themes ETFs": THEMES,
    "Commodity ETFs": COMMODITIES,
    "Country ETFs": COUNTRIES,
}

DEFAULT_BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA"]

# Plotly qualitative palette (stable)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2",
]


# ----------------------------
# Manage Universe upload
# ----------------------------

def _read_universe_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"])

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            return pd.DataFrame(columns=["Group", "Ticker", "Name"])
    except Exception:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"])

    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    def pick(*options):
        for o in options:
            if o in col_map:
                return col_map[o]
        return None

    c_group = pick("group", "universe", "category")
    c_ticker = pick("ticker", "symbol")
    c_name = pick("name", "description", "label")

    if c_ticker is None or c_group is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"])

    out = pd.DataFrame()
    out["Group"] = df[c_group].astype(str).str.strip()
    out["Ticker"] = df[c_ticker].astype(str).str.strip().str.upper()
    if c_name is not None:
        out["Name"] = df[c_name].astype(str).str.strip()
    else:
        out["Name"] = out["Ticker"]

    out = out.replace({"": np.nan}).dropna(subset=["Group", "Ticker"])
    # Remove duplicates (keep first: "first group wins")
    out = out.drop_duplicates(subset=["Ticker"], keep="first")
    return out[["Group", "Ticker", "Name"]]


def build_master_universe(upload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a master universe table:
      Group | Ticker | Name | Source
    De-dupe by Ticker: first occurrence wins (predefined first, then upload, or vice versa).
    We’ll use: predefined first, then upload -> upload will NOT override predefined by default.
    """
    rows = []
    for uni_name, mapping in UNIVERSES.items():
        for name, ticker in mapping.items():
            rows.append({"Group": uni_name, "Ticker": ticker, "Name": name, "Source": "Predefined"})

    base = pd.DataFrame(rows)

    if upload_df is not None and not upload_df.empty:
        up = upload_df.copy()
        up["Source"] = "Uploaded"
        # IMPORTANT: Option A "first group wins" -> keep first by Ticker
        combined = pd.concat([base, up], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Ticker"], keep="first")
        return combined.reset_index(drop=True)

    return base.reset_index(drop=True)


# ----------------------------
# Yahoo Finance download (robust)
# ----------------------------

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(
    tickers: Tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with one column per ticker.
    Prefers Close; falls back to Adj Close.
    Handles:
      - single ticker => flat columns
      - multi-ticker => MultiIndex columns
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in set(lvl0):
            close = df["Close"].copy()
        elif "Adj Close" in set(lvl0):
            close = df["Adj Close"].copy()
        else:
            return pd.DataFrame()

        close = close.reindex(columns=list(tickers))
        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    # single ticker
    cols = list(df.columns)
    t0 = tickers[0]
    if "Close" in cols:
        close = df[["Close"]].rename(columns={"Close": t0}).copy()
    elif "Adj Close" in cols:
        close = df[["Adj Close"]].rename(columns={"Adj Close": t0}).copy()
    else:
        return pd.DataFrame()

    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    return prices.resample("W-FRI").last().dropna(how="all")


# ----------------------------
# RRG math (pragmatic approximation)
# ----------------------------

def _zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(52, min_periods=20).mean()
    sd = s.rolling(52, min_periods=20).std()
    return (s - mu) / sd


def compute_rrg_series(
    asset_prices: pd.DataFrame,
    bench_prices: pd.Series,
    lookback: int,
    momentum: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proxy approximation of JdK RS-Ratio / RS-Momentum:
      RS = asset / benchmark
      RS-Ratio = zscore(rolling mean of RS over lookback)
      RS-Momentum = zscore(diff(RS-Ratio, momentum))
    """
    df = asset_prices.join(bench_prices.rename("BENCH"), how="inner")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bench = df["BENCH"]
    assets = df.drop(columns=["BENCH"])

    rs = assets.divide(bench, axis=0)
    rs_smooth = rs.rolling(lookback, min_periods=max(10, lookback // 3)).mean()

    rs_ratio_z = rs_smooth.apply(_zscore)
    rs_mom_z = rs_ratio_z.diff(momentum).apply(_zscore)

    return rs_ratio_z, rs_mom_z


# ----------------------------
# Interpretation helpers
# ----------------------------

def state_from_value(v: float, flat_band: float = 0.25) -> str:
    if pd.isna(v):
        return "N/A"
    if abs(v) <= flat_band:
        return "Flat"
    return "Improving" if v > 0 else "Weakening"


def quadrant(rs: float, mom: float) -> str:
    if pd.isna(rs) or pd.isna(mom):
        return "Unknown"
    if rs >= 0 and mom >= 0:
        return "Leading"
    if rs < 0 and mom >= 0:
        return "Improving"
    if rs < 0 and mom < 0:
        return "Lagging"
    return "Weakening"


def direction_arrow(dx: float, dy: float) -> str:
    if pd.isna(dx) or pd.isna(dy) or (dx == 0 and dy == 0):
        return "•"
    ang = math.degrees(math.atan2(dy, dx))
    bins = [
        (-22.5, 22.5, "→"),
        (22.5, 67.5, "↗"),
        (67.5, 112.5, "↑"),
        (112.5, 157.5, "↖"),
        (157.5, 180.0, "←"),
        (-180.0, -157.5, "←"),
        (-157.5, -112.5, "↙"),
        (-112.5, -67.5, "↓"),
        (-67.5, -22.5, "↘"),
    ]
    for lo, hi, a in bins:
        if lo <= ang < hi:
            return a
    return "•"


def speed_bucket_label(speeds: pd.Series, symbol_speed: float) -> str:
    speeds = speeds.dropna()
    if speeds.empty or pd.isna(symbol_speed):
        return "N/A"

    # percentile rank (0..1)
    pct = float((speeds < symbol_speed).mean())
    if pct < 0.25:
        return "Slow"
    if pct < 0.60:
        return "Medium"
    if pct < 0.85:
        return "Fast"
    return "Hot/Climactic"


# ----------------------------
# Plot: RRG with dynamic axis bounds + tail thinning
# ----------------------------

def _compute_axis_bounds(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    tail_len: int,
    pad_frac: float = 0.10,
    min_span: float = 2.5,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []

    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]
    for sym in symbols:
        x = rs_ratio_z[sym].dropna()
        y = rs_mom_z[sym].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < max(3, tail_len):
            continue
        tidx = idx[-tail_len:]
        xt = rs_ratio_z.loc[tidx, sym].astype(float).values
        yt = rs_mom_z.loc[tidx, sym].astype(float).values
        xs.extend([v for v in xt if np.isfinite(v)])
        ys.extend([v for v in yt if np.isfinite(v)])

    if not xs or not ys:
        return (-3.0, 3.0), (-3.0, 3.0)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    xspan = max(xmax - xmin, min_span)
    yspan = max(ymax - ymin, min_span)

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    xspan *= (1.0 + pad_frac)
    yspan *= (1.0 + pad_frac)

    return (xmid - xspan / 2.0, xmid + xspan / 2.0), (ymid - yspan / 2.0, ymid + yspan / 2.0)


def _thin_tail(xt: pd.Series, yt: pd.Series, max_points: int) -> Tuple[pd.Series, pd.Series]:
    """
    Reduce clutter by keeping <= max_points points while preserving order and head.
    Always keeps the most recent point.
    """
    n = len(xt)
    if n <= max_points:
        return xt, yt
    # evenly spaced indices, keep last
    keep_idx = np.linspace(0, n - 1, max_points).round().astype(int)
    keep_idx[-1] = n - 1
    keep_idx = np.unique(keep_idx)
    return xt.iloc[keep_idx], yt.iloc[keep_idx]


def make_rrg_figure(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    tail_len: int,
    title: str,
    auto_zoom: bool = True,
    zoom_padding: float = 0.10,
    fixed_range: bool = False,
    tail_max_points: int = 18,
) -> go.Figure:
    """
    Tail connects to head:
      - Tail trace includes points up through most recent point
      - Head trace overlays a large diamond marker on most recent point
    Cleaner output:
      - tail_max_points thins tail if tail_len is large (prevents spaghetti)
    """
    fig = go.Figure()

    if rs_ratio_z.empty or rs_mom_z.empty:
        fig.update_layout(title=title, height=540, margin=dict(l=30, r=30, t=70, b=40))
        return fig

    if fixed_range:
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
    else:
        if auto_zoom:
            (x_min, x_max), (y_min, y_max) = _compute_axis_bounds(
                rs_ratio_z, rs_mom_z, tail_len=tail_len, pad_frac=zoom_padding
            )
        else:
            x_min, x_max = -3.0, 3.0
            y_min, y_max = -3.0, 3.0

    # Quadrant shading
    fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max,
                  fillcolor="rgba(120,150,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max,
                  fillcolor="rgba(120,255,150,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0,
                  fillcolor="rgba(255,120,120,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0,
                  fillcolor="rgba(255,210,120,0.08)", line_width=0, layer="below")

    # Crosshairs
    fig.add_shape(type="line", x0=x_min, y0=0, x1=x_max, y1=0,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))
    fig.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))

    # Quadrant labels
    fig.add_annotation(x=x_min + 0.03 * (x_max - x_min), y=y_max - 0.05 * (y_max - y_min),
                       text="Improving", showarrow=False, font=dict(size=12, color="blue"))
    fig.add_annotation(x=x_max - 0.03 * (x_max - x_min), y=y_max - 0.05 * (y_max - y_min),
                       text="Leading", showarrow=False, xanchor="right",
                       font=dict(size=12, color="green"))
    fig.add_annotation(x=x_min + 0.03 * (x_max - x_min), y=y_min + 0.05 * (y_max - y_min),
                       text="Lagging", showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(x=x_max - 0.03 * (x_max - x_min), y=y_min + 0.05 * (y_max - y_min),
                       text="Weakening", showarrow=False, xanchor="right",
                       font=dict(size=12, color="orange"))

    # Stable color mapping by sorted symbol list
    symbols = sorted([c for c in rs_ratio_z.columns if c in rs_mom_z.columns])
    color_map = {sym: PALETTE[i % len(PALETTE)] for i, sym in enumerate(symbols)}

    # Plot each symbol
    for sym in symbols:
        x = rs_ratio_z[sym].dropna()
        y = rs_mom_z[sym].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < max(3, tail_len):
            continue

        tail_idx = idx[-tail_len:]
        xt = rs_ratio_z.loc[tail_idx, sym]
        yt = rs_mom_z.loc[tail_idx, sym]

        # Thin tail to reduce clutter while preserving tail→head structure
        xt2, yt2 = _thin_tail(xt, yt, max_points=tail_max_points)

        col = color_map[sym]

        # Tail
        fig.add_trace(
            go.Scatter(
                x=xt2,
                y=yt2,
                mode="lines+markers",
                name=sym,
                line=dict(width=2, color=col),
                marker=dict(size=5, color=col),
                hovertemplate=(
                    f"<b>{sym}</b><br>"
                    "RS-Ratio(z): %{x:.2f}<br>"
                    "RS-Mom(z): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Head (latest)
        xh = float(xt.iloc[-1])
        yh = float(yt.iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[xh],
                y=[yh],
                mode="markers",
                name=f"{sym} (latest)",
                marker=dict(
                    size=14,
                    symbol="diamond",
                    color=col,
                    line=dict(width=2, color="black"),
                ),
                hovertemplate=(
                    f"<b>{sym} (latest)</b><br>"
                    "RS-Ratio(z): %{x:.2f}<br>"
                    "RS-Mom(z): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        height=540,
        margin=dict(l=30, r=30, t=70, b=40),
        xaxis=dict(title="RS-Ratio (standardized)", range=[x_min, x_max], zeroline=False),
        yaxis=dict(title="RS-Momentum (standardized)", range=[y_min, y_max], zeroline=False),
    )
    return fig


# ----------------------------
# Snapshot tables
# ----------------------------

@dataclass
class RRGRow:
    symbol: str
    name: str
    group: str
    rs_state: str
    mom_state: str
    arrow: str
    speed_label: str
    quad: str
    rs_val: float
    mom_val: float
    color: str


def build_snapshot_table(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    meta: pd.DataFrame,  # Group/Ticker/Name
    color_map: Dict[str, str],
) -> pd.DataFrame:
    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]

    meta_idx = meta.set_index("Ticker")

    # speed distribution across selected symbols (last-step speed)
    spds = {}
    for sym in symbols:
        s1 = rs_ratio_z[sym].dropna()
        s2 = rs_mom_z[sym].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 3:
            continue
        x_last, y_last = float(s1.loc[idx[-1]]), float(s2.loc[idx[-1]])
        x_prev, y_prev = float(s1.loc[idx[-2]]), float(s2.loc[idx[-2]])
        spds[sym] = math.sqrt((x_last - x_prev) ** 2 + (y_last - y_prev) ** 2)

    spds_s = pd.Series(spds, dtype="float64")

    rows: List[dict] = []
    for sym in symbols:
        s1 = rs_ratio_z[sym].dropna()
        s2 = rs_mom_z[sym].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 3:
            continue

        x_last, y_last = float(s1.loc[idx[-1]]), float(s2.loc[idx[-1]])
        x_prev, y_prev = float(s1.loc[idx[-2]]), float(s2.loc[idx[-2]])
        dx, dy = x_last - x_prev, y_last - y_prev

        rs_state = state_from_value(x_last, flat_band=0.25)
        mom_state = state_from_value(y_last, flat_band=0.25)
        arr = direction_arrow(dx, dy)
        spd_label = speed_bucket_label(spds_s, spds.get(sym, np.nan))
        q = quadrant(x_last, y_last)

        if sym in meta_idx.index:
            g = str(meta_idx.loc[sym, "Group"])
            nm = str(meta_idx.loc[sym, "Name"])
        else:
            g = "Extra"
            nm = sym

        rows.append(
            {
                "Color": "■",
                "Ticker": sym,
                "Name": nm,
                "Group": g,
                "RS-Ratio": rs_state,
                "Momentum": mom_state,
                "Direction": arr,
                "Rotation Speed": spd_label,
                "Quadrant": q,
                "_RS": x_last,
                "_MOM": y_last,
                "_COLORHEX": color_map.get(sym, "#444444"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    quad_order = {"Leading": 0, "Improving": 1, "Weakening": 2, "Lagging": 3, "Unknown": 4}
    df["_QO"] = df["Quadrant"].map(quad_order).fillna(9)
    df = df.sort_values(["_QO", "_RS", "_MOM"], ascending=[True, False, False]).reset_index(drop=True)
    return df


def top3_tables(snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    show_cols = ["Color", "Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]

    if snapshot.empty:
        return pd.DataFrame(), pd.DataFrame()

    leading = snapshot[snapshot["Quadrant"] == "Leading"].copy()
    improving = snapshot[snapshot["Quadrant"] == "Improving"].copy()

    leading = leading.sort_values(["_RS", "_MOM"], ascending=[False, False]).head(3)
    improving = improving.sort_values(["_MOM", "_RS"], ascending=[False, False]).head(3)

    return leading[show_cols].reset_index(drop=True), improving[show_cols].reset_index(drop=True)


def colorize_table(df: pd.DataFrame) -> Optional[pd.io.formats.style.Styler]:
    if df is None or df.empty:
        return None

    # Color the "Color" swatch to match line color
    def _style_row(row):
        hexcol = row.get("_COLORHEX", "#444444")
        return ["color: %s; font-weight: 900;" % hexcol if c == "Color" else "" for c in row.index]

    # Build styler; include hidden _COLORHEX in styling, then drop it for display
    styled = df.style.apply(_style_row, axis=1)
    return styled


# ----------------------------
# Main app
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")

    st.title("Relative Rotation Graph (RRG)")
    st.caption(
        "Track sector, theme, commodity, and country rotation vs a benchmark. "
        "Approximation of JdK RS-Ratio and RS-Momentum."
    )

    # ---- Sidebar: Manage Universe upload ----
    st.sidebar.header("RRG Settings")

    with st.sidebar.expander("Manage Universe (CSV/XLSX upload)", expanded=False):
        st.markdown(
            "**Template columns**: `Group`, `Ticker`, `Name` (Name optional)\n\n"
            "Example:\n"
            "```\n"
            "Group,Ticker,Name\n"
            "Themes,SOXX,Semiconductors\n"
            "Commodities,COPX,Copper Miners\n"
            "Countries,MCHI,China\n"
            "```\n"
        )
        uploaded = st.file_uploader("Upload Universe CSV/XLSX", type=["csv", "xlsx", "xls"])

    upload_df = _read_universe_upload(uploaded)
    master = build_master_universe(upload_df)

    # ---- Universe group filter ----
    # Keep your organizing drop-down (Groups)
    group_options = sorted(master["Group"].dropna().unique().tolist())
    selected_group = st.sidebar.selectbox("Universe group filter", group_options, index=group_options.index("SPDR Sectors") if "SPDR Sectors" in group_options else 0)

    group_df = master[master["Group"] == selected_group].copy()
    # labels
    group_df["Label"] = group_df["Name"].astype(str) + " (" + group_df["Ticker"].astype(str) + ")"

    benchmark = st.sidebar.selectbox("Benchmark", DEFAULT_BENCHMARKS, index=0)

    timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=1)
    default_hist_years = 1 if timeframe == "Daily" else 3
    history_years = st.sidebar.slider("History (years, daily data)", 1, 5, default_hist_years)

    lookback_weeks = st.sidebar.slider("Lookback (weeks)", 12, 104, 52, 1)
    momentum_weeks = st.sidebar.slider("Momentum (weeks)", 4, 52, 13, 1)
    tail_len_weeks = st.sidebar.slider("Tail length (weeks)", 4, 52, 13, 1)

    st.sidebar.subheader("Chart view")
    fixed_range = st.sidebar.checkbox("Fixed range (±3)", value=False)
    auto_zoom = st.sidebar.checkbox("Auto-zoom to fit points", value=True)
    zoom_padding = st.sidebar.slider("Zoom padding", 0.05, 0.35, 0.10, 0.01)
    tail_max_points = st.sidebar.slider("Tail point cap (reduces clutter)", 8, 30, 18, 1)

    # Choose tickers
    default_n = min(12, len(group_df))
    default_labels = group_df["Label"].iloc[:default_n].tolist()
    chosen_labels = st.sidebar.multiselect(
        "Choose ETFs",
        options=group_df["Label"].tolist(),
        default=default_labels,
    )

    extra = st.sidebar.text_input("Extra tickers (comma-separated)", value="")
    extra_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]

    chosen_tickers = group_df.set_index("Label").loc[chosen_labels, "Ticker"].tolist() if chosen_labels else []
    # De-dupe (keep first)
    tickers: List[str] = []
    seen = set()
    for t in chosen_tickers + extra_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        st.warning("Select at least one ETF/ticker.")
        return

    # Date range
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(history_years * 365.25) + 7)

    # Download
    all_tickers = tuple(sorted(set(tickers + [benchmark])))
    prices = download_prices(all_tickers, start=start, end=end)

    if prices.empty:
        st.error("No data returned from Yahoo Finance for the selected tickers.")
        return

    if benchmark not in prices.columns:
        st.error(f"Benchmark '{benchmark}' data missing. Try a different benchmark.")
        return

    # Remove dead columns
    prices = prices[[c for c in prices.columns if prices[c].notna().sum() > 0]].copy()

    # Timeframe transforms
    if timeframe == "Weekly":
        panel = to_weekly(prices)
        lookback = lookback_weeks
        momentum = momentum_weeks
        tail_points = tail_len_weeks
        required_points = lookback_weeks + momentum_weeks + tail_len_weeks + 10
        chart_title = f"{selected_group} vs {benchmark} (Weekly)"
    else:
        # Daily view: keep daily but convert weeks->trading-days approximations
        panel = prices.dropna(how="all").copy()
        lookback = int(lookback_weeks * 5)
        momentum = int(momentum_weeks * 5)
        tail_points = int(tail_len_weeks * 5)
        required_points = int((lookback_weeks + momentum_weeks + tail_len_weeks) * 5 + 30)
        chart_title = f"{selected_group} vs {benchmark} (Daily)"

    bench_s = panel[benchmark].dropna()

    # Filter assets with sufficient history
    asset_cols = [t for t in tickers if t in panel.columns and t != benchmark]

    dropped: List[str] = []
    kept: List[str] = []
    for t in asset_cols:
        n = int(panel[t].dropna().shape[0])
        if n < required_points:
            dropped.append(f"{t} (have {n} pts, need ≥ {required_points})")
        else:
            kept.append(t)

    if dropped:
        st.sidebar.warning("Dropped (insufficient history):\n\n" + "\n".join(dropped))

    if not kept:
        st.warning("Not enough data to build RRG (shorten lookback/momentum/tail or increase history).")
        return

    assets = panel[kept].dropna(how="all")
    bench_s = bench_s.reindex(assets.index).dropna()
    assets = assets.reindex(bench_s.index).dropna(how="all")

    if assets.empty or bench_s.empty:
        st.warning("No overlapping data between benchmark and selected ETFs.")
        return

    rs_ratio_z, rs_mom_z = compute_rrg_series(assets, bench_s, lookback=lookback, momentum=momentum)

    # Stable colors
    symbols = sorted([c for c in rs_ratio_z.columns if c in rs_mom_z.columns])
    color_map = {sym: PALETTE[i % len(PALETTE)] for i, sym in enumerate(symbols)}

    fig = make_rrg_figure(
        rs_ratio_z=rs_ratio_z,
        rs_mom_z=rs_mom_z,
        tail_len=tail_points,
        title=chart_title,
        auto_zoom=auto_zoom,
        zoom_padding=zoom_padding,
        fixed_range=fixed_range,
        tail_max_points=tail_max_points,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Snapshot table
    meta_for_selected = master[master["Ticker"].isin(kept)].copy()
    snapshot = build_snapshot_table(rs_ratio_z, rs_mom_z, meta_for_selected, color_map)
    if snapshot.empty:
        st.warning("Could not compute snapshot table (insufficient valid data after alignment).")
        return

    top_leading, top_improving = top3_tables(snapshot)

    st.subheader("Latest RRG Snapshot (interpreted)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 3 Leading**")
        if top_leading.empty:
            st.info("No symbols currently in the Leading quadrant.")
        else:
            sty = colorize_table(top_leading.assign(_COLORHEX=top_leading["_COLORHEX"] if "_COLORHEX" in top_leading.columns else "#444444"))
            # ensure the hidden helper exists for styling
            df_disp = top_leading.copy()
            df_disp["_COLORHEX"] = top_leading.get("_COLORHEX", "#444444")
            st.dataframe(colorize_table(df_disp), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Top 3 Improving**")
        if top_improving.empty:
            st.info("No symbols currently in the Improving quadrant.")
        else:
            df_disp = top_improving.copy()
            df_disp["_COLORHEX"] = top_improving.get("_COLORHEX", "#444444")
            st.dataframe(colorize_table(df_disp), use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("Universe Snapshot Table (all selected symbols)", expanded=False):
        show_cols = ["Color", "Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant"]
        df_disp = snapshot[show_cols + ["_COLORHEX"]].copy()
        st.dataframe(colorize_table(df_disp), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Note: If the chart still feels cluttered, reduce Tail length, increase Lookback, "
        "or lower the Tail point cap. Weekly generally looks cleaner than Daily for rotation."
    )


if __name__ == "__main__":
    main()
