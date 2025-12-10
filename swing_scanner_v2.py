# -*- coding: utf-8 -*-
"""
ALL-US Breakout Screen v2 (Polygon-only, CSV + optional Telegram)

Goal:
- Find stocks with:
    • Strong 6-month relative strength vs SPY (top X%)
    • Sufficient liquidity (volume + dollar volume)
    • Healthy volatility (ADR% between min/max)
    • Price above key moving averages (10 & 22 day)
    • Within a controlled depth of recent 120-day high (base)
    • Volatility contraction (recent vol < long-term vol)
- Produce:
    • Main breakout list (strict criteria)
    • Preview / near-miss watchlist (slightly relaxed around edges)
    • CSV files + Telegram summary.

Data source: Polygon.io (no Yahoo Finance at all)
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from ta.trend import SMAIndicator

# ---------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------

ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_FAST = int(os.getenv("MA_FAST", "10"))
MA_SLOW = int(os.getenv("MA_SLOW", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))
RS_LOOKBACK_MONTHS = int(os.getenv("RS_LOOKBACK_MONTHS", "6"))
HIGH_LOOKBACK_DAYS = int(os.getenv("HIGH_LOOKBACK_DAYS", "120"))

MIN_ADR_PCT = float(os.getenv("MIN_ADR_PCT", "5"))
MAX_ADR_PCT = float(os.getenv("MAX_ADR_PCT", "12"))
MIN_AVG_VOL_30D = float(os.getenv("MIN_AVG_VOL_30D", "30000000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_RS_PCTILE = float(os.getenv("MIN_RS_PCTILE", "0.98"))

PREFILTER_MIN_PRICE = float(os.getenv("PREFILTER_MIN_PRICE", "3.0"))
PREFILTER_MIN_AVG_VOL5 = float(os.getenv("PREFILTER_MIN_AVG_VOL5", "5000000"))
PREFILTER_TOP_N_BY_DOLLAR_VOL = int(
    os.getenv("PREFILTER_TOP_N_BY_DOLLAR_VOL", "1200")
)

MAX_BASE_DEPTH_PCT = float(os.getenv("MAX_BASE_DEPTH_PCT", "35"))
MAX_DIST_FROM_HIGH_PCT = float(os.getenv("MAX_DIST_FROM_HIGH_PCT", "8"))
MAX_DIST_FROM_22SMA_PCT = float(os.getenv("MAX_DIST_FROM_22SMA_PCT", "15"))

VCP_SHORT = int(os.getenv("VCP_SHORT", "10"))
VCP_LONG = int(os.getenv("VCP_LONG", "40"))
MIN_VCP_RATIO = float(os.getenv("MIN_VCP_RATIO", "0.2"))
MAX_VCP_RATIO = float(os.getenv("MAX_VCP_RATIO", "0.7"))

TOP_N_TO_REPORT = int(os.getenv("TOP_N_TO_REPORT", "40"))
TOP_N_PREVIEW = int(os.getenv("TOP_N_PREVIEW", "20"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_SLEEP_SECONDS = float(os.getenv("POLYGON_SLEEP_SECONDS", "0.02"))

pd.options.display.float_format = "{:,.2f}".format


# ---------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------

def _polygon_get(url: str, params: dict | None = None) -> dict:
    """Low-level Polygon GET with API key + basic error handling."""
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY not set in environment.")
    if params is None:
        params = {}
    params = dict(params)
    params["apiKey"] = POLYGON_API_KEY
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    finally:
        # Be nice to rate limits
        time.sleep(POLYGON_SLEEP_SECONDS)


def polygon_download_daily(ticker: str, lookback_days: int) -> pd.DataFrame:
    """
    Download daily OHLCV bars for the last `lookback_days` using Polygon aggs.
    Returns DataFrame indexed by UTC datetime with columns:
    Open, High, Low, Close, Volume
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"

    try:
        data = _polygon_get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000},
        )
    except Exception:
        return pd.DataFrame()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    records = []
    for bar in results:
        ts = datetime.utcfromtimestamp(bar["t"] / 1000.0)
        records.append(
            {
                "Date": ts,
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.set_index("Date").sort_index()
    return df


def get_all_us_tickers() -> list[str]:
    """
    Universe: active US common stocks from Polygon.
    Filters to:
      - market = stocks
      - type   = CS (common stock)
      - active = true
      - ticker is alphabetic (no ETFs/units/+.A, etc.)
    """
    print("Fetching US tickers from Polygon...")
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": 1000,
    }

    tickers: list[str] = []
    while True:
        data = _polygon_get(url, params=params)
        for item in data.get("results", []):
            sym = item.get("ticker", "")
            # keep things like RGTI, HOOD; drop weird stuff
            if sym.isalpha():
                tickers.append(sym)

        next_url = data.get("next_url")
        if not next_url:
            break

        # for next_url Polygon wants apiKey again; we handle in _polygon_get
        url = next_url
        params = {}

    unique = sorted(set(tickers))
    print(f"Polygon tickers fetched: {len(unique)}")
    return unique


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def compute_adr_pct(df: pd.DataFrame, period: int = 14, ref_price: float | None = None) -> float:
    """Average daily range % based on High–Low."""
    if df.empty or len(df) < period + 1:
        return np.nan
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    price = float(ref_price) if ref_price is not None else float(df["Close"].iloc[-1])
    if price <= 0 or pd.isna(rng):
        return np.nan
    return float(rng / price * 100.0)


def prefilter_universe(
    tickers: list[str],
    min_price: float = 3.0,
    min_avg_vol5: float = 5_000_000,
    cap: int = 1500,
) -> list[str]:
    """
    Coarse universe filter to reduce API load using Polygon:
      - price >= min_price
      - 5-day avg volume >= min_avg_vol5
      - rank by 5-day dollar volume (price * volume), keep top `cap`.
    """
    print("Prefiltering universe (Polygon)...")
    scores: list[tuple[str, float]] = []

    for t in tickers:
        df = polygon_download_daily(t, lookback_days=15)
        if df.empty or len(df) < 5:
            continue

        px = float(df["Close"].iloc[-1])
        vol5 = float(df["Volume"].tail(5).mean())
        if px >= min_price and vol5 >= min_avg_vol5:
            dollar_vol5 = px * vol5
            scores.append((t, dollar_vol5))

    scores.sort(key=lambda x: x[1], reverse=True)
    trimmed = [t for t, _ in scores[:cap]]
    print(f"Universe after prefilter+cap: {len(trimmed)}")
    return trimmed


def rs_percentile(
    tickers: list[str],
    months: int = 6,
    benchmark: str = "SPY",
) -> pd.Series:
    """
    Relative strength vs benchmark over N months using Polygon:
      RS_raw = (stock_return - benchmark_return)
      RS_pctile = percentile rank of RS_raw across universe.
    """
    lookback_days = months * 32

    bench_df = polygon_download_daily(benchmark, lookback_days=lookback_days)
    if bench_df.empty:
        print("WARNING: failed to load benchmark data; RS will be empty.")
        return pd.Series(dtype=float)

    bench_close = bench_df["Close"].dropna()
    out: dict[str, float] = {}

    for t in tickers:
        df = polygon_download_daily(t, lookback_days=lookback_days)
        if df.empty:
            continue
        px = df["Close"].dropna()
        # align dates
        idx = px.index.intersection(bench_close.index)
        if len(idx) < 20:
            continue

        pxa = px.reindex(idx).ffill()
        bxa = bench_close.reindex(idx).ffill()

        r_stock = float(pxa.iloc[-1] / pxa.iloc[0] - 1.0)
        r_bench = float(bxa.iloc[-1] / bxa.iloc[0] - 1.0)
        out[t] = r_stock - r_bench

    if not out:
        return pd.Series(dtype=float)

    rs_raw = pd.Series(out)
    rs_pct = rs_raw.rank(pct=True)
    return rs_pct


def send_telegram(text: str) -> bool:
    """Send a simple text message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------
# Core screening logic
# ---------------------------------------------------------------------

def compute_breakout_score(
    rs_val: float,
    adr_pct: float,
    base_depth_pct: float,
    vcp_ratio: float | float,
    dist_from_sma_pct: float,
) -> float:
    """Quant-ish composite score 0-100."""
    # RS already 0–1
    rs_score = max(0.0, min(1.0, rs_val))

    # ADR: prefer mid-range between MIN and MAX
    adr_mid = 0.5 * (MIN_ADR_PCT + MAX_ADR_PCT)
    adr_span = max(1e-6, MAX_ADR_PCT - MIN_ADR_PCT)
    adr_score = 1.0 - abs(adr_pct - adr_mid) / adr_span
    adr_score = max(0.0, min(1.0, adr_score))

    # Base depth: shallower is better
    base_score = 1.0 - base_depth_pct / max(MAX_BASE_DEPTH_PCT, 1e-6)
    base_score = max(0.0, min(1.0, base_score))

    # VCP: more contraction (lower ratio) is better
    if np.isnan(vcp_ratio):
        vcp_score = 0.0
    else:
        if vcp_ratio <= MIN_VCP_RATIO:
            vcp_score = 1.0
        elif vcp_ratio >= MAX_VCP_RATIO:
            vcp_score = 0.0
        else:
            vcp_score = (MAX_VCP_RATIO - vcp_ratio) / max(
                MAX_VCP_RATIO - MIN_VCP_RATIO, 1e-6
            )
        vcp_score = max(0.0, min(1.0, vcp_score))

    # Distance from 22-day MA: closer is better
    dist_sma_score = 1.0 - abs(dist_from_sma_pct) / max(
        MAX_DIST_FROM_22SMA_PCT, 1e-6
    )
    dist_sma_score = max(0.0, min(1.0, dist_sma_score))

    score = (
        0.45 * rs_score
        + 0.20 * adr_score
        + 0.15 * base_score
        + 0.10 * vcp_score
        + 0.10 * dist_sma_score
    ) * 100.0
    return score


def is_near_miss(
    adr_pct: float,
    avg_vol_30d: float,
    price_now: float,
    rs_val: float,
    base_depth_pct: float,
    dist_from_high_pct: float,
    dist_from_sma_pct: float,
    vcp_ratio: float,
) -> bool:
    """
    Define a "near-miss" as:
      - Still liquid enough and price > 1
      - Structural picture okay
      - One or two metrics just outside strict ranges.
    This powers the Preview Watchlist.
    """
    if np.isnan(adr_pct) or np.isnan(rs_val):
        return False
    if avg_vol_30d < 0.8 * MIN_AVG_VOL_30D:
        return False
    if price_now < MIN_PRICE:
        return False

    rs_ok = rs_val >= (MIN_RS_PCTILE - 0.03)
    adr_ok = (MIN_ADR_PCT - 1.0) <= adr_pct <= (MAX_ADR_PCT + 2.0)
    base_ok = 0 <= base_depth_pct <= (MAX_BASE_DEPTH_PCT + 10.0)
    prox_ok = abs(dist_from_high_pct) <= (MAX_DIST_FROM_HIGH_PCT + 5.0)
    dist_sma_ok = abs(dist_from_sma_pct) <= (MAX_DIST_FROM_22SMA_PCT + 5.0)

    if np.isnan(vcp_ratio):
        vcp_ok = False
    else:
        vcp_ok = (
            MIN_VCP_RATIO * 0.7
            <= vcp_ratio
            <= MAX_VCP_RATIO * 1.3
        )

    flags = [rs_ok, adr_ok, base_ok, prox_ok, dist_sma_ok, vcp_ok]
    # near-miss = at least 4 of 6 "soft" conditions are true
    return sum(flags) >= 4


def run_screen() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    print("Building ALL_US universe from Polygon...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    universe = prefilter_universe(
        all_us,
        min_price=PREFILTER_MIN_PRICE,
        min_avg_vol5=PREFILTER_MIN_AVG_VOL5,
        cap=PREFILTER_TOP_N_BY_DOLLAR_VOL,
    )

    print("Computing RS percentiles vs SPY (Polygon)...")
    rs = rs_percentile(universe, months=RS_LOOKBACK_MONTHS, benchmark="SPY")
    rs_dict = rs.to_dict()

    print("Running final filters...")
    strict_rows: list[dict] = []
    preview_rows: list[dict] = []
    skips = {"empty": 0, "short": 0, "filters": 0}

    # how far back we need to cover all indicators
    min_len_required = max(
        ADR_PERIOD, MA_SLOW, HIGH_LOOKBACK_DAYS, AVG_VOL_LOOKBACK
    ) + 5

    lookback_days_main = max(min_len_required + 20, RS_LOOKBACK_MONTHS * 32)

    for t in universe:
        df = polygon_download_daily(t, lookback_days=lookback_days_main)
        if df.empty:
            skips["empty"] += 1
            continue

        if len(df) < min_len_required:
            skips["short"] += 1
            continue

        try:
            price_now = float(df["Close"].iloc[-1])

            # Volatility (ADR%)
            adr_pct = compute_adr_pct(df, ADR_PERIOD, price_now)

            # Liquidity: 30-day avg volume
            avg_vol_30d = float(
                df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1]
            )

            # Moving averages
            ma_fast = float(
                SMAIndicator(df["Close"], window=MA_FAST)
                .sma_indicator()
                .iloc[-1]
            )
            ma_slow = float(
                SMAIndicator(df["Close"], window=MA_SLOW)
                .sma_indicator()
                .iloc[-1]
            )

            # Recent high (base reference)
            recent = df.tail(HIGH_LOOKBACK_DAYS)
            high_lookback = float(recent["Close"].max())
            if high_lookback <= 0:
                continue

            dist_from_high_pct = (price_now / high_lookback - 1.0) * 100.0
            base_depth_pct = -dist_from_high_pct  # positive = below high
            dist_from_sma_pct = (
                (price_now / ma_slow - 1.0) * 100.0 if ma_slow != 0 else 0.0
            )

            # Volatility contraction pattern (VCP)
            ret = df["Close"].pct_change()
            vol_short = float(ret.rolling(VCP_SHORT).std().iloc[-1] * 100.0)
            vol_long = float(ret.rolling(VCP_LONG).std().iloc[-1] * 100.0)
            if (
                vol_short <= 0
                or vol_long <= 0
                or np.isnan(vol_short)
                or np.isnan(vol_long)
            ):
                vcp_ratio = np.nan
            else:
                vcp_ratio = vol_short / vol_long

            rs_val = float(rs_dict.get(t, np.nan))

            # --- strict filter conditions ---
            conds = []

            # Volatility sweet spot
            conds.append(adr_pct >= MIN_ADR_PCT)
            conds.append(adr_pct <= MAX_ADR_PCT)

            # Liquidity
            conds.append(avg_vol_30d >= MIN_AVG_VOL_30D)

            # Price / MAs
            conds.append(price_now >= MIN_PRICE)
            conds.append(price_now > ma_fast)
            conds.append(price_now > ma_slow)

            # Relative strength
            conds.append(not np.isnan(rs_val) and rs_val >= MIN_RS_PCTILE)

            # Base / proximity to high
            conds.append(base_depth_pct <= MAX_BASE_DEPTH_PCT)
            conds.append(base_depth_pct >= 0)
            conds.append(abs(dist_from_high_pct) <= MAX_DIST_FROM_HIGH_PCT)

            # Not too extended from 22-day MA
            conds.append(abs(dist_from_sma_pct) <= MAX_DIST_FROM_22SMA_PCT)

            # Volatility contraction
            conds.append(not np.isnan(vcp_ratio))
            if not np.isnan(vcp_ratio):
                conds.append(vcp_ratio >= MIN_VCP_RATIO)
                conds.append(vcp_ratio <= MAX_VCP_RATIO)

            passed = all(conds)

            # Breakout score (used for both strict + preview)
            score = compute_breakout_score(
                rs_val=rs_val,
                adr_pct=adr_pct,
                base_depth_pct=base_depth_pct,
                vcp_ratio=vcp_ratio,
                dist_from_sma_pct=dist_from_sma_pct,
            )

            row = {
                "Ticker": t,
                "Price": round(price_now, 2),
                "ADR%": round(adr_pct, 2),
                "AvgVol(30d)": int(avg_vol_30d),
                "RS_pctile": round(rs_val, 4),
                "DistHi_%": round(dist_from_high_pct, 2),
                "BaseDepth_%": round(base_depth_pct, 2),
                "Dist22SMA_%": round(dist_from_sma_pct, 2),
                "VCP_ratio": round(float(vcp_ratio), 3)
                if not np.isnan(vcp_ratio)
                else np.nan,
                "Score": round(score, 2),
            }

            if passed:
                strict_rows.append(row)
            else:
                # check for near-miss candidates
                if is_near_miss(
                    adr_pct=adr_pct,
                    avg_vol_30d=avg_vol_30d,
                    price_now=price_now,
                    rs_val=rs_val,
                    base_depth_pct=base_depth_pct,
                    dist_from_high_pct=dist_from_high_pct,
                    dist_from_sma_pct=dist_from_sma_pct,
                    vcp_ratio=vcp_ratio,
                ):
                    preview_rows.append(row)
                else:
                    skips["filters"] += 1

        except Exception:
            # Skip any problematic ticker entirely
            continue

    strict_df = pd.DataFrame(strict_rows)
    preview_df = pd.DataFrame(preview_rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    main_csv = f"allus_breakout_v2_{ts}.csv"
    strict_df.to_csv(main_csv, index=False)

    preview_csv = f"allus_breakout_v2_preview_{ts}.csv"
    if not preview_df.empty:
        preview_df.to_csv(preview_csv, index=False)

    if strict_df.empty:
        print("No tickers passed the STRICT breakout screen today.")
        print(f"Saved empty strict results to {main_csv}")
    else:
        strict_df_sorted = strict_df.sort_values(
            ["Score", "RS_pctile", "AvgVol(30d)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        strict_df = strict_df_sorted
        print("\nBreakout Screen v2 (strict, top results):")
        print(strict_df.head(TOP_N_TO_REPORT).to_string(index=False))
        print(f"\nSaved {len(strict_df)} strict results to {main_csv}")

    if not preview_df.empty:
        preview_df_sorted = preview_df.sort_values(
            ["Score", "RS_pctile", "AvgVol(30d)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        preview_df = preview_df_sorted
        print("\nPreview Watchlist (near-misses, top results):")
        print(preview_df.head(TOP_N_PREVIEW).to_string(index=False))
        print(f"\nSaved {len(preview_df)} preview results to {preview_csv}")

    return strict_df, preview_df, main_csv


# ---------------------------------------------------------------------
# Telegram summary
# ---------------------------------------------------------------------

def send_summary(strict_df: pd.DataFrame, preview_df: pd.DataFrame):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; skipping Telegram.")
        return

    def fmt_vol(v):
        try:
            return f"{float(v) / 1e6:.1f}M"
        except Exception:
            return str(v)

    # Nothing at all
    if strict_df.empty and preview_df.empty:
        send_telegram("📭 No tickers passed the breakout screen today.")
        return

    # No strict setups but we have near-misses (like your old screenshot)
    if strict_df.empty and not preview_df.empty:
        lines = ["📭 No pre-breakout setups today.\n"]
        lines.append("🔍 <b>Preview Watchlist (Near-Misses)</b>")
        top_prev = preview_df.head(TOP_N_PREVIEW)
        for _, r in top_prev.iterrows():
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"RS {r['RS_pctile']:.2f} | DistSMA {r['Dist22SMA_%']:.2f}% | "
                f"BaseDepth {r['BaseDepth_%']:.2f}% | "
                f"VCP {r['VCP_ratio']:.3f} | Score {r['Score']:.1f}"
            )
        send_telegram("\n".join(lines))
        return

    # We have strict hits (and maybe preview as well)
    lines = ["<b>🚀 Breakout Screen v2 (Top)</b>"]
    top_main = strict_df.head(15)
    for _, r in top_main.iterrows():
        lines.append(
            f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
            f"Vol {fmt_vol(r['AvgVol(30d)'])} | RS {r['RS_pctile']:.2f} | "
            f"DistHi {r['DistHi_%']:.2f}% | "
            f"BaseDepth {r['BaseDepth_%']:.2f}% | "
            f"VCP {r['VCP_ratio']:.3f} | "
            f"Score {r['Score']:.1f}"
        )

    if not preview_df.empty:
        lines.append("\n🔍 <b>Preview Watchlist (Near-Misses)</b>")
        top_prev = preview_df.head(TOP_N_PREVIEW)
        for _, r in top_prev.iterrows():
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"RS {r['RS_pctile']:.2f} | DistSMA {r['Dist22SMA_%']:.2f}% | "
                f"BaseDepth {r['BaseDepth_%']:.2f}% | "
                f"VCP {r['VCP_ratio']:.3f} | Score {r['Score']:.1f}"
            )

    send_telegram("\n".join(lines))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    strict_df, preview_df, _ = run_screen()
    send_summary(strict_df, preview_df)


if __name__ == "__main__":
    main()
