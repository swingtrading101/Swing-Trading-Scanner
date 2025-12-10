# -*- coding: utf-8 -*-
"""
ALL-US Breakout Screen v2 (Polygon-powered, with Near Miss preview)

Core idea (same spirit as your original model, but cleaner and stricter):

- Universe: liquid US stocks (price + recent volume + dollar volume prefilter)
- Strong 6-month RS vs SPY (top X%)
- Healthy but controlled volatility (ADR% between min/max)
- Above key MAs (10 & 22 day)
- Not too far from 120-day high (in a base, not extended)
- Volatility contraction (short-term vol < long-term vol, VCP-style)
- Ranked by a breakout score (0–100)
- Also produce a "near-miss" preview list (failed only 1–2 conditions by small margins)

Data source: Polygon.io daily aggregates
"""

import os
import math
import time
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from ta.trend import SMAIndicator
from yahoo_fin import stock_info as si

# =======================
# Config from environment
# =======================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

if not POLYGON_API_KEY:
    raise RuntimeError(
        "POLYGON_API_KEY is not set in environment. "
        "Add it as a GitHub secret and expose it in the workflow env."
    )

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

# Slow down a tiny bit to respect Polygon limits if needed
POLYGON_SLEEP_SECONDS = float(os.getenv("POLYGON_SLEEP_SECONDS", "0.02"))

pd.options.display.float_format = "{:,.2f}".format


# ==============
# Polygon helper
# ==============

def polygon_get_agg_daily(
    ticker: str,
    days_back: int,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Polygon for the past `days_back` calendar days.

    Returns DataFrame indexed by datetime with columns:
    Open, High, Low, Close, Volume
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days_back * 1.6))  # pad for weekends/holidays

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY}

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if "results" not in data or not data["results"]:
            return pd.DataFrame()
        rows = data["results"]
        df = pd.DataFrame(rows)
        # Polygon keys: t (ms), o,h,l,c,v
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("date", inplace=True)
        df.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
            },
            inplace=True,
        )
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        # tiny sleep helps avoid hammering free-tier limits
        if POLYGON_SLEEP_SECONDS > 0:
            time.sleep(POLYGON_SLEEP_SECONDS)


def safe_download_daily(ticker: str, period_days: int) -> pd.DataFrame:
    """
    Wrapper to fetch daily data safely. Returns empty DF on any error.
    """
    return polygon_get_agg_daily(ticker, days_back=period_days)


# ==================
# Universe / helpers
# ==================

def get_all_us_tickers() -> List[str]:
    """Universe: NASDAQ + others, filter to alpha-only tickers (no weird suffixes)."""
    try:
        tickers = si.tickers_nasdaq() + si.tickers_other()
    except Exception:
        tickers = []
    return sorted({t for t in tickers if t.isalpha()})


def compute_adr_pct(df: pd.DataFrame, period: int = 14, ref_price: float = None) -> float:
    """Average daily range % based on High–Low."""
    if df.empty or len(df) < period + 1:
        return float("nan")
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    if ref_price is None:
        ref_price = float(df["Close"].iloc[-1])
    if ref_price <= 0 or pd.isna(rng):
        return float("nan")
    return float(rng / ref_price * 100.0)


def prefilter_universe(
    tickers: List[str],
    min_price: float,
    min_avg_vol5: float,
    cap: int,
) -> List[str]:
    """
    Coarse filter using Polygon:
      - price >= min_price
      - 5-day avg volume >= min_avg_vol5
      - rank by 5-day dollar volume, keep top `cap`
    """
    scores: List[Tuple[str, float]] = []

    for i, t in enumerate(tickers):
        df = safe_download_daily(t, period_days=15)
        if df.empty or len(df) < 5:
            continue
        px = float(df["Close"].iloc[-1])
        vol5 = float(df["Volume"].tail(5).mean())
        if px >= min_price and vol5 >= min_avg_vol5:
            dollar_vol5 = px * vol5
            scores.append((t, dollar_vol5))

        # Soft stop if we already have plenty of tickers and we've scanned a lot
        if len(scores) >= cap * 1.5 and i > cap * 5:
            break

    scores.sort(key=lambda x: x[1], reverse=True)
    trimmed = [t for t, _ in scores[:cap]]
    return trimmed


def rs_percentile(
    tickers: List[str],
    months: int,
    benchmark: str = "SPY",
) -> pd.Series:
    """
    Relative strength vs benchmark over N months:
      RS = (stock_return - benchmark_return)
      then percentile rank across universe.
    Uses Polygon for both benchmark and stocks.
    """
    days_back = months * 32  # loose approx

    bench_df = safe_download_daily(benchmark, period_days=days_back)
    if bench_df.empty:
        return pd.Series(dtype=float)

    bench_close = bench_df["Close"].dropna()
    if bench_close.empty:
        return pd.Series(dtype=float)

    out: Dict[str, float] = {}

    for t in tickers:
        df = safe_download_daily(t, period_days=days_back)
        if df.empty:
            continue
        px = df["Close"].dropna()
        if px.empty:
            continue

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

    return pd.Series(out).rank(pct=True)


# ================
# Telegram helper
# ================

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


# ========================
# Core screening logic v2
# ========================

def run_screen() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    print("Building ALL_US universe...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    print("Prefiltering (price & recent volume & dollar volume via Polygon)...")
    universe = prefilter_universe(
        all_us,
        min_price=PREFILTER_MIN_PRICE,
        min_avg_vol5=PREFILTER_MIN_AVG_VOL5,
        cap=PREFILTER_TOP_N_BY_DOLLAR_VOL,
    )
    print(f"Universe after prefilter+cap: {len(universe)}")

    print("Computing RS percentiles vs SPY (Polygon)...")
    rs = rs_percentile(universe, months=RS_LOOKBACK_MONTHS, benchmark="SPY")
    rs_dict = rs.to_dict()

    print("Running final filters...")
    rows_pass: List[Dict] = []
    rows_preview: List[Dict] = []
    skips = {"empty": 0, "short": 0, "filters": 0}

    min_len_required = max(
        ADR_PERIOD, MA_SLOW, HIGH_LOOKBACK_DAYS, AVG_VOL_LOOKBACK, VCP_LONG
    ) + 2

    for t in universe:
        df = safe_download_daily(t, period_days=260)  # ~1 year of data
        if df.empty:
            skips["empty"] += 1
            continue
        if len(df) < min_len_required:
            skips["short"] += 1
            continue

        try:
            price_now = float(df["Close"].iloc[-1])
            adr_pct = compute_adr_pct(df, ADR_PERIOD, price_now)
            avg_vol_30d = float(
                df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1]
            )

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

            recent = df.tail(HIGH_LOOKBACK_DAYS)
            high_lookback = float(recent["Close"].max())
            if high_lookback <= 0:
                continue

            dist_from_high_pct = (price_now / high_lookback - 1.0) * 100.0
            base_depth_pct = -dist_from_high_pct  # positive = how far below high
            dist_from_sma_pct = (price_now / ma_slow - 1.0) * 100.0 if ma_slow != 0 else 0

            # VCP: volatility contraction
            ret = df["Close"].pct_change()
            vol_short = float(ret.rolling(VCP_SHORT).std().iloc[-1] * 100.0)
            vol_long = float(ret.rolling(VCP_LONG).std().iloc[-1] * 100.0)

            if (
                vol_short <= 0
                or vol_long <= 0
                or np.isnan(vol_short)
                or np.isnan(vol_long)
            ):
                vcp_ratio = float("nan")
            else:
                vcp_ratio = vol_short / vol_long

            rs_val = float(rs_dict.get(t, float("nan")))

            # ------------- CONDITIONS -------------
            conds: Dict[str, bool] = {}

            # Volatility sweet spot
            conds["adr_min"] = adr_pct >= MIN_ADR_PCT
            conds["adr_max"] = adr_pct <= MAX_ADR_PCT

            # Liquidity
            conds["liquidity"] = avg_vol_30d >= MIN_AVG_VOL_30D

            # Price/MAs
            conds["price_min"] = price_now >= MIN_PRICE
            conds["above_fast"] = price_now > ma_fast
            conds["above_slow"] = price_now > ma_slow

            # Relative strength
            conds["rs"] = (not np.isnan(rs_val)) and rs_val >= MIN_RS_PCTILE

            # Base structure / proximity to high
            conds["base_shallow"] = base_depth_pct <= MAX_BASE_DEPTH_PCT
            conds["below_high"] = base_depth_pct >= 0
            conds["near_high"] = abs(dist_from_high_pct) <= MAX_DIST_FROM_HIGH_PCT

            # Not too extended from 22d MA
            conds["near_sma22"] = (
                abs(dist_from_sma_pct) <= MAX_DIST_FROM_22SMA_PCT
            )

            # Volatility contraction
            conds["vcp_defined"] = not np.isnan(vcp_ratio)
            if not np.isnan(vcp_ratio):
                conds["vcp_min"] = vcp_ratio >= MIN_VCP_RATIO
                conds["vcp_max"] = vcp_ratio <= MAX_VCP_RATIO
            else:
                conds["vcp_min"] = False
                conds["vcp_max"] = False

            passed = all(conds.values())
            fail_keys = [k for k, v in conds.items() if not v]

            # ------------- SCORES -------------
            # RS already 0–1
            rs_score = max(0.0, min(1.0, rs_val))

            # ADR sweet spot (mid-range)
            adr_mid = 0.5 * (MIN_ADR_PCT + MAX_ADR_PCT)
            adr_span = max(1e-6, MAX_ADR_PCT - MIN_ADR_PCT)
            adr_score = 1.0 - abs(adr_pct - adr_mid) / adr_span
            adr_score = max(0.0, min(1.0, adr_score))

            # Shallower bases score higher
            base_score = 1.0 - base_depth_pct / max(MAX_BASE_DEPTH_PCT, 1e-6)
            base_score = max(0.0, min(1.0, base_score))

            # VCP: lower ratio = better contraction
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

            # Distance from 22d MA: closer is better
            dist_sma_score = 1.0 - abs(dist_from_sma_pct) / max(
                MAX_DIST_FROM_22SMA_PCT, 1e-6
            )
            dist_sma_score = max(0.0, min(1.0, dist_sma_score))

            # Final breakout score (0–100)
            score = (
                0.45 * rs_score
                + 0.20 * adr_score
                + 0.15 * base_score
                + 0.10 * vcp_score
                + 0.10 * dist_sma_score
            ) * 100.0

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
                "Failed_Conds": ",".join(fail_keys),
                "Fail_Count": len(fail_keys),
            }

            if passed:
                rows_pass.append(row)
            else:
                # Near-miss logic: failed only 1–2 conditions
                # and score is still reasonably high.
                if len(fail_keys) <= 2 and score >= 50:
                    rows_preview.append(row)
                else:
                    skips["filters"] += 1

        except Exception:
            skips["filters"] += 1
            continue

    results_pass = pd.DataFrame(rows_pass)
    results_preview = pd.DataFrame(rows_preview)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_name_main = f"allus_breakout_v2_{ts}.csv"
    csv_name_prev = f"allus_breakout_v2_preview_{ts}.csv"

    results_pass.to_csv(csv_name_main, index=False)
    results_preview.to_csv(csv_name_prev, index=False)

    if results_pass.empty:
        print("No tickers passed the breakout screen today.")
    else:
        results_pass = results_pass.sort_values(
            ["Score", "RS_pctile", "AvgVol(30d)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        print("\nBreakout Screen v2 (top results):")
        print(results_pass.head(TOP_N_TO_REPORT).to_string(index=False))

    if not results_preview.empty:
        results_preview = results_preview.sort_values(
            ["Score", "RS_pctile", "AvgVol(30d)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        print("\nPreview Watchlist (Near-Misses):")
        print(results_preview.head(TOP_N_PREVIEW).to_string(index=False))

    print(f"\nSaved {len(results_pass)} main results to {csv_name_main}")
    print(f"Saved {len(results_preview)} preview results to {csv_name_prev}")
    print("Skips summary:", skips)

    return results_pass, results_preview, csv_name_main


# =====================
# Telegram summaries
# =====================

def send_summary(results: pd.DataFrame, preview: pd.DataFrame):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; skipping Telegram.")
        return

    if results.empty and preview.empty:
        send_telegram("📭 No tickers passed the breakout screen today.")
        return

    lines: List[str] = []

    if not results.empty:
        top = results.head(15).copy()

        def fmt_vol(v):
            try:
                return f"{float(v) / 1e6:.1f}M"
            except Exception:
                return str(v)

        lines.append("<b>🚀 Breakout Screen v2 (Top)</b>")
        for _, r in top.iterrows():
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"Vol {fmt_vol(r['AvgVol(30d)'])} | RS {r['RS_pctile']:.2f} | "
                f"DistHi {r['DistHi_%']:.2f}% | "
                f"BaseDepth {r['BaseDepth_%']:.2f}% | "
                f"VCP {r['VCP_ratio']:.3f} | "
                f"Score {r['Score']:.1f}"
            )

    if not preview.empty:
        prev = preview.head(10).copy()

        def fmt_vol2(v):
            try:
                return f"{float(v) / 1e6:.1f}M"
            except Exception:
                return str(v)

        lines.append("\n<b>🔍 Preview Watchlist (Near-Misses)</b>")
        for _, r in prev.iterrows():
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"Vol {fmt_vol2(r['AvgVol(30d)'])} | RS {r['RS_pctile']:.2f} | "
                f"Score {r['Score']:.1f} | "
                f"Missed: {r['Failed_Conds']}"
            )

    send_telegram("\n".join(lines))


# =============
# Main entrypoint
# =============

def main():
    results, preview, _ = run_screen()
    send_summary(results, preview)


if __name__ == "__main__":
    main()
