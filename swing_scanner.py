# -*- coding: utf-8 -*-
"""
ALL-US Breakout Screen v2 (CSV + optional Telegram)

Goal:
- Find stocks with:
    • Strong 6-month relative strength vs SPY (top X%)
    • Sufficient liquidity (volume + dollar volume)
    • Healthy volatility (ADR% between min/max)
    • Price above key moving averages (10 & 22 day)
    • Within a controlled depth of recent 120-day high (base)
    • Volatility contraction (recent vol < long-term vol)
- Output ranked list + optional Telegram summary.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from ta.trend import SMAIndicator
from yahoo_fin import stock_info as si

# ---------- Config from environment ----------

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
    os.getenv("PREFILTER_TOP_N_BY_DOLLAR_VOL", "1500")
)

MAX_BASE_DEPTH_PCT = float(os.getenv("MAX_BASE_DEPTH_PCT", "35"))
MAX_DIST_FROM_HIGH_PCT = float(os.getenv("MAX_DIST_FROM_HIGH_PCT", "8"))
MAX_DIST_FROM_22SMA_PCT = float(os.getenv("MAX_DIST_FROM_22SMA_PCT", "15"))

VCP_SHORT = int(os.getenv("VCP_SHORT", "10"))
VCP_LONG = int(os.getenv("VCP_LONG", "40"))
MIN_VCP_RATIO = float(os.getenv("MIN_VCP_RATIO", "0.2"))
MAX_VCP_RATIO = float(os.getenv("MAX_VCP_RATIO", "0.7"))

TOP_N_TO_REPORT = int(os.getenv("TOP_N_TO_REPORT", "40"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

pd.options.display.float_format = "{:,.2f}".format


# ---------- Helper functions ----------

def get_all_us_tickers():
    """Universe: NASDAQ + others, alpha-only tickers (no ETFs/ADRs with dots)."""
    try:
        tickers = si.tickers_nasdaq() + si.tickers_other()
    except Exception:
        tickers = []
    # Keep only pure alphabetic tickers (filters out ADRs with '.' etc.)
    return sorted({t for t in tickers if t.isalpha()})


def safe_download_daily(ticker, period="6mo"):
    """Wrapper for yfinance daily data."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        # Handle MultiIndex from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[ticker]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def compute_adr_pct(df, period=14, ref_price=None):
    """Average daily range % based on High–Low."""
    if df.empty or len(df) < period + 1:
        return np.nan
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    price = float(ref_price) if ref_price is not None else float(df["Close"].iloc[-1])
    if price <= 0 or pd.isna(rng):
        return np.nan
    return float(rng / price * 100.0)


def prefilter_universe(
    tickers, min_price=3.0, min_avg_vol5=5_000_000, cap=1500
):
    """
    Quick coarse universe filter to reduce API load:
      - price >= min_price
      - 5-day avg volume >= min_avg_vol5
      - rank by 5-day dollar volume (price * volume)
    """
    scores = []
    for t in tickers:
        try:
            d = yf.download(
                t, period="10d", interval="1d", auto_adjust=False, progress=False
            )
            if d.empty or len(d) < 5:
                continue
            px = float(d["Close"].iloc[-1])
            vol5 = float(d["Volume"].tail(5).mean())
            if px >= min_price and vol5 >= min_avg_vol5:
                dollar_vol5 = px * vol5
                scores.append((t, dollar_vol5))
        except Exception:
            # Ignore problematic tickers silently
            pass

    scores.sort(key=lambda x: x[1], reverse=True)
    trimmed = [t for t, _ in scores[:cap]]
    return trimmed


def rs_percentile(tickers, months=6, benchmark="SPY"):
    """
    Relative strength vs benchmark over N months:
      RS = (stock_return - benchmark_return)
      then percentile rank across universe.
    """
    try:
        bench = yf.download(
            benchmark,
            period=f"{months*32}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"].dropna()
    except Exception:
        bench = pd.Series(dtype=float)

    out = {}
    for t in tickers:
        try:
            px = yf.download(
                t,
                period=f"{months*32}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )["Close"].dropna()
            if px.empty or bench.empty:
                continue
            idx = px.index.intersection(bench.index)
            if len(idx) < 20:
                continue
            pxa = px.reindex(idx).ffill()
            bxa = bench.reindex(idx).ffill()
            r_stock = float(pxa.iloc[-1] / pxa.iloc[0] - 1)
            r_bench = float(bxa.iloc[-1] / bxa.iloc[0] - 1)
            out[t] = r_stock - r_bench
        except Exception:
            pass

    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out).rank(pct=True)


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


# ---------- Core screening logic ----------

def run_screen():
    print("Building ALL_US universe...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    print("Prefiltering (price & recent volume & dollar volume)...")
    universe = prefilter_universe(
        all_us,
        min_price=PREFILTER_MIN_PRICE,
        min_avg_vol5=PREFILTER_MIN_AVG_VOL5,
        cap=PREFILTER_TOP_N_BY_DOLLAR_VOL,
    )
    print(f"Universe after prefilter+cap: {len(universe)}")

    print("Computing RS percentiles vs SPY...")
    rs = rs_percentile(universe, months=RS_LOOKBACK_MONTHS, benchmark="SPY")
    rs_dict = rs.to_dict()

    print("Running final filters...")
    rows = []
    skips = {"empty": 0, "short": 0, "filters": 0}

    for t in universe:
        df = safe_download_daily(t, period="8mo")
        if df.empty:
            skips["empty"] += 1
            continue

        min_len = max(ADR_PERIOD, MA_SLOW, HIGH_LOOKBACK_DAYS, AVG_VOL_LOOKBACK) + 2
        if len(df) < min_len:
            skips["short"] += 1
            continue

        try:
            price_now = float(df["Close"].iloc[-1])

            # ADR% (volatility)
            adr_pct = compute_adr_pct(df, ADR_PERIOD, price_now)

            # Liquidity: 30-day avg volume
            avg_vol_30d = float(
                df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1]
            )

            # Moving averages
            ma_fast = float(
                SMAIndicator(df["Close"], window=MA_FAST).sma_indicator().iloc[-1]
            )
            ma_slow = float(
                SMAIndicator(df["Close"], window=MA_SLOW).sma_indicator().iloc[-1]
            )

            # 120-day high (recent base)
            recent = df.tail(HIGH_LOOKBACK_DAYS)
            high_lookback = float(recent["Close"].max())
            if high_lookback <= 0:
                continue

            # Distances
            dist_from_high_pct = (price_now / high_lookback - 1.0) * 100.0
            base_depth_pct = -dist_from_high_pct  # positive = how far below the high
            dist_from_sma_pct = (price_now / ma_slow - 1.0) * 100.0 if ma_slow != 0 else 0

            # Volatility contraction pattern
            ret = df["Close"].pct_change()
            vol_short = float(ret.rolling(VCP_SHORT).std().iloc[-1] * 100.0)
            vol_long = float(ret.rolling(VCP_LONG).std().iloc[-1] * 100.0)

            if vol_short <= 0 or vol_long <= 0 or np.isnan(vol_short) or np.isnan(vol_long):
                vcp_ratio = np.nan
            else:
                vcp_ratio = vol_short / vol_long

            rs_val = float(rs_dict.get(t, np.nan))

            # --- Filter conditions (core of the model) ---

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

            # Base structure / proximity to high
            conds.append(base_depth_pct <= MAX_BASE_DEPTH_PCT)
            conds.append(base_depth_pct >= 0)  # below or at high, not above
            conds.append(abs(dist_from_high_pct) <= MAX_DIST_FROM_HIGH_PCT)

            # Not too extended from 22-day MA
            conds.append(abs(dist_from_sma_pct) <= MAX_DIST_FROM_22SMA_PCT)

            # Volatility contraction (VCP-style)
            conds.append(not np.isnan(vcp_ratio))
            if not np.isnan(vcp_ratio):
                conds.append(vcp_ratio >= MIN_VCP_RATIO)
                conds.append(vcp_ratio <= MAX_VCP_RATIO)

            passed = all(conds)
            if not passed:
                skips["filters"] += 1
                continue

            # --- Breakout score (quant-ish weighting) ---

            # RS already in 0–1
            rs_score = max(0.0, min(1.0, rs_val))

            # ADR: prefer mid-range between MIN and MAX
            adr_mid = 0.5 * (MIN_ADR_PCT + MAX_ADR_PCT)
            adr_span = max(1e-6, MAX_ADR_PCT - MIN_ADR_PCT)
            adr_score = 1.0 - abs(adr_pct - adr_mid) / adr_span
            adr_score = max(0.0, min(1.0, adr_score))

            # Base depth: shallower is better
            base_score = 1.0 - base_depth_pct / max(MAX_BASE_DEPTH_PCT, 1e-6)
            base_score = max(0.0, min(1.0, base_score))

            # VCP: lower ratio (more contraction) is better
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

            # Combine into a final score (0–100)
            score = (
                0.45 * rs_score
                + 0.20 * adr_score
                + 0.15 * base_score
                + 0.10 * vcp_score
                + 0.10 * dist_sma_score
            ) * 100.0

            rows.append(
                {
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
            )

        except Exception:
            # Skip any problematic ticker
            continue

    results = pd.DataFrame(rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_name = f"allus_breakout_v2_{ts}.csv"
    results.to_csv(csv_name, index=False)

    if results.empty:
        print("No tickers passed the breakout screen today.")
        print(f"Saved empty results to {csv_name}")
        return results, csv_name

    # Rank by Score, then RS and volume as tie-breakers
    results = results.sort_values(
        ["Score", "RS_pctile", "AvgVol(30d)"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print("\nBreakout Screen v2 (top results):")
    print(results.head(TOP_N_TO_REPORT).to_string(index=False))
    print(f"\nSaved {len(results)} results to {csv_name}")

    return results, csv_name


# ---------- Telegram summary ----------

def send_summary(results: pd.DataFrame):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; skipping Telegram.")
        return

    if results.empty:
        send_telegram("📭 No tickers passed the breakout screen today.")
        return

    top = results.head(15).copy()

    def fmt_vol(v):
        try:
            return f"{float(v) / 1e6:.1f}M"
        except Exception:
            return str(v)

    lines = ["<b>🚀 Breakout Screen v2 (Top)</b>"]
    for _, r in top.iterrows():
        lines.append(
            f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
            f"Vol {fmt_vol(r['AvgVol(30d)'])} | RS {r['RS_pctile']:.2f} | "
            f"DistHi {r['DistHi_%']:.2f}% | "
            f"BaseDepth {r['BaseDepth_%']:.2f}% | "
            f"VCP {r['VCP_ratio']:.3f} | "
            f"Score {r['Score']:.1f}"
        )

    send_telegram("\n".join(lines))


# ---------- Main ----------

def main():
    results, _ = run_screen()
    send_summary(results)


if __name__ == "__main__":
    main()
