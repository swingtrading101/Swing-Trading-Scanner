# -*- coding: utf-8 -*-
"""
ALL-US Pre-Market Screen
Criteria:
  • ADR(14) > 5%
  • 30-day avg volume > 30,000,000 shares
  • Price > $1  (optionally uses premarket price if available)
  • RS (6 months) in top 2% vs SPY
  • Above 22-day SMA

Outputs: printed table + CSV.
Deps: yfinance pandas numpy ta scipy requests yahoo_fin lxml
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator
from yahoo_fin import stock_info as si

# ---------------- CONFIG (tweakable via env) ----------------
ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_PERIOD = int(os.getenv("MA_PERIOD", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))
RS_LOOKBACK_MONTHS = int(os.getenv("RS_LOOKBACK_MONTHS", "6"))

# Hard filter thresholds for this screen
MIN_ADR_PCT = float(os.getenv("MIN_ADR_PCT", "5"))
MIN_AVG_VOL_30D = float(os.getenv("MIN_AVG_VOL_30D", "30000000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_RS_PCTILE = float(os.getenv("MIN_RS_PCTILE", "0.98"))  # top 2%

# Universe speed knobs
PREFILTER_MIN_PRICE = float(os.getenv("PREFILTER_MIN_PRICE", "3.0"))       # quick pass using recent data
PREFILTER_MIN_AVG_VOL5 = float(os.getenv("PREFILTER_MIN_AVG_VOL5", "5000000"))
PREFILTER_TOP_N_BY_VOL = int(os.getenv("PREFILTER_TOP_N_BY_VOL", "1500"))

# Use premarket/real-time price if available via fast_info (fallback to last close)
USE_PREMARKET_PRICE = os.getenv("USE_PREMARKET_PRICE", "1") == "1"

pd.options.display.float_format = "{:,.2f}".format


# ---------------- Helpers ----------------
def get_all_us_tickers():
    tickers = si.tickers_nasdaq() + si.tickers_other()
    # Keep only plain alphabetic symbols (drop weird/ETF-like tickers with punctuation)
    return sorted({t for t in tickers if t.isalpha()})


def safe_download_daily(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[ticker]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def recent_price_fast(ticker, fallback_close):
    """Try premarket/real-time price via fast_info; fall back to last close."""
    if not USE_PREMARKET_PRICE:
        return float(fallback_close)
    try:
        ti = yf.Ticker(ticker)
        fi = getattr(ti, "fast_info", None)
        if fi is None:
            return float(fallback_close)
        # Try several fields fast_info may expose
        for key in ("last_price", "regular_market_price", "previous_close"):
            val = getattr(fi, key, None)
            if val is not None and np.isfinite(val) and val > 0:
                return float(val)
        return float(fallback_close)
    except Exception:
        return float(fallback_close)


def compute_adr_pct(df, period=14, ref_price=None):
    # Average Daily Range = mean(High-Low) over N days
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    price = float(ref_price) if ref_price is not None else float(df["Close"].iloc[-1])
    if price <= 0 or pd.isna(rng):
        return np.nan
    return float(rng / price * 100.0)


def prefilter_universe(tickers, min_price=3.0, min_avg_vol5=5_000_000, cap=1500):
    """Lightweight 7-day check to keep only liquid candidates; then cap by recent volume."""
    kept = []
    vols = []
    for t in tickers:
        try:
            d = yf.download(t, period="7d", interval="1d", progress=False)
            if d.empty or len(d) < 2:
                continue
            px = float(d["Close"].iloc[-1])
            v5 = float(d["Volume"].tail(5).mean())
            if px >= min_price and v5 >= min_avg_vol5:
                kept.append(t)
                vols.append((t, v5))
        except Exception:
            pass
    vols.sort(key=lambda x: x[1], reverse=True)
    top = [t for t, _ in vols[:cap]]
    return top


def rs_percentile(tickers, months=6, benchmark="SPY"):
    """Relative strength vs SPY over ~months window using adjusted closes."""
    try:
        bench = yf.download(benchmark, period=f"{months*32}d", interval="1d",
                            auto_adjust=True, progress=False)["Close"].dropna()
    except Exception:
        bench = pd.Series(dtype=float)
    out = {}
    for t in tickers:
        try:
            px = yf.download(t, period=f"{months*32}d", interval="1d",
                             auto_adjust=True, progress=False)["Close"].dropna()
            if px.empty or bench.empty:
                continue
            # Align dates
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
    return pd.Series(out).rank(pct=True)  # 0..1


# ---------------- Main scan ----------------
def run_screen():
    print("Building ALL_US universe...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    print("Prefiltering (price & recent volume)...")
    universe = prefilter_universe(
        all_us,
        min_price=PREFILTER_MIN_PRICE,
        min_avg_vol5=PREFILTER_MIN_AVG_VOL5,
        cap=PREFILTER_TOP_N_BY_VOL
    )
    print(f"Universe after prefilter+cap: {len(universe)}")

    print("Computing RS percentiles...")
    rs = rs_percentile(universe, months=RS_LOOKBACK_MONTHS, benchmark="SPY")
    rs_dict = rs.to_dict()

    print("Running final filters...")
    rows = []
    skips = {"empty": 0, "short": 0, "filters": 0}
    for t in universe:
        df = safe_download_daily(t, period="6mo")
        if df.empty:
            skips["empty"] += 1
            continue
        if len(df) < max(ADR_PERIOD, MA_PERIOD) + 2:
            skips["short"] += 1
            continue

        try:
            # price: try premarket/real-time, fallback to last close
            last_close = float(df["Close"].iloc[-1])
            price_now = recent_price_fast(t, last_close)

            # ADR%
            adr_pct = compute_adr_pct(df, period=ADR_PERIOD, ref_price=price_now)

            # 30d avg volume
            avg_vol_30d = float(df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1])

            # 22-day SMA
            ma22 = float(SMAIndicator(df["Close"], window=MA_PERIOD).sma_indicator().iloc[-1])

            # RS percentile
            rs_val = float(rs_dict.get(t, np.nan))

            # Apply hard criteria
            passed = (
                (adr_pct is not None and adr_pct > MIN_ADR_PCT) and
                (avg_vol_30d > MIN_AVG_VOL_30D) and
                (price_now > MIN_PRICE) and
                (not np.isnan(rs_val) and rs_val >= MIN_RS_PCTILE) and
                (price_now > ma22)
            )
            if not passed:
                skips["filters"] += 1
                continue

            rows.append({
                "Ticker": t,
                "Price": round(price_now, 2),
                "ADR%": round(adr_pct, 2),
                "AvgVol(30d)": int(avg_vol_30d),
                "RS_pctile": round(rs_val, 4),
                "Dist_22SMA_%": round((price_now - ma22) / max(ma22, 1e-9) * 100.0, 2)
            })
        except Exception as e:
            # Just skip noisy tickers
            continue

    results = pd.DataFrame(rows)
    if results.empty:
        print("No tickers passed the pre-market screen today.")
        print("Skips:", skips)
        return results

    results = results.sort_values(
        ["RS_pctile", "ADR%", "AvgVol(30d)"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print("\nPre-Market Screen (top 50):")
    print(results.head(50).to_string(index=False))

    # Save CSV
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    fn = f"allus_premarket_screen_{ts}.csv"
    results.to_csv(fn, index=False)
    print(f"\nSaved {len(results)} results to {fn}")
    return results


if __name__ == "__main__":
    run_screen()
