# -*- coding: utf-8 -*-
"""
ALL‑US Pre‑Market Screen (CSV + optional Telegram)

Filters:
  • ADR(14) > 5%
  • Avg volume(30d) > 30,000,000 shares
  • Price > $1
  • RS (6 months, vs SPY) in top 2% (>= 0.98)
  • Above 22‑day SMA

Outputs:
  • Prints a ranked table (top rows)
  • ALWAYS saves CSV: allus_premarket_screen_YYYYMMDD_HHMM.csv
  • Optional Telegram summary if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set

Deps required (installed by workflow):
  yfinance pandas numpy ta yahoo_fin requests beautifulsoup4 html5lib lxml websockets
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from ta.trend import SMAIndicator
from yahoo_fin import stock_info as si

# ---------- CONFIG (env overrides supported) ----------
ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_PERIOD = int(os.getenv("MA_PERIOD", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))
RS_LOOKBACK_MONTHS = int(os.getenv("RS_LOOKBACK_MONTHS", "6"))

# Hard filters
MIN_ADR_PCT = float(os.getenv("MIN_ADR_PCT", "5"))
MIN_AVG_VOL_30D = float(os.getenv("MIN_AVG_VOL_30D", "30000000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_RS_PCTILE = float(os.getenv("MIN_RS_PCTILE", "0.98"))  # top 2%

# Universe speed knobs (keep scan time sane)
PREFILTER_MIN_PRICE = float(os.getenv("PREFILTER_MIN_PRICE", "3.0"))
PREFILTER_MIN_AVG_VOL5 = float(os.getenv("PREFILTER_MIN_AVG_VOL5", "5000000"))
PREFILTER_TOP_N_BY_VOL = int(os.getenv("PREFILTER_TOP_N_BY_VOL", "1200"))

# Optional Telegram (set in repo → Settings → Secrets and variables → Actions)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

pd.options.display.float_format = "{:,.2f}".format


# ---------- Helpers ----------
def get_all_us_tickers():
    """NASDAQ + other (NYSE/AMEX). Keep only simple alphabetic symbols."""
    try:
        tickers = si.tickers_nasdaq() + si.tickers_other()
    except Exception:
        tickers = []
    return sorted({t for t in tickers if t.isalpha()})


def safe_download_daily(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[ticker]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def compute_adr_pct(df, period=14, ref_price=None):
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    price = float(ref_price) if ref_price is not None else float(df["Close"].iloc[-1])
    if price <= 0 or pd.isna(rng):
        return np.nan
    return float(rng / price * 100.0)


def prefilter_universe(tickers, min_price=3.0, min_avg_vol5=5_000_000, cap=1200):
    """
    Quick 7‑day pass to remove illiquid names, then keep top N by recent volume.
    """
    scores = []
    for t in tickers:
        try:
            d = yf.download(t, period="7d", interval="1d", progress=False)
            if d.empty or len(d) < 2:
                continue
            px = float(d["Close"].iloc[-1])
            v5 = float(d["Volume"].tail(5).mean())
            if px >= min_price and v5 >= min_avg_vol5:
                scores.append((t, v5))
        except Exception:
            pass
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scores[:cap]]


def rs_percentile(tickers, months=6, benchmark="SPY"):
    """
    Relative strength vs SPY over ~months using adjusted closes.
    Returns Series of 0..1 percentile ranks keyed by ticker.
    """
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


def send_telegram(text: str) -> bool:
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


# ---------- Main scan ----------
def run_screen():
    print("Building ALL_US universe...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    print("Prefiltering (price & recent volume)...")
    universe = prefilter_universe(
        all_us,
        min_price=PREFILTER_MIN_PRICE,
        min_avg_vol5=PREFILTER_MIN_AVG_VOL5,
        cap=PREFILTER_TOP_N_BY_VOL,
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
            price_now = float(df["Close"].iloc[-1])  # latest close (stable, no websockets)
            adr_pct = compute_adr_pct(df, period=ADR_PERIOD, ref_price=price_now)
            avg_vol_30d = float(df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1])
            ma22 = float(SMAIndicator(df["Close"], window=MA_PERIOD).sma_indicator().iloc[-1])
            rs_val = float(rs_dict.get(t, np.nan))

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
                "Dist_22SMA_%": round((price_now - ma22) / max(ma22, 1e-9) * 100.0, 2),
            })
        except Exception:
            continue

    results = pd.DataFrame(rows)

    # ALWAYS save CSV so the workflow can upload an artifact even if empty
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_name = f"allus_premarket_screen_{ts}.csv"
    results.to_csv(csv_name, index=False)

    if results.empty:
        print("No tickers passed the pre‑market screen today.")
        print(f"Saved empty results to {csv_name}")
        return results, csv_name

    results = results.sort_values(
        ["RS_pctile", "ADR%", "AvgVol(30d)"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print("\nPre‑Market Screen (top 50):")
    print(results.head(50).to_string(index=False))
    print(f"\nSaved {len(results)} results to {csv_name}")
    return results, csv_name


def send_summary(results: pd.DataFrame):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; skipping Telegram.")
        return
    if results.empty:
        send_telegram("📭 No tickers passed the pre‑market screen today.")
        return

    top = results.head(15).copy()

    def fmt_vol(v):
        try:
            return f"{float(v)/1e6:.1f}M"
        except Exception:
            return str(v)

    lines = ["<b>📣 Pre‑Market Screen (Top)</b>"]
    for _, r in top.iterrows():
        lines.append(
            f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
            f"Vol {fmt_vol(r['AvgVol(30d)'])} | RS {r['RS_pctile']:.2f} | "
            f"Dist22SMA {r['Dist_22SMA_%']:.2f}%"
        )
    msg = "\n".join(lines)
    ok = send_telegram(msg)
    print("Telegram sent." if ok else "Telegram failed.")


def main():
    results, _ = run_screen()
    send_summary(results)


if __name__ == "__main__":
    main()
