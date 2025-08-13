# -*- coding: utf-8 -*-
"""
Swing Scanner — PRE‑BREAKOUT WATCHLIST with Preview (CI‑ready)

Finds liquid, high‑RS US stocks that are:
  • ADR(14) >= MIN_ADR_PCT (default 5%)
  • Avg volume(30d) >= MIN_AVG_VOL_30D (default 30M)
  • Price > $1 and (optionally) above 22‑SMA
  • RS percentile >= MIN_RS_PCTILE (default top 2% = 0.98)
  • In a tightening consolidation (higher lows + lower highs)
  • NOT yet broken out; within PROX_TO_BREAKOUT_PCT% below trigger
If none pass, sends a **Preview Watchlist**: top K near‑misses closest to breakout.

Outputs: Telegram summary + CSV artifact(s).
Deps: yfinance pandas numpy ta scipy requests yahoo_fin lxml
"""

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from ta.trend import SMAIndicator
from scipy.stats import linregress
from yahoo_fin import stock_info as si

pd.options.display.float_format = '{:,.2f}'.format

# ========= CONFIG (override via env) =========
UNIVERSE = os.getenv("UNIVERSE", "ALL_US")        # "ALL_US" | "S&P500" | "Custom List"
CUSTOM_TICKERS = os.getenv("CUSTOM_TICKERS", "")  # CSV list for "Custom List"

LOOKBACK_MONTHS_RS = int(os.getenv("LOOKBACK_MONTHS_RS", "6"))
ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_PERIOD = int(os.getenv("MA_PERIOD", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))

# Core gates (tunable)
MIN_ADR_PCT      = float(os.getenv("MIN_ADR_PCT", "5"))
MIN_AVG_VOL_30D  = float(os.getenv("MIN_AVG_VOL_30D", "30000000"))
MIN_RS_PCTILE    = float(os.getenv("MIN_RS_PCTILE", "0.98"))
REQUIRE_ABOVE_SMA = os.getenv("REQUIRE_ABOVE_SMA", "1") == "1"

# Pre‑breakout pattern tuning
MAX_DIST_SMA_PCT      = float(os.getenv("MAX_DIST_SMA_PCT", "12"))   # exclude extended > +12% vs 22‑SMA
MIN_DIST_SMA_PCT      = float(os.getenv("MIN_DIST_SMA_PCT", "-3"))   # allow slightly below
PROX_TO_BREAKOUT_PCT  = float(os.getenv("PROX_TO_BREAKOUT_PCT", "2.0"))  # must be within X% under trigger
ADR_CONTRACTION_RATIO = float(os.getenv("ADR_CONTRACTION_RATIO", "0.85")) # ADR14/ADR50 <= 0.85

# Prefilter (keeps ALL_US fast)
PREFILTER_MIN_PRICE     = float(os.getenv("PREFILTER_MIN_PRICE", "3.0"))         # only scan $3+
PREFILTER_MIN_AVG_VOL   = float(os.getenv("PREFILTER_MIN_AVG_VOL", "5000000"))   # 5M (5‑day avg)
PREFILTER_TOP_N_BY_VOL  = int(os.getenv("PREFILTER_TOP_N_BY_VOL", "1500"))       # cap universe post‑prefilter

# Preview watchlist (when strict list is empty)
PREVIEW_WATCHLIST = os.getenv("PREVIEW_WATCHLIST", "1") == "1"
PREVIEW_TOP_K     = int(os.getenv("PREVIEW_TOP_K", "10"))

# Telegram creds (set as GH Secrets in CI)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ========= HELPERS =========
def get_universe(universe="S&P500", custom_list=None):
    if universe == "S&P500":
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = [t.replace('.', '-') for t in tables[0]["Symbol"].tolist()]
    elif universe == "ALL_US":
        tickers = si.tickers_nasdaq() + si.tickers_other()
        tickers = [t for t in tickers if t.isalpha()]  # clean weirds
    elif universe == "Custom List":
        tickers = [t.strip().upper() for t in (custom_list or "").split(",") if t.strip()]
    else:
        tickers = []
    return sorted(list(set(tickers)))

def safe_download(t, period="6mo", interval="1d"):
    try:
        df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[t]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def compute_adr(df, period=14):
    dr = df['High'] - df['Low']
    return dr.rolling(period).mean()

def rs_percentile(tickers, months=6, benchmark='SPY'):
    bench = yf.download(benchmark, period=f"{months*32}d", interval="1d",
                        auto_adjust=True, progress=False)['Close'].dropna()
    results = {}
    for t in tickers:
        try:
            px = yf.download(t, period=f"{months*32}d", interval="1d",
                             auto_adjust=True, progress=False)['Close'].dropna()
            if len(px) < 20:
                continue
            bench_sync = bench.reindex(px.index).ffill().dropna()
            px = px.reindex(bench_sync.index).ffill().dropna()
            if len(px) < 2:
                continue
            r_stock = float(px.iloc[-1] / px.iloc[0] - 1)
            r_bench = float(bench_sync.iloc[-1] / bench_sync.iloc[0] - 1)
            results[t] = r_stock - r_bench
        except Exception:
            pass
    return pd.Series(results).rank(pct=True)  # 0..1

def check_consolidation_and_breakout(df, lookback=15, max_range_pct=12.0,
                                     require_triangle=True):
    """
    Returns: cons(bool), brk(bool), hi_recent, lo_recent, breakout_level, range_pct
      cons = tightening (higher lows + lower highs) and range% <= max_range_pct
      brk  = price > breakout_level (0.3% buffer)
    """
    if len(df) < lookback + 2:
        return False, False, np.nan, np.nan, np.nan, np.nan

    window = df.iloc[-lookback:]
    hi = window['High'].values
    lo = window['Low'].values
    close = float(df['Close'].iloc[-1])

    high_recent = float(np.max(hi))
    low_recent  = float(np.min(lo))
    range_pct = (high_recent - low_recent) / max(1e-9, close) * 100.0

    tight = range_pct <= max_range_pct

    triangle_ok = True
    if require_triangle:
        x = np.arange(len(window))
        slope_high = linregress(x, hi).slope
        slope_low  = linregress(x, lo).slope
        triangle_ok = (slope_high <= 0) and (slope_low >= 0)

    cons = tight and triangle_ok

    breakout_level = high_recent * 1.003  # +0.3% buffer
    brk = close > breakout_level if cons else False

    return cons, brk, high_recent, low_recent, breakout_level, range_pct

def send_telegram_message(token, chat_id, text):
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15
        )
        return r.ok
    except Exception:
        return False

# ---- Prefilter helpers (speed for ALL_US) ----
def prefilter_tickers(tickers, min_price=3.0, min_avg_vol=5_000_000):
    kept = []
    for t in tickers:
        try:
            df = yf.download(t, period="7d", interval="1d", auto_adjust=False, progress=False)
            if df.empty or len(df) < 2:
                continue
            price = float(df["Close"].iloc[-1])
            avg5  = float(df["Volume"].tail(5).mean())
            if (price >= min_price) and (avg5 >= min_avg_vol):
                kept.append(t)
        except Exception:
            pass
    return kept

def cap_by_recent_volume(tickers, top_n=1500):
    scored = []
    for t in tickers:
        try:
            d = yf.download(t, period="7d", interval="1d", progress=False)
            if d.empty:
                continue
            scored.append((t, float(d["Volume"].tail(5).mean())))
        except Exception:
            pass
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]

# ========= SCAN (pre‑breakout only; collect near‑misses) =========
def run_scan():
    tickers = get_universe(UNIVERSE, CUSTOM_TICKERS)
    print(f"Universe size (raw): {len(tickers)}")

    if UNIVERSE == "ALL_US":
        tickers = prefilter_tickers(tickers, PREFILTER_MIN_PRICE, PREFILTER_MIN_AVG_VOL)
        print(f"Universe after prefilter: {len(tickers)}")
        tickers = cap_by_recent_volume(tickers, PREFILTER_TOP_N_BY_VOL)
        print(f"Universe after cap: {len(tickers)}")

    rs_dict = rs_percentile(tickers, months=LOOKBACK_MONTHS_RS, benchmark='SPY').to_dict()

    candidates = []
    near_miss = []  # collect almost-there with metrics
    skips = {"empty":0, "short":0, "price":0, "filters":0}

    for t in tickers:
        df = safe_download(t, period="6mo", interval="1d")
        if df.empty:
            skips["empty"] += 1;  continue
        if len(df) < max(MA_PERIOD, ADR_PERIOD) + 2:
            skips["short"] += 1;  continue

        try:
            price = float(df['Close'].iloc[-1])
            if price <= 1:
                skips["price"] += 1;  continue

            adr14 = compute_adr(df, period=ADR_PERIOD).iloc[-1]
            if pd.isna(adr14):  continue
            adr50 = compute_adr(df, period=50).iloc[-1]
            adr_pct = float((adr14 / price) * 100.0)

            avg_vol = float(df['Volume'].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1])
            ma = float(SMAIndicator(df['Close'], window=MA_PERIOD).sma_indicator().iloc[-1])

            rs = float(rs_dict.get(t, np.nan))
            if np.isnan(rs):  continue

            # Core gates
            base_pass = (
                (adr_pct >= MIN_ADR_PCT) and
                (avg_vol >= MIN_AVG_VOL_30D) and
                ((price > ma) if REQUIRE_ABOVE_SMA else True) and
                (rs >= MIN_RS_PCTILE)
            )
            if not base_pass:
                skips["filters"] += 1;  continue

            cons, brk, hi_c, lo_c, breakout_level, base_range = check_consolidation_and_breakout(
                df, lookback=15, max_range_pct=12.0, require_triangle=True
            )

            dist_ma_pct = (price - ma) / max(ma, 1e-9) * 100.0
            prox_to_breakout = ((breakout_level - price) / breakout_level * 100.0) if np.isfinite(breakout_level) else np.nan
            adr_contraction_ok = (pd.notna(adr50) and float(adr14 / max(adr50, 1e-9)) <= ADR_CONTRACTION_RATIO)

            # Decide pass/fail for pre-breakout watchlist
            passes = (
                cons and (not brk) and
                (MIN_DIST_SMA_PCT <= dist_ma_pct <= MAX_DIST_SMA_PCT) and
                (0.0 <= prox_to_breakout <= PROX_TO_BREAKOUT_PCT) and
                adr_contraction_ok
            )

            # Collect near-misses with metrics (to rank later if needed)
            if not passes:
                near_miss.append({
                    "Ticker": t,
                    "Price": round(price, 2),
                    "ADR%": round(adr_pct, 2),
                    "AvgVol(30d)": int(avg_vol),
                    "RS_pctile": round(rs, 4),
                    f"Dist_{MA_PERIOD}SMA_%": round(dist_ma_pct, 2),
                    "BaseRange%": round(base_range, 2) if np.isfinite(base_range) else np.nan,
                    "Prox2BO%": round(prox_to_breakout, 2) if np.isfinite(prox_to_breakout) else np.nan,
                    "cons": bool(cons), "brk": bool(brk),
                    "adr_contr": bool(adr_contraction_ok),
                })
                skips["filters"] += 1
                continue

            # If passes, add to candidates
            candidates.append({
                "Ticker": t,
                "Price": round(price, 2),
                "ADR%": round(adr_pct, 2),
                "AvgVol(30d)": int(avg_vol),
                "RS_pctile": round(rs, 4),
                f"Dist_{MA_PERIOD}SMA_%": round(dist_ma_pct, 2),
                "BaseRange%": round(base_range, 2),
                "Prox2BO%": round(prox_to_breakout, 2),
                "Signal": "Watch"
            })

        except Exception as e:
            print(f"[calc error] {t}: {type(e).__name__}: {e}")

    results = pd.DataFrame(candidates)
    preview = pd.DataFrame()

    if results.empty and PREVIEW_WATCHLIST and len(near_miss) > 0:
        # Rank near-misses by "closest to breakout" then RS, then ADR (prefer viable liquid names)
        preview = pd.DataFrame(near_miss)
        # Prox2BO% can be NaN; we want smallest positive (closest under breakout)
        preview["Prox2BO%"] = pd.to_numeric(preview["Prox2BO%"], errors="coerce")
        preview = preview[preview["Prox2BO%"].notna() & (preview["Prox2BO%"] >= 0)]
        if not preview.empty:
            preview = preview.sort_values(
                ["Prox2BO%", "RS_pctile", "ADR%"],
                ascending=[True, False, False]
            ).head(PREVIEW_TOP_K).reset_index(drop=True)

    if not results.empty:
        results = results.sort_values(
            ["Prox2BO%", "RS_pctile", "ADR%"], ascending=[True, False, False]
        ).reset_index(drop=True)
        print("\nPre‑Breakout Watchlist (top):")
        print(results.head(30).to_string(index=False))
    else:
        print("No strict pre‑breakout matches today.")
        if not preview.empty:
            print("\nPreview Watchlist (near‑misses, closest to breakout):")
            print(preview.to_string(index=False))
    print("Skipped counts:", skips)

    return results, preview

# ========= OUTPUTS =========
def send_results_to_telegram(results, preview):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_* env vars; skipping Telegram.")
        return

    def build_lines(df, title):
        dist_col = next((c for c in df.columns if c.startswith("Dist_") and c.endswith("SMA_%")), None)
        lines = [f"<b>{title}</b>"]
        for _, r in df.iterrows():
            dist_val = r.get(dist_col, np.nan) if dist_col else np.nan
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"RS {r['RS_pctile']:.2f} | DistSMA {float(dist_val):.2f}% | "
                f"Prox {r.get('Prox2BO%', np.nan):.2f}%"
            )
        return "\n".join(lines)

    if results is not None and not results.empty:
        msg = build_lines(results.head(15), "🧭 Pre‑Breakout Watchlist")
        ok = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
        print("Telegram (strict) sent." if ok else "Telegram failed (strict).")
    elif preview is not None and not preview.empty:
        msg = build_lines(preview, "🔎 Preview Watchlist (Near‑Misses)")
        ok = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
        print("Telegram (preview) sent." if ok else "Telegram failed (preview).")
    else:
        ok = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                   "🧭 No pre‑breakout setups or near‑misses today.")
        print("Telegram (none) sent." if ok else "Telegram failed (none).")

def export_csv(results, preview):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if results is not None and not results.empty:
        fn = f"swing_watchlist_strict_{ts}.csv"
        results.to_csv(fn, index=False)
        print(f"Saved strict list: {fn}")
    if preview is not None and not preview.empty:
        fnp = f"swing_watchlist_preview_{ts}.csv"
        preview.to_csv(fnp, index=False)
        print(f"Saved preview list: {fnp}")

def main():
    results, preview = run_scan()
    send_results_to_telegram(results, preview)
    export_csv(results, preview)

if __name__ == "__main__":
    main()