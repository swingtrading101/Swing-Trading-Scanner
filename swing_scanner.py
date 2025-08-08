# -*- coding: utf-8 -*-
"""
Swing Scanner (CI-ready)

- Scans a universe (S&P 500 by default) for your swing criteria
- Sends Telegram alert using env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- Prints a compact table to CI logs
- Exports a CSV artifact to the run directory

Dependencies (install in CI):
  pip install yfinance pandas numpy ta scipy requests
"""

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from ta.trend import SMAIndicator
from scipy.stats import linregress

pd.options.display.float_format = '{:,.2f}'.format

# ========= CONFIG =========
UNIVERSE = os.getenv("UNIVERSE", "ALL_US")
   # "S&P500" or "Custom List"
CUSTOM_TICKERS = os.getenv("CUSTOM_TICKERS", "")  # CSV, e.g. "AAPL,MSFT,NVDA"

LOOKBACK_MONTHS_RS = int(os.getenv("LOOKBACK_MONTHS_RS", "6"))
ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_PERIOD = int(os.getenv("MA_PERIOD", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))

# Consolidation detection
CONS_LOOKBACK = int(os.getenv("CONS_LOOKBACK", "15"))
MAX_RANGE_PCT = float(os.getenv("MAX_RANGE_PCT", "12.0"))
REQUIRE_LOWER_HIGHS_HIGHER_LOWS = os.getenv("REQUIRE_TRIANGLE", "1") == "1"
BREAKOUT_BUFFER_PCT = float(os.getenv("BREAKOUT_BUFFER_PCT", "0.3"))

# Risk (not used in CI, but kept for completeness)
MAX_RISK_DOLLARS = float(os.getenv("MAX_RISK_DOLLARS", "500"))

# Telegram creds from env (set as GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ========= CORE HELPERS =========
from yahoo_fin import stock_info as si

def get_universe(universe="S&P500", custom_list=None):
    if universe == "S&P500":
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
    elif universe == "ALL_US":
        tickers = si.tickers_nasdaq() + si.tickers_other()
        tickers = [t for t in tickers if t.isalpha()]  # filter out weird tickers
    elif universe == "Custom List":
        return [t.strip().upper() for t in (custom_list or "").split(",") if t.strip()]
    else:
        tickers = []
    return sorted(list(set(tickers)))


def compute_adr(df, period=14):
    dr = df['High'] - df['Low']
    return dr.rolling(period).mean()

def rs_percentile(tickers, months=6, benchmark='SPY'):
    """
    Robust RS percentile vs SPY:
    - uses adjusted close
    - aligns dates to avoid 'identically-labeled' errors
    """
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
            r_stock = (px.iloc[-1] / px.iloc[0]) - 1
            r_bench = (bench_sync.iloc[-1] / bench_sync.iloc[0]) - 1
            results[t] = float(r_stock - r_bench)
        except Exception:
            pass
    ser = pd.Series(results)
    return ser.rank(pct=True)  # 0..1

def safe_download(t, period="6mo", interval="1d"):
    try:
        df = yf.download(t, period=period, interval=interval,
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[t]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def check_consolidation_and_breakout(df, lookback=15, max_range_pct=12.0,
                                     require_triangle=True, breakout_buffer_pct=0.3):
    if len(df) < lookback + 2:
        return False, False, np.nan, np.nan
    window = df.iloc[-lookback:]
    hi = window['High'].values
    lo = window['Low'].values
    close = df['Close'].iloc[-1]
    high_recent = float(np.max(hi))
    low_recent  = float(np.min(lo))
    rng_pct = (high_recent - low_recent) / max(1e-9, close) * 100.0
    tight = rng_pct <= max_range_pct
    triangle_ok = True
    if require_triangle:
        x = np.arange(len(window))
        slope_high = linregress(x, hi).slope
        slope_low  = linregress(x, lo).slope
        triangle_ok = (slope_high < 0) and (slope_low > 0)
    cons = tight and triangle_ok
    brk = False
    if cons:
        breakout_level = high_recent * (1 + breakout_buffer_pct/100.0)
        brk = df['Close'].iloc[-1] > breakout_level
    return cons, brk, high_recent, low_recent

def position_size(entry, stop, max_risk_dollars=500):
    risk_per_share = max(1e-8, abs(entry - stop))
    qty = int(max_risk_dollars // risk_per_share)
    return max(qty, 0)

def send_telegram_message(token, chat_id, text):
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.ok
    except Exception:
        return False

# ========= SCAN / ALERT / EXPORT =========
def run_scan():
    tickers = get_universe(UNIVERSE, CUSTOM_TICKERS)
    print(f"Universe size: {len(tickers)}")

    rs_series = rs_percentile(tickers, months=LOOKBACK_MONTHS_RS, benchmark='SPY')
    rs_dict = rs_series.to_dict()

    candidates = []
    skips = {"empty":0, "short":0, "price":0, "filters":0}

    for t in tickers:
        df = safe_download(t, period="6mo", interval="1d")
        if df.empty:
            skips["empty"] += 1
            continue
        if len(df) < max(MA_PERIOD, ADR_PERIOD) + 2:
            skips["short"] += 1
            continue

        try:
            price = float(df['Close'].iloc[-1])
            if price <= 1:
                skips["price"] += 1
                continue

            adr_val = compute_adr(df, period=ADR_PERIOD).iloc[-1]
            if pd.isna(adr_val):
                continue
            adr_pct = float((adr_val / price) * 100.0)

            avg_vol = float(df['Volume'].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1])
            ma = SMAIndicator(df['Close'], window=MA_PERIOD).sma_indicator().iloc[-1]

            rs = float(rs_dict.get(t, np.nan))
            if np.isnan(rs):
                continue

            passed = (adr_pct > 5) and (avg_vol > 30_000_000) and (price > ma) and (rs >= 0.98)
            if not passed:
                skips["filters"] += 1
                continue

            cons, brk, hi_c, lo_c = check_consolidation_and_breakout(
                df,
                lookback=CONS_LOOKBACK,
                max_range_pct=MAX_RANGE_PCT,
                require_triangle=REQUIRE_LOWER_HIGHS_HIGHER_LOWS,
                breakout_buffer_pct=BREAKOUT_BUFFER_PCT
            )
            dist_ma_pct = (price - ma) / ma * 100.0
            signal = "Breakout" if brk else ("Consolidation" if cons else "")

            candidates.append({
                "Ticker": t,
                "Price": price,
                "ADR%": round(adr_pct, 2),
                "AvgVol(30d)": int(avg_vol),
                "RS_pctile": round(rs, 4),
                f"Dist_{MA_PERIOD}SMA_%": round(dist_ma_pct, 2),
                "Signal": signal
            })
        except Exception as e:
            print(f"[calc error] {t}: {type(e).__name__}: {e}")

    results = pd.DataFrame(candidates)
    if results.empty:
        print("No tickers passed all filters today.")
        print("Skipped counts:", skips)
        return results

    results = results.sort_values(
        ["Signal", "RS_pctile", "ADR%"], ascending=[True, False, False]
    ).reset_index(drop=True)

    print("\nTop matches:")
    print(results.head(30).to_string(index=False))
    print("Skipped counts:", skips)
    return results

def send_results_to_telegram(results):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars; skipping Telegram.")
        return
    if results is None or results.empty:
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                              "📉 No tickers passed all filters today.")
        print("No matches — alert sent.")
        return
    dist_col = next((c for c in results.columns if c.startswith("Dist_") and c.endswith("SMA_%")), None)
    lines = ["<b>📈 Swing Scanner Matches</b>"]
    for _, r in results.iterrows():
        dist_val = r[dist_col] if dist_col else np.nan
        lines.append(
            f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
            f"RS {r['RS_pctile']:.2f} | DistSMA {dist_val:.2f}% | {r['Signal'] or '—'}"
        )
    ok = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, "\n".join(lines))
    print("Telegram sent." if ok else "Telegram failed.")

def export_csv(results):
    if results is None or results.empty:
        print("No matches to export.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"swing_scanner_{ts}.csv"
    results.to_csv(filename, index=False)
    print(f"Saved {len(results)} results to {filename}")

def main():
    results = run_scan()
    send_results_to_telegram(results)
    export_csv(results)

if __name__ == "__main__":
    main()
