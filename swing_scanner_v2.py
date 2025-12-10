# -*- coding: utf-8 -*-
"""
ALL-US Breakout Screen v2 (Polygon.io ONLY)

- Fetch US stock tickers from Polygon
- Download OHLCV via Polygon aggregates
- Compute:
    • ADR%
    • Liquidity (30d avg volume)
    • Moving averages (10 & 22 SMA)
    • Relative strength vs SPY (6 mo)
    • Distance from recent 120-day high
    • Base depth
    • VCP (volatility contraction)
    • Breakout score

- Output:
    • CSV file
    • Telegram summary (optional)
"""

import os, time, requests, numpy as np, pandas as pd
from ta.trend import SMAIndicator

# ==============================
# ENV VAR CONFIG
# ==============================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_SLEEP_SECONDS = float(os.getenv("POLYGON_SLEEP_SECONDS", "0.7"))

ADR_PERIOD = int(os.getenv("ADR_PERIOD", "14"))
MA_FAST = int(os.getenv("MA_FAST", "10"))
MA_SLOW = int(os.getenv("MA_SLOW", "22"))
AVG_VOL_LOOKBACK = int(os.getenv("AVG_VOL_LOOKBACK", "30"))
HIGH_LOOKBACK_DAYS = int(os.getenv("HIGH_LOOKBACK_DAYS", "120"))
RS_LOOKBACK_MONTHS = int(os.getenv("RS_LOOKBACK_MONTHS", "6"))

MIN_ADR_PCT = float(os.getenv("MIN_ADR_PCT", "5"))
MAX_ADR_PCT = float(os.getenv("MAX_ADR_PCT", "12"))
MIN_AVG_VOL_30D = float(os.getenv("MIN_AVG_VOL_30D", "30000000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1.0"))
MIN_RS_PCTILE = float(os.getenv("MIN_RS_PCTILE", "0.98"))

PREFILTER_MIN_PRICE = float(os.getenv("PREFILTER_MIN_PRICE", "3.0"))
PREFILTER_MIN_AVG_VOL5 = float(os.getenv("PREFILTER_MIN_AVG_VOL5", "5000000"))
PREFILTER_TOP_N_BY_DOLLAR_VOL = int(os.getenv("PREFILTER_TOP_N_BY_DOLLAR_VOL", "1200"))

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


# ==============================
# POLYGON REQUEST WRAPPER
# ==============================

def _polygon_get(url, params=None, max_retries=6):
    if params is None:
        params = {}
    params.setdefault("apiKey", POLYGON_API_KEY)

    backoff = POLYGON_SLEEP_SECONDS

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                wait = float(retry_after)
            else:
                wait = backoff

            print(f"[429] Rate limited. Sleeping {wait:.2f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
            backoff *= 1.5
            continue

        try:
            r.raise_for_status()
        except Exception:
            print(f"HTTP error: {r.status_code} | URL: {r.url}")
            raise

        return r.json()

    raise RuntimeError("Polygon 429: exhausted retries")


# ==============================
# TICKER UNIVERSE FROM POLYGON
# ==============================

def get_all_us_tickers():
    print("Fetching US tickers from Polygon...")
    url = "https://api.polygon.io/v3/reference/tickers"

    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": 500,
    }

    tickers = []
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        data = _polygon_get(url, params=params)

        for item in data.get("results", []):
            t = item.get("ticker", "")
            if t and t.isalpha():
                tickers.append(t)

        cursor = data.get("next_cursor")
        if not cursor:
            break

        time.sleep(POLYGON_SLEEP_SECONDS)

    print(f"US ticker count: {len(tickers)}")
    return sorted(set(tickers))


# ==============================
# FETCH DAILY BARS FROM POLYGON
# ==============================

def fetch_daily_ohlcv(ticker, months=8):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2020-01-01/9999-12-31"
    params = {"adjusted": "true", "limit": 50000}

    data = _polygon_get(url, params=params)
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)

    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


# ==============================
# CALCULATIONS
# ==============================

def compute_adr_pct(df, period):
    if len(df) < period + 1:
        return np.nan
    rng = (df["High"] - df["Low"]).rolling(period).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    return float(rng / price * 100.0)


def rs_vs_spy(ticker_df, spy_df):
    if ticker_df.empty or spy_df.empty:
        return np.nan
    idx = ticker_df.index.intersection(spy_df.index)
    if len(idx) < 20:
        return np.nan
    t0, t1 = ticker_df["Close"].iloc[0], ticker_df["Close"].iloc[-1]
    s0, s1 = spy_df["Close"].iloc[0], spy_df["Close"].iloc[-1]
    return (t1 / t0 - 1) - (s1 / s0 - 1)


# ==============================
# RUN SCREEN
# ==============================

def run_screen():
    all_tickers = get_all_us_tickers()

    # Prefilter by recent dollar volume
    pre_universe = []
    for t in all_tickers:
        df = fetch_daily_ohlcv(t, months=1)
        if df.empty or len(df) < 5:
            continue
        px = df["Close"].iloc[-1]
        v5 = df["Volume"].tail(5).mean()
        if px >= PREFILTER_MIN_PRICE and v5 >= PREFILTER_MIN_AVG_VOL5:
            pre_universe.append((t, px * v5))

    pre_universe = sorted(pre_universe, key=lambda x: x[1], reverse=True)
    universe = [t for t, _ in pre_universe[:PREFILTER_TOP_N_BY_DOLLAR_VOL]]

    print(f"Universe after volume filter: {len(universe)}")

    spy = fetch_daily_ohlcv("SPY", months=RS_LOOKBACK_MONTHS)
    rows = []

    for t in universe:
        df = fetch_daily_ohlcv(t, months=8)
        if df.empty or len(df) < 150:
            continue

        price = df["Close"].iloc[-1]
        adr = compute_adr_pct(df, ADR_PERIOD)
        avg_vol_30 = df["Volume"].rolling(AVG_VOL_LOOKBACK).mean().iloc[-1]

        ma_fast = SMAIndicator(df["Close"], window=MA_FAST).sma_indicator().iloc[-1]
        ma_slow = SMAIndicator(df["Close"], window=MA_SLOW).sma_indicator().iloc[-1]

        recent = df.tail(HIGH_LOOKBACK_DAYS)
        high = recent["Close"].max()
        dist_high = (price / high - 1) * 100
        base_depth = -dist_high

        # RS
        rs_val = rs_vs_spy(df.tail(RS_LOOKBACK_MONTHS * 22), spy)

        if any([
            adr < MIN_ADR_PCT,
            adr > MAX_ADR_PCT,
            avg_vol_30 < MIN_AVG_VOL_30D,
            price < MIN_PRICE,
            price <= ma_fast,
            price <= ma_slow,
            base_depth < 0,
            base_depth > MAX_BASE_DEPTH_PCT,
            abs(dist_high) > MAX_DIST_FROM_HIGH_PCT,
            rs_val is None or np.isnan(rs_val),
        ]):
            continue

        rows.append({
            "Ticker": t,
            "Price": price,
            "ADR%": adr,
            "AvgVol30": avg_vol_30,
            "RS": rs_val,
            "DistHigh%": dist_high,
            "BaseDepth%": base_depth,
        })

    df_out = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    name = f"allus_breakout_v2_{ts}.csv"
    df_out.to_csv(name, index=False)
    print(f"Saved {len(df_out)} results to {name}")

    return df_out


# ==============================
# TELEGRAM SUMMARY
# ==============================

def send_summary(df):
    if TELEGRAM_BOT_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        return
    if df.empty:
        msg = "📭 No swing setups today."
    else:
        top = df.head(15)
        lines = ["<b>📈 Swing Breakout v2 (Top)</b>"]
        for _, r in top.iterrows():
            lines.append(
                f"{r['Ticker']}: ${r['Price']:.2f} | ADR {r['ADR%']:.2f}% | "
                f"RS {r['RS']:.2f} | DistHigh {r['DistHigh%']:.2f}%"
            )
        msg = "\n".join(lines)

    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
    )


# ==============================
# MAIN
# ==============================

def main():
    df = run_screen()
    send_summary(df)

if __name__ == "__main__":
    main()
