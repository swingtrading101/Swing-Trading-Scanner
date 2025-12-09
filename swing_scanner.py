# -*- coding: utf-8 -*-
"""
After-Hours Gap Scan (ALL-US) + Telegram Summary

What this does:
- Builds a large US stock universe (NASDAQ + "other" from yahoo_fin)
- Prefilters by price + recent volume so runtime stays reasonable
- Pulls daily history via yfinance for:
    - last close
    - last day's volume
    - 20-day average volume
- Pulls quote snapshot via yahoo_fin for:
    - market cap
    - after-hours % move (postMarketChangePercent)
    - regular-day high
- Applies filters to keep only:
    - micro/small caps in your range
    - strong regular-session volume + RVOL
    - solid close near day highs
    - meaningful AFTER-HOURS % move
- Scores candidates and keeps the top names
- Saves to CSV + sends a Telegram message with the top setups

This is built to predict **next-day gappers** based on **after-hours momentum**.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from yahoo_fin import stock_info as si

# --------- ENV CONFIG (set from GitHub Actions or defaults) ---------

PREFILTER_MIN_PRICE = float(os.getenv("PREFILTER_MIN_PRICE", "0.75"))
PREFILTER_MIN_AVG_VOL5 = float(os.getenv("PREFILTER_MIN_AVG_VOL5", "150000"))
PREFILTER_TOP_N_BY_VOL = int(os.getenv("PREFILTER_TOP_N_BY_VOL", "1200"))

AH_MIN_PRICE = float(os.getenv("AH_MIN_PRICE", "0.75"))
AH_MAX_PRICE = float(os.getenv("AH_MAX_PRICE", "20.0"))
AH_MIN_MC = float(os.getenv("AH_MIN_MC", "20000000"))       # 20M
AH_MAX_MC = float(os.getenv("AH_MAX_MC", "750000000"))      # 750M
AH_MIN_REG_VOL = float(os.getenv("AH_MIN_REG_VOL", "300000"))
AH_MIN_AVG_VOL_20D = float(os.getenv("AH_MIN_AVG_VOL_20D", "200000"))
AH_MIN_AH_PCT = float(os.getenv("AH_MIN_AH_PCT", "2.0"))
AH_MIN_RVOL = float(os.getenv("AH_MIN_RVOL", "1.5"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

pd.options.display.float_format = "{:,.2f}".format


# --------- DATA STRUCTURES ---------

@dataclass
class Snapshot:
    symbol: str
    price: float
    day_high: float
    reg_volume: int
    avg_vol_20: int
    market_cap: float
    ah_pct: float   # after-hours % change vs regular close


# --------- HELPER FUNCTIONS ---------

def get_all_us_tickers():
    """Get a big list of US tickers (NASDAQ + 'other')."""
    try:
        tickers = si.tickers_nasdaq() + si.tickers_other()
    except Exception:
        tickers = []
    # Keep only simple alphabetic tickers
    return sorted({t for t in tickers if t.isalpha()})


def safe_download_daily(ticker, period="3mo"):
    """Daily candles via yfinance, auto-adjusted."""
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
        # Handle multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1)[ticker]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def prefilter_universe(
    tickers,
    min_price: float = PREFILTER_MIN_PRICE,
    min_avg_vol5: float = PREFILTER_MIN_AVG_VOL5,
    cap: int = PREFILTER_TOP_N_BY_VOL,
):
    """
    Quick prefilter using last close + 5-day avg volume so we only
    run the more expensive after-hours logic on a tighter universe.
    """
    scores = []
    for t in tickers:
        try:
            d = yf.download(
                t,
                period="7d",
                interval="1d",
                progress=False,
            )
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


def get_ah_pct_from_quote_data(qd: dict) -> float:
    """
    Yahoo sometimes returns postMarketChangePercent as either
    0.0521 (5.21%) or 5.21. This normalizes it to 'percent'.
    """
    raw = qd.get("postMarketChangePercent", 0.0)
    if raw is None:
        return 0.0
    try:
        raw = float(raw)
    except Exception:
        return 0.0

    if abs(raw) < 1:  # probably fractional
        return raw * 100.0
    return raw


def fetch_snapshot(symbol: str) -> Snapshot | None:
    """
    Pull daily data (yfinance) + quote snapshot (yahoo_fin) and
    combine into a Snapshot object. Returns None if anything critical
    is missing.
    """
    df = safe_download_daily(symbol, period="3mo")
    if df.empty or len(df) < 25:
        return None

    price = float(df["Close"].iloc[-1])
    day_high = float(df["High"].iloc[-1])
    reg_vol = int(df["Volume"].iloc[-1])
    avg_vol_20 = int(df["Volume"].tail(20).mean())

    try:
        qd = si.get_quote_data(symbol)
    except Exception:
        qd = {}

    market_cap = qd.get("marketCap", np.nan)
    ah_pct = get_ah_pct_from_quote_data(qd)

    return Snapshot(
        symbol=symbol,
        price=price,
        day_high=day_high,
        reg_volume=reg_vol,
        avg_vol_20=avg_vol_20,
        market_cap=market_cap,
        ah_pct=ah_pct,
    )


def passes_filters(snap: Snapshot) -> bool:
    """Apply your after-hours / micro-cap breakout filters."""
    # Price range
    if not (AH_MIN_PRICE <= snap.price <= AH_MAX_PRICE):
        return False

    # Market cap range
    if np.isnan(snap.market_cap):
        return False
    if not (AH_MIN_MC <= snap.market_cap <= AH_MAX_MC):
        return False

    # Volume + average volume
    if snap.reg_volume < AH_MIN_REG_VOL:
        return False
    if snap.avg_vol_20 < AH_MIN_AVG_VOL_20D:
        return False

    # After-hours % move
    if snap.ah_pct < AH_MIN_AH_PCT:
        return False

    # Relative volume
    rvol = snap.reg_volume / max(snap.avg_vol_20, 1)
    if rvol < AH_MIN_RVOL:
        return False

    # Close near day high (strong regular-session close)
    if snap.day_high <= 0:
        return False
    close_vs_high = snap.price / snap.day_high
    if close_vs_high < 0.96:  # within ~4% of high
        return False

    return True


def score_snapshot(snap: Snapshot) -> float:
    """
    Scoring function to rank candidates.
    Higher = stronger after-hours + volume + close.
    """
    rvol = snap.reg_volume / max(snap.avg_vol_20, 1)
    close_vs_high = snap.price / snap.day_high if snap.day_high > 0 else 0.0

    score = (
        1.5 * snap.ah_pct +                  # after-hours move weight
        1.0 * (rvol * 10.0) +                # RVOL
        0.75 * (close_vs_high * 100.0) +     # how close to day high (%)
        0.5 * (snap.reg_volume / 1e6)        # reward high liquidity
    )
    return score


def send_telegram(text: str) -> bool:
    """Send a message to your Telegram chat."""
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


# --------- MAIN SCAN LOGIC ---------

def run_screen():
    print("Building ALL-US universe...")
    all_us = get_all_us_tickers()
    print(f"ALL_US raw size: {len(all_us)}")

    print("Prefiltering (price & recent volume)...")
    universe = prefilter_universe(
        all_us,
        PREFILTER_MIN_PRICE,
        PREFILTER_MIN_AVG_VOL5,
        PREFILTER_TOP_N_BY_VOL,
    )
    print(f"Universe after prefilter+cap: {len(universe)}")

    print("Running after-hours gap filters...")
    rows = []

    for i, symbol in enumerate(universe, start=1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(universe)} tickers...")

        snap = fetch_snapshot(symbol)
        if snap is None:
            continue

        if not passes_filters(snap):
            continue

        rvol = snap.reg_volume / max(snap.avg_vol_20, 1)
        close_vs_high = snap.price / snap.day_high if snap.day_high > 0 else 0.0
        score = score_snapshot(snap)

        rows.append({
            "Ticker": snap.symbol,
            "Price": round(snap.price, 2),
            "AH_%Change": round(snap.ah_pct, 2),
            "Reg_Volume": int(snap.reg_volume),
            "AvgVol_20d": int(snap.avg_vol_20),
            "RVOL": round(rvol, 2),
            "Close_vs_High_%": round(close_vs_high * 100.0, 2),
            "MarketCap": int(snap.market_cap),
            "Score": round(score, 2),
        })

        # polite delay to avoid hammering Yahoo
        time.sleep(0.05)

    results = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    csv_name = f"afterhours_gap_scan_{ts}.csv"
    results.to_csv(csv_name, index=False)

    if results.empty:
        print("No tickers passed the after-hours gap scan.")
        print(f"Saved empty results to {csv_name}")
        return results, csv_name

    results = results.sort_values(
        ["Score", "AH_%Change", "RVOL", "Reg_Volume"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    print("\nAfter-Hours Gap Scan (top 50):")
    print(results.head(50).to_string(index=False))
    print(f"\nSaved {len(results)} results to {csv_name}")
    return results, csv_name


def send_summary(results: pd.DataFrame):
    """Send a concise summary of the best setups to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; skipping Telegram.")
        return

    if results.empty:
        send_telegram("📭 No tickers passed the after-hours gap scan today.")
        return

    top = results.head(15).copy()

    def fmt_vol(v):
        try:
            return f"{float(v) / 1e6:.1f}M"
        except Exception:
            return str(v)

    lines = ["<b>🔥 After-Hours Gap Scan – Top Candidates</b>"]
    for _, r in top.iterrows():
        lines.append(
            f"{r['Ticker']}: ${r['Price']:.2f} | "
            f"AH {r['AH_%Change']:.2f}% | "
            f"RVOL {r['RVOL']:.2f} | "
            f"Vol {fmt_vol(r['Reg_Volume'])} | "
            f"MC {fmt_vol(r['MarketCap'])} | "
            f"Score {r['Score']:.1f}"
        )

    send_telegram("\n".join(lines))


def main():
    results, _ = run_screen()
    send_summary(results)


if __name__ == "__main__":
    main()
