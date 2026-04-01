"""
Stock screener: filters US-listed stocks by P/E, volume spike, market cap,
and 5-year high premium.

Universe
--------
All publicly traded US equities sourced (in priority order) from:
  1. SEC EDGAR company_tickers.json  (~10 000+ tickers)
  2. Wikipedia composite fallback    (S&P 500 + S&P 400 + S&P 600)
  3. Hardcoded large-cap fallback

Volume is batch-downloaded in chunks of 400 for speed. Only tickers that pass
the volume pre-filter proceed to per-ticker fundamentals calls.
"""
import re
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------------

def _edgar_tickers() -> list[str]:
    """
    Fetch all SEC-registered US company tickers from EDGAR.
    Returns simple alphabetic tickers only (excludes warrants, rights, units).
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "InvestmentAgent/1.0 research@example.com"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        raw = [v["ticker"] for v in data.values() if isinstance(v.get("ticker"), str)]
        # Keep only clean alpha-numeric tickers (exclude warrants/rights: W, R, U suffixes etc.)
        clean = [
            t.upper().replace(".", "-")
            for t in raw
            if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", t.upper())
        ]
        return sorted(set(clean))
    except Exception:
        return []


def _wikipedia_tickers() -> list[str]:
    """Scrape S&P 500, S&P 400, and S&P 600 constituent lists from Wikipedia."""
    urls = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",  "Symbol"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",  "Ticker symbol"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",  "Ticker symbol"),
    ]
    tickers: list[str] = []
    opts = {"User-Agent": "Mozilla/5.0"}
    for url, col in urls:
        try:
            tables = pd.read_html(url, storage_options=opts)
            for tbl in tables:
                if col in tbl.columns:
                    tickers += tbl[col].str.replace(".", "-", regex=False).tolist()
                    break
        except Exception:
            continue
    return sorted(set(tickers))


_FALLBACK_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","BRK-B","LLY","V","JPM",
    "UNH","XOM","TSLA","MA","JNJ","PG","HD","MRK","AVGO","CVX",
    "ABBV","PEP","COST","KO","ADBE","WMT","BAC","TMO","CRM","MCD",
    "CSCO","ACN","ABT","NFLX","LIN","DHR","TXN","AMD","QCOM","PM",
    "NKE","INTC","MS","GS","AMGN","BMY","LOW","SPGI","BLK","AXP",
    "DE","ISRG","SYK","GILD","VRTX","PLD","AMT","CI","CB","SO",
    "DUK","NEE","ELV","HUM","CVS","WFC","USB","TGT","MDLZ","ZTS",
    "EOG","SLB","MO","CME","PNC","ICE","MCO","AON","MMC","ADP",
    "REGN","BIIB","ILMN","MRNA","DXCM","IDXX","EW","BSX","BDX","ZBH",
]


def get_stock_universe() -> list[str]:
    """
    Return the broadest available list of US-listed equity tickers.
    Falls back progressively if sources are unreachable.
    """
    tickers = _edgar_tickers()
    if len(tickers) > 2000:
        return tickers

    # EDGAR unavailable — try Wikipedia composites
    tickers = _wikipedia_tickers()
    if len(tickers) > 200:
        return tickers

    return _FALLBACK_TICKERS


# ---------------------------------------------------------------------------
# Volume batch download
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 400  # yfinance handles ~400 tickers per batch reliably


def _fetch_volume_data(
    tickers: list[str],
    min_avg_volume: int = 100_000,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, dict]:
    """
    Batch-download 30 days of volume data in chunks.

    Returns {ticker: {current_volume, avg_volume, vol_ratio}}.
    Only tickers with avg_volume >= min_avg_volume are included (liquidity gate).
    """
    volume_map: dict[str, dict] = {}
    chunks = [tickers[i:i + _CHUNK_SIZE] for i in range(0, len(tickers), _CHUNK_SIZE)]
    total_chunks = len(chunks)

    for chunk_idx, chunk in enumerate(chunks):
        if progress_callback:
            done = chunk_idx * _CHUNK_SIZE
            progress_callback(
                done, len(tickers),
                f"Volume download {chunk_idx + 1}/{total_chunks} "
                f"({done:,} / {len(tickers):,} tickers)..."
            )
        try:
            raw = yf.download(
                chunk,
                period="30d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            vol_df = raw["Volume"]
        else:
            vol_df = raw[["Volume"]].rename(columns={"Volume": chunk[0]})

        for ticker in chunk:
            if ticker not in vol_df.columns:
                continue
            series = vol_df[ticker].dropna()
            if len(series) < 21:
                continue
            avg_vol = series.iloc[-21:-1].mean()
            current_vol = series.iloc[-1]
            if avg_vol < min_avg_volume:
                continue  # liquidity gate
            vol_ratio = current_vol / avg_vol
            volume_map[ticker] = {
                "current_volume": int(current_vol),
                "avg_volume":     int(avg_vol),
                "vol_ratio":      round(vol_ratio, 3),
            }

    return volume_map


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

def screen_stocks(
    min_market_cap: float = 5e9,
    max_pe: float = 50.0,
    min_vol_ratio: float = 1.2,
    max_vol_ratio: float = 5.0,
    min_5yr_high_pct: float = 0.0,
    min_avg_volume: int = 100_000,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Screens all US-listed equities and returns a DataFrame of candidates.

    Pipeline
    --------
    1. Build universe (SEC EDGAR → Wikipedia → fallback)
    2. Batch-download 30-day volume; apply liquidity + vol-ratio pre-filter
    3. Per-ticker fundamentals (market cap, P/E, 5yr high) on survivors only

    Args:
        min_market_cap:   Min market cap in USD (0 = disabled).
        max_pe:           Max trailing P/E (99999 = disabled).
        min_vol_ratio:    Min today-vol / 20d-avg-vol.
        max_vol_ratio:    Max today-vol / 20d-avg-vol.
        min_5yr_high_pct: 5yr high must be >= this % above current price (0 = off).
        min_avg_volume:   Minimum 20-day avg daily volume (liquidity gate).
        progress_callback: Optional fn(current, total, message).
    """
    # --- Step 1: universe ---
    if progress_callback:
        progress_callback(0, 100, "Building US equity universe...")

    universe = get_stock_universe()

    if progress_callback:
        progress_callback(2, 100, f"Universe: {len(universe):,} US-listed tickers. Downloading volume data...")

    # --- Step 2: volume pre-filter (batched, fast) ---
    def _vol_cb(done, total, msg):
        if progress_callback:
            pct = 2 + int(done / total * 50) if total else 2
            progress_callback(pct, 100, msg)

    volume_map = _fetch_volume_data(universe, min_avg_volume=min_avg_volume, progress_callback=_vol_cb)

    vol_candidates = [
        t for t, v in volume_map.items()
        if min_vol_ratio <= v["vol_ratio"] <= max_vol_ratio
    ]

    if progress_callback:
        progress_callback(
            55, 100,
            f"{len(vol_candidates)} tickers passed volume filter "
            f"(from {len(volume_map):,} liquid tickers). Fetching fundamentals..."
        )

    # --- Step 3: fundamentals (per-ticker) ---
    results = []
    total = len(vol_candidates)

    for i, ticker in enumerate(vol_candidates):
        if progress_callback and i % 5 == 0:
            pct = 55 + int(i / total * 40) if total else 55
            progress_callback(pct, 100, f"Fundamentals {i + 1}/{total}: {ticker}")

        try:
            info = yf.Ticker(ticker).info

            market_cap = info.get("marketCap") or 0
            if market_cap < min_market_cap:
                continue

            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe is None or pe <= 0 or pe > max_pe:
                continue

            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0
            )

            # 5-year high
            five_yr_high = None
            five_yr_high_pct = None
            try:
                hist_5y = yf.Ticker(ticker).history(period="5y")
                if not hist_5y.empty:
                    five_yr_high = round(float(hist_5y["High"].max()), 2)
                    if price > 0:
                        five_yr_high_pct = round(
                            (five_yr_high - price) / price * 100, 1
                        )
            except Exception:
                pass

            if min_5yr_high_pct > 0:
                if five_yr_high_pct is None or five_yr_high_pct < min_5yr_high_pct:
                    continue

            vol_info = volume_map[ticker]
            results.append({
                "ticker":            ticker,
                "name":              info.get("shortName", ticker),
                "sector":            info.get("sector", "N/A"),
                "price":             round(price, 2),
                "market_cap":        market_cap,
                "pe_ratio":          round(pe, 2),
                "current_volume":    vol_info["current_volume"],
                "avg_volume":        vol_info["avg_volume"],
                "vol_ratio":         vol_info["vol_ratio"],
                "52w_high":          info.get("fiftyTwoWeekHigh"),
                "52w_low":           info.get("fiftyTwoWeekLow"),
                "5yr_high":          five_yr_high,
                "5yr_high_pct_above":five_yr_high_pct,
            })
        except Exception:
            continue

    if progress_callback:
        progress_callback(98, 100, f"Done — {len(results)} candidates before sentiment filter.")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("vol_ratio", ascending=False).reset_index(drop=True)

    return df
