"""
Stock screener: filters large-cap stocks by P/E, volume spike, and market cap.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Callable, Optional


def get_stock_universe() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia as the screening universe."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"},
        )
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        # Hardcoded fallback: broad set of large-cap tickers
        return [
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


def _fetch_volume_data(tickers: list[str]) -> dict[str, dict]:
    """
    Batch-download 30 days of OHLCV data for all tickers.
    Returns a dict of {ticker: {current_volume, avg_volume, vol_ratio}}.
    """
    raw = yf.download(
        tickers,
        period="30d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    volume_map: dict[str, dict] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        vol_df = raw["Volume"]
    else:
        # Single ticker returned flat frame
        vol_df = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    for ticker in tickers:
        if ticker not in vol_df.columns:
            continue
        series = vol_df[ticker].dropna()
        if len(series) < 21:
            continue
        avg_vol = series.iloc[-21:-1].mean()
        current_vol = series.iloc[-1]
        if avg_vol == 0:
            continue
        volume_map[ticker] = {
            "current_volume": int(current_vol),
            "avg_volume": int(avg_vol),
            "vol_ratio": round(current_vol / avg_vol, 3),
        }

    return volume_map


def screen_stocks(
    min_market_cap: float = 5e9,
    max_pe: float = 35.0,
    min_vol_ratio: float = 1.5,
    max_vol_ratio: float = 2.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Run the full screening pipeline and return a DataFrame of candidates.

    Args:
        min_market_cap: Minimum market capitalisation in USD.
        max_pe: Maximum trailing P/E ratio (>0).
        min_vol_ratio: Minimum today-volume / 20-day-avg-volume.
        max_vol_ratio: Maximum today-volume / 20-day-avg-volume.
        progress_callback: Optional fn(current, total, message) for UI progress.
    """
    universe = get_stock_universe()

    # --- Step 1: volume pre-filter (batch, fast) ---
    if progress_callback:
        progress_callback(0, 4, "Downloading volume data for full universe...")

    volume_map = _fetch_volume_data(universe)

    vol_candidates = [
        t for t, v in volume_map.items()
        if min_vol_ratio <= v["vol_ratio"] <= max_vol_ratio
    ]

    if progress_callback:
        progress_callback(1, 4, f"{len(vol_candidates)} tickers passed volume filter. Fetching fundamentals...")

    # --- Step 2: fundamentals filter (per-ticker, slower) ---
    results = []
    total = len(vol_candidates)

    for i, ticker in enumerate(vol_candidates):
        if progress_callback and i % 5 == 0:
            progress_callback(
                2, 4,
                f"Checking fundamentals {i+1}/{total}: {ticker}",
            )
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

            vol_info = volume_map[ticker]
            results.append(
                {
                    "ticker": ticker,
                    "name": info.get("shortName", ticker),
                    "sector": info.get("sector", "N/A"),
                    "price": round(price, 2),
                    "market_cap": market_cap,
                    "pe_ratio": round(pe, 2),
                    "current_volume": vol_info["current_volume"],
                    "avg_volume": vol_info["avg_volume"],
                    "vol_ratio": vol_info["vol_ratio"],
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                }
            )
        except Exception:
            continue

    if progress_callback:
        progress_callback(3, 4, f"Fundamental screen done — {len(results)} candidates before sentiment filter.")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("vol_ratio", ascending=False).reset_index(drop=True)

    return df
