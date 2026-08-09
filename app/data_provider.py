from __future__ import annotations

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataProvider:
    """Abstract provider interface."""

    def get_history(self, symbol: str, period: Optional[str] = "1y", interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError

    def get_history_bulk(self, symbols: List[str], period: Optional[str] = "1y", interval: str = "1d", workers: int = 4) -> dict:
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.get_history, s, period, interval): s for s in symbols}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    results[s] = fut.result()
                except Exception:
                    results[s] = pd.DataFrame()
        return results


class YFinanceProvider(DataProvider):
    def get_history(self, symbol: str, period: Optional[str] = "1y", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return df
        df = df.rename(columns={
            "Adj Close": "AdjClose",
            "Close": "Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Volume": "Volume",
        })
        # Ensure required columns exist
        if "Close" not in df.columns:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df


class PolygonProvider(DataProvider):
    def __init__(self, api_key: Optional[str] = None, fallback: bool = True):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self.fallback = fallback
        if not self.api_key and not self.fallback:
            raise RuntimeError("POLYGON_API_KEY is required for PolygonProvider")

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        safe = symbol.replace("/", "_")
        return CACHE_DIR / f"{safe}_{start}_{end}.parquet"

    def _fetch_agg(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # Use daily aggregates range endpoint
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        attempts = 3
        for attempt in range(attempts):
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if "results" not in data:
                    return pd.DataFrame()
                rows = []
                for ritem in data["results"]:
                    # t is epoch milliseconds
                    dt = pd.to_datetime(ritem.get("t"), unit="ms")
                    rows.append({
                        "Date": dt,
                        "Open": ritem.get("o"),
                        "High": ritem.get("h"),
                        "Low": ritem.get("l"),
                        "Close": ritem.get("c"),
                        "Volume": ritem.get("v"),
                    })
                df = pd.DataFrame(rows).set_index("Date")
                return df
            # rate limit/backoff
            if r.status_code in (429, 503):
                time.sleep(1 + attempt * 2)
                continue
            else:
                r.raise_for_status()
        return pd.DataFrame()

    def get_history(self, symbol: str, period: Optional[str] = "1y", interval: str = "1d") -> pd.DataFrame:
        # polygon requires explicit start/end dates. Interpret period like '1y' or '365d'
        end = pd.Timestamp.today().normalize()
        if isinstance(period, str) and period.endswith("y"):
            years = int(period[:-1])
            start = end - pd.DateOffset(years=years)
        elif isinstance(period, str) and period.endswith("d"):
            days = int(period[:-1])
            start = end - pd.Timedelta(days=days)
        else:
            # default to 365 days
            start = end - pd.Timedelta(days=365)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        cache = self._cache_path(symbol, start_str, end_str)
        if cache.exists():
            try:
                return pd.read_parquet(cache)
            except Exception:
                cache.unlink(missing_ok=True)

        if not self.api_key:
            if self.fallback:
                return YFinanceProvider().get_history(symbol, period=period, interval=interval)
            else:
                raise RuntimeError("POLYGON_API_KEY not set")

        df = self._fetch_agg(symbol, start_str, end_str)
        if not df.empty:
            try:
                df.to_parquet(cache)
            except Exception:
                pass
            return df
        # fallback to yfinance if polygon returned empty
        if self.fallback:
            return YFinanceProvider().get_history(symbol, period=period, interval=interval)
        return pd.DataFrame()

