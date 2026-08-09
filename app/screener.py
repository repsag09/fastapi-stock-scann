from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from app.data_provider import DataProvider, YFinanceProvider, PolygonProvider

TEMPLATES_DIR = Path("templates")


def load_template(name: str = "minervini") -> Dict[str, Any]:
    path = TEMPLATES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()
    df["ma50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["ma150"] = df["Close"].rolling(150, min_periods=1).mean()
    df["ma200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["vol50"] = df["Volume"].rolling(50, min_periods=1).mean()
    return df


def ma_slope(series: pd.Series, lookback: int) -> float:
    arr = series.dropna().values
    if len(arr) < lookback:
        return float("nan")
    y = arr[-lookback:]
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def compute_relative_strength_from_provider(provider: DataProvider, symbol_df: pd.DataFrame, benchmark: str = "SPY", period_days: int = 252) -> float:
    # compute total return over period_days for symbol and benchmark and return a relative score
    try:
        end = symbol_df.index[-1]
        start = end - pd.Timedelta(days=period_days)
        sym_slice = symbol_df["Close"].loc[start:end]
        if len(sym_slice) < 2:
            return float("nan")
        sym_ret = sym_slice.iloc[-1] / sym_slice.iloc[0] - 1
        bench_df = provider.get_history(benchmark, period=f"{period_days}d")
        if bench_df.empty:
            return float("nan")
        bench_slice = bench_df["Close"].loc[start:end]
        if len(bench_slice) < 2:
            return float("nan")
        bench_ret = bench_slice.iloc[-1] / bench_slice.iloc[0] - 1
        if bench_ret == 0:
            return float("nan")
        # Return percentile-like metric: (sym_ret - bench_ret)/abs(bench_ret)
        return float((sym_ret - bench_ret) / (abs(bench_ret) if bench_ret != 0 else 1))
    except Exception:
        return float("nan")


def apply_template(df: pd.DataFrame, template: Dict[str, Any], provider: DataProvider | None = None) -> Tuple[bool, Dict[str, Any]]:
    params = template["params"]
    df = compute_indicators(df)
    latest = df.iloc[-1]
    flags: Dict[str, Any] = {}

    # price vs MAs
    flags["price_above_50"] = bool(latest["Close"] > latest["ma50"] * params["min_price_vs_ma"])
    flags["price_above_150"] = bool(latest["Close"] > latest["ma150"] * params["min_price_vs_ma"])
    flags["price_above_200"] = bool(latest["Close"] > latest["ma200"] * params["min_price_vs_ma"])

    # MA ordering
    if params.get("ma_ordering", True):
        flags["ma_50_above_150"] = bool(latest["ma50"] > latest["ma150"])
        flags["ma_150_above_200"] = bool(latest["ma150"] > latest["ma200"])

    # MA slopes
    flags["ma200_slope"] = ma_slope(df["ma200"], params["ma_slope_lookback"])
    flags["ma200_slope_positive"] = bool(flags["ma200_slope"] > params["min_ma200_slope"])
    flags["ma50_slope"] = ma_slope(df["ma50"], params["ma_slope_lookback"])
    flags["ma50_slope_positive"] = bool(flags["ma50_slope"] > 0)

    # breakout detection (prior high in lookback window)
    lookback = params["breakout_lookback"]
    if len(df) > lookback:
        prior_high = df["Close"].iloc[-(lookback + 1):-1].max()
    else:
        prior_high = df["Close"].iloc[:-1].max() if len(df) > 1 else float("nan")
    if pd.isna(prior_high):
        flags["breakout"] = False
    else:
        flags["breakout"] = bool(latest["Close"] > prior_high * params["breakout_multiplier"])

    # volume on breakout
    vol_avg = df["Volume"].rolling(params["volume_avg_lookback"], min_periods=1).mean().iloc[-1]
    flags["volume_on_breakout"] = bool(latest["Volume"] > vol_avg * params["volume_multiplier"])

    # relative strength proxy vs SPY
    rs = None
    if provider is not None:
        rs = compute_relative_strength_from_provider(provider, df, benchmark="SPY", period_days=params["rs_period_days"])
    else:
        # without provider we cannot compute RS reliably; set NaN
        rs = float("nan")
    flags["rs_score"] = rs
    flags["rs_pass"] = bool((rs >= params["min_rs_pct"]) if not pd.isna(rs) else False)

    # overall pass
    musts = [
        "price_above_50", "price_above_150", "price_above_200",
        "ma_50_above_150", "ma_150_above_200",
        "ma200_slope_positive", "breakout", "volume_on_breakout", "rs_pass"
    ]
    overall = True
    for key in musts:
        if key in flags:
            overall = overall and bool(flags[key])

    return overall, flags

