from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)


def robust_zscore(s: pd.Series) -> pd.Series:
    s = winsorize_series(s.dropna())
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    z = 0.6745 * (s - med) / mad
    return z.reindex(s.index)


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.resample("M").last()
    return m.pct_change().dropna(how="all")


def annualize_vol(ret: pd.Series, periods: int) -> float:
    if ret.dropna().empty:
        return np.nan
    return ret.std() * np.sqrt(periods)


def drawdown(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1


def pct_fmt(x: float, dec: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.{dec}f}%"


def num_fmt(x: float, dec: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{dec}f}"


def asof_date(*idx: pd.Index) -> pd.Timestamp:
    vals = [i.max() for i in idx if len(i) > 0]
    return max(vals) if vals else pd.NaT
