from __future__ import annotations

import numpy as np
import pandas as pd


def safe_get_latest(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def to_monthly(series: pd.Series, how: str = "eom") -> pd.Series:
    if series.empty:
        return series
    if how == "avg":
        return series.resample("M").mean()
    return series.resample("M").last()


def robust_zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def pct(x: float, n: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x*100:.{n}f}%"
