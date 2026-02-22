from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last().pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def sanitize_returns(df: pd.DataFrame, min_obs_per_asset: int = 24, min_rows_for_cov: int = 36, min_assets_for_opt: int = 3) -> tuple[pd.DataFrame, dict]:
    x = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    dropped_cols = [c for c in x.columns if x[c].dropna().shape[0] < min_obs_per_asset]
    x = x.drop(columns=dropped_cols, errors="ignore")
    diag = {
        "rows": int(x.shape[0]),
        "assets": int(x.shape[1]),
        "dropped_assets": dropped_cols,
        "can_cov": bool(x.shape[0] >= min_rows_for_cov and x.shape[1] >= min_assets_for_opt),
    }
    return x, diag


def fallback_covariance(returns: pd.DataFrame, eps: float = 1e-5) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    c = returns.cov().fillna(0)
    return c + np.eye(c.shape[0]) * eps


def rolling_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    peak = prices.rolling(window, min_periods=1).max()
    return prices / peak - 1
