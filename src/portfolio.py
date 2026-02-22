from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.config import Settings
from src.universe import ANCHOR_PROFILES, BUCKET_MAP
from src.risk import fallback_covariance, sanitize_returns


def benchmark_60_40(monthly_ret: pd.DataFrame) -> pd.Series:
    if not {"SPY", "IEF"}.issubset(monthly_ret.columns):
        return pd.Series(dtype=float)
    r = monthly_ret[["SPY", "IEF"]].dropna()
    return 0.6 * r["SPY"] + 0.4 * r["IEF"]


def optimize_weights(monthly_ret: pd.DataFrame, mu: pd.Series, profile: str, flexibility: float, stress_z: float, regime_name: str) -> dict:
    cfg = Settings()
    clean, diag = sanitize_returns(monthly_ret, cfg.min_obs_per_asset, cfg.min_rows_for_cov, cfg.min_assets_for_opt)

    cols = [c for c in clean.columns if c in mu.index]
    clean = clean[cols]
    if clean.shape[1] < cfg.min_assets_for_opt or clean.shape[0] < 12:
        return fallback_anchor(profile, cols, "insufficient clean returns for optimization", diag)

    if diag["can_cov"]:
        try:
            cov = pd.DataFrame(LedoitWolf().fit(clean.values).covariance_, index=clean.columns, columns=clean.columns)
            cov_mode = "LedoitWolf"
        except Exception:
            cov = fallback_covariance(clean)
            cov_mode = "sample+ridgeterm"
    else:
        cov = fallback_covariance(clean)
        cov_mode = "sample+ridgeterm"

    anchor = _anchor_vector(clean.columns.tolist(), profile)
    x0 = np.array(anchor.values)

    hyg_cap = 0.05 if (stress_z > 0.5 or regime_name in ["Slowdown", "Stagflation"]) else 0.25
    bounds = [(0.0, 0.25) for _ in clean.columns]
    if "HYG" in clean.columns:
        bounds[list(clean.columns).index("HYG")] = (0.0, hyg_cap)

    def bucket_sum(x: np.ndarray, bucket: str) -> float:
        idx = [i for i, t in enumerate(clean.columns) if t in BUCKET_MAP[bucket]]
        return float(x[idx].sum())

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    for b, target in ANCHOR_PROFILES[profile].items():
        constraints.append({"type": "ineq", "fun": lambda x, b=b, target=target: bucket_sum(x, b) - max(0, target - flexibility)})
        constraints.append({"type": "ineq", "fun": lambda x, b=b, target=target: min(1, target + flexibility) - bucket_sum(x, b)})

    mu_v = mu.reindex(clean.columns).fillna(mu.mean()).values

    def obj(x: np.ndarray) -> float:
        p_ret = x @ mu_v
        p_var = x @ cov.values @ x
        p_cvar = np.percentile(-(clean.values @ x), 95)
        ridge = np.sum((x - anchor.values) ** 2)
        turn = np.sum(np.abs(x - anchor.values))
        return -(p_ret - 3 * p_var - 0.4 * p_cvar - 0.5 * ridge - 0.1 * turn)

    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        w = pd.Series(res.x if res.success else anchor.values, index=clean.columns)
        reason = "optimized" if res.success else f"solver fallback: {res.message}"
    except Exception as exc:
        return fallback_anchor(profile, clean.columns.tolist(), f"optimization exception: {exc}", diag)

    turnover = float(np.abs(w - anchor).sum())
    return {
        "weights": w,
        "anchor": anchor,
        "turnover": turnover,
        "annual_turnover": turnover * 12,
        "tx_cost": turnover * 0.001,
        "hyg_cap_triggered": hyg_cap <= 0.05,
        "reason": reason,
        "cov_mode": cov_mode,
        "diag": diag,
        "cov": cov,
    }


def fallback_anchor(profile: str, cols: list[str], reason: str, diag: dict) -> dict:
    anchor = _anchor_vector(cols, profile)
    return {
        "weights": anchor.copy(),
        "anchor": anchor,
        "turnover": 0.0,
        "annual_turnover": 0.0,
        "tx_cost": 0.0,
        "hyg_cap_triggered": False,
        "reason": reason,
        "cov_mode": "fallback",
        "diag": diag,
        "cov": pd.DataFrame(),
    }


def _anchor_vector(cols: list[str], profile: str) -> pd.Series:
    v = pd.Series(0.0, index=cols)
    for bucket, target in ANCHOR_PROFILES[profile].items():
        bucket_cols = [c for c in cols if c in BUCKET_MAP[bucket]]
        if bucket_cols:
            v[bucket_cols] = target / len(bucket_cols)
    s = v.sum()
    return (v / s) if s > 0 else v


def stress_scenarios(weights: pd.Series, regime_mu: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    rows = []
    names = ["Current", "Goldilocks", "Reflation", "Slowdown", "Stagflation"]
    for n in names:
        mu = regime_mu.mean(axis=1) if (regime_mu.empty or n == "Current") else regime_mu.get(n, regime_mu.mean(axis=1))
        mu = mu if isinstance(mu, pd.Series) else pd.Series(dtype=float)
        wp = weights.reindex(mu.index).fillna(0)
        er = float(wp.dot(mu)) if not mu.empty else 0.0
        if cov.empty:
            vol = np.nan
        else:
            wc = weights.reindex(cov.index).fillna(0).values
            vol = float(np.sqrt(wc @ cov.values @ wc))
        rows.append({"Scenario": n, "Exp Return (m)": er, "Vol (m)": vol})
    return pd.DataFrame(rows)
