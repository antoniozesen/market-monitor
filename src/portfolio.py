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


def _anchor_vector(cols: list[str], profile: str) -> pd.Series:
    v = pd.Series(0.0, index=cols)
    for bucket, target in ANCHOR_PROFILES[profile].items():
        bcols = [c for c in cols if c in BUCKET_MAP[bucket]]
        if bcols:
            v[bcols] = target / len(bcols)
    return v / v.sum() if v.sum() > 0 else v


def optimize_weights(monthly_ret: pd.DataFrame, mu: pd.Series, profile: str, flexibility: float, stress_z: float, regime_name: str) -> dict:
    cfg = Settings()
    clean, diag = sanitize_returns(monthly_ret, cfg.min_obs_per_asset, cfg.min_rows_for_cov, cfg.min_assets_for_opt)
    cols = [c for c in clean.columns if c in mu.index]
    clean = clean[cols]
    if clean.shape[1] < cfg.min_assets_for_opt or clean.shape[0] < 12:
        return _fallback(profile, cols, "fallback: insufficient returns", diag, [])

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
    x0 = anchor.values.copy()
    hyg_cap = 0.05 if (stress_z > 0.5 or regime_name in ["Slowdown", "Stagflation"]) else 0.25

    bounds = [(0.0, 0.25) for _ in clean.columns]
    if "HYG" in clean.columns:
        bounds[list(clean.columns).index("HYG")] = (0.0, hyg_cap)

    def bucket_sum(x: np.ndarray, b: str) -> float:
        idx = [i for i, t in enumerate(clean.columns) if t in BUCKET_MAP[b]]
        return float(x[idx].sum())

    def build_constraints(flex: float):
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
        for b, target in ANCHOR_PROFILES[profile].items():
            cons.append({"type": "ineq", "fun": lambda x, b=b, target=target: bucket_sum(x, b) - max(0, target - flex)})
            cons.append({"type": "ineq", "fun": lambda x, b=b, target=target: min(1, target + flex) - bucket_sum(x, b)})
        return cons

    mu_v = mu.reindex(clean.columns).fillna(mu.mean()).values
    binding = []

    def objective(x: np.ndarray) -> float:
        p_ret = x @ mu_v
        p_var = x @ cov.values @ x
        tail = np.percentile(-(clean.values @ x), 95)
        ridge = np.sum((x - anchor.values) ** 2)
        turn = np.sum(np.abs(x - anchor.values))
        return -(p_ret - 3 * p_var - 0.4 * tail - 0.5 * ridge - 0.1 * turn)

    for flex_try in [flexibility, min(0.2, flexibility + 0.03), min(0.2, flexibility + 0.06)]:
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=build_constraints(flex_try))
            if res.success:
                w = pd.Series(res.x, index=clean.columns)
                turnover = float(np.abs(w - anchor).sum())
                return {
                    "weights": w,
                    "anchor": anchor,
                    "turnover": turnover,
                    "annual_turnover": turnover * 12,
                    "tx_cost": turnover * 0.001,
                    "hyg_cap_triggered": hyg_cap <= 0.05,
                    "reason": f"optimized (flex={flex_try:.2f})",
                    "cov_mode": cov_mode,
                    "diag": diag,
                    "cov": cov,
                    "binding_constraints": binding,
                }
            binding.append(f"solver_failed_flex_{flex_try:.2f}: {res.message}")
        except Exception as exc:
            binding.append(f"solver_exception_flex_{flex_try:.2f}: {exc}")

    return _fallback(profile, clean.columns.tolist(), "fallback: incompatible constraints", diag, binding)


def _fallback(profile: str, cols: list[str], reason: str, diag: dict, binding: list[str]) -> dict:
    a = _anchor_vector(cols, profile)
    return {
        "weights": a.copy(),
        "anchor": a,
        "turnover": 0.0,
        "annual_turnover": 0.0,
        "tx_cost": 0.0,
        "hyg_cap_triggered": False,
        "reason": reason,
        "cov_mode": "fallback",
        "diag": diag,
        "cov": pd.DataFrame(),
        "binding_constraints": binding,
    }


def stress_scenarios(weights: pd.Series, regime_mu: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    rows = []
    for name in ["Current", "Goldilocks", "Reflation", "Slowdown", "Stagflation"]:
        if regime_mu.empty:
            mu = pd.Series(0.0, index=weights.index)
        else:
            mu = regime_mu.mean(axis=1) if name == "Current" else regime_mu.get(name, regime_mu.mean(axis=1))
        wp = weights.reindex(mu.index).fillna(0)
        er = float(wp.dot(mu)) if not mu.empty else 0.0
        vol = np.nan
        if not cov.empty:
            wc = weights.reindex(cov.index).fillna(0).values
            vol = float(np.sqrt(wc @ cov.values @ wc))
        rows.append({"Scenario": name, "Exp Return (m)": er, "Vol (m)": vol})
    return pd.DataFrame(rows)
