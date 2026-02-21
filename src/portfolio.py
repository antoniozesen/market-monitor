from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.config import ANCHOR_PROFILES, BUCKET_MAP


def benchmark_60_40(monthly_ret: pd.DataFrame) -> pd.Series:
    r = monthly_ret[["SPY", "IEF"]].dropna()
    return 0.6 * r["SPY"] + 0.4 * r["IEF"]


def optimize_allocation(
    monthly_ret: pd.DataFrame,
    mu: pd.Series,
    profile: str,
    flexibility: float,
    stress_z: float,
    regime_name: str,
) -> dict:
    cols = [c for c in monthly_ret.columns if c in mu.index]
    r = monthly_ret[cols].dropna()
    x0 = np.repeat(1 / len(cols), len(cols))

    lw = LedoitWolf().fit(r.values)
    cov = lw.covariance_
    anchor_bucket = ANCHOR_PROFILES[profile]
    anchor = np.zeros(len(cols))
    for i, t in enumerate(cols):
        for b, members in BUCKET_MAP.items():
            if t in members:
                anchor[i] = anchor_bucket[b] / len([m for m in members if m in cols])

    hyg_idx = cols.index("HYG") if "HYG" in cols else None
    hyg_cap = 0.05 if (stress_z > 0.5 or regime_name in ["Slowdown", "Stagflation"]) else 0.25

    bounds = [(0, 0.25) for _ in cols]
    if hyg_idx is not None:
        bounds[hyg_idx] = (0, hyg_cap)

    def bucket_constraint(x, bucket):
        idx = [i for i, t in enumerate(cols) if t in BUCKET_MAP[bucket]]
        return x[idx].sum()

    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    for b, w in anchor_bucket.items():
        cons += [
            {"type": "ineq", "fun": lambda x, b=b, w=w: bucket_constraint(x, b) - max(0, w - flexibility)},
            {"type": "ineq", "fun": lambda x, b=b, w=w: min(1, w + flexibility) - bucket_constraint(x, b)},
        ]

    mu_v = mu[cols].fillna(mu.mean()).values

    def objective(x):
        ret = x @ mu_v
        var = x @ cov @ x
        ridge = np.sum((x - anchor) ** 2)
        cvar_proxy = np.percentile(-(r.values @ x), 95)
        turn = np.sum(np.abs(x - anchor))
        return -(ret - 3 * var - 0.5 * cvar_proxy - 0.5 * ridge - 0.1 * turn)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = pd.Series(res.x if res.success else anchor, index=cols)
    turnover = float(np.sum(np.abs(w.values - anchor)))
    tc_impact = turnover * 0.001
    return {
        "weights": w,
        "success": bool(res.success),
        "message": res.message,
        "turnover": turnover,
        "annual_turnover": turnover * 12,
        "tc_impact": tc_impact,
        "hyg_cap_triggered": hyg_cap <= 0.05,
    }


def stress_flip_table(weights: pd.Series, regime_mu: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["Current", "Goldilocks", "Reflation", "Slowdown", "Stagflation"]:
        m = regime_mu.mean(axis=1) if regime == "Current" else regime_mu.get(regime, regime_mu.mean(axis=1))
        mu_p = float(weights.reindex(m.index).fillna(0).dot(m))
        vol = float(np.sqrt(weights.reindex(cov.index).fillna(0).values @ cov.values @ weights.reindex(cov.index).fillna(0).values))
        rows.append({"Scenario": regime, "Exp Return (m)": mu_p, "Vol (m)": vol})
    return pd.DataFrame(rows)
