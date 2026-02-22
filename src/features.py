from __future__ import annotations

import pandas as pd

from src.utils import robust_zscore


def build_regime_features(macro: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if macro.empty:
        return pd.DataFrame(), {"reason": "macro empty"}

    m = macro.resample("M").last()
    yoy = m.pct_change(12)

    growth = yoy.filter(regex="growth|INDPRO|PAYEMS|HOUST|leading").mean(axis=1)
    inflation = yoy.filter(regex="inflation|CPI|PCE").mean(axis=1)
    labor = yoy.filter(regex="labor|UNRATE|ICSA").mean(axis=1)
    stress = m.filter(regex="conditions|NFCI|BAA10YM|VIXCLS").mean(axis=1)
    real_rate = m.filter(regex="DFII10").mean(axis=1)
    slope = m.filter(regex="DGS10").mean(axis=1) - m.filter(regex="DGS2").mean(axis=1)

    feats = pd.DataFrame(
        {
            "growth_z": robust_zscore(growth),
            "inflation_z": robust_zscore(inflation),
            "labor_z": robust_zscore(labor),
            "stress_z": robust_zscore(stress),
            "real_rate_z": robust_zscore(real_rate),
            "slope_z": robust_zscore(slope),
        }
    )
    return feats, {"sample_rows": int(feats.dropna(how="all").shape[0]), "missing_pct": float(feats.isna().mean().mean() * 100)}


def regime_expected_returns(monthly_returns: pd.DataFrame, probs: pd.DataFrame, shrinkage: float = 0.6) -> pd.DataFrame:
    if monthly_returns.empty or probs.empty:
        return pd.DataFrame()
    mu_lr = monthly_returns.mean()
    out = {}
    for r in probs.columns:
        w = probs[r].reindex(monthly_returns.index).fillna(0)
        den = w.sum()
        mu_r = (monthly_returns.mul(w, axis=0).sum() / den) if den > 0 else mu_lr
        out[r] = (1 - shrinkage) * mu_r + shrinkage * mu_lr
    return pd.DataFrame(out)
