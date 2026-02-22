from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.utils import robust_zscore


def _transform_by_concept(series: pd.Series, concept: str) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s

    if concept in ["growth", "inflation"]:
        t = s.pct_change(12)  # YoY for level series
    elif concept in ["labor", "rates"]:
        t = s  # levels for rates
    elif concept in ["conditions", "stress"]:
        t = s
    elif concept in ["leading"]:
        t = s
    else:
        t = s
    return t


def build_regime_features(macro: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if macro.empty:
        return pd.DataFrame(), {"effective_sample": 0, "reason": "macro empty"}

    m = macro.resample("M").last()
    transformed = {}
    for c in m.columns:
        concept = c.split("_")[-1]
        transformed[c] = _transform_by_concept(m[c], concept)
    tdf = pd.DataFrame(transformed)

    growth = tdf.filter(regex="_growth$", axis=1).mean(axis=1)
    inflation = tdf.filter(regex="_inflation$", axis=1).mean(axis=1)
    labor = tdf.filter(regex="_labor$", axis=1).mean(axis=1)
    stress = tdf.filter(regex="_stress$|_conditions$", axis=1).mean(axis=1)
    real_rate = m.filter(regex="DFII10", axis=1).mean(axis=1)
    slope = m.filter(regex="DGS10", axis=1).mean(axis=1) - m.filter(regex="DGS2", axis=1).mean(axis=1)

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

    feats = feats.replace([np.inf, -np.inf], np.nan)
    # effective sample after alignment/cleaning
    cleaned = feats.dropna(thresh=4)
    cleaned = cleaned.fillna(cleaned.median())

    return cleaned, {
        "raw_rows": int(feats.shape[0]),
        "effective_sample": int(cleaned.shape[0]),
        "effective_start": str(cleaned.index.min().date()) if not cleaned.empty else "n/a",
        "effective_end": str(cleaned.index.max().date()) if not cleaned.empty else "n/a",
        "missing_pct_preclean": float(feats.isna().mean().mean() * 100),
    }


def regime_expected_returns(monthly_returns: pd.DataFrame, probs: pd.DataFrame, signals: pd.DataFrame | None = None, shrinkage: float = 0.6) -> pd.DataFrame:
    if monthly_returns.empty:
        return pd.DataFrame()
    mu_lr = monthly_returns.mean()

    if probs.empty:
        # fallback expected returns: long-run means + small momentum tilt
        tilt = pd.Series(0.0, index=monthly_returns.columns)
        if signals is not None and not signals.empty and "mom_6m" in signals.columns:
            m = signals.set_index("ticker")["mom_6m"].reindex(monthly_returns.columns).fillna(0)
            tilt = 0.10 * (m - m.mean())
        return pd.DataFrame({"Fallback": (mu_lr + tilt).clip(-0.03, 0.03)})

    out = {}
    for r in probs.columns:
        w = probs[r].reindex(monthly_returns.index).fillna(0)
        den = w.sum()
        mu_r = (monthly_returns.mul(w, axis=0).sum() / den) if den > 0 else mu_lr
        out[r] = (1 - shrinkage) * mu_r + shrinkage * mu_lr
    return pd.DataFrame(out)
