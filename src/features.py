from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import robust_zscore


def build_monthly_macro_features(fred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if fred_df.empty:
        return pd.DataFrame(), {}
    m = fred_df.resample("M").last()
    yoy = m.pct_change(12)

    growth = yoy.filter(regex="growth|INDPRO").mean(axis=1)
    inflation = yoy.filter(regex="inflation|CPILFESL").mean(axis=1)
    labor = yoy.filter(regex="labor|UNRATE|ICSA").mean(axis=1)
    conditions = m.filter(regex="conditions|NFCI|BAA10YM|VIXCLS").mean(axis=1)

    features = pd.DataFrame(
        {
            "growth_z": robust_zscore(growth),
            "inflation_z": robust_zscore(inflation),
            "labor_z": robust_zscore(labor),
            "conditions_z": robust_zscore(conditions),
        }
    )
    if "US_rates" in m.columns:
        features["real_rate_z"] = robust_zscore(m["US_rates"])
    # deterministic slope proxy from available columns
    d10 = m.filter(regex="DGS10").mean(axis=1)
    d2 = m.filter(regex="DGS2").mean(axis=1)
    if not d10.empty and not d2.empty:
        features["slope_z"] = robust_zscore(d10 - d2)

    audit = {
        "winsorization": "1% tails",
        "robust_scaling": "median/MAD",
        "effective_sample": str(int(features.dropna().shape[0])),
    }
    return features.dropna(how="all"), audit


def regime_conditioned_mu(monthly_returns: pd.DataFrame, probs: pd.DataFrame, shrink: float = 0.6) -> pd.DataFrame:
    out = {}
    long_run = monthly_returns.mean()
    for regime in probs.columns:
        w = probs[regime].reindex(monthly_returns.index).fillna(0)
        den = w.sum()
        cond = (monthly_returns.mul(w, axis=0).sum() / den) if den > 0 else long_run
        out[regime] = (1 - shrink) * cond + shrink * long_run
    return pd.DataFrame(out)
