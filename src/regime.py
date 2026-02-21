from __future__ import annotations

import pandas as pd
from sklearn.mixture import GaussianMixture

REGIME_NAMES = ["Goldilocks", "Reflation", "Slowdown", "Stagflation"]


def fit_regime_gmm(features: pd.DataFrame, random_state: int = 42) -> dict:
    x = features.dropna()
    if x.shape[0] < 48:
        return {"probs": pd.DataFrame(), "state": pd.Series(dtype=str), "diagnostics": {"warning": "insufficient sample"}}

    model = GaussianMixture(n_components=4, covariance_type="full", random_state=random_state, n_init=10)
    model.fit(x.values)
    p = pd.DataFrame(model.predict_proba(x.values), index=x.index, columns=[f"S{i}" for i in range(4)])

    centroids = pd.DataFrame(model.means_, columns=x.columns, index=p.columns)
    mapping = label_states(centroids)
    probs = p.rename(columns=mapping).reindex(columns=REGIME_NAMES).fillna(0)
    probs = probs.ewm(span=3).mean().clip(lower=0.02)
    probs = probs.div(probs.sum(axis=1), axis=0)
    state = probs.idxmax(axis=1)

    share = state.value_counts(normalize=True)
    diagnostics = {
        "converged": bool(model.converged_),
        "sample_size": int(x.shape[0]),
        "switches_12m": int((state.tail(12) != state.shift(1).tail(12)).sum()),
        "regime_share": share.to_dict(),
        "imbalance_warning": bool((share < 0.05).any()),
        "flip_warning": bool(((state.tail(12) != state.shift(1).tail(12)).sum()) > 6),
        "last_refit": str(x.index.max().date()),
        "refit_frequency": "quarterly",
        "missingness": float(features.isna().mean().mean()),
    }
    return {"probs": probs, "state": state, "centroids": centroids, "diagnostics": diagnostics}


def label_states(centroids: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used = set()
    for state, row in centroids.iterrows():
        g = row.get("growth_z", 0)
        i = row.get("inflation_z", 0)
        if g > 0.3 and i < 0.3:
            label = "Goldilocks"
        elif g > 0.3 and i > 0.3:
            label = "Reflation"
        elif g < -0.3 and i < 0.3:
            label = "Slowdown"
        else:
            label = "Stagflation"
        if label in used:
            label = next(r for r in REGIME_NAMES if r not in used)
        used.add(label)
        mapping[state] = label
    return mapping
