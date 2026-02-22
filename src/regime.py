from __future__ import annotations

import pandas as pd
from sklearn.mixture import GaussianMixture

from src.config import REGIME_NAMES, Settings


def _label_states(centroids: pd.DataFrame) -> dict[str, str]:
    used = set()
    mapping = {}
    for s, row in centroids.iterrows():
        g = row.get("growth_z", 0)
        i = row.get("inflation_z", 0)
        stress = row.get("stress_z", 0)
        if g > 0.3 and i < 0.3:
            label = "Goldilocks"
        elif g > 0.3 and i > 0.3:
            label = "Reflation"
        elif g < -0.3 and i < 0.3:
            label = "Slowdown"
        else:
            label = "Stagflation" if stress > 0 else "Reflation"
        if label in used:
            label = next(x for x in REGIME_NAMES if x not in used)
        used.add(label)
        mapping[s] = label
    return mapping


def _proxy_regime(features: pd.DataFrame) -> tuple[str, pd.Series]:
    if features.empty:
        return "Unavailable", pd.Series(dtype=float)
    last = features.iloc[-1]
    g, i, s = last.get("growth_z", 0), last.get("inflation_z", 0), last.get("stress_z", 0)
    if g > 0.3 and i < 0.3:
        reg = "Goldilocks"
    elif g > 0.3 and i > 0.3:
        reg = "Reflation"
    elif g < -0.3 and i < 0.3:
        reg = "Slowdown"
    else:
        reg = "Stagflation" if s > 0 else "Reflation"
    probs = pd.Series({r: 0.1 for r in REGIME_NAMES})
    probs[reg] = 0.7
    return reg, probs / probs.sum()


def run_regime_model(features: pd.DataFrame, smooth_span: int = 3) -> dict:
    cfg = Settings()
    x = features.replace([float("inf"), float("-inf")], pd.NA).dropna()
    eff = int(x.shape[0])
    if eff < cfg.min_regime_months:
        pr, p = _proxy_regime(features)
        return {
            "available": False,
            "mode": "proxy",
            "reason": f"effective sample {eff} < {cfg.min_regime_months}",
            "effective_sample": eff,
            "proxy_regime": pr,
            "proxy_probs": p,
            "probs": pd.DataFrame(),
            "state": pd.Series(dtype=str),
        }

    model = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=8)
    model.fit(x.values)
    p = pd.DataFrame(model.predict_proba(x.values), index=x.index, columns=[f"S{i}" for i in range(4)])
    cent = pd.DataFrame(model.means_, index=p.columns, columns=x.columns)
    mapping = _label_states(cent)

    probs = p.rename(columns=mapping).reindex(columns=REGIME_NAMES).fillna(0)
    probs = probs.ewm(span=smooth_span).mean().clip(lower=cfg.prob_floor)
    probs = probs.div(probs.sum(axis=1), axis=0)
    state = probs.idxmax(axis=1)

    share = state.value_counts(normalize=True)
    flips_12m = int((state.tail(12) != state.shift(1).tail(12)).sum())
    avg_duration = float(12 / max(flips_12m, 1))

    return {
        "available": True,
        "mode": "gmm",
        "effective_sample": eff,
        "probs": probs,
        "state": state,
        "centroids": cent,
        "diagnostics": {
            "converged": bool(model.converged_),
            "effective_sample": eff,
            "regime_share": share.to_dict(),
            "avg_duration_months": avg_duration,
            "flips_12m": flips_12m,
            "degenerate_warning": bool((share < 0.05).any()),
        },
    }
