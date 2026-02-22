from __future__ import annotations

import pandas as pd


def committee_text(regime: str, stress_z: float, top_assets: list[str], weak_assets: list[str], top_w: list[str]) -> str:
    return (
        f"Regime assessment: {regime}. Stress z-score at {stress_z:.2f}. "
        f"Signal leaders: {', '.join(top_assets) if top_assets else 'n/a'}; laggards: {', '.join(weak_assets) if weak_assets else 'n/a'}. "
        f"Portfolio overweights: {', '.join(top_w) if top_w else 'n/a'}."
    )


def banker_text(regime: str, top_w: list[str]) -> str:
    return (
        f"Hoy leemos el mercado como {regime}. Mantenemos cartera diversificada y ajustamos riesgo de forma prudente. "
        f"Ahora pesan más: {', '.join(top_w) if top_w else 'n/a'}."
    )


def allocation_takeaways(weights: pd.Series, reason: str, hyg_flag: bool) -> list[str]:
    if weights.empty:
        return ["No allocation available.", "Fallback used.", f"Reason: {reason}."]
    return [
        f"Top weight is {weights.idxmax()} at {weights.max():.1%}.",
        f"Allocation mode: {reason}.",
        f"Defensive HY cap active: {'Yes' if hyg_flag else 'No'}.",
    ]
