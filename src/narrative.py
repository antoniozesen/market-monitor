from __future__ import annotations

import pandas as pd


def build_takeaways_regime(current_regime: str, probs: pd.Series, drivers: pd.Series) -> list[str]:
    top = probs.sort_values(ascending=False).head(2)
    return [
        f"Current dominant regime is {current_regime} with probability {top.iloc[0]:.1%}.",
        f"Second regime risk is {top.index[1]} at {top.iloc[1]:.1%}.",
        f"Growth z-score is {drivers.get('growth_z', 0):.2f} and inflation z-score is {drivers.get('inflation_z', 0):.2f}.",
    ]


def build_committee_text(regime: str, signal_table: pd.DataFrame, alloc: pd.Series) -> str:
    best = signal_table.head(3)["ticker"].tolist() if not signal_table.empty else []
    worst = signal_table.tail(3)["ticker"].tolist() if not signal_table.empty else []
    top_w = alloc.sort_values(ascending=False).head(5)
    return (
        f"Regime assessment: {regime}. Leading indicators and financial conditions imply disciplined risk budgeting. "
        f"Preferred exposures by signal strength: {', '.join(best)}; weak tails: {', '.join(worst)}. "
        f"Portfolio implementation overweights {', '.join(top_w.index.tolist())} with explicit drawdown/turnover controls."
    )


def build_banker_text(regime: str, alloc: pd.Series) -> str:
    return (
        f"Today we read markets as {regime}. We keep diversification first, add risk where momentum is healthy, "
        f"and cap fragile credit when stress rises. Largest allocations now: {', '.join(alloc.sort_values(ascending=False).head(3).index.tolist())}."
    )
