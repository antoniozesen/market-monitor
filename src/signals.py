from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import drawdown


def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 60:
            continue
        r1 = s.pct_change(21).iloc[-1]
        r3 = s.pct_change(63).iloc[-1]
        r6 = s.pct_change(126).iloc[-1]
        r12 = s.pct_change(252).iloc[-1]
        ma3y = s.rolling(756).mean().iloc[-1]
        dist_ma = s.iloc[-1] / ma3y - 1 if pd.notna(ma3y) and ma3y != 0 else np.nan
        dd = drawdown(s)
        dd_pct = dd.rank(pct=True).iloc[-1]
        vol3 = s.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
        vol12 = s.pct_change().rolling(252).std().iloc[-1] * np.sqrt(252)
        out.append({
            "ticker": c,
            "ret_1m": r1,
            "ret_3m": r3,
            "mom_6m": r6,
            "mom_12m": r12,
            "dist_3y_ma": dist_ma,
            "drawdown_pctile": dd_pct,
            "vol_3m": vol3,
            "vol_12m": vol12,
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    val_pct = df["dist_3y_ma"].rank(pct=True)
    df["valuation_bucket"] = pd.cut(val_pct, bins=[0, 0.1, 0.9, 1], labels=["Cheap", "Neutral", "Rich"], include_lowest=True)
    return df.sort_values("mom_6m", ascending=False)
