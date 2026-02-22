from __future__ import annotations

import numpy as np
import pandas as pd


def build_signals(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.shape[0] < 90:
            continue
        ret = s.pct_change()
        ma200 = s.rolling(200).mean()
        rows.append(
            {
                "ticker": t,
                "ret_1m": s.pct_change(21).iloc[-1],
                "ret_3m": s.pct_change(63).iloc[-1],
                "mom_6m": s.pct_change(126).iloc[-1],
                "mom_12m": s.pct_change(252).iloc[-1],
                "vol_3m": ret.rolling(63).std().iloc[-1] * np.sqrt(252),
                "vol_12m": ret.rolling(252).std().iloc[-1] * np.sqrt(252),
                "drawdown": (s / s.cummax() - 1).iloc[-1],
                "above_200d": float(s.iloc[-1] > ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan,
                "ma_slope": (ma200.iloc[-1] / ma200.iloc[-20] - 1) if s.shape[0] > 220 and ma200.iloc[-20] != 0 else np.nan,
                "dist_3y_ma": (s.iloc[-1] / s.rolling(756).mean().iloc[-1] - 1) if s.shape[0] > 756 else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    p = df["dist_3y_ma"].rank(pct=True)
    df["valuation_bucket"] = pd.cut(p, [0, 0.1, 0.9, 1], labels=["Cheap", "Neutral", "Rich"], include_lowest=True)
    return df.sort_values("mom_6m", ascending=False)
