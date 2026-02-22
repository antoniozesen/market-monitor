from __future__ import annotations

import pandas as pd


def panel_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "cols": 0, "missing_pct": 100.0}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_pct": float(df.isna().mean().mean() * 100),
        "last_date": str(df.index.max().date()) if hasattr(df.index, "max") and len(df.index) > 0 else "n/a",
    }


def stale_summary(meta: pd.DataFrame, stale_days: int = 60) -> dict:
    if meta.empty or "staleness_days" not in meta.columns:
        return {"stale_count": 0, "stale_series": []}
    stale = meta.loc[meta["staleness_days"] > stale_days, "series_id"].fillna("").tolist()
    return {"stale_count": len(stale), "stale_series": stale}
