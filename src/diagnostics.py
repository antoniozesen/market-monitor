from __future__ import annotations

import pandas as pd

from src.config import FRED_CANDIDATES


def panel_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "cols": 0, "missing_pct": 100.0}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_pct": float(df.isna().mean().mean() * 100),
        "last_date": str(df.index.max().date()) if len(df.index) > 0 else "n/a",
    }


def stale_summary(meta: pd.DataFrame, stale_days: int = 60) -> dict:
    if meta.empty or "staleness_days" not in meta.columns:
        return {"stale_count": 0, "stale_series": []}
    stale = meta.loc[meta["staleness_days"] > stale_days, "series_id"].fillna("").tolist()
    return {"stale_count": len(stale), "stale_series": stale}


def missing_checklist(meta: pd.DataFrame, yf_missing: list[str], regime_available: bool, curve_available: bool) -> pd.DataFrame:
    rows: list[dict] = []
    present = set(meta["series_id"].dropna().astype(str).tolist()) if not meta.empty and "series_id" in meta.columns else set()
    for region, concepts in FRED_CANDIDATES.items():
        for concept, ids in concepts.items():
            hit = any(i in present for i in ids)
            rows.append({
                "type": "macro",
                "region": region,
                "item": concept,
                "status": "OK" if hit else "MISSING",
                "what_to_share": "Provide alternative FRED IDs for this concept" if not hit else "",
            })
    for t in yf_missing:
        rows.append({"type": "market", "region": "global", "item": t, "status": "MISSING", "what_to_share": "Ticker missing in yfinance period"})
    rows.append({"type": "engine", "region": "global", "item": "regime_model", "status": "OK" if regime_available else "MISSING", "what_to_share": "Need effective monthly sample >= minimum"})
    rows.append({"type": "engine", "region": "US", "item": "curve_2y_10y_slope", "status": "OK" if curve_available else "MISSING", "what_to_share": "Need DGS2 and DGS10"})
    return pd.DataFrame(rows)


def data_quality_summary(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame(columns=["region", "concept", "usable"])
    g = meta.assign(usable=lambda x: x["series_id"].astype(str).str.len() > 0).groupby(["region", "concept"], dropna=False)["usable"].max().reset_index()
    return g
