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
        "last_date": str(df.index.max().date()) if hasattr(df.index, "max") and len(df.index) > 0 else "n/a",
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
            rows.append(
                {
                    "type": "macro",
                    "region": region,
                    "item": f"{concept}",
                    "status": "OK" if hit else "MISSING",
                    "what_to_share": "Provide alternative FRED series id(s) for this concept" if not hit else "",
                }
            )

    for t in yf_missing:
        rows.append({
            "type": "market",
            "region": "global",
            "item": t,
            "status": "MISSING",
            "what_to_share": "Ticker unavailable in yfinance for selected period; share replacement ETF if needed",
        })

    rows.append({
        "type": "engine",
        "region": "global",
        "item": "regime_model",
        "status": "OK" if regime_available else "MISSING",
        "what_to_share": "Need more usable macro history or additional monthly series" if not regime_available else "",
    })
    rows.append({
        "type": "engine",
        "region": "US",
        "item": "curve_2y_10y_slope",
        "status": "OK" if curve_available else "MISSING",
        "what_to_share": "Need FRED DGS2 and DGS10 (or valid substitutes)",
    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["status", "type", "region", "item"], ascending=[True, True, True, True]).reset_index(drop=True)
