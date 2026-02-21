from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from fredapi import Fred

from src.config import FRED_CORE_SERIES, SEARCH_KEYWORDS


@dataclass
class SeriesMeta:
    indicator: str
    region: str
    series_id: str
    transformation: str
    frequency: str
    units: str
    last_obs: pd.Timestamp
    staleness_days: int
    missingness: float
    source_mode: str
    rationale: str


def _fred_client() -> Fred | None:
    key = st.secrets.get("FRED_API_KEY", None)
    if not key:
        return None
    try:
        return Fred(api_key=key)
    except Exception:
        return None


def _score_search_row(row: pd.Series, keywords: list[str]) -> float:
    title = str(row.get("title", "")).lower()
    score = sum(int(k in title) for k in keywords)
    if row.get("frequency", "") == "Monthly":
        score += 2
    return float(score)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_fred_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fred = _fred_client()
    if fred is None:
        return pd.DataFrame(), pd.DataFrame(), ["Missing FRED API key in st.secrets['FRED_API_KEY']"]

    frames, meta_rows, warnings = [], [], []
    for region, concepts in FRED_CORE_SERIES.items():
        for concept, candidates in concepts.items():
            sid = None
            source_mode = "candidate"
            rationale = ""
            for c in candidates:
                try:
                    s = fred.get_series(c)
                    if s is not None and s.dropna().shape[0] > 12:
                        sid = c
                        data = s
                        break
                except Exception:
                    continue
            if sid is None and region in SEARCH_KEYWORDS and concept in SEARCH_KEYWORDS[region]:
                source_mode = "search_fallback"
                kws = SEARCH_KEYWORDS[region][concept]
                best = None
                best_score = -1.0
                for kw in kws:
                    try:
                        search = fred.search(kw)
                        if search.empty:
                            continue
                        search = search.copy()
                        search["_score"] = search.apply(lambda r: _score_search_row(r, kws), axis=1)
                        top = search.sort_values("_score", ascending=False).iloc[0]
                        if top["_score"] > best_score:
                            best_score = top["_score"]
                            best = top
                    except Exception:
                        continue
                if best is not None:
                    sid = best["id"]
                    rationale = f"fallback search via {', '.join(kws)}"
                    try:
                        data = fred.get_series(sid)
                    except Exception:
                        sid = None
            if sid is None:
                warnings.append(f"{region}-{concept}: no usable series")
                continue

            series = pd.Series(data).dropna()
            series.index = pd.to_datetime(series.index)
            series.name = f"{region}_{concept}"
            frames.append(series)
            last = series.index.max()
            meta_rows.append(SeriesMeta(
                indicator=concept,
                region=region,
                series_id=sid,
                transformation="YoY for level indicators, z-score in model",
                frequency="Monthly",
                units="index/rate",
                last_obs=last,
                staleness_days=(dt.datetime.utcnow().date() - last.date()).days,
                missingness=float(series.isna().mean()),
                source_mode=source_mode,
                rationale=rationale or "curated candidate",
            ).__dict__)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(meta_rows), warnings
    df = pd.concat(frames, axis=1).sort_index()
    return df, pd.DataFrame(meta_rows), warnings
