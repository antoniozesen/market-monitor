from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import streamlit as st
from fredapi import Fred

from src.config import FRED_CANDIDATES, FRED_SEARCH, Settings, TRANSFORM_RULES


@dataclass
class MetaRow:
    concept: str
    region: str
    series_id: str
    title: str
    units: str
    frequency: str
    transformation: str
    last_obs_date: str
    staleness_days: int
    missingness_pct: float
    source_mode: str
    excluded_reason: str
    selection_why: str


def _score_candidate(meta_row: pd.Series, keywords: list[str], staleness_days: int, missingness: float, history: int) -> float:
    title = str(meta_row.get("title", "")).lower()
    freq = str(meta_row.get("frequency", "")).lower()
    f_score = 30 if "month" in freq else (15 if "week" in freq else 0)
    kw_score = sum(int(k in title) for k in keywords) * 4
    stale_pen = -40 if staleness_days > 120 else 0
    stale_pen += -80 if staleness_days > 365 else 0
    miss_pen = -min(30.0, missingness * 100)
    hist_score = min(20, history // 24)
    return float(f_score + kw_score + stale_pen + miss_pen + hist_score)


def _series_meta(fred: Fred, sid: str) -> pd.Series:
    info = fred.get_series_info(sid)
    return info if info is not None else pd.Series(dtype=object)


def _try_series(fred: Fred, sid: str) -> tuple[pd.Series | None, pd.Series]:
    try:
        s = fred.get_series(sid)
        if s is None:
            return None, pd.Series(dtype=object)
        s = pd.Series(s).dropna()
        s.index = pd.to_datetime(s.index)
        return s, _series_meta(fred, sid)
    except Exception:
        return None, pd.Series(dtype=object)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_macro_library() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = st.secrets.get("FRED_API_KEY", None)
    if not key:
        return pd.DataFrame(), pd.DataFrame(), {"warnings": ["Missing FRED_API_KEY in Streamlit secrets"]}

    cfg = Settings()
    fred = Fred(api_key=key)
    panel: list[pd.Series] = []
    meta_rows: list[dict] = []
    warnings: list[str] = []
    dropped_for_quality: list[str] = []

    for region, concepts in FRED_CANDIDATES.items():
        for concept, candidates in concepts.items():
            best = None
            best_score = -1e9
            mode = "candidate"
            keywords = [region.lower(), concept.lower(), "monthly"]

            candidate_pool = list(candidates)
            if region in FRED_SEARCH and concept in FRED_SEARCH[region]:
                for query in FRED_SEARCH[region][concept]:
                    try:
                        sr = fred.search(query)
                        if sr is not None and not sr.empty:
                            candidate_pool.extend(sr.head(15)["id"].astype(str).tolist())
                    except Exception:
                        continue

            seen = set()
            for sid in candidate_pool:
                if sid in seen:
                    continue
                seen.add(sid)
                s, info = _try_series(fred, sid)
                if s is None or s.empty:
                    continue

                last = s.index.max()
                stale = (dt.date.today() - last.date()).days
                missingness = float(1 - (s.shape[0] / max(1, len(pd.date_range(s.index.min(), s.index.max(), freq="M")))))
                score = _score_candidate(info, keywords, stale, missingness, s.shape[0])
                if score > best_score:
                    best_score = score
                    best = (sid, s, info, stale, missingness)

            if best is None:
                warnings.append(f"No usable series: {region}-{concept}")
                meta_rows.append(asdict(MetaRow(concept, region, "", "", "", "", TRANSFORM_RULES.get(concept, ""), "", -1, 100.0, "search_fallback", "candidate+search failed", "none")))
                continue

            sid, s, info, stale, missingness = best
            mode = "candidate" if sid in candidates else "search_fallback"
            if stale > cfg.severe_stale_days:
                dropped_for_quality.append(f"{region}-{concept}:{sid}")

            s.name = f"{region}_{concept}"
            panel.append(s)

            title = str(info.get("title", ""))
            units = str(info.get("units", ""))
            freq = str(info.get("frequency", ""))
            why = f"score={best_score:.1f}; stale={stale}d; missing={missingness:.1%}; hist={s.shape[0]}"

            meta_rows.append(
                asdict(
                    MetaRow(
                        concept=concept,
                        region=region,
                        series_id=sid,
                        title=title,
                        units=units,
                        frequency=freq,
                        transformation=TRANSFORM_RULES.get(concept, "level_z"),
                        last_obs_date=str(s.index.max().date()),
                        staleness_days=int(stale),
                        missingness_pct=float(missingness * 100),
                        source_mode=mode,
                        excluded_reason="stale>365d" if stale > cfg.severe_stale_days else "",
                        selection_why=why,
                    )
                )
            )

    macro = pd.concat(panel, axis=1).sort_index() if panel else pd.DataFrame()
    return macro, pd.DataFrame(meta_rows), {"warnings": warnings, "dropped_for_quality": dropped_for_quality}
