from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict

import pandas as pd
import streamlit as st
from fredapi import Fred

from src.config import FRED_CANDIDATES, FRED_SEARCH


@dataclass
class MetaRow:
    indicator: str
    region: str
    series_id: str
    transformation: str
    frequency: str
    units: str
    last_obs_date: str
    staleness_days: int
    missingness_pct: float
    source_mode: str
    excluded_reason: str


def _score(row: pd.Series, keywords: list[str]) -> float:
    title = str(row.get("title", "")).lower()
    k = sum(int(w in title) for w in keywords)
    freq = 2 if str(row.get("frequency", "")).lower() == "monthly" else 0
    return float(k + freq)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_macro_library() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = st.secrets.get("FRED_API_KEY", None)
    if not key:
        return pd.DataFrame(), pd.DataFrame(), {"warnings": ["Missing FRED_API_KEY in Streamlit secrets"]}

    fred = Fred(api_key=key)
    series_list: list[pd.Series] = []
    meta: list[dict] = []
    warns: list[str] = []

    for region, concepts in FRED_CANDIDATES.items():
        for indicator, candidates in concepts.items():
            sid, data, mode, ex = None, None, "candidate", ""
            for c in candidates:
                try:
                    s = fred.get_series(c)
                    if s is not None and s.dropna().shape[0] >= 24:
                        sid, data = c, s
                        break
                except Exception:
                    continue
            if sid is None and region in FRED_SEARCH and indicator in FRED_SEARCH[region]:
                mode = "search_fallback"
                kws = FRED_SEARCH[region][indicator]
                best = None
                best_score = -1.0
                for kw in kws:
                    try:
                        r = fred.search(kw)
                        if r.empty:
                            continue
                        r = r.copy()
                        r["score"] = r.apply(lambda x: _score(x, kws), axis=1)
                        top = r.sort_values("score", ascending=False).iloc[0]
                        if float(top["score"]) > best_score:
                            best = top
                            best_score = float(top["score"])
                    except Exception:
                        continue
                if best is not None:
                    sid = str(best["id"])
                    try:
                        data = fred.get_series(sid)
                    except Exception:
                        sid = None

            if sid is None or data is None:
                warns.append(f"No usable series: {region}-{indicator}")
                ex = "candidate+search failed"
                meta.append(asdict(MetaRow(indicator, region, "", "", "", "", "", -1, 100.0, mode, ex)))
                continue

            s = pd.Series(data).dropna()
            s.index = pd.to_datetime(s.index)
            name = f"{region}_{indicator}"
            s.name = name
            series_list.append(s)

            last = s.index.max()
            meta.append(
                asdict(
                    MetaRow(
                        indicator=indicator,
                        region=region,
                        series_id=sid,
                        transformation="YoY for level series; z-score for model",
                        frequency="Monthly",
                        units="rate/index",
                        last_obs_date=str(last.date()),
                        staleness_days=(dt.date.today() - last.date()).days,
                        missingness_pct=float(s.isna().mean() * 100),
                        source_mode=mode,
                        excluded_reason="",
                    )
                )
            )

    panel = pd.concat(series_list, axis=1).sort_index() if series_list else pd.DataFrame()
    return panel, pd.DataFrame(meta), {"warnings": warns}
