from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=4 * 3600, show_spinner=False)
def get_prices(tickers: list[str], period: str = "25y") -> tuple[pd.DataFrame, dict]:
    diagnostics = {"warnings": [], "missing_tickers": []}
    try:
        raw = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            diagnostics["warnings"].append("yfinance returned empty response")
            return pd.DataFrame(), diagnostics
        px = raw["Close"] if "Close" in raw else raw
        if isinstance(px, pd.Series):
            px = px.to_frame()
        px = px.sort_index().dropna(how="all")
        diagnostics["missing_tickers"] = [t for t in tickers if t not in px.columns]
        return px, diagnostics
    except Exception as exc:
        diagnostics["warnings"].append(f"yfinance error: {exc}")
        return pd.DataFrame(), diagnostics
