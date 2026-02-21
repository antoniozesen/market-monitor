from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def download_prices(tickers: list[str], period: str = "25y") -> tuple[pd.DataFrame, list[str]]:
    try:
        px = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame()
        px = px.dropna(how="all").sort_index()
        missing = [t for t in tickers if t not in px.columns]
        return px, missing
    except Exception:
        return pd.DataFrame(), tickers
