from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ALL_TICKERS, ANCHOR_PROFILES, ETF_NAMES
from src.data_fred import fetch_fred_panel
from src.data_yf import download_prices
from src.features import build_monthly_macro_features, regime_conditioned_mu
from src.narrative import build_banker_text, build_committee_text, build_takeaways_regime
from src.plots import plot_curve_spread, plot_regime_probs, plot_timeline
from src.portfolio import benchmark_60_40, optimize_allocation, stress_flip_table
from src.regime import fit_regime_gmm
from src.signals import compute_signals
from src.utils import asof_date, pct_fmt

st.set_page_config(page_title="Cross-Asset Market Monitor", layout="wide")
st.title("Institutional Cross-Asset Market Monitor")

with st.sidebar:
    profile = st.selectbox("Profile", list(ANCHOR_PROFILES.keys()), index=1)
    flex = st.slider("Bucket flexibility (±)", 0.0, 0.20, 0.10, 0.01)
    include_mchi = st.toggle("Include MCHI", value=True)
    client_safe = st.toggle("Client-safe mode", value=False)

tickers = [t for t in ALL_TICKERS if include_mchi or t != "MCHI"]
prices, missing_tickers = download_prices(tickers)
fred_df, data_dict, fred_warnings = fetch_fred_panel()

if prices.empty:
    st.error("No market data available from yfinance currently.")
    st.stop()

monthly_ret = prices.resample("M").last().pct_change().dropna(how="all")
signals = compute_signals(prices)
features, feature_audit = build_monthly_macro_features(fred_df)
regime = fit_regime_gmm(features)

if regime["probs"].empty:
    st.warning("Regime engine has insufficient sample. Showing data and signals only.")
    current_regime = "Unavailable"
    probs_now = pd.Series(dtype=float)
else:
    current_regime = regime["state"].iloc[-1]
    probs_now = regime["probs"].iloc[-1]

mu_regime = regime_conditioned_mu(monthly_ret, regime["probs"] if not regime["probs"].empty else pd.DataFrame(index=monthly_ret.index))
mu_current = mu_regime.get(current_regime, monthly_ret.mean()) if not mu_regime.empty else monthly_ret.mean()
stress_z = float(features.get("conditions_z", pd.Series([0])).dropna().iloc[-1]) if not features.empty else 0.0
alloc = optimize_allocation(monthly_ret, mu_current, profile, flex, stress_z, current_regime) if not monthly_ret.empty else None

asof = asof_date(prices.index, fred_df.index)
st.caption(f"As-of date: {asof.date() if pd.notna(asof) else 'n/a'}")

if missing_tickers:
    st.warning(f"Insufficient history/missing tickers: {', '.join(missing_tickers)}")
if fred_warnings:
    st.warning("FRED fallbacks/issues: " + " | ".join(fred_warnings[:6]))
if not data_dict.empty and (data_dict["staleness_days"] > 60).any():
    stale = data_dict.loc[data_dict["staleness_days"] > 60, "series_id"].tolist()
    st.warning(f"Stale macro series (>60d): {', '.join(stale[:8])}")

ov, reg_tab, mkt, sig_tab, alloc_tab, narr, data_tab = st.tabs([
    "Overview", "Regime", "Markets / Curves & Spreads", "Signals", "Allocation", "Narrative", "Data Dictionary / Inputs"
])

with ov:
    c1, c2, c3 = st.columns(3)
    c1.metric("Current regime", current_regime)
    c2.metric("Stress z", f"{stress_z:.2f}")
    spy12 = prices["SPY"].pct_change(252).iloc[-1] if "SPY" in prices else np.nan
    c3.metric("SPY 12m", pct_fmt(spy12))
    if not signals.empty:
        heat = signals.set_index("ticker")[["mom_6m", "mom_12m", "dist_3y_ma"]]
        st.plotly_chart(px.imshow(heat.T, aspect="auto", title="Signal heatmap"), use_container_width=True)
    st.markdown("**Key takeaways**")
    takes = [
        f"Regime now: {current_regime}.",
        f"Profile selected: {profile} with ±{int(flex*100)}pp flexibility.",
        f"Top momentum ETF: {signals.iloc[0]['ticker'] if not signals.empty else 'n/a'}.",
    ]
    for t in takes:
        st.write(f"- {t}")

with reg_tab:
    if not regime["probs"].empty:
        st.plotly_chart(plot_regime_probs(regime["probs"].tail(12)), use_container_width=True)
        st.plotly_chart(plot_timeline(regime["state"].tail(180)), use_container_width=True)
        st.dataframe(features.tail(24).round(2))
        for t in build_takeaways_regime(current_regime, probs_now, features.iloc[-1]):
            st.write(f"- {t}")
    if not client_safe:
        with st.expander("Model diagnostics"):
            st.json(regime["diagnostics"])
            st.json(feature_audit)

with mkt:
    curve = fred_df[[c for c in fred_df.columns if c.endswith("rates") or "DGS" in c]].copy() if not fred_df.empty else pd.DataFrame()
    if not curve.empty:
        curve = curve.rename(columns={c: c.split("_")[-1] for c in curve.columns})
        if "DGS10" in curve.columns and "DGS2" in curve.columns:
            curve["Slope"] = curve["DGS10"] - curve["DGS2"]
            st.plotly_chart(plot_curve_spread(curve.dropna().tail(240)), use_container_width=True)
    hyg_lqd = (prices["HYG"] / prices["LQD"]).dropna() if {"HYG", "LQD"}.issubset(prices.columns) else pd.Series(dtype=float)
    if not hyg_lqd.empty:
        st.line_chart(hyg_lqd.tail(756), height=220)
    st.markdown("**Key takeaways**")
    st.write(f"- Curve slope latest: {curve['Slope'].dropna().iloc[-1]:.2f}" if not curve.empty else "- Curve unavailable")
    st.write(f"- HYG/LQD trend 6m: {((hyg_lqd.iloc[-1]/hyg_lqd.iloc[-126])-1):.1%}" if len(hyg_lqd) > 126 else "- HYG/LQD insufficient history")
    st.write(f"- Real-rate proxy stress z: {stress_z:.2f}")

with sig_tab:
    if signals.empty:
        st.info("Signals unavailable")
    else:
        out = signals.copy()
        out["name"] = out["ticker"].map(ETF_NAMES)
        st.dataframe(out[["ticker", "name", "valuation_bucket", "mom_6m", "mom_12m", "vol_3m", "drawdown_pctile"]].round(3))
        hm = out.set_index("ticker")[["mom_6m", "mom_12m", "drawdown_pctile"]]
        st.plotly_chart(px.imshow(hm.T, aspect="auto", title="Signals percentiles"), use_container_width=True)
        with st.expander("Definitions"):
            st.write("Valuation: percentile of 3y MA distance (Cheap p<=10, Rich p>=90).")
            st.write("Momentum: total return over 6m/12m.")
    st.markdown("**Key takeaways**")
    if not signals.empty:
        st.write(f"- Rich bucket count: {(signals['valuation_bucket']=='Rich').sum()} of {len(signals)}.")
        st.write(f"- Cheapest asset now: {signals.sort_values('dist_3y_ma').iloc[0]['ticker']}.")
        st.write(f"- Highest 12m momentum: {signals.sort_values('mom_12m', ascending=False).iloc[0]['ticker']}.")

with alloc_tab:
    if alloc is None:
        st.info("Allocation unavailable.")
    else:
        w = alloc["weights"].sort_values(ascending=False)
        res = pd.DataFrame({"ticker": w.index, "name": [ETF_NAMES.get(t, t) for t in w.index], "weight": w.values})
        st.dataframe(res.style.format({"weight": "{:.2%}"}))
        st.bar_chart(w)
        bmk = benchmark_60_40(monthly_ret) if {"SPY", "IEF"}.issubset(monthly_ret.columns) else pd.Series(dtype=float)
        st.write(f"Turnover: {alloc['turnover']:.2f} | Annualized: {alloc['annual_turnover']:.2f} | TC impact: {alloc['tc_impact']:.2%}")
        st.write(f"HYG defensive cap triggered: {'Yes' if alloc['hyg_cap_triggered'] else 'No'}")
        cov = monthly_ret.cov()
        stf = stress_flip_table(w, mu_regime, cov)
        st.dataframe(stf.style.format({"Exp Return (m)": "{:.2%}", "Vol (m)": "{:.2%}"}))
        st.plotly_chart(px.bar(stf, x="Scenario", y="Exp Return (m)", title="What-if regime flip"), use_container_width=True)
    st.markdown("**Key takeaways**")
    if alloc is not None:
        st.write(f"- Profile-sensitive anchor: {profile}.")
        st.write(f"- Top weight: {alloc['weights'].idxmax()} at {alloc['weights'].max():.1%}.")
        st.write(f"- Defensive HY rule active: {'Yes' if alloc['hyg_cap_triggered'] else 'No'}.")

with narr:
    if alloc is not None:
        st.subheader("Comité (technical)")
        st.text_area("", build_committee_text(current_regime, signals, alloc["weights"]), height=160)
        st.subheader("Banqueros (simple)")
        st.text_area(" ", build_banker_text(current_regime, alloc["weights"]), height=120)
    st.markdown("**Key takeaways**")
    st.write("- Narrative stays consistent with regime, drivers, signals, and allocation.")
    st.write("- Text is deterministic and copy/paste ready.")
    st.write("- Risks include data gaps, regime shifts, and turnover costs.")

with data_tab:
    st.subheader("Data Dictionary")
    if data_dict.empty:
        st.info("No FRED metadata available.")
    else:
        st.dataframe(data_dict)
        csv = data_dict.to_csv(index=False).encode()
        st.download_button("Download data dictionary CSV", csv, "data_dictionary.csv", "text/csv")
    st.subheader("Audit trail")
    st.write({
        "features_used": list(features.columns),
        "missing_features": [c for c in ["growth_z", "inflation_z", "conditions_z", "slope_z", "real_rate_z"] if c not in features.columns],
        "last_refit": regime.get("diagnostics", {}).get("last_refit", "n/a"),
    })

# quick self-checks
if not set(tickers).issubset(set(ETF_NAMES.keys())):
    st.error("Name mapping incomplete.")
if not regime["probs"].empty:
    ok = np.allclose(regime["probs"].sum(axis=1).tail(3).values, 1.0, atol=1e-6)
    if not ok:
        st.error("Regime probabilities do not sum to 1.")
