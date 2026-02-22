from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Settings
from src.data_fred import get_macro_library
from src.data_yf import get_prices
from src.diagnostics import data_quality_summary, missing_checklist, panel_stats, stale_summary
from src.features import build_regime_features, regime_expected_returns
from src.narrative import allocation_takeaways, banker_text, committee_text
from src.plots import curve_schema, fig_curve, fig_heatmap
from src.portfolio import benchmark_60_40, optimize_weights, stress_scenarios
from src.regime import run_regime_model
from src.risk import monthly_returns
from src.signals import build_signals
from src.universe import ALL_TICKERS, ANCHOR_PROFILES, ETF_NAMES
from src.utils import pct, safe_get_latest

st.set_page_config(page_title="Cross-Asset Market Monitor", layout="wide")
st.title("Cross-Asset Market Monitor")

cfg = Settings()

with st.sidebar:
    profile = st.selectbox("Profile", list(ANCHOR_PROFILES.keys()), index=1)
    flexibility = st.slider("Bucket flexibility ±", 0.0, 0.20, 0.10, 0.01)
    include_mchi = st.toggle("Include MCHI", True)
    client_safe = st.toggle("Client-safe mode", False)

# Data
prices, yf_diag = get_prices([t for t in ALL_TICKERS if include_mchi or t != "MCHI"])
macro, meta, fred_diag = get_macro_library()

if prices.empty:
    st.error("No ETF data available from yfinance. Please rerun later.")
    st.stop()

mret = monthly_returns(prices)
signals = build_signals(prices)
features, feat_diag = build_regime_features(macro)
reg = run_regime_model(features)
curve = curve_schema(macro)
curve_available = not curve[["2Y", "10Y"]].dropna(how="all").empty

if reg.get("available"):
    regime_name = reg["state"].iloc[-1]
    reg_probs_now = reg["probs"].iloc[-1]
else:
    regime_name = reg.get("proxy_regime", "Unavailable")
    reg_probs_now = reg.get("proxy_probs", pd.Series(dtype=float))

reg_mu = regime_expected_returns(mret, reg["probs"] if reg.get("available") else pd.DataFrame(), signals)
mu_now = reg_mu.get(regime_name, reg_mu.iloc[:, 0] if not reg_mu.empty else mret.mean()) if not reg_mu.empty else mret.mean()
stress_z = float(features["stress_z"].dropna().iloc[-1]) if "stress_z" in features and not features["stress_z"].dropna().empty else 0.0
alloc = optimize_weights(mret, mu_now, profile, flexibility, stress_z, regime_name)

missing_items = missing_checklist(meta, yf_diag.get("missing_tickers", []), reg.get("available", False), curve_available)
stale = stale_summary(meta, cfg.stale_days)

asof_etf = prices.index.max()
asof_macro = macro.index.max() if not macro.empty else pd.NaT
st.caption(f"As-of ETF: {asof_etf.date() if pd.notna(asof_etf) else 'n/a'} | As-of Macro: {asof_macro.date() if pd.notna(asof_macro) else 'n/a'}")

for w in yf_diag.get("warnings", []) + fred_diag.get("warnings", []):
    st.warning(w)
if stale["stale_count"] > 0:
    st.warning(f"Stale macro series >{cfg.stale_days}d: {stale['stale_count']}")


tabs = st.tabs(["Overview", "Regime", "Markets", "Macro Dashboard", "Signals", "Risk", "Allocation", "Narrative", "Data Dictionary", "Data Quality"])
over, reg_tab, mkt, macro_tab, sig_tab, risk_tab, alloc_tab, narrative_tab, dict_tab, quality_tab = tabs

with over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime_name)
    c2.metric("Mode", reg.get("mode", "unknown"))
    c3.metric("Effective sample (months)", f"{reg.get('effective_sample', feat_diag.get('effective_sample', 0))}")
    breadth = float(signals["above_200d"].mean()) if not signals.empty else np.nan
    c4.metric("Breadth >200d", pct(breadth))
    if not signals.empty:
        st.plotly_chart(fig_heatmap(signals.set_index("ticker")[["mom_6m", "mom_12m", "dist_3y_ma"]].T, "Overview heatmap"), use_container_width=True)
    fac = [t for t in ["QUAL", "MTUM", "VLUE", "USMV", "VUG", "SPY"] if t in prices.columns]
    if len(fac) >= 2:
        r12 = prices[fac].pct_change(252).iloc[-1].sort_values(ascending=False)
        st.dataframe(r12.rename("12m_return").to_frame().style.format("{:.2%}"))
    st.markdown("**Key takeaways**")
    st.write(f"- Regime/proxy: {regime_name} ({reg.get('mode','n/a')}).")
    st.write(f"- Effective regime sample: {reg.get('effective_sample', feat_diag.get('effective_sample',0))} months.")
    st.write(f"- Breadth above 200d: {pct(breadth)}.")
    st.write(f"- Missing checklist items: {int((missing_items['status']=='MISSING').sum()) if not missing_items.empty else 0}.")

with reg_tab:
    if reg.get("available"):
        st.plotly_chart(px.area(reg["probs"].tail(24), x=reg["probs"].tail(24).index, y=reg["probs"].columns, title="Regime probabilities"), use_container_width=True)
    else:
        st.info(f"GMM unavailable ({reg.get('reason','n/a')}); showing proxy regime.")
    st.dataframe(features.tail(24).round(2))
    st.markdown("**Key takeaways**")
    st.write(f"- Regime currently: {regime_name}.")
    st.write(f"- Model mode: {reg.get('mode','n/a')}.")
    st.write(f"- Effective sample (not raw index): {reg.get('effective_sample', feat_diag.get('effective_sample',0))}.")
    if not client_safe:
        st.json(reg.get("diagnostics", {}))

with mkt:
    if curve_available:
        st.plotly_chart(fig_curve(curve.tail(300)), use_container_width=True)
        slope = safe_get_latest(curve, "Slope")
        slope_txt = f"{slope:.2f}" if not np.isnan(slope) else "n/a"
    else:
        st.info("Curve unavailable (missing DGS2 or DGS10)")
        slope_txt = "n/a"
    real = safe_get_latest(macro.filter(regex="DFII10"), macro.filter(regex="DFII10").columns[0]) if not macro.filter(regex="DFII10").empty else np.nan
    hyg_lqd = (prices["HYG"] / prices["LQD"]).dropna() if {"HYG", "LQD"}.issubset(prices.columns) else pd.Series(dtype=float)
    if not hyg_lqd.empty:
        pctl = hyg_lqd.rank(pct=True).iloc[-1]
        st.line_chart(hyg_lqd.tail(756), height=220)
    else:
        pctl = np.nan
    st.markdown("**Key takeaways**")
    st.write(f"- Curve slope (10Y-2Y): {slope_txt}.")
    st.write(f"- Real yield DFII10: {real:.2f}." if not np.isnan(real) else "- Real yield unavailable.")
    st.write(f"- HYG/LQD percentile: {pct(pctl)}." if not np.isnan(pctl) else "- HYG/LQD unavailable.")

with macro_tab:
    if macro.empty:
        st.info("Macro library unavailable")
    else:
        for region in ["US", "Europe", "Japan", "EM"]:
            cols = [c for c in macro.columns if c.startswith(f"{region}_")]
            st.subheader(region)
            if cols:
                sub = macro[cols].resample("M").last().tail(24)
                st.line_chart(sub)
                st.caption(f"Freshness: {str(sub.dropna(how='all').index.max().date()) if not sub.dropna(how='all').empty else 'n/a'}")
            else:
                st.info(f"No series for {region}")
    st.markdown("**Key takeaways**")
    st.write(f"- Total macro series loaded: {macro.shape[1] if not macro.empty else 0}.")
    st.write(f"- Stale series count: {stale['stale_count']}.")
    st.write("- Region gaps are listed in Data Quality -> missing checklist.")

with sig_tab:
    if not signals.empty:
        out = signals.copy()
        out["name"] = out["ticker"].map(ETF_NAMES)
        st.dataframe(out[["ticker", "name", "valuation_bucket", "mom_6m", "mom_12m", "vol_3m", "vol_12m", "drawdown", "ma_slope"]].round(3))
        bucket_map = {
            "Regions": ["SPY", "VGK", "EWJ", "IEMG", "MCHI"],
            "Sectors": ["XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLE", "XLB"],
            "Factors": ["QUAL", "MTUM", "USMV", "VLUE", "VUG"],
            "Bonds": ["TLT", "IEF", "LQD", "HYG"],
            "Gold": ["GLD"],
        }
        hm = pd.DataFrame({k: out.set_index("ticker").reindex(v)["mom_6m"].mean() for k, v in bucket_map.items()}, index=["mom_6m"]).T
        st.plotly_chart(fig_heatmap(hm.T, "Bucket leadership heatmap"), use_container_width=True)
    else:
        st.info("Signals unavailable")
    st.markdown("**Key takeaways**")
    st.write(f"- Signals computed for {len(signals)} ETFs.")
    st.write(f"- Rich assets: {int((signals['valuation_bucket']=='Rich').sum()) if not signals.empty else 0}.")
    st.write(f"- Top momentum ticker: {signals.iloc[0]['ticker'] if not signals.empty else 'n/a'}.")

with risk_tab:
    w36 = mret.tail(36) if len(mret) >= 36 else mret
    if not w36.empty:
        st.plotly_chart(px.imshow(w36.corr(), title="Monthly correlation"), use_container_width=True)
        rv = w36.std() * np.sqrt(12)
        st.dataframe(rv.rename("ann_vol").to_frame().style.format("{:.2%}"))
    st.markdown("**Key takeaways**")
    st.write(f"- Correlation window: {len(w36)} months.")
    st.write(f"- Benchmark 60/40 available: {'Yes' if not benchmark_60_40(mret).empty else 'No'}.")
    st.write("- If regime unavailable, rolling risk diagnostics are used.")

with alloc_tab:
    w = alloc["weights"].sort_values(ascending=False)
    anchor = alloc["anchor"].reindex(w.index).fillna(0)
    at = pd.DataFrame({"ticker": w.index, "name": [ETF_NAMES.get(t, t) for t in w.index], "anchor": anchor.values, "recommended": w.values, "delta": (w-anchor).values})
    st.dataframe(at.style.format({"anchor": "{:.2%}", "recommended": "{:.2%}", "delta": "{:+.2%}"}))
    st.write(f"Mode: {alloc['reason']} | Covariance: {alloc['cov_mode']}")
    st.write(f"Turnover: {alloc['turnover']:.2f} | Annualized: {alloc['annual_turnover']:.2f} | Cost: {alloc['tx_cost']:.2%}")
    if alloc.get("binding_constraints"):
        st.write("Binding/solver notes:")
        for b in alloc["binding_constraints"]:
            st.write(f"- {b}")
    stf = stress_scenarios(w, reg_mu, alloc.get("cov", pd.DataFrame()))
    if not stf.empty:
        st.dataframe(stf.style.format({"Exp Return (m)": "{:.2%}", "Vol (m)": "{:.2%}"}))
    st.markdown("**Key takeaways**")
    for t in allocation_takeaways(w, alloc["reason"], alloc["hyg_cap_triggered"]):
        st.write(f"- {t}")

with narrative_tab:
    top_assets = signals.head(3)["ticker"].tolist() if not signals.empty else []
    weak_assets = signals.tail(3)["ticker"].tolist() if not signals.empty else []
    top_w = w.head(3).index.tolist() if 'w' in locals() and not w.empty else []
    ctext = committee_text(regime_name, stress_z, top_assets, weak_assets, top_w)
    btext = banker_text(regime_name, top_w)
    st.text_area("Comité", ctext, height=120)
    st.text_area("Banqueros", btext, height=120)
    st.download_button("Download narrative", f"## Comité\n{ctext}\n\n## Banqueros\n{btext}\n", "narrative.md")
    st.markdown("**Key takeaways**")
    st.write(f"- Narrative anchored to regime: {regime_name}.")
    st.write(f"- Top exposures: {', '.join(top_w) if top_w else 'n/a'}.")
    st.write("- Narrative is deterministic.")

with dict_tab:
    st.dataframe(meta)
    st.download_button("Download data dictionary CSV", meta.to_csv(index=False).encode(), "data_dictionary.csv", "text/csv")
    st.markdown("**Key takeaways**")
    st.write(f"- Dictionary entries: {len(meta)}.")
    st.write(f"- Search fallback used in {int((meta['source_mode']=='search_fallback').sum()) if not meta.empty else 0} rows.")
    st.write(f"- Entries marked excluded: {int((meta['excluded_reason'].astype(str).str.len()>0).sum()) if not meta.empty else 0}.")

with quality_tab:
    st.subheader("Missing checklist")
    st.dataframe(missing_items)
    miss = missing_items[missing_items["status"] == "MISSING"] if not missing_items.empty else pd.DataFrame()
    if not miss.empty:
        st.download_button("Download missing checklist CSV", miss.to_csv(index=False).encode(), "missing_checklist.csv", "text/csv")

    st.subheader("Quality summary")
    st.dataframe(data_quality_summary(meta))
    if not meta.empty and "staleness_days" in meta.columns:
        st.plotly_chart(px.histogram(meta, x="staleness_days", title="Staleness distribution"), use_container_width=True)

    st.json(
        {
            "prices": panel_stats(prices),
            "monthly_returns": panel_stats(mret),
            "macro": panel_stats(macro),
            "features": panel_stats(features),
            "effective_regime_sample": reg.get("effective_sample", feat_diag.get("effective_sample", 0)),
            "dropped_for_quality": fred_diag.get("dropped_for_quality", []),
            "optimization_diag": alloc.get("diag", {}),
        }
    )
    st.markdown("**Key takeaways**")
    st.write(f"- Missing items detected: {len(miss)}.")
    st.write(f"- Effective regime sample: {reg.get('effective_sample', feat_diag.get('effective_sample',0))} months.")
    st.write(f"- Stale series >{cfg.stale_days}d: {stale['stale_count']}.")

# quick self-checks
if not curve.empty:
    assert list(curve.columns) == ["2Y", "10Y", "Slope"]
if not mret.empty:
    _finite_ok = np.isfinite(mret.replace([np.inf, -np.inf], np.nan).fillna(0).values).all()
    if not _finite_ok:
        st.warning("Non-finite returns detected; fallback mode may be used.")
