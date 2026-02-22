from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Settings
from src.data_fred import get_macro_library
from src.data_yf import get_prices
from src.diagnostics import panel_stats, stale_summary
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
    advanced = st.toggle("Advanced diagnostics", False)

tickers = [t for t in ALL_TICKERS if include_mchi or t != "MCHI"]
prices, yf_diag = get_prices(tickers)
macro, meta, fred_diag = get_macro_library()

if prices.empty:
    st.error("No ETF data available from yfinance. Please rerun later.")
    st.stop()

mret = monthly_returns(prices)
signals = build_signals(prices)
features, feat_diag = build_regime_features(macro)
reg = run_regime_model(features)
regime_name = reg["state"].iloc[-1] if reg.get("available") else "Unavailable"
reg_probs_now = reg["probs"].iloc[-1] if reg.get("available") else pd.Series(dtype=float)
reg_mu = regime_expected_returns(mret, reg["probs"] if reg.get("available") else pd.DataFrame())
mu_now = reg_mu.get(regime_name, mret.mean()) if not reg_mu.empty else mret.mean()
stress_z = float(features["stress_z"].dropna().iloc[-1]) if "stress_z" in features and not features["stress_z"].dropna().empty else 0.0
alloc = optimize_weights(mret, mu_now, profile, flexibility, stress_z, regime_name)

curve = curve_schema(macro)

asof_etf = prices.index.max()
asof_macro = macro.index.max() if not macro.empty else pd.NaT
st.caption(f"As-of ETF: {asof_etf.date() if pd.notna(asof_etf) else 'n/a'} | As-of Macro: {asof_macro.date() if pd.notna(asof_macro) else 'n/a'}")

for w in yf_diag.get("warnings", []) + fred_diag.get("warnings", []):
    st.warning(w)
if yf_diag.get("missing_tickers"):
    st.warning("Missing tickers: " + ", ".join(yf_diag["missing_tickers"]))
ss = stale_summary(meta, cfg.stale_days)
if ss["stale_count"] > 0:
    st.warning(f"Stale macro series >{cfg.stale_days}d: {ss['stale_count']}")


over, reg_tab, mkt, macro_tab, sig_tab, risk_tab, alloc_tab, narrative_tab, dict_tab, quality_tab = st.tabs(
    ["Overview", "Regime", "Markets", "Macro Dashboard", "Signals", "Risk", "Allocation", "Narrative", "Data Dictionary", "Data Quality"]
)

with over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime_name)
    c2.metric("Stress z", f"{stress_z:.2f}")
    breadth = float(signals["above_200d"].mean()) if not signals.empty else np.nan
    c3.metric("Breadth >200d", pct(breadth))
    c4.metric("SPY 12m", pct(prices["SPY"].pct_change(252).iloc[-1]) if "SPY" in prices else "n/a")
    if not signals.empty:
        st.plotly_chart(fig_heatmap(signals.set_index("ticker")[["mom_6m", "mom_12m", "dist_3y_ma"]].T, "Overview heatmap"), use_container_width=True)
        top = signals.nlargest(3, "ret_1m")["ticker"].tolist()
        bot = signals.nsmallest(3, "ret_1m")["ticker"].tolist()
    else:
        top, bot = [], []
    st.markdown("**Key takeaways**")
    st.write(f"- Regime is {regime_name}.")
    st.write(f"- Market breadth is {pct(breadth)} above 200-day average.")
    st.write(f"- Top 1m performers: {', '.join(top) if top else 'n/a'}.")
    st.write(f"- Bottom 1m performers: {', '.join(bot) if bot else 'n/a'}.")

with reg_tab:
    if reg.get("available"):
        st.plotly_chart(px.area(reg["probs"].tail(12), x=reg["probs"].tail(12).index, y=reg["probs"].columns, title="Regime probabilities (1Y)"), use_container_width=True)
        timeline = reg["state"].tail(180).to_frame("regime")
        timeline["code"] = timeline["regime"].map({k: i for i, k in enumerate(reg["probs"].columns)})
        st.plotly_chart(px.scatter(timeline, x=timeline.index, y="code", color="regime", title="Regime timeline"), use_container_width=True)
    else:
        st.info(f"Regime unavailable: {reg.get('reason','unknown')}")
    st.markdown("**Key takeaways**")
    st.write(f"- Current regime: {regime_name}.")
    st.write(f"- Regime sample size: {feat_diag.get('sample_rows',0)} months.")
    st.write(f"- Missingness in feature matrix: {feat_diag.get('missing_pct',100):.1f}%.")
    if not client_safe:
        with st.expander("Diagnostics"):
            st.json(reg.get("diagnostics", {}))
            st.json(feat_diag)

with mkt:
    if curve.empty or curve[["2Y", "10Y"]].dropna(how="all").empty:
        st.info("Curve unavailable (missing DGS2/DGS10).")
    else:
        st.plotly_chart(fig_curve(curve.tail(300)), use_container_width=True)
    real_yield = safe_get_latest(macro.filter(regex="DFII10"), macro.filter(regex="DFII10").columns[0]) if not macro.filter(regex="DFII10").empty else np.nan
    hyg_lqd = (prices["HYG"] / prices["LQD"]).dropna() if {"HYG", "LQD"}.issubset(prices.columns) else pd.Series(dtype=float)
    if not hyg_lqd.empty:
        st.line_chart(hyg_lqd.tail(756), height=220)
    st.markdown("**Key takeaways**")
    st.write(f"- Latest slope (10Y-2Y): {safe_get_latest(curve, 'Slope'):.2f}.")
    st.write(f"- Real yield (DFII10 latest): {real_yield:.2f}.")
    st.write(f"- HYG/LQD available: {'Yes' if not hyg_lqd.empty else 'No'}.")

with macro_tab:
    if macro.empty:
        st.info("Macro library unavailable.")
    else:
        z = features.tail(24).round(2)
        st.dataframe(z)
        avail = [c for c in macro.columns if any(r in c for r in ["US_", "Europe_", "Japan_", "EM_"])]
        st.write(f"Available macro series: {len(avail)}")
    st.markdown("**Key takeaways**")
    st.write(f"- US/Europe/Japan/EM coverage rows: {len(macro.columns) if not macro.empty else 0}.")
    st.write(f"- Macro stale count: {ss['stale_count']}.")
    st.write("- Missing regional series are disclosed in Data Dictionary.")

with sig_tab:
    if signals.empty:
        st.info("Signals unavailable")
    else:
        out = signals.copy()
        out["name"] = out["ticker"].map(ETF_NAMES)
        st.dataframe(out[["ticker", "name", "valuation_bucket", "mom_6m", "mom_12m", "vol_3m", "vol_12m", "drawdown", "above_200d"]].round(3))
        st.plotly_chart(fig_heatmap(out.set_index("ticker")[["mom_6m", "mom_12m", "drawdown"]].T, "Signal heatmap"), use_container_width=True)
    st.markdown("**Key takeaways**")
    st.write(f"- Signals computed for {len(signals)} ETFs.")
    st.write(f"- Rich bucket count: {int((signals['valuation_bucket']=='Rich').sum()) if not signals.empty else 0}.")
    st.write(f"- Highest 12m momentum: {signals.sort_values('mom_12m', ascending=False).iloc[0]['ticker'] if not signals.empty else 'n/a'}.")

with risk_tab:
    window = mret.tail(36) if len(mret) >= 36 else mret
    if not window.empty:
        st.plotly_chart(px.imshow(window.corr(), title="Correlation heatmap (monthly)"), use_container_width=True)
    if "SPY" in mret.columns:
        beta = mret.cov().get("SPY", pd.Series(dtype=float)) / mret["SPY"].var() if mret["SPY"].var() != 0 else pd.Series(dtype=float)
        st.dataframe(beta.rename("beta_to_SPY").to_frame().round(2))
    st.markdown("**Key takeaways**")
    st.write(f"- Correlation window used: {len(window)} months.")
    st.write(f"- Benchmark available: {'Yes' if not benchmark_60_40(mret).empty else 'No'}.")
    st.write("- Risk views degrade gracefully with limited sample.")

with alloc_tab:
    w = alloc["weights"].sort_values(ascending=False)
    out = pd.DataFrame({"ticker": w.index, "name": [ETF_NAMES.get(x, x) for x in w.index], "weight": w.values})
    st.dataframe(out.style.format({"weight": "{:.2%}"}))
    st.bar_chart(out.set_index("ticker")["weight"])
    st.write(f"Mode: {alloc['reason']} | Covariance: {alloc['cov_mode']}")
    st.write(f"Turnover: {alloc['turnover']:.2f} | Annualized: {alloc['annual_turnover']:.2f} | Tx cost: {alloc['tx_cost']:.2%}")
    st.write(f"HYG defensive cap triggered: {'Yes' if alloc['hyg_cap_triggered'] else 'No'}")
    stf = stress_scenarios(w, reg_mu, alloc.get("cov", pd.DataFrame()))
    if not stf.empty:
        st.dataframe(stf.style.format({"Exp Return (m)": "{:.2%}", "Vol (m)": "{:.2%}"}))
        st.plotly_chart(px.bar(stf, x="Scenario", y="Exp Return (m)", title="What-if regime flip"), use_container_width=True)
    st.markdown("**Key takeaways**")
    for t in allocation_takeaways(w, alloc["reason"], alloc["hyg_cap_triggered"]):
        st.write(f"- {t}")

with narrative_tab:
    top_assets = signals.head(3)["ticker"].tolist() if not signals.empty else []
    weak_assets = signals.tail(3)["ticker"].tolist() if not signals.empty else []
    top_w = w.head(3).index.tolist() if not w.empty else []
    committee = committee_text(regime_name, stress_z, top_assets, weak_assets, top_w)
    banker = banker_text(regime_name, top_w)
    st.subheader("Comité")
    st.text_area(" ", committee, height=120)
    st.subheader("Banqueros")
    st.text_area("  ", banker, height=120)
    md = f"## Comité\n{committee}\n\n## Banqueros\n{banker}\n"
    st.download_button("Download narrative (markdown)", md.encode(), "narrative.md", "text/markdown")
    st.markdown("**Key takeaways**")
    st.write(f"- Narrative regime anchor: {regime_name}.")
    st.write(f"- Top overweight names: {', '.join(top_w) if top_w else 'n/a'}.")
    st.write("- Narrative is deterministic and copy/paste friendly.")

with dict_tab:
    st.dataframe(meta)
    st.download_button("Download data dictionary CSV", meta.to_csv(index=False).encode(), "data_dictionary.csv", "text/csv")
    etf_cov = pd.DataFrame({"ticker": prices.columns, "start": [str(prices[c].dropna().index.min().date()) for c in prices.columns], "end": [str(prices[c].dropna().index.max().date()) for c in prices.columns]})
    st.dataframe(etf_cov)
    st.markdown("**Key takeaways**")
    st.write(f"- Macro records listed: {len(meta)}.")
    st.write(f"- Stale macro series: {ss['stale_count']}.")
    st.write(f"- ETF coverage table rows: {len(etf_cov)}.")

with quality_tab:
    st.json({
        "prices": panel_stats(prices),
        "monthly_returns": panel_stats(mret),
        "macro": panel_stats(macro),
        "features": panel_stats(features),
        "optimization_diag": alloc.get("diag", {}),
    })
    st.markdown("**Key takeaways**")
    st.write(f"- Returns rows for risk: {mret.shape[0]}.")
    st.write(f"- Assets in optimization input: {alloc.get('diag',{}).get('assets',0)}.")
    st.write(f"- Dropped assets (<{cfg.min_obs_per_asset} months): {len(alloc.get('diag',{}).get('dropped_assets',[]))}.")

if advanced and not client_safe:
    with st.expander("Advanced diagnostics"):
        st.json({"yfinance": yf_diag, "fred": fred_diag, "feature_diag": feat_diag, "alloc_diag": alloc.get("diag", {})})
