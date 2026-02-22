# Cross-Asset Market Monitor (Streamlit)

Institutional market monitor using only free sources:
- FRED (`fredapi`) for macro
- Yahoo Finance (`yfinance`) for ETFs

## Click-by-click deployment (no terminal)
1. Create a **public** GitHub repository.
2. Upload all files via **Add file → Upload files**.
3. Open [Streamlit Community Cloud](https://share.streamlit.io), create new app.
4. Set **Main file path** to `app.py`.
5. Add secrets in **Settings → Secrets**:
```toml
FRED_API_KEY="YOUR_KEY"
```
6. Deploy.

## Fixed ETF universe (exact)
- Equity regions: SPY, VGK, EWJ, IEMG, MCHI (optional toggle)
- US sectors: XLK, XLF, XLI, XLV, XLP, XLU, XLE, XLB
- Factors: QUAL, MTUM, USMV, VLUE, VUG
- Bonds/Credit: TLT, IEF, LQD, HYG
- Gold: GLD

## Transformation rules (now corrected)
- **Level series** (growth/inflation: INDPRO, CPI/PCE): YoY then z-score.
- **Rate series** (UNRATE, DGS2, DGS10, policy rates): level z-score + change features (not YoY).
- **Spread/conditions/stress** (BAA10YM, NFCI, VIX): level z-score.
- **Leading/sentiment** (UMCSENT/CLI): level z-score + short-horizon change.

## Troubleshooting
- KeyError curve columns: fixed with canonical curve schema [`2Y`,`10Y`,`Slope`].
- LedoitWolf ValueError: fixed with `sanitize_returns` thresholds + covariance fallback + anchor fallback.
- Regime "Unavailable" inconsistency: app now uses **effective sample after cleaning/alignment**, not raw index length.
- Missing Europe/Japan leading data: candidate+search picker with stronger staleness penalties.
- Stale macro selections: picker penalizes staleness >120d and heavily >365d.
- Missing content diagnosis: check **Data Quality** tab and export `missing_checklist.csv`.

## CHANGELOG (important fixes)
- Fixed internal inconsistency between raw sample and effective regime sample.
- Enforced robust curve engine with canonical schema and safe, non-NaN takeaways.
- Hardened optimizer pipeline with deterministic fallback modes.
- Expanded series picker scoring and metadata transparency (`selection_why`, freshness, fallback mode).
- Added Data Quality summaries: staleness distribution, usable concept coverage, missing checklist.

## Disclaimer
For informational/internal research use only. Free data can be delayed, revised, or incomplete.

## Academic references
Hamilton (1989), Ang & Bekaert (2002), Ledoit & Wolf (2004), Rockafellar & Uryasev (2000), Jagannathan & Ma (2003), Stock & Watson (1999), DeMiguel et al. (2009), Black & Litterman (1992).
