# Cross-Asset Market Monitor (Streamlit)

Institutional-style market monitor using only free data:
- **FRED** (`fredapi`) for macro
- **Yahoo Finance** (`yfinance`) for ETF prices

## What the app includes
- 4-state regime engine (Goldilocks / Reflation / Slowdown / Stagflation)
- Signals dashboard across the exact ETF universe
- Profile-sensitive allocation (Conservative/Balanced/Growth)
- Robust fallback modes (no hard crash on missing data)
- Data Dictionary + Data Quality diagnostics + staleness transparency

## Exact ETF universe
- Equity regions: SPY, VGK, EWJ, IEMG, MCHI (MCHI optional in UI)
- US sectors: XLK, XLF, XLI, XLV, XLP, XLU, XLE, XLB
- Factors: QUAL, MTUM, USMV, VLUE, VUG
- Bonds/Credit: TLT, IEF, LQD, HYG
- Gold: GLD

## Click-by-click (no terminal required)
### 1) Create a public GitHub repo
1. Go to GitHub and click **New repository**.
2. Name it (example: `market-monitor`).
3. Set visibility to **Public**.
4. Click **Create repository**.

### 2) Upload files in web UI
1. In the repo, click **Add file** → **Upload files**.
2. Drag all files/folders from this project.
3. Add commit message and click **Commit changes**.

### 3) Deploy on Streamlit Community Cloud
1. Open [https://share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Select your repo + branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

### 4) Add secrets (TOML)
1. Streamlit app → **Settings** → **Secrets**.
2. Paste:
```toml
FRED_API_KEY="YOUR_KEY"
```
3. Save and redeploy if needed.

## Troubleshooting
- **KeyError missing columns**: fixed by canonical schema in curve panel (`2Y`, `10Y`, `Slope`).
- **LedoitWolf ValueError**: fixed by `sanitize_returns` and fallback covariance/anchor allocation.
- **Missing FRED series / stale macro**: app logs fallbacks in Data Dictionary and shows staleness warnings.
- **yfinance timeouts**: app degrades gracefully and reports missing tickers.
- **Regime unavailable**: app still renders and allocation falls back to anchor logic.
- **Missing content diagnosis**: open **Data Quality** tab to see a checklist of missing macro/ticker/engine inputs and download `missing_checklist.csv` to share exactly what is missing.

## Data disclaimer
For informational/internal research use only. Free sources can be delayed, revised, or incomplete.

## Academic references
- Hamilton (1989), *Econometrica*.
- Ang & Bekaert (2002), *Review of Financial Studies*.
- Ledoit & Wolf (2004), *Journal of Multivariate Analysis*.
- Rockafellar & Uryasev (2000), *Journal of Risk*.
- Jagannathan & Ma (2003), *Journal of Finance*.
- Stock & Watson (1999), *Journal of Monetary Economics*.
- DeMiguel, Garlappi, Uppal (2009), *Review of Financial Studies*.
- Black & Litterman (1992), *Financial Analysts Journal*.
