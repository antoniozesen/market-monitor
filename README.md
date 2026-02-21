# Cross-Asset Market Monitor (Streamlit)

Institutional-style **Cross-Asset Market Monitor** using only free data:
- **FRED** macro series (via `fredapi`)
- **ETF prices** from Yahoo Finance (via `yfinance`)

It provides:
- A 4-state regime engine (Goldilocks / Reflation / Slowdown / Stagflation)
- Regime-conditioned expectations and diagnostics
- Cross-asset signals for the required ETF universe
- Profile-sensitive allocation (Conservative / Balanced / Growth)
- Stress test for regime flips
- Data governance: transparency on sources, transformations, freshness, and fallbacks

## Required ETF universe (exact)
- Equity regions: SPY, VGK, EWJ, IEMG, MCHI (optional toggle in UI)
- US sectors: XLK, XLF, XLI, XLV, XLP, XLU, XLE, XLB
- Factors: QUAL, MTUM, USMV, VLUE, VUG
- Bonds/Credit: TLT, IEF, LQD, HYG
- Gold: GLD

## Click-by-click setup (no terminal required)

### 1) Create a public GitHub repository
1. Open GitHub in your browser and sign in.
2. Click **New repository**.
3. Repository name: `market-monitor` (or your preferred name).
4. Set visibility to **Public**.
5. Click **Create repository**.

### 2) Upload files via GitHub Web UI
1. In your new repository page, click **Add file** → **Upload files**.
2. Drag and drop all files/folders from this project.
3. Scroll down, add a commit message like `Initial app upload`.
4. Click **Commit changes**.

### 3) Deploy on Streamlit Community Cloud
1. Go to [https://share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Choose your GitHub repo and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

### 4) Add secrets (FRED key)
1. In Streamlit Cloud, open your app.
2. Click **Settings** → **Secrets**.
3. Paste:
   ```toml
   FRED_API_KEY="YOUR_KEY"
   ```
4. Save.

### 5) Troubleshooting
- If Streamlit says the app file is missing, set **Main file path** to `app.py` (or use `streamlit_app.py` compatibility entrypoint).
- If macro panels are empty: confirm `FRED_API_KEY` is set exactly in Secrets.
- If some ETFs are missing: Yahoo may have intermittent outages; retry/redeploy.
- If history is short for an ETF: app degrades gracefully and flags warnings.

## Design notes / safeguards
- Avoids “everything is expensive” by mixing valuation, drawdown percentile, and trend-adjusted context.
- Avoids regime degeneracy by smoothing/posterior floor and stability diagnostics.
- Avoids unintended HY overweight with explicit risk-off cap on HYG.
- Weights are profile-sensitive via bucket anchor + flexibility.
- Shows ticker + ETF name using internal mapping.
- Logs fallbacks and freshness in Data Dictionary.
- Every panel ends with deterministic key takeaways.

## Data & model disclaimer
This app is for **informational/internal research use only**. Free data may be revised, delayed, missing, or inconsistent. Not investment advice.

## Academic references
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica.
- Ang, A. & Bekaert, G. (2002). *International Asset Allocation with Regime Shifts*. Review of Financial Studies.
- Ledoit, O. & Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis.
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
- Jagannathan, R. & Ma, T. (2003). *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*. Journal of Finance.
- Stock, J.H. & Watson, M.W. (1999). *Forecasting Inflation*. Journal of Monetary Economics.
- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification*. Review of Financial Studies.
