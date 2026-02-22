from __future__ import annotations

ETF_UNIVERSE = {
    "equity_regions": ["SPY", "VGK", "EWJ", "IEMG", "MCHI"],
    "us_sectors": ["XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLE", "XLB"],
    "factors": ["QUAL", "MTUM", "USMV", "VLUE", "VUG"],
    "bonds_credit": ["TLT", "IEF", "LQD", "HYG"],
    "gold": ["GLD"],
}

ETF_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "VGK": "Vanguard FTSE Europe ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "IEMG": "iShares Core MSCI Emerging Markets ETF",
    "MCHI": "iShares MSCI China ETF",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "VLUE": "iShares MSCI USA Value Factor ETF",
    "VUG": "Vanguard Growth ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "GLD": "SPDR Gold Shares",
}

ALL_TICKERS = sum(ETF_UNIVERSE.values(), [])

ANCHOR_PROFILES = {
    "Conservative": {"Equity": 0.35, "Bonds": 0.55, "Gold": 0.10},
    "Balanced": {"Equity": 0.50, "Bonds": 0.40, "Gold": 0.10},
    "Growth": {"Equity": 0.65, "Bonds": 0.25, "Gold": 0.10},
}

BUCKET_MAP = {
    "Equity": ETF_UNIVERSE["equity_regions"] + ETF_UNIVERSE["us_sectors"] + ETF_UNIVERSE["factors"],
    "Bonds": ETF_UNIVERSE["bonds_credit"],
    "Gold": ETF_UNIVERSE["gold"],
}
