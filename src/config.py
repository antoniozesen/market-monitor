from __future__ import annotations

from dataclasses import dataclass

REGIME_NAMES = ["Goldilocks", "Reflation", "Slowdown", "Stagflation"]

FRED_CANDIDATES = {
    "US": {
        "growth": ["INDPRO", "PAYEMS", "HOUST"],
        "inflation": ["CPILFESL", "CPIAUCSL", "PCEPILFE", "PCEPI"],
        "labor": ["UNRATE", "ICSA"],
        "sentiment": ["UMCSENT"],
        "conditions": ["NFCI", "BAA10YM", "VIXCLS"],
        "rates": ["DGS2", "DGS10", "DFII10"],
    },
    "Europe": {
        "growth": ["CLVMNACSCAB1GQEA19", "OECDPRINTO01GYSAM"],
        "inflation": ["CP0000EZ19M086NEST"],
        "labor": ["LRHUTTTTEZM156S"],
        "leading": ["OECDCLI"],
        "rates": ["IR3TIB01EZM156N"],
    },
    "Japan": {
        "growth": ["JPNPROINDMISMEI"],
        "inflation": ["JPNCPIALLMINMEI"],
        "labor": ["LRUNTTTTJPM156S"],
        "leading": ["JPNLOLITONOSTSAM"],
        "rates": ["IR3TIB01JPM156N"],
    },
    "EM": {
        "growth": ["OECDPRINTO01GYSAM"],
        "inflation": ["FPCPITOTLZGEMU"],
    },
}

FRED_SEARCH = {
    "Europe": {
        "growth": ["euro area industrial production", "euro area oecd cli"],
        "inflation": ["euro area cpi core", "euro area inflation"],
        "labor": ["euro area unemployment rate"],
    },
    "Japan": {
        "growth": ["japan industrial production", "japan oecd cli"],
        "inflation": ["japan cpi"],
        "labor": ["japan unemployment rate"],
    },
    "EM": {
        "growth": ["emerging markets industrial production"],
        "inflation": ["emerging markets inflation"],
    },
}

@dataclass
class Settings:
    stale_days: int = 60
    min_obs_per_asset: int = 24
    min_rows_for_cov: int = 36
    min_assets_for_opt: int = 3
    prob_floor: float = 0.02
