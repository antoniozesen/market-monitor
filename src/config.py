from __future__ import annotations

from dataclasses import dataclass

REGIME_NAMES = ["Goldilocks", "Reflation", "Slowdown", "Stagflation"]

# Broad candidate sets (candidate-first, then search fallback)
FRED_CANDIDATES = {
    "US": {
        "growth": ["INDPRO", "PAYEMS", "HOUST", "RRSFS", "IPMAN"],
        "inflation": ["CPILFESL", "CPIAUCSL", "PCEPILFE", "PCEPI", "MEDCPIM158SFRBCLE"],
        "labor": ["UNRATE", "ICSA", "UEMPLT5", "CIVPART", "AWHMAN"],
        "leading": ["ICSA", "UMCSENT", "PERMIT", "AMTMNO", "CFNAI"],
        "rates": ["DGS2", "DGS10", "DFII10", "FEDFUNDS", "T10Y2Y"],
        "conditions": ["NFCI", "ANFCI", "BAA10YM", "VIXCLS", "STLFSI4"],
        "stress": ["VIXCLS", "NFCI", "STLFSI4", "BAA10YM", "TEDRATE"],
    },
    "Europe": {
        "growth": ["CLVMNACSCAB1GQEA19", "OECDPRINTO01GYSAM", "EUROSTATIP", "NAEXKP01EZM661S"],
        "inflation": ["CP0000EZ19M086NEST", "CPHPTT01EZM661N", "CPALTT01EZM657N"],
        "labor": ["LRHUTTTTEZM156S", "LRUN64TTEZM156S", "LRUN25TTEZM156S"],
        "leading": ["OECDCLI", "LOLITOAAEZM156N", "BSCICP03EZM665S"],
        "rates": ["IR3TIB01EZM156N", "IRLTLT01EZM156N", "ECBDFR"],
    },
    "Japan": {
        "growth": ["JPNPROINDMISMEI", "JPNRGDPEXP", "JPNPRMNTO01IXOBSAM"],
        "inflation": ["JPNCPIALLMINMEI", "CPALTT01JPM657N", "CPGRLE01JPM659N"],
        "labor": ["LRUNTTTTJPM156S", "JPNURTOTQDSMEI", "LREM64TTJPM156S"],
        "leading": ["JPNLOLITONOSTSAM", "LOLITOAAJPM156N", "JPNCPHPLA01GYM"],
        "rates": ["IR3TIB01JPM156N", "IRLTLT01JPM156N", "INTDSRJPM193N"],
    },
    "EM": {
        "growth": ["OECDPRINTO01GYSAM", "RGDPNACNA666NRUG"],
        "inflation": ["FPCPITOTLZGEMU", "CPALTT01ZAM659N"],
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
