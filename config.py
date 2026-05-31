"""
config.py — Configuration centrale du pipeline GSR
"""
from pathlib import Path
import os

# ── Chemins ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_FIG = BASE_DIR / "outputs" / "figures"
OUTPUT_TAB = BASE_DIR / "outputs" / "tables"
OUTPUT_MOD = BASE_DIR / "outputs" / "models"

for d in [DATA_RAW, DATA_PROCESSED, OUTPUT_FIG, OUTPUT_TAB, OUTPUT_MOD]:
    d.mkdir(parents=True, exist_ok=True)

# ── Période d'étude ──────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE = "2026-05-07"

# ── Mapping fichiers FactSet → nom interne ───────────────────────
# Chaque entrée : (nom_fichier, colonne_prix, nom_interne)
DAILY_FILES = {
    "gold":     ("GC1_Gold_Futures.xlsx",                    "Settlement Price", "gold"),
    "silver":   ("SI1_Silver_Futures.xlsx",                  "Settlement Price", "silver"),
    "gld":      ("GLD-US_SPDR_Gold_Shares_ETF.xlsx",         "Last",            "gld"),
    "slv":      ("SLV-US_iShares_Silver_Trust_ETF.xlsx",     "Last",            "slv"),
    "xau_spot": ("XAUUSD_Gold_Spot.xlsx",                    "Last",            "xau_spot"),
    "xag_spot": ("XAGUSD_Silver_Spot.xlsx",                  "Last",            "xag_spot"),
    "us10y":    ("US10YT_US_Treasury_10Y_Yield.xlsx",        "Yield",           "us10y"),
    "tips10y":  ("TIPS10Y_US_TIPS_10Y_Real_Yield.xlsx",      "Yield",           "tips10y"),
    "be10y":    ("BE10Y_US_10Y_Breakeven_Inflation.xlsx",    "Yield",           "be10y"),
    "fed_funds":("Fed_Funds_Fed_Funds_Effective_Rate.xlsx",  "Yield",           "fed_funds"),
    "dxy":      ("DXY-US_US_Dollar_Index.xlsx",              "Price",           "dxy"),
    "vix":      ("VIX-US_CBOE_VIX_Index.xlsx",               "Price",           "vix"),
    "spx":      ("SPX_S&P_500_Index.xlsx",                   "Price",           "spx"),
    "copper":   ("HG1_Copper_Futures (front-month).xlsx",    "Settlement Price", "copper"),
    "oil":      ("CL1_WTI_Crude_Oil_Futures.xlsx",           "Settlement Price", "oil"),
    "mxwo":     ("MXWO_MSCI_World_Index.xlsx",               "Price",           "mxwo"),
}

# Fichiers mensuels (format horizontal FactSet)
MONTHLY_FILES = {
    "m2":  ("M2_US_US_M2_Money_Supply.xlsx",  "M2",       "m2"),
    "cpi": ("CPI_US_US_CPI_YoY.xlsx",         "All items", "cpi"),
}

# Fichiers Bloomberg (à ajouter quand disponibles)
BBG_FILES = {
    "ig_oas":   None,  # LF98OAS
    "hy_oas":   None,  # HY OAS
    "cftc_gold": None, # CFTC gold net spec
    "cftc_silver": None,
    "gld_tons": None,
    "slv_tons": None,
    "gvz":      None,
    "move":     None,
}

# ── Walk-Forward ─────────────────────────────────────────────────
TRAIN_YEARS = 5
TEST_MONTHS = 6
MIN_TRAIN_OBS = 1000

# ── Triple-Barrier ───────────────────────────────────────────────
TB_HORIZON = 21          # jours (1 mois trading)
TB_PT_MULT = 1.5         # take-profit = 1.5 × ATR
TB_SL_MULT = 1.0         # stop-loss = 1.0 × ATR
TB_ATR_PERIOD = 14

# ── Modèles ──────────────────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS = -1

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

# ── Backtest ─────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 10  # 10 bps aller-retour
SLIPPAGE_BPS = 5
INITIAL_CAPITAL = 1_000_000
