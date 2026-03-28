"""
config.py
=========
Central configuration for the F1 GRU forecasting project.
All team names, year ranges, hyperparameters, and paths are defined here.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [RAW_DIR, PROC_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# FastF1 cache (avoids re-downloading)
CACHE_DIR = os.path.join(BASE_DIR, "data", "fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# YEARS
# ─────────────────────────────────────────────────────────────────────────────
ALL_YEARS        = list(range(2014, 2026))   # 2014–2025 (inclusive)
FORECAST_YEAR    = 2026

# Walk-forward validation splits
SPLITS = [
    {"train": list(range(2014, 2017)), "test": [2017], "label": "Split_1"},
    {"train": list(range(2014, 2022)), "test": [2022], "label": "Split_2"},
]

# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME NORMALISATION
# FastF1 returns slightly different names across years — map all to canonical.
# Cadillac is excluded (new team in 2026 only).
# ─────────────────────────────────────────────────────────────────────────────
TEAM_ALIASES: dict[str, str] = {
    # Mercedes
    "Mercedes":                     "Mercedes",
    "Mercedes-AMG Petronas F1 Team":"Mercedes",
    "Mercedes AMG Petronas F1 Team":"Mercedes",

    # Red Bull
    "Red Bull Racing":              "Red Bull",
    "Red Bull":                     "Red Bull",
    "Oracle Red Bull Racing":       "Red Bull",

    # Ferrari
    "Ferrari":                      "Ferrari",
    "Scuderia Ferrari":             "Ferrari",

    # McLaren
    "McLaren":                      "McLaren",
    "McLaren F1 Team":              "McLaren",
    "McLaren Mercedes":             "McLaren",

    # Alpine / Renault
    "Alpine":                       "Alpine",
    "Alpine F1 Team":               "Alpine",
    "Renault":                      "Alpine",
    "Renault F1 Team":              "Alpine",

    # AlphaTauri / RB / Toro Rosso / Visa RB
    "AlphaTauri":                   "RB",
    "Scuderia AlphaTauri":          "RB",
    "Toro Rosso":                   "RB",
    "Scuderia Toro Rosso":          "RB",
    "RB":                           "RB",
    "Visa Cash App RB":             "RB",
    "Visa Cash App RB F1 Team":     "RB",

    # Aston Martin / Racing Point / Force India
    "Aston Martin":                 "Aston Martin",
    "Aston Martin F1 Team":         "Aston Martin",
    "Racing Point":                 "Aston Martin",
    "Racing Point F1 Team":         "Aston Martin",
    "Force India":                  "Aston Martin",
    "Sahara Force India F1 Team":   "Aston Martin",
    "BWT Racing Point F1 Team":     "Aston Martin",

    # Williams
    "Williams":                     "Williams",
    "Williams Racing":              "Williams",

    # Haas
    "Haas F1 Team":                 "Haas",
    "Haas":                         "Haas",
    "MoneyGram Haas F1 Team":       "Haas",

    # Sauber / Alfa Romeo / Kick Sauber
    "Alfa Romeo":                   "Sauber",
    "Alfa Romeo F1 Team ORLEN":     "Sauber",
    "Alfa Romeo Racing":            "Sauber",
    "Sauber":                       "Sauber",
    "Stake F1 Team Kick Sauber":    "Sauber",
    "Kick Sauber":                  "Sauber",
}

# The 10 teams that will race in 2026 (canonical names, no Cadillac)
CANONICAL_TEAMS = [
    "Mercedes",
    "Red Bull",
    "Ferrari",
    "McLaren",
    "Alpine",
    "RB",
    "Aston Martin",
    "Williams",
    "Haas",
    "Sauber",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Minimum laps a driver must have in a session to be included
MIN_LAPS = 5
# Lap-time outlier filter: keep laps within this factor of median lap time
OUTLIER_FACTOR = 1.07   # drop laps > 7% slower than median (pit laps, SC etc.)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (built during preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
# The model receives these features per (team, season) timestep
FEATURE_COLS = [
    "mean_delta",          # target: mean gap to fastest team (seconds)
    "median_delta",        # robustness feature
    "delta_std",           # consistency / variance
    "reg_era",             # ordinal era index (0=2014–16, 1=2017–21, 2=2022–25)
    "seasons_in_era",      # how many seasons into this era
    "prior_year_delta",    # lag-1 feature
    "rolling3_delta",      # 3-year rolling mean gap
]

TARGET_COL = "mean_delta"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN        = 3        # look-back window (seasons)
INPUT_SIZE     = len(FEATURE_COLS)
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
DROPOUT        = 0.3
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS         = 300
BATCH_SIZE     = 16       # (team, window) samples per batch
PATIENCE       = 40       # early stopping patience

# ─────────────────────────────────────────────────────────────────────────────
# REGULATION ERAS (for feature encoding)
# ─────────────────────────────────────────────────────────────────────────────
ERA_MAP = {
    2014: 0, 2015: 0, 2016: 0,
    2017: 1, 2018: 1, 2019: 1, 2020: 1, 2021: 1,
    2022: 2, 2023: 2, 2024: 2, 2025: 2,
    2026: 3,   # new era — model must generalise to this
}

ERA_START = {0: 2014, 1: 2017, 2: 2022, 3: 2026}

RANDOM_SEED = 42
