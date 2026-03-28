"""
02_preprocessing.py
===================
Reads raw per-season CSVs, computes per-team-per-season aggregated pace metrics,
engineers temporal features, scales them, and builds fixed-length sequence
windows for the GRU.

Outputs
-------
data/processed/team_season_features.csv   — unscaled feature table
data/processed/sequences.npz             — X (N, SEQ_LEN, F), y (N,), metadata
data/processed/scaler_params.npz         — mean/std for inverse-transform
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    ALL_YEARS, RAW_DIR, PROC_DIR,
    CANONICAL_TEAMS, FEATURE_COLS, TARGET_COL,
    ERA_MAP, ERA_START, SEQ_LEN, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load raw CSVs and compute per-team-per-season pace delta
# ─────────────────────────────────────────────────────────────────────────────

def compute_season_deltas(year: int) -> pd.DataFrame:
    """
    For each race in the season:
      1. Compute median lap time per team.
      2. Subtract the fastest team's median → delta (0 = best team).
    
    Then aggregate across all races for the season → per-team stats.
    Returns DataFrame: team, year, mean_delta, median_delta, delta_std
    """
    path = os.path.join(RAW_DIR, f"season_{year}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found — skipping {year}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    race_team_medians = (
        df.groupby(["race_round", "team"])["lap_time_s"]
          .median()
          .reset_index()
          .rename(columns={"lap_time_s": "median_lap"})
    )

    # Per-race: compute delta vs fastest team that race
    def add_delta(grp):
        grp = grp.copy()
        grp["delta"] = grp["median_lap"] - grp["median_lap"].min()
        return grp

    race_team_medians = (
        race_team_medians.groupby("race_round", group_keys=False)
                         .apply(add_delta)
    )

    # Aggregate to season level
    season_stats = (
        race_team_medians.groupby("team")["delta"]
                         .agg(mean_delta="mean", median_delta="median", delta_std="std")
                         .reset_index()
    )
    season_stats["year"] = year

    # Keep only canonical teams
    season_stats = season_stats[season_stats["team"].isin(CANONICAL_TEAMS)]

    return season_stats


def build_feature_table() -> pd.DataFrame:
    """
    Build the full (team × year) feature table across ALL_YEARS.
    """
    print("Building season-level pace delta table …")
    parts = []
    for year in ALL_YEARS:
        df = compute_season_deltas(year)
        if not df.empty:
            parts.append(df)
            print(f"  {year}: {len(df)} teams")

    if not parts:
        raise RuntimeError("No raw data found — run 01_data_collection.py first.")

    master = pd.concat(parts, ignore_index=True)
    master = master.sort_values(["team", "year"]).reset_index(drop=True)

    # ── Regulation era features ──────────────────────────────────────────────
    master["reg_era"] = master["year"].map(ERA_MAP)
    master["era_start_year"] = master["reg_era"].map(ERA_START)
    master["seasons_in_era"] = master["year"] - master["era_start_year"]

    # ── Lag features (per team) ──────────────────────────────────────────────
    master = master.sort_values(["team", "year"])
    master["prior_year_delta"] = master.groupby("team")["mean_delta"].shift(1)
    master["rolling3_delta"] = (
        master.groupby("team")["mean_delta"]
              .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Fill NaN lags with team mean (for first years)
    for col in ["prior_year_delta", "rolling3_delta"]:
        master[col] = master.groupby("team")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    # Drop rows still missing any feature (shouldn't happen after fillna above)
    master = master.dropna(subset=FEATURE_COLS)

    return master


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build fixed-length sequences per team
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(master: pd.DataFrame):
    """
    For each team, slide a window of length SEQ_LEN over the year axis.
    Each sample: X = features over [t-SEQ_LEN, t-1], y = mean_delta at t.

    Returns:
        X         np.ndarray  (N, SEQ_LEN, num_features)
        y         np.ndarray  (N,)
        meta      list of dicts  — {team, year} for each sample
        scaler    fitted StandardScaler
    """
    # Fit scaler on all feature columns (using full dataset — we re-fit per split
    # in training, but we save a global scaler here for reference)
    scaler = StandardScaler()
    master_scaled = master.copy()
    master_scaled[FEATURE_COLS] = scaler.fit_transform(master[FEATURE_COLS])

    X_list, y_list, meta_list = [], [], []

    for team in CANONICAL_TEAMS:
        team_df = master_scaled[master_scaled["team"] == team].sort_values("year")

        years  = team_df["year"].tolist()
        X_team = team_df[FEATURE_COLS].values
        y_team = team_df[TARGET_COL].values   # scaled target

        for i in range(SEQ_LEN, len(years)):
            X_window = X_team[i - SEQ_LEN : i]   # (SEQ_LEN, F)
            y_val    = y_team[i]
            target_year = years[i]

            X_list.append(X_window)
            y_list.append(y_val)
            meta_list.append({"team": team, "year": target_year})

    X = np.array(X_list, dtype=np.float32)   # (N, SEQ_LEN, F)
    y = np.array(y_list, dtype=np.float32)   # (N,)

    return X, y, meta_list, scaler, master_scaled


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Persist
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(master_raw, X, y, meta_list, scaler):
    # Unscaled feature table
    feat_path = os.path.join(PROC_DIR, "team_season_features.csv")
    master_raw.to_csv(feat_path, index=False)
    print(f"\nSaved feature table → {feat_path}")

    # Sequences
    meta_teams = np.array([m["team"] for m in meta_list])
    meta_years = np.array([m["year"] for m in meta_list])
    seq_path = os.path.join(PROC_DIR, "sequences.npz")
    np.savez(seq_path, X=X, y=y, teams=meta_teams, years=meta_years)
    print(f"Saved sequences    → {seq_path}  (shape X={X.shape}, y={y.shape})")

    # Scaler
    scaler_path = os.path.join(PROC_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler       → {scaler_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: train/test split helpers (used by training script)
# ─────────────────────────────────────────────────────────────────────────────

def split_by_years(X, y, teams, years, train_years, test_years):
    """
    Split pre-built sequence arrays by which year each sample's target falls in.
    Re-fits the scaler ONLY on training data (data-leakage-safe).
    Returns scaled X_train, X_test and original-scale y_train, y_test.
    """
    train_mask = np.isin(years, train_years)
    test_mask  = np.isin(years, test_years)

    X_tr_raw = X[train_mask]
    X_te_raw = X[test_mask]
    y_tr     = y[train_mask]
    y_te     = y[test_mask]
    teams_tr = teams[train_mask]
    teams_te = teams[test_mask]
    years_te = years[test_mask]

    # Re-scale: fit on train only
    N_tr, S, F = X_tr_raw.shape
    scaler = StandardScaler()
    X_tr_2d = X_tr_raw.reshape(-1, F)
    X_te_2d = X_te_raw.reshape(-1, F)

    X_tr_scaled = scaler.fit_transform(X_tr_2d).reshape(N_tr, S, F).astype(np.float32)
    X_te_scaled = scaler.transform(X_te_2d).reshape(X_te_raw.shape[0], S, F).astype(np.float32)

    # y is already in original seconds scale (load from raw feature table)
    return (
        X_tr_scaled, y_tr.astype(np.float32),
        X_te_scaled, y_te.astype(np.float32),
        teams_tr, teams_te, years_te, scaler
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("F1 GRU Forecast — Step 2: Preprocessing")
    print("=" * 60)

    master_raw = build_feature_table()
    print(f"\nTotal (team, year) rows: {len(master_raw)}")
    print(master_raw.groupby("year")["team"].count().to_string())

    X, y, meta_list, scaler, _ = build_sequences(master_raw)
    save_outputs(master_raw, X, y, meta_list, scaler)

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
