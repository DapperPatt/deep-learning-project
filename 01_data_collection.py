"""
01_data_collection.py
=====================
Downloads F1 race lap-time data for every season in ALL_YEARS using FastF1,
normalises team names, removes outlier laps, and saves one CSV per season.

Output
------
data/raw/season_{year}.csv
    Columns: year, race_round, team, driver, lap, lap_time_s
"""

import os
import warnings
import time
import pandas as pd
import numpy as np
import fastf1
from tqdm import tqdm

from config import (
    ALL_YEARS, CACHE_DIR, RAW_DIR,
    TEAM_ALIASES, MIN_LAPS, OUTLIER_FACTOR,
)

# ─────────────────────────────────────────────────────────────────────────────
# FastF1 setup
# ─────────────────────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache(CACHE_DIR)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_team(raw_name: str) -> str | None:
    """Map a raw FastF1 team name to a canonical team name, or None to drop."""
    return TEAM_ALIASES.get(raw_name, None)


def timedelta_to_seconds(td) -> float | None:
    """Convert a pandas Timedelta to float seconds; return None if NaT/NaN."""
    try:
        return td.total_seconds()
    except Exception:
        return None


def filter_outlier_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-race, per-driver: drop laps whose lap time exceeds
    OUTLIER_FACTOR × median lap time for that driver in that race.
    This removes pit-in/pit-out laps, SC laps, and formation laps.
    """
    def _filter(grp):
        med = grp["lap_time_s"].median()
        return grp[grp["lap_time_s"] <= med * OUTLIER_FACTOR]

    return (
        df.groupby(["race_round", "driver"], group_keys=False)
          .apply(_filter)
          .reset_index(drop=True)
    )


def collect_season(year: int) -> pd.DataFrame:
    """
    Load all race sessions for a given year, extract clean race laps,
    return a tidy DataFrame.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Keep only conventional race rounds
    schedule = schedule[schedule["EventFormat"].str.lower().str.contains("conventional|sprint")]

    rows = []
    rounds = schedule["RoundNumber"].tolist()

    for rnd in tqdm(rounds, desc=f"  Rounds ({year})", leave=False):
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"    [WARN] year={year} round={rnd}: {e}")
            continue

        laps = session.laps[["Driver", "Team", "LapNumber", "LapTime"]].copy()
        laps = laps.dropna(subset=["LapTime", "Team"])

        laps["lap_time_s"] = laps["LapTime"].apply(timedelta_to_seconds)
        laps = laps.dropna(subset=["lap_time_s"])
        laps = laps[laps["lap_time_s"] > 0]

        laps["team_canonical"] = laps["Team"].apply(normalise_team)
        laps = laps.dropna(subset=["team_canonical"])

        laps = laps.rename(columns={
            "Driver": "driver",
            "LapNumber": "lap",
            "team_canonical": "team",
        })
        laps["year"] = year
        laps["race_round"] = rnd

        rows.append(laps[["year", "race_round", "team", "driver", "lap", "lap_time_s"]])

        # Be polite — avoid hammering the cache unnecessarily
        time.sleep(0.05)

    if not rows:
        print(f"  [WARN] No data collected for {year}")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = filter_outlier_laps(df)

    # Drop drivers with very few laps (retirements, etc.)
    lap_counts = df.groupby(["race_round", "driver"])["lap"].count()
    valid_combos = lap_counts[lap_counts >= MIN_LAPS].index
    df = df.set_index(["race_round", "driver"])
    df = df[df.index.isin(valid_combos)].reset_index()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("F1 GRU Forecast — Step 1: Data Collection")
    print("=" * 60)

    for year in ALL_YEARS:
        out_path = os.path.join(RAW_DIR, f"season_{year}.csv")

        if os.path.exists(out_path):
            print(f"[{year}] Already exists → skipping ({out_path})")
            continue

        print(f"\n[{year}] Fetching data …")
        df = collect_season(year)

        if df.empty:
            print(f"[{year}] Empty dataset — skipping save.")
            continue

        df.to_csv(out_path, index=False)
        print(f"[{year}] Saved {len(df):,} laps → {out_path}")

    print("\n✓ Data collection complete.")
    print(f"  Raw CSVs in: {RAW_DIR}")


if __name__ == "__main__":
    main()
