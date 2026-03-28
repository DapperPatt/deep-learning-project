"""
05_forecast_2026.py
===================
Loads the final GRU model (trained on 2014–2025) and constructs 2026 input
sequences for each team.  Outputs:

  - Console table: predicted mean lap-time delta per team for 2026
  - plots/forecast_2026_ranking.png     — horizontal bar chart
  - plots/forecast_2026_vs_history.png  — team trajectories + 2026 dot
  - plots/forecast_2026_heatmap.png     — full heatmap including 2026 column

The 2026 season introduces a completely new regulation era (era index 3).
We build the input window as the last SEQ_LEN years of each team's scaled
history and encode the regulation reset signal explicitly in the features.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from config import (
    PROC_DIR, MODELS_DIR, PLOTS_DIR,
    CANONICAL_TEAMS, FEATURE_COLS, TARGET_COL,
    ERA_MAP, ERA_START, SEQ_LEN, ALL_YEARS, FORECAST_YEAR,
    RANDOM_SEED,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Import sibling modules dynamically (filenames start with digits)
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(name, filename):
    spec   = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_model_mod = _load_module("model", "03_model.py")
F1GRU      = _model_mod.F1GRU
get_device = _model_mod.get_device


# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────

def load_final_model():
    path = os.path.join(MODELS_DIR, "model_final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("model_final.pt not found — run 04_train_evaluate.py first.")
    ckpt   = torch.load(path, map_location="cpu")
    model  = F1GRU()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = ckpt["scaler"]
    return model, scaler


def load_raw_features() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "team_season_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("team_season_features.csv not found — run 02_preprocessing.py first.")
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Build 2026 input sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_2026_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct synthetic 2026 feature rows for each team.

    Since 2026 hasn't happened yet, we extrapolate each team's features:
      - mean_delta / median_delta / delta_std : taken as the last known value
        (2025) — the model learns whether to trust or override this based on
        its regulation-reset training signal.
      - reg_era        = 3  (new 2026 era)
      - seasons_in_era = 0  (first season of the new era)
      - prior_year_delta = team's 2025 mean_delta
      - rolling3_delta   = 3-year rolling mean from 2023–2025
    """
    rows = []
    for team in CANONICAL_TEAMS:
        team_df = raw_df[raw_df["team"] == team].sort_values("year")
        if team_df.empty:
            print(f"  [WARN] No history for {team} — skipping.")
            continue

        last = team_df.iloc[-1]  # most recent available (2025 or earlier)

        # Rolling 3-year delta
        recent = team_df["mean_delta"].values[-3:]
        roll3  = float(np.mean(recent))

        row = {
            "team":              team,
            "year":              FORECAST_YEAR,
            "mean_delta":        last["mean_delta"],    # placeholder — overwritten by model
            "median_delta":      last["median_delta"],
            "delta_std":         last["delta_std"],
            "reg_era":           3,
            "seasons_in_era":    0,
            "prior_year_delta":  last["mean_delta"],
            "rolling3_delta":    roll3,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_sequences_for_forecast(raw_df: pd.DataFrame, feats_2026: pd.DataFrame, scaler: StandardScaler):
    """
    For each team, build a (1, SEQ_LEN, F) tensor from the last SEQ_LEN−1
    years of real history + the 2026 synthetic row as the final context step.

    The scaler fitted during training is applied here.
    """
    X_list, team_list = [], []

    for team in CANONICAL_TEAMS:
        team_hist = raw_df[raw_df["team"] == team].sort_values("year")
        row_2026  = feats_2026[feats_2026["team"] == team]

        if team_hist.empty or row_2026.empty:
            continue

        # We need SEQ_LEN steps: the last SEQ_LEN-1 from history + 2026 row
        # (2026 row acts as the "current" context; model predicts y at t+1
        #  but here we want the model to predict 2026 directly — so we feed
        #  the last SEQ_LEN historical rows as the window.)
        # Strategy: feed last SEQ_LEN years of history (up to & including 2025).
        hist_features = team_hist[FEATURE_COLS].values
        if len(hist_features) < SEQ_LEN:
            # Pad with the earliest row if not enough history
            pad_n  = SEQ_LEN - len(hist_features)
            pad    = np.tile(hist_features[0], (pad_n, 1))
            hist_features = np.vstack([pad, hist_features])

        window = hist_features[-SEQ_LEN:]  # (SEQ_LEN, F)

        # Override the LAST timestep's reg_era & seasons_in_era to signal
        # the upcoming regulation reset (the model has seen this pattern before
        # at 2017 and 2022 resets during training).
        # Index positions in FEATURE_COLS:
        era_idx  = FEATURE_COLS.index("reg_era")
        sia_idx  = FEATURE_COLS.index("seasons_in_era")
        window[-1, era_idx] = 3   # 2026 era
        window[-1, sia_idx] = 0   # first season

        X_list.append(window)
        team_list.append(team)

    X = np.array(X_list, dtype=np.float32)  # (N, SEQ_LEN, F)
    N, S, F = X.shape
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, S, F).astype(np.float32)

    return X_scaled, team_list


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_2026(model: F1GRU, X_scaled: np.ndarray, team_list: list) -> pd.DataFrame:
    device = get_device()
    model.to(device)
    X_tensor = torch.from_numpy(X_scaled).to(device)
    preds    = model(X_tensor).cpu().numpy()   # (N,)

    # Predictions are in scaled space — inverse transform the target column
    # We only need the mean_delta column (index 0 in FEATURE_COLS)
    target_idx = FEATURE_COLS.index(TARGET_COL)

    # To inverse-transform a single column, reconstruct a full-feature array
    dummy = np.zeros((len(preds), len(FEATURE_COLS)), dtype=np.float32)
    dummy[:, target_idx] = preds

    # NOTE: The scaler was fit on all features; we need the scaler from the checkpoint.
    # The predictions are already in the scaler's output space — they need
    # inverse transformation. We pass back to the caller; caller holds the scaler.
    return pd.DataFrame({"team": team_list, "pred_delta_scaled": preds})


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-scale predictions back to seconds
# ─────────────────────────────────────────────────────────────────────────────

def inverse_scale_predictions(pred_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    target_idx = FEATURE_COLS.index(TARGET_COL)
    dummy = np.zeros((len(pred_df), len(FEATURE_COLS)), dtype=np.float32)
    dummy[:, target_idx] = pred_df["pred_delta_scaled"].values
    inv = scaler.inverse_transform(dummy)
    pred_df = pred_df.copy()
    pred_df["pred_delta_s"] = inv[:, target_idx]
    # Clip to physical plausibility (gap can't be negative or > 5s)
    pred_df["pred_delta_s"] = pred_df["pred_delta_s"].clip(lower=0.0, upper=5.0)
    return pred_df


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

# F1 team colours (approximate official palette)
TEAM_COLORS = {
    "Mercedes":    "#00D2BE",
    "Red Bull":    "#0600EF",
    "Ferrari":     "#DC0000",
    "McLaren":     "#FF8700",
    "Alpine":      "#0090FF",
    "RB":          "#2B4562",
    "Aston Martin":"#006F62",
    "Williams":    "#005AFF",
    "Haas":        "#B6BABD",
    "Sauber":      "#52E252",
}


def plot_2026_ranking(pred_df: pd.DataFrame):
    """Horizontal bar chart of predicted 2026 pace deltas."""
    df = pred_df.sort_values("pred_delta_s").reset_index(drop=True)
    colors = [TEAM_COLORS.get(t, "#888") for t in df["team"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["team"], df["pred_delta_s"], color=colors, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, df["pred_delta_s"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}s", va="center", ha="left", fontsize=9
        )

    ax.set_xlabel("Predicted Mean Lap-Time Delta vs Fastest Team (s)", fontsize=11)
    ax.set_title("2026 F1 Season — GRU Forecast: Predicted Team Pace", fontsize=13, fontweight="bold")
    ax.set_xlim(left=0, right=df["pred_delta_s"].max() * 1.2)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "forecast_2026_ranking.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")


def plot_team_trajectories(raw_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Line chart: each team's historical delta + 2026 forecast dot."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()

    all_years = sorted(raw_df["year"].unique().tolist() + [FORECAST_YEAR])

    for ax, team in zip(axes, CANONICAL_TEAMS):
        hist = raw_df[raw_df["team"] == team].sort_values("year")
        fc   = pred_df[pred_df["team"] == team]

        color = TEAM_COLORS.get(team, "#888")

        ax.plot(hist["year"], hist["mean_delta"], color=color, linewidth=2, marker="o", markersize=4)

        if not fc.empty:
            ax.scatter(
                [FORECAST_YEAR], fc["pred_delta_s"].values,
                color=color, s=120, zorder=5, marker="*",
                edgecolors="black", linewidths=0.8, label="2026 forecast"
            )

        # Regulation reset markers
        for reset in [2017, 2022, 2026]:
            ax.axvline(x=reset, color="#999", linestyle=":", linewidth=0.8)

        ax.set_title(team, fontweight="bold", fontsize=10, color=color)
        ax.set_xlabel("Season", fontsize=8)
        ax.set_ylabel("Δ vs Fastest (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "F1 Team Pace Trajectory (2014–2025) + 2026 GRU Forecast ★",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "forecast_2026_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def plot_full_heatmap(raw_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Heatmap of all years including 2026 column."""
    hist_pivot = raw_df.pivot(index="team", columns="year", values="mean_delta")

    # Add 2026 column
    fc_series = pred_df.set_index("team")["pred_delta_s"]
    hist_pivot[FORECAST_YEAR] = fc_series

    # Reorder rows
    hist_pivot = hist_pivot.loc[[t for t in CANONICAL_TEAMS if t in hist_pivot.index]]
    hist_pivot = hist_pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        hist_pivot, ax=ax,
        cmap="RdYlGn_r",
        linewidths=0.4, linecolor="#444",
        annot=True, fmt=".2f",
        cbar_kws={"label": "Mean Delta vs Fastest (s)"},
        vmin=0, vmax=hist_pivot.values.max(),
    )

    # Highlight 2026 column
    cols = list(hist_pivot.columns)
    x2026 = cols.index(FORECAST_YEAR)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x2026, 0), 1, len(hist_pivot),
            boxstyle="square,pad=0",
            linewidth=3, edgecolor="gold", facecolor="none", zorder=5,
        )
    )

    ax.set_title(
        f"F1 Team Pace Delta — 2014–2025 (actual) + 2026 (GRU forecast ★)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Season")
    ax.set_ylabel("Team")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "forecast_2026_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_forecast_table(pred_df: pd.DataFrame):
    df = pred_df.sort_values("pred_delta_s").reset_index(drop=True)
    df.index += 1
    print("\n" + "=" * 50)
    print("  2026 F1 Season — GRU Pace Forecast")
    print("=" * 50)
    print(f"  {'Rank':<6} {'Team':<18} {'Pred. Δ vs Fastest (s)'}")
    print(f"  {'-'*46}")
    for rank, row in df.iterrows():
        bar = "█" * max(1, int(row["pred_delta_s"] / 0.1))
        print(f"  {rank:<6} {row['team']:<18} {row['pred_delta_s']:>6.3f}s  {bar}")
    print("=" * 50)
    print("  ★ = 0.000s means predicted to be the benchmark team.")
    print("  Predicted gaps are relative — not absolute lap times.")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("F1 GRU Forecast — Step 5: 2026 Forecast")
    print("=" * 60)

    model, scaler = load_final_model()
    raw_df        = load_raw_features()

    feats_2026    = build_2026_features(raw_df)
    X_scaled, team_list = build_sequences_for_forecast(raw_df, feats_2026, scaler)

    pred_df_raw   = predict_2026(model, X_scaled, team_list)
    pred_df       = inverse_scale_predictions(pred_df_raw, scaler)

    # Normalise: set minimum to 0 (the "fastest" team)
    pred_df["pred_delta_s"] -= pred_df["pred_delta_s"].min()

    print_forecast_table(pred_df)

    plot_2026_ranking(pred_df)
    plot_team_trajectories(raw_df, pred_df)
    plot_full_heatmap(raw_df, pred_df)

    # Save forecast CSV
    out_path = os.path.join(PROC_DIR, "forecast_2026.csv")
    pred_df.to_csv(out_path, index=False)
    print(f"\nForecast saved → {out_path}")
    print("\n✓ Forecasting complete.")


if __name__ == "__main__":
    main()
