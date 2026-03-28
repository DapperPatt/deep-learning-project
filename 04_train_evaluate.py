"""
04_train_evaluate.py
====================
Walk-forward validation:
  Split 1 — Train 2014-2016 → Test 2017
  Split 2 — Train 2014-2021 → Test 2022

Then trains the final model on 2014-2025 and saves it for forecasting.

Outputs
-------
models/model_Split_1.pt
models/model_Split_2.pt
models/model_final.pt
plots/training_curves_Split_{1,2}.png
plots/val_predictions_Split_{1,2}_{year}.png
plots/historical_delta_heatmap.png
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

from config import (
    SPLITS, ALL_YEARS, PROC_DIR, MODELS_DIR, PLOTS_DIR,
    FEATURE_COLS, TARGET_COL, CANONICAL_TEAMS,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    RANDOM_SEED,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic imports  (filenames start with digits — can't use regular import)
# ─────────────────────────────────────────────────────────────────────────────

def _load(alias, filename):
    if alias in sys.modules:
        return sys.modules[alias]
    base = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(base, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_deps():
    pre   = _load("preprocessing", "02_preprocessing.py")
    model = _load("model",         "03_model.py")
    return pre, model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences():
    path = os.path.join(PROC_DIR, "sequences.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "sequences.npz not found. Run 02_preprocessing.py first."
        )
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["teams"], data["years"]


def load_raw_features():
    return pd.read_csv(os.path.join(PROC_DIR, "team_season_features.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one split
# ─────────────────────────────────────────────────────────────────────────────

def train_split(X, y, teams, years, train_years, test_years, label, save_path):
    pre, model_mod = _get_deps()
    device = model_mod.get_device()

    print(f"\n{'─'*55}")
    print(f"  {label}: Train {train_years[0]}-{train_years[-1]}  |  Test {test_years}")
    print(f"{'─'*55}")

    (X_tr, y_tr,
     X_te, y_te,
     teams_tr, teams_te,
     years_te, scaler) = pre.split_by_years(
        X, y, teams, years, train_years, test_years
    )

    print(f"  Train samples: {len(X_tr)}  |  Test samples: {len(X_te)}")

    train_loader = model_mod.make_loader(X_tr, y_tr, shuffle=True)
    test_loader  = model_mod.make_loader(X_te, y_te, shuffle=False)

    model, optimizer, scheduler, criterion = model_mod.build_model_and_optim(
        LEARNING_RATE, WEIGHT_DECAY, device
    )
    early_stop = model_mod.EarlyStopping(patience=PATIENCE)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss                  = model_mod.train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_mae, _, _  = model_mod.evaluate(
            model, test_loader, criterion, device
        )

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:>4} | "
                f"TrainMSE={tr_loss:.4f}  "
                f"ValMSE={val_loss:.4f}  "
                f"ValMAE={val_mae:.3f}s"
            )

        if early_stop.step(val_loss, model):
            print(f"    Early stop at epoch {epoch}")
            break

    early_stop.restore_best(model)
    torch.save({"model_state": model.state_dict(), "scaler": scaler}, save_path)
    print(f"  Saved -> {save_path}")

    _, final_mae, preds, targets = model_mod.evaluate(
        model, test_loader, criterion, device
    )
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    print(f"\n  Test MAE  = {final_mae:.4f} s")
    print(f"  Test RMSE = {rmse:.4f} s")

    _plot_training_curves(train_losses, val_losses, label)
    _plot_val_predictions(preds, targets, teams_te, years_te, label)

    return {
        "label":       label,
        "train_years": train_years,
        "test_years":  test_years,
        "mae":         final_mae,
        "rmse":        rmse,
        "preds":       preds,
        "targets":     targets,
        "teams_te":    teams_te,
        "years_te":    years_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_curves(train_losses, val_losses, label):
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", color="#1f77b4")
    ax.plot(epochs, val_losses,   label="Val MSE",   color="#d62728", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curves - {label}")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"training_curves_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot -> {path}")


def _plot_val_predictions(preds, targets, teams_te, years_te, label):
    df = pd.DataFrame({
        "team": teams_te, "year": years_te,
        "actual": targets, "pred": preds,
    })
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].sort_values("actual").reset_index(drop=True)
        x   = np.arange(len(sub))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w/2, sub["actual"], w, label="Actual",    color="#2196F3", alpha=0.85)
        ax.bar(x + w/2, sub["pred"],   w, label="Predicted", color="#FF5722", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["team"], rotation=30, ha="right")
        ax.set_ylabel("Mean Lap-Time Delta vs Fastest (s)")
        ax.set_title(f"Actual vs Predicted - {yr}  [{label}]")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"val_predictions_{label}_{yr}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Plot -> {path}")


def plot_combined_heatmap(raw_features_df):
    pivot = raw_features_df.pivot(index="team", columns="year", values="mean_delta")
    pivot = pivot.loc[[t for t in CANONICAL_TEAMS if t in pivot.index]]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r",
        linewidths=0.4, linecolor="#333",
        annot=True, fmt=".2f",
        cbar_kws={"label": "Mean Delta vs Fastest (s)"},
    )
    for reset_year in [2017, 2022]:
        cols = list(pivot.columns)
        if reset_year in cols:
            ax.axvline(x=cols.index(reset_year), color="white", linewidth=2.5, linestyle="--")
    ax.set_title("F1 Team Pace Delta (vs Fastest Team) - 2014-2025", fontsize=14)
    ax.set_xlabel("Season")
    ax.set_ylabel("Team")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "historical_delta_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Final model: train on full 2014-2025
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X, y, teams, years):
    _, model_mod = _get_deps()
    device       = model_mod.get_device()

    print(f"\n{'─'*55}")
    print(f"  FINAL MODEL: Train {ALL_YEARS[0]}-{ALL_YEARS[-1]}")
    print(f"{'─'*55}")

    train_mask = np.isin(years, ALL_YEARS)
    X_tr = X[train_mask].astype(np.float32)
    y_tr = y[train_mask].astype(np.float32)

    N, S, F = X_tr.shape
    scaler  = StandardScaler()
    X_tr_s  = (
        scaler.fit_transform(X_tr.reshape(-1, F))
              .reshape(N, S, F)
              .astype(np.float32)
    )

    loader  = model_mod.make_loader(X_tr_s, y_tr, shuffle=True)
    model, optimizer, scheduler, criterion = model_mod.build_model_and_optim(
        LEARNING_RATE, WEIGHT_DECAY, device
    )

    best_loss, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = model_mod.train_one_epoch(
            model, loader, optimizer, criterion, device
        )
        scheduler.step(tr_loss)

        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4} | TrainMSE={tr_loss:.4f}")

    model.load_state_dict(best_state)
    save_path = os.path.join(MODELS_DIR, "model_final.pt")
    torch.save({"model_state": model.state_dict(), "scaler": scaler}, save_path)
    print(f"  Final model saved -> {save_path}")
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results_list):
    print("\n" + "=" * 55)
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  {'Split':<12} {'Train':<14} {'Test':<10} {'MAE (s)':<10} {'RMSE (s)'}")
    print(f"  {'-'*52}")
    for r in results_list:
        tr = f"{r['train_years'][0]}-{r['train_years'][-1]}"
        te = str(r['test_years'])
        print(
            f"  {r['label']:<12} {tr:<14} {te:<10} "
            f"{r['mae']:<10.4f} {r['rmse']:.4f}"
        )
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("F1 GRU Forecast — Step 4: Training & Validation")
    print("=" * 60)

    X, y, teams, years = load_sequences()
    raw_features        = load_raw_features()

    results_list = []
    for split in SPLITS:
        save_path = os.path.join(MODELS_DIR, f"model_{split['label']}.pt")
        result = train_split(
            X, y, teams, years,
            train_years = split["train"],
            test_years  = split["test"],
            label       = split["label"],
            save_path   = save_path,
        )
        results_list.append(result)

    plot_combined_heatmap(raw_features)
    print_summary(results_list)
    train_final_model(X, y, teams, years)

    print("\n Training complete.")


if __name__ == "__main__":
    main()
