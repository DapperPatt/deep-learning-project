"""
run_pipeline.py
===============
Convenience script that runs the entire pipeline in order.
Usage:
    python run_pipeline.py [--skip-collection]

--skip-collection : skip step 1 if raw CSVs already exist
"""

import sys
import os
import importlib.util
import argparse


def load_module(name: str, filename: str):
    """Load a module from a file path (handles digit-prefixed filenames)."""
    base = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(base, filename))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def banner(text: str):
    print("\n" + "█" * 60)
    print(f"  {text}")
    print("█" * 60)


def main():
    parser = argparse.ArgumentParser(description="F1 GRU Forecast Pipeline")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip FastF1 data download (use existing raw CSVs)")
    args = parser.parse_args()

    # ── Step 0: make sure deps are importable ────────────────────────────────
    try:
        import fastf1, torch, sklearn, seaborn  # noqa
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Step 1: Data collection ───────────────────────────────────────────────
    if not args.skip_collection:
        banner("Step 1 / 5 — Data Collection (FastF1)")
        dc = load_module("data_collection", "01_data_collection.py")
        dc.main()
    else:
        print("\n[Skipping Step 1 — data collection]")

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    banner("Step 2 / 5 — Preprocessing & Feature Engineering")
    pre = load_module("preprocessing", "02_preprocessing.py")
    pre.main()

    # ── Step 3: Model is just a definition — nothing to run ─────────────────
    banner("Step 3 / 5 — Model architecture loaded ✓")
    load_module("model", "03_model.py")
    print("  GRU model class registered.")

    # ── Step 4: Training & validation ────────────────────────────────────────
    banner("Step 4 / 5 — Walk-Forward Training & Validation")
    tr = load_module("train_evaluate", "04_train_evaluate.py")
    # Patch imports that the training module expects
    tr.split_by_years        = pre.split_by_years
    _model_mod               = sys.modules["model"]
    tr.F1GRU                 = _model_mod.F1GRU
    tr.make_loader           = _model_mod.make_loader
    tr.get_device            = _model_mod.get_device
    tr.EarlyStopping         = _model_mod.EarlyStopping
    tr.train_one_epoch       = _model_mod.train_one_epoch
    tr.evaluate              = _model_mod.evaluate
    tr.build_model_and_optim = _model_mod.build_model_and_optim
    tr.main()

    # ── Step 5: 2026 Forecast ─────────────────────────────────────────────────
    banner("Step 5 / 5 — 2026 Season Forecast")
    fc = load_module("forecast", "05_forecast_2026.py")
    fc.main()

    print("\n" + "=" * 60)
    print("  ✅ Full pipeline complete!")
    print(f"  Plots  → {os.path.join(os.path.dirname(__file__), 'plots')}")
    print(f"  Models → {os.path.join(os.path.dirname(__file__), 'models')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
