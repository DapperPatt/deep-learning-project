# F1 Team Performance Forecasting with GRU

**Research Question:** Can a GRU trained on historical F1 team pace data across multiple regulation resets learn performance patterns and generalize to forecast relative team pace for the 2026 season?

## Project Structure

```
f1_gru_forecast/
├── README.md
├── requirements.txt
├── config.py                  # Central config (teams, years, hyperparams)
├── 01_data_collection.py      # FastF1 data pipeline → raw CSVs
├── 02_preprocessing.py        # Feature engineering, scaling, sequence building
├── 03_model.py                # GRU model definition (PyTorch)
├── 04_train_evaluate.py       # Training loop + walk-forward validation
├── 05_forecast_2026.py        # Final model → 2026 forecast + plots
└── data/
    ├── raw/                   # Raw per-race CSVs from FastF1
    └── processed/             # Scaled sequences ready for model
```

## Pipeline (run in order)

```bash
pip install -r requirements.txt
python 01_data_collection.py    # ~30-60 min (FastF1 caches data)
python 02_preprocessing.py
python 03_model.py              # (defines model, no direct run needed)
python 04_train_evaluate.py     # Trains val splits + prints metrics
python 05_forecast_2026.py      # Produces 2026 forecast plots
```

## Methodology

| Training Window   | Test Set | Purpose                        |
|-------------------|----------|--------------------------------|
| 2014–2016         | 2017     | Validate across reg reset      |
| 2014–2021         | 2022     | Validate across reg reset      |
| 2014–2025         | 2026     | Final forecast                 |

**Target variable:** Mean lap-time delta (seconds) relative to the fastest team per race, averaged per team per season.

## Key Dependencies
- `fastf1` — F1 timing data
- `torch` — GRU model
- `pandas`, `numpy`, `scikit-learn` — preprocessing
- `matplotlib`, `seaborn` — visualisation
