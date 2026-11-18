# PathSnap: Distributed Map-Matching & Route Reconstruction

Matches noisy GPS observations to road segments using an ML classifier followed by HMM/Viterbi decoding, served through a FastAPI backend (Laptop 1) and a UI (Laptop 2).

## Folder Structure

| Folder | Purpose |
|---|---|
| `train/` | 80 training trips (raw data) |
| `student_test/` | 10 student-testing trips (raw data) |
| `data_processing/` | Reading, inspecting, cleaning and preprocessing raw files |
| `features/` | Candidate generation and feature engineering |
| `models/` | Training and evaluation of all compared models |
| `tuning/` | Bayesian hyperparameter optimisation |
| `hmm/` | HMM and Viterbi route reconstruction |
| `visualisation/` | Folium maps and plotting scripts |
| `results/` | CSVs, logs, plots and result summaries |
| `artifacts/` | Saved model, scaler and HMM config used by the API |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator interface (Laptop 2) |

Each trip folder contains `.track` (lon, lat, time), `.nodes` (lon, lat), `.arcs` (from_node, to_node) and `.route` (sequence of arc ids).

## Installation

```bash
pip install -r requirements.txt
```

## Execution

Run every script from the project root with `python -m` so the folders import each other correctly.

| Step | Command | Output |
|---|---|---|
| 1. Dataset exploration | `python -m data_processing.eda` | `results/eda/eda_summary.csv` |
| 2. Candidates + features | `python -m features.build_features` | `results/features/*.parquet`, `candidate_summary.csv`, `candidate_miss_rate_per_trip.csv` |
| 3. EDA plots | `python -m visualisation.eda_plots` | `results/eda/plots/*.png` |
| 4. EDA Folium maps | `python -m visualisation.folium_maps` | `results/maps/*.html` |
| 5. Phase 1 summary | `python -m data_processing.phase1_summary` | console summary |
| 6. Train + compare 5 models | `python -m models.train_models` | `results/models/model_comparison.csv`, `cv_folds.csv`, `history_*.csv`, `predictions/`, `plots/` |
| 7. Comparison plots | `python -m models.plot_model_comparison` | `results/models/plots/*.png` |

Key settings (candidate radius, max candidates, seed, folds) are in `config.py`.

## Deployment

Laptop 1 (API) and Laptop 2 (UI) instructions are added once the API and UI are built.
