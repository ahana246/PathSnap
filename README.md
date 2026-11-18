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
| 8. Bayesian tuning | `python -m tuning.bayesian_optimisation --trials 20` | `results/tuning/optuna_trials.csv`, `best_params.json` |
| 9. Tuned vs untuned + export | `python -m tuning.export_model` | `results/tuning/tuned_vs_untuned.csv`, `artifacts/final_model.joblib`, `artifacts/model_config.json` |
| 10. HMM decoder tuning | `python -m hmm.tune_hmm` | `results/hmm/decoder_tuning.csv`, `artifacts/hmm_config.json` |
| 11. Classifier vs HMM | `python -m hmm.evaluate_hmm` | `results/hmm/classifier_vs_hmm.csv`, `per_trip.csv`, `student_test_predictions.parquet` |
| 12. Prediction maps | `python -m visualisation.prediction_maps` | `results/maps/prediction_*.html` |

Key settings (candidate radius, max candidates, HMM gap split, seed, folds) are in `config.py`. The `artifacts/` folder already holds the trained model and HMM settings, so the API can run without repeating steps 1–12.

## Deployment (two laptops)

Both laptops need the repo and `pip install -r requirements.txt`, and must be on the same network.

**Laptop 1: API server.** Put the `evaluator_test/` folder (10 trips) in the project root, then:

```bash
python -m api.main
```

The API listens on port 8000 on all network interfaces and serves the trips in `evaluator_test/`. Use `--data-dir student_test` to rehearse with the student-test trips. Every request is logged in this window. Find this laptop's IP address with `ipconfig` (Windows) and allow Python through the firewall when Windows asks.

**Laptop 2: UI.**

```bash
streamlit run ui/app.py
```

Open the page it prints (http://localhost:8501), enter `http://<laptop-1-ip>:8000` as the API address in the sidebar, choose one of the ten trips and click **Run map matching**. The UI shows the predicted route on a map and the accuracy computed by the API.

The reference routes stay on laptop 1: the API only uses them to score the prediction and never sends them to the UI.
