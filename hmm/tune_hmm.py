# Member 2: chooses the HMM decoder parameters on training trips and saves artifacts/hmm_config.json
import os
import json
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import TRAIN_DIR, FEATURES_DIR, HMM_DIR, HMM_CONFIG_PATH, MODEL_CONFIG_PATH, N_FOLDS
from data_processing.data_loader import load_trip
from hmm.transition import RoadGraph
from hmm.viterbi import build_steps, viterbi
from models.train_models import labelled_only
from tuning.bayesian_optimisation import make_model

BETAS = [1, 2, 5, 10, 20, 40]                 # metres of route/GPS mismatch per unit of log-probability
UNREACHABLE_LOGS = [-10, -20, -40, -80, -160]  # log-probability of a move the road network does not allow
TRIP_STEP = 4                         # decode every 4th training trip (20 trips) to keep the search short


def out_of_fold_probs(df, features, params):
    # each trip is scored by a model that never saw it, so the emissions are as honest as on new trips
    probs = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(df, groups=df['trip_id']):
        fit_df = labelled_only(df.iloc[tr])
        model = make_model(params).fit(fit_df[features], fit_df['target'])
        probs[va] = model.predict_proba(df.iloc[va][features])[:, 1]
    return probs


def main():
    os.makedirs(HMM_DIR, exist_ok=True)
    with open(MODEL_CONFIG_PATH) as f:
        model_cfg = json.load(f)

    df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    print("Out-of-fold classifier probabilities on the 80 train trips...")
    df['prob'] = out_of_fold_probs(df, model_cfg['features'], model_cfg['params'])

    trip_ids = sorted(df['trip_id'].unique())[::TRIP_STEP]
    print(f"Building HMM steps for {len(trip_ids)} train trips...")
    start = time.perf_counter()
    prepared = []
    for tid in trip_ids:
        trip = load_trip(TRAIN_DIR, tid)
        cands = df[df['trip_id'] == tid]
        graph = RoadGraph(trip['nodes'], trip['arcs'], trip['track']['lat'].mean())
        steps, diffs = build_steps(cands, cands['prob'].values, trip['track'], graph)
        truth = cands.groupby('obs_idx')['true_arc'].first()
        prepared.append((steps, diffs, truth[truth >= 0]))
    print(f"  done in {time.perf_counter() - start:.0f} s")

    # Top-1 accuracy of the decoded route on labelled observations, for every parameter pair
    rows = []
    for beta, unreachable in itertools.product(BETAS, UNREACHABLE_LOGS):
        hits = total = 0
        for steps, diffs, truth in prepared:
            route = viterbi(steps, diffs, beta, unreachable)
            hits += sum(route[o] == arc for o, arc in truth.items())
            total += len(truth)
        rows.append({'beta': beta, 'unreachable_log': unreachable, 'train_top1': round(hits / total, 4)})
        print(f"  beta={beta}, unreachable_log={unreachable}: top-1={hits / total:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(HMM_DIR / "decoder_tuning.csv", index=False)
    best = results.loc[results['train_top1'].idxmax()]
    hmm_cfg = {'beta': float(best['beta']), 'unreachable_log': float(best['unreachable_log']),
               'chosen_by': f"highest Top-1 on {len(trip_ids)} train trips with out-of-fold probabilities",
               'train_top1': float(best['train_top1'])}
    with open(HMM_CONFIG_PATH, 'w') as f:
        json.dump(hmm_cfg, f, indent=2)
    print(f"\nChosen: beta={hmm_cfg['beta']}, unreachable_log={hmm_cfg['unreachable_log']} -> {HMM_CONFIG_PATH}")


if __name__ == "__main__":
    main()
