# Member 2: full prediction for one trip: candidates and features -> classifier probabilities -> HMM route
import json
import joblib
import pandas as pd

from config import MODEL_PATH, MODEL_CONFIG_PATH, HMM_CONFIG_PATH
from data_processing.data_loader import load_trip
from features.feature_builder import build_dataset_for_trip
from hmm.viterbi import decode_trip


def load_artifacts():
    with open(MODEL_CONFIG_PATH) as f:
        model_cfg = json.load(f)
    with open(HMM_CONFIG_PATH) as f:
        hmm_cfg = json.load(f)
    return {'model': joblib.load(MODEL_PATH), 'features': model_cfg['features'], 'hmm': hmm_cfg}


def predict_trip(partition_dir, trip_id, artifacts):
    """
    Returns the loaded trip and one row per GPS observation with the classifier-only arc and the HMM arc.
    The reference route is removed before prediction, so it can only be used afterwards for scoring.
    """
    trip = load_trip(partition_dir, trip_id)
    hidden = dict(trip, route=None)

    cands = build_dataset_for_trip(hidden, sample_rate=1)
    probs = artifacts['model'].predict_proba(cands[artifacts['features']])[:, 1]

    # classifier only: the highest-probability candidate of each observation
    best = cands.assign(prob=probs).loc[lambda d: d.groupby('obs_idx')['prob'].idxmax()]
    classifier = dict(zip(best['obs_idx'], best['arc_id']))
    hmm = decode_trip(cands, probs, hidden, artifacts['hmm'])

    track = trip['track']
    preds = pd.DataFrame({'obs_idx': range(len(track)), 'lon': track['lon'], 'lat': track['lat']})
    preds['classifier_arc'] = preds['obs_idx'].map(classifier)
    preds['hmm_arc'] = preds['obs_idx'].map(hmm)
    return trip, preds
