# Turns a trip into a table with one row per (GPS observation, candidate arc) pair
import numpy as np
import pandas as pd

from config import CANDIDATE_RADIUS_M, MAX_CANDIDATES
from data_processing.data_loader import load_trip
from data_processing.geo_utils import gps_kinematics
from data_processing.ground_truth_aligner import align_track_to_route
from features.candidate_generator import FastRoadNetworkIndex

FEATURE_COLS = [
    'dist_to_segment',
    'dist_to_from',
    'dist_to_to',
    'proj_ratio',
    'segment_length',
    'heading_diff',
    'sin_heading_diff',
    'cos_heading_diff',
    'gps_speed',
    'dt',
    'dist_rank',
    'dist_minus_best',
    'num_candidates',
]


def build_dataset_for_trip(trip_data, sample_rate=1):
    """target = 1 if the candidate arc is the reference arc of that GPS observation"""
    track = trip_data['track']
    mean_lat = track['lat'].mean()
    true_arcs = align_track_to_route(trip_data)

    index = FastRoadNetworkIndex(trip_data['nodes'], trip_data['arcs'], mean_lat)
    speed, heading, dt = gps_kinematics(track, mean_lat)

    parts = []
    for i in range(0, len(track), sample_rate):
        cands = index.get_candidates_for_point(track['lon'].iat[i], track['lat'].iat[i],
                                               heading[i], speed[i], dt[i],
                                               radius=CANDIDATE_RADIUS_M, max_k=MAX_CANDIDATES)
        cands['obs_idx'] = i
        cands['true_arc'] = true_arcs[i] if true_arcs is not None else -1
        parts.append(cands)

    df = pd.concat(parts, ignore_index=True)
    df['trip_id'] = trip_data['trip_id']
    df['target'] = (df['arc_id'] == df['true_arc']).astype(np.int8)
    return df[['trip_id', 'obs_idx', 'arc_id', 'true_arc', 'target'] + FEATURE_COLS]


def build_multi_trip_dataset(partition_dir, trip_ids, sample_rate=1):
    all_dfs = []
    print(f"Building features for {len(trip_ids)} trips (sample rate={sample_rate})...")
    for n, tid in enumerate(trip_ids, 1):
        all_dfs.append(build_dataset_for_trip(load_trip(partition_dir, tid), sample_rate))
        if n % 5 == 0 or n == len(trip_ids):
            print(f"  {n}/{len(trip_ids)} trips done ({sum(len(d) for d in all_dfs)} rows)")
    return pd.concat(all_dfs, ignore_index=True)
