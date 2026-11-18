# Builds the candidate feature tables for train and student_test and reports candidate miss rate
import os
import pandas as pd

from config import PARTITIONS, FEATURES_DIR, TRAIN_SAMPLE_RATE, CANDIDATE_RADIUS_M, MAX_CANDIDATES
from data_processing.data_loader import list_trip_ids
from features.feature_builder import build_multi_trip_dataset


def candidate_report(df, partition):
    # per observation: was the reference arc among the candidates, and at which distance rank
    all_obs = df.groupby(['trip_id', 'obs_idx']).agg(true_arc=('true_arc', 'first'), n_cands=('arc_id', 'size'))
    labelled = df[df['true_arc'] >= 0]  # unlabelled points (not covered by the route) are left out
    obs = labelled.groupby(['trip_id', 'obs_idx']).agg(hit=('target', 'max'), n_cands=('arc_id', 'size'))
    true_rows = labelled[labelled['target'] == 1]

    per_trip = all_obs.groupby('trip_id').agg(observations=('true_arc', 'size'),
                                              unlabelled=('true_arc', lambda s: (s < 0).sum()))
    per_trip['hits'] = obs.groupby('trip_id')['hit'].sum()
    per_trip['hits'] = per_trip['hits'].fillna(0).astype(int)
    per_trip['candidate_miss_rate'] = 1 - per_trip['hits'] / (per_trip['observations'] - per_trip['unlabelled'])
    per_trip = per_trip.reset_index()
    per_trip['partition'] = partition

    summary = {
        'partition': partition,
        'trips': df['trip_id'].nunique(),
        'observations': len(all_obs),
        'unlabelled_observations': int((all_obs['true_arc'] < 0).sum()),
        'unlabelled_pct': round((all_obs['true_arc'] < 0).mean() * 100, 2),
        'labelled_observations': len(obs),
        'candidate_rows': len(df),
        'mean_candidates_per_obs': round(obs['n_cands'].mean(), 2),
        'positive_row_pct': round(labelled['target'].mean() * 100, 2),
        'candidate_miss_rate': round(1 - obs['hit'].mean(), 4),
        'true_arc_is_nearest_pct': round((true_rows['dist_rank'] == 0).sum() / len(obs) * 100, 2),
        'true_arc_median_dist_m': round(true_rows['dist_to_segment'].median(), 2),
        'true_arc_p95_dist_m': round(true_rows['dist_to_segment'].quantile(0.95), 2),
        'radius_m': CANDIDATE_RADIUS_M,
        'max_candidates': MAX_CANDIDATES,
    }
    return per_trip, summary


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    per_trip_all, summaries = [], []

    for partition, part_dir in PARTITIONS.items():
        # training trips are subsampled to keep the table manageable; test trips keep every point
        rate = TRAIN_SAMPLE_RATE if partition == 'train' else 1
        df = build_multi_trip_dataset(part_dir, list_trip_ids(part_dir), sample_rate=rate)
        out_path = FEATURES_DIR / f"{partition}_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[+] Saved {out_path} ({len(df)} rows)")

        per_trip, summary = candidate_report(df, partition)
        per_trip_all.append(per_trip)
        summaries.append(summary)

    pd.concat(per_trip_all).to_csv(FEATURES_DIR / "candidate_miss_rate_per_trip.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(FEATURES_DIR / "candidate_summary.csv", index=False)
    print("\n" + summary_df.T.to_string())


if __name__ == "__main__":
    main()
