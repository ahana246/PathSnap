# Prints a Phase 1 checkpoint summary using only numbers read from the generated result files
import pandas as pd

from config import ROOT, EDA_DIR, EDA_PLOTS_DIR, FEATURES_DIR, MAPS_DIR, LARGE_GAP_S
from features.feature_builder import FEATURE_COLS


def main():
    print("=" * 70)
    print("  PATHSNAP: PHASE 1 CHECKPOINT")
    print("=" * 70)

    eda = pd.read_csv(EDA_DIR / "eda_summary.csv", dtype={'trip_id': str})
    print("\n[1] DATASET AUDIT")
    for part, sub in eda.groupby('partition'):
        print(f"  {part}: {len(sub)} trips | GPS obs per trip mean={sub['num_gps_observations'].mean():.0f} "
              f"(min {sub['num_gps_observations'].min()}, max {sub['num_gps_observations'].max()})")
    print(f"  NaN values: {eda['nan_count'].sum()} | invalid coordinates: {eda['invalid_coords'].sum()}")
    print(f"  Steps exactly 1 s apart: {eda['pct_exact_1s'].mean():.2f}%")
    print(f"  Trips with gaps > {LARGE_GAP_S:.0f} s: {(eda['num_large_gaps'] > 0).sum()} | "
          f"largest gap: {eda['max_dt_sec'].max():.0f} s")
    print(f"  Road network: mean {eda['num_nodes'].mean():.0f} nodes, {eda['num_arcs'].mean():.0f} arcs")
    print(f"  Trips whose reference route has breaks: {(eda['route_breaks'] > 0).sum()}")

    cand_path = FEATURES_DIR / "candidate_summary.csv"
    print("\n[2] CANDIDATE GENERATION")
    if cand_path.exists():
        for _, r in pd.read_csv(cand_path).iterrows():
            print(f"  {r['partition']}: {r['observations']} obs, {r['mean_candidates_per_obs']} candidates/obs, "
                  f"miss rate {r['candidate_miss_rate'] * 100:.2f}%, nearest arc is true arc "
                  f"{r['true_arc_is_nearest_pct']:.2f}%")
    else:
        print("  not generated yet (run features/build_features.py)")
    print(f"  Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")

    print("\n[3] GENERATED FILES")
    for folder in [EDA_PLOTS_DIR, MAPS_DIR, FEATURES_DIR]:
        files = sorted(p.name for p in folder.glob('*')) if folder.exists() else []
        print(f"  {folder.relative_to(ROOT)}: {', '.join(files) or 'none'}")


if __name__ == "__main__":
    main()
