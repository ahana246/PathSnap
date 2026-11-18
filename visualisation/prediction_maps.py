# Member 3: Folium maps comparing classifier-only and HMM routes for the trips where the HMM helped most and least
import pandas as pd

from config import STUDENT_TEST_DIR, HMM_DIR, MAPS_DIR
from data_processing.data_loader import load_trip
from visualisation.folium_maps import create_folium_map


def main():
    per_trip = pd.read_csv(HMM_DIR / "per_trip.csv", dtype={'trip_id': str})
    preds = pd.read_parquet(HMM_DIR / "student_test_predictions.parquet")

    # largest and smallest Top-1 gain from HMM decoding
    picks = {'most_improved': per_trip.loc[per_trip['top1_gain'].idxmax(), 'trip_id'],
             'least_improved': per_trip.loc[per_trip['top1_gain'].idxmin(), 'trip_id']}
    for label, tid in picks.items():
        p = preds[preds['trip_id'] == tid].sort_values('obs_idx')
        routes = {'Classifier-only route': (p['classifier_arc'].tolist(), '#FB8C00'),
                  'HMM / Viterbi route': (p['hmm_arc'].tolist(), '#E53935')}
        create_folium_map(load_trip(STUDENT_TEST_DIR, tid), str(MAPS_DIR / f"prediction_{label}_{tid}.html"), routes)


if __name__ == "__main__":
    main()
