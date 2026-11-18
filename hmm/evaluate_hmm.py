# Member 3: scores classifier-only vs classifier + HMM routes on the student_test trips
import os
import time
import pandas as pd

from config import STUDENT_TEST_DIR, HMM_DIR
from data_processing.data_loader import list_trip_ids
from data_processing.ground_truth_aligner import align_track_to_route
from hmm.pipeline import load_artifacts, predict_trip


def continuity(arcs, trip):
    # share of consecutive predictions on the same arc or on directly connected arcs
    f, t = trip['arcs']['from_node'].values, trip['arcs']['to_node'].values
    a1, a2 = arcs[:-1], arcs[1:]
    return float(((a1 == a2) | (t[a1] == f[a2])).mean()) if len(a1) else 1.0


def score_trip(trip, preds):
    """
    Uses the reference route (server-side only) to score both predicted routes:
    Top-1 accuracy on labelled points, route continuity, and route arc recall/precision (sequence level).
    """
    if trip['route'] is None:
        return None
    true_arcs = pd.Series(align_track_to_route(trip))
    labelled = true_arcs >= 0
    route = set(trip['route'])

    scores = {'labelled_points': int(labelled.sum())}
    for name in ['classifier', 'hmm']:
        arcs = preds[f'{name}_arc'].values
        predicted = set(arcs)
        scores[f'{name}_top1'] = float((arcs[labelled] == true_arcs[labelled].values).mean())
        scores[f'{name}_continuity'] = continuity(arcs, trip)
        scores[f'{name}_route_recall'] = len(route & predicted) / len(route)
        scores[f'{name}_route_precision'] = len(route & predicted) / len(predicted)
    return scores


def main():
    os.makedirs(HMM_DIR, exist_ok=True)
    artifacts = load_artifacts()
    rows, all_preds = [], []
    for tid in list_trip_ids(STUDENT_TEST_DIR):
        start = time.perf_counter()
        trip, preds = predict_trip(STUDENT_TEST_DIR, tid, artifacts)
        scores = score_trip(trip, preds)
        rows.append({'trip_id': tid, 'points': len(preds), **scores, 'runtime_s': round(time.perf_counter() - start, 2)})
        all_preds.append(preds.assign(trip_id=tid))
        print(f"  {tid}: top-1 classifier {scores['classifier_top1']:.3f} -> HMM {scores['hmm_top1']:.3f}, "
              f"continuity {scores['classifier_continuity']:.3f} -> {scores['hmm_continuity']:.3f}")

    per_trip = pd.DataFrame(rows)
    per_trip['top1_gain'] = per_trip['hmm_top1'] - per_trip['classifier_top1']
    per_trip.round(4).to_csv(HMM_DIR / "per_trip.csv", index=False)
    pd.concat(all_preds).to_parquet(HMM_DIR / "student_test_predictions.parquet", index=False)

    # overall: Top-1 weighted by labelled points, continuity weighted by points, route scores averaged over trips
    overall = []
    for name in ['classifier', 'hmm']:
        overall.append({
            'method': 'classifier only' if name == 'classifier' else 'classifier + HMM/Viterbi',
            'top1_accuracy': (per_trip[f'{name}_top1'] * per_trip['labelled_points']).sum() / per_trip['labelled_points'].sum(),
            'route_continuity': (per_trip[f'{name}_continuity'] * per_trip['points']).sum() / per_trip['points'].sum(),
            'route_recall': per_trip[f'{name}_route_recall'].mean(),
            'route_precision': per_trip[f'{name}_route_precision'].mean(),
            'trips_improved': int((per_trip['top1_gain'] > 0).sum()) if name == 'hmm' else None,
        })
    overall = pd.DataFrame(overall).round(4)
    overall.to_csv(HMM_DIR / "classifier_vs_hmm.csv", index=False)
    print("\n" + overall.to_string(index=False))
    print(f"[+] Saved results to {HMM_DIR}")


if __name__ == "__main__":
    main()
