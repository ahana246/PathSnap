# Computes candidate-level and observation-level metrics for a set of predicted probabilities
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from data_processing.data_loader import load_arcs


def load_arcs_by_trip(partition_dir, trip_ids):
    # arc_ids are local to each trip, so connectivity must be checked with that trip's own arcs
    arcs_by_trip = {}
    for tid in trip_ids:
        arcs = load_arcs(partition_dir / tid / f"{tid}.arcs")
        arcs_by_trip[tid] = (arcs['from_node'].values, arcs['to_node'].values)
    return arcs_by_trip


def top1_predictions(df, probs):
    # for each GPS observation keep the candidate with the highest probability
    scored = df[['trip_id', 'obs_idx', 'arc_id', 'true_arc']].copy()
    scored['prob'] = probs
    best = scored.loc[scored.groupby(['trip_id', 'obs_idx'])['prob'].idxmax()]
    return best.sort_values(['trip_id', 'obs_idx']).rename(columns={'arc_id': 'pred_arc'})


def route_continuity(best, arcs_by_trip):
    # share of consecutive predictions that stay on the same arc or move to a directly connected arc
    connected, total = 0, 0
    for tid, grp in best.groupby('trip_id'):
        from_nodes, to_nodes = arcs_by_trip[tid]
        a1, a2 = grp['pred_arc'].values[:-1], grp['pred_arc'].values[1:]
        connected += int(((a1 == a2) | (to_nodes[a1] == from_nodes[a2])).sum())
        total += len(a1)
    return connected / total if total else 0.0


def evaluate(df, probs, arcs_by_trip, threshold=0.5):
    # continuity uses every point; the other metrics only use points that have a reference label
    best_all = top1_predictions(df, probs)
    lab = (df['true_arc'] >= 0).values
    df, probs = df[lab], probs[lab]

    y_true = df['target'].values
    y_pred = (probs >= threshold).astype(int)

    best = top1_predictions(df, probs)
    obs_hit = df.groupby(['trip_id', 'obs_idx'])['target'].max()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'log_loss': log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1]),
        'top1_accuracy': (best['pred_arc'] == best['true_arc']).mean(),
        'candidate_miss_rate': 1 - obs_hit.mean(),
        'route_continuity': route_continuity(best_all, arcs_by_trip),
        'observations': len(best_all),
        'labelled_observations': len(best),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}, best_all
