# Member 2: HMM over candidate arcs decoded with the Viterbi algorithm (log space)
import numpy as np

from data_processing.geo_utils import to_meters
from hmm.transition import RoadGraph, step_differences, transition_log


def build_steps(cands, probs, track, graph):
    """
    One step per GPS observation: its candidate arcs (hidden states), their position along the arc,
    the log emission (classifier probability) and the route/GPS differences to the previous step.
    """
    cands = cands.assign(prob=probs).sort_values(['obs_idx', 'dist_rank'])
    x, y = to_meters(track['lon'].values, track['lat'].values, track['lat'].mean())
    times = track['time'].values

    steps, diffs, prev = [], [], None
    for obs, grp in cands.groupby('obs_idx', sort=True):
        step = {'obs': obs,
                'arc': grp['arc_id'].values,
                'pos': (grp['proj_ratio'] * grp['segment_length']).values,
                'emission': np.log(np.clip(grp['prob'].values, 1e-6, 1.0))}
        if prev is not None:
            gps_step = float(np.hypot(x[obs] - x[prev['obs']], y[obs] - y[prev['obs']]))
            diffs.append(step_differences(graph, prev, step, gps_step, times[obs] - times[prev['obs']]))
        steps.append(step)
        prev = step
    return steps, diffs


def viterbi(steps, diffs, beta, unreachable_log):
    # score[j] = best log-probability of any path ending in candidate j of the current step
    score = steps[0]['emission'].copy()
    back = []
    for step, d in zip(steps[1:], diffs):
        if d is None:  # after a long gap every previous state leads equally to every current one
            trans = np.zeros((len(score), len(step['arc'])))
        else:
            trans = transition_log(d, beta, unreachable_log)
        total = score[:, None] + trans  # previous x current
        back.append(total.argmax(axis=0))
        score = total.max(axis=0) + step['emission']

    # trace the best path backwards
    best = [int(score.argmax())]
    for pointers in reversed(back):
        best.append(int(pointers[best[-1]]))
    best.reverse()
    return {s['obs']: int(s['arc'][k]) for s, k in zip(steps, best)}


def decode_trip(cands, probs, trip, hmm_cfg):
    """Returns {obs_idx: arc_id} for every GPS observation that has candidates"""
    graph = RoadGraph(trip['nodes'], trip['arcs'], trip['track']['lat'].mean())
    steps, diffs = build_steps(cands, probs, trip['track'], graph)
    return viterbi(steps, diffs, hmm_cfg['beta'], hmm_cfg['unreachable_log'])
