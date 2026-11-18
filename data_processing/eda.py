# Dataset exploration: per-trip inventory, GPS quality, road-network connectivity and route checks
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config import PARTITIONS, EDA_DIR, LARGE_GAP_S
from data_processing.data_loader import list_trip_ids, load_trip
from data_processing.geo_utils import to_meters

MAX_PLAUSIBLE_SPEED = 60.0  # m/s (216 km/h), faster jumps are treated as GPS outliers


def file_info(partition_dir, trip_id):
    # availability and size (KB) of each raw file
    info = {}
    for ext in ['track', 'nodes', 'arcs', 'route']:
        path = os.path.join(partition_dir, trip_id, f"{trip_id}.{ext}")
        info[f'has_{ext}'] = os.path.exists(path)
        info[f'{ext}_kb'] = round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0
    return info


def track_stats(track):
    dt = track['time'].diff().dropna()
    x, y = to_meters(track['lon'].values, track['lat'].values, track['lat'].mean())
    step_m = np.hypot(np.diff(x), np.diff(y))
    speed = step_m / np.maximum(dt.values, 0.1)

    invalid = ((track['lon'].abs() > 180) | (track['lat'].abs() > 90) |
               ((track['lon'] == 0) & (track['lat'] == 0))).sum()
    return {
        'num_gps_observations': len(track),
        'nan_count': int(track.isna().sum().sum()),
        'invalid_coords': int(invalid),
        'duplicate_timestamps': int((dt == 0).sum()),
        'non_increasing_time': int((dt < 0).sum()),
        'duplicate_positions': int(track.duplicated(subset=['lon', 'lat']).sum()),
        'mean_dt_sec': round(float(dt.mean()), 3) if len(dt) else 0,
        'max_dt_sec': round(float(dt.max()), 3) if len(dt) else 0,
        'pct_exact_1s': round(float((np.abs(dt - 1.0) < 0.05).mean() * 100), 2) if len(dt) else 0,
        'num_large_gaps': int((dt > LARGE_GAP_S).sum()),
        'num_speed_outliers': int((speed > MAX_PLAUSIBLE_SPEED).sum()),
        'duration_min': round(float(track['time'].iloc[-1] - track['time'].iloc[0]) / 60, 1),
        'track_length_km': round(float(step_m.sum()) / 1000, 2),
        'min_lon': track['lon'].min(), 'max_lon': track['lon'].max(),
        'min_lat': track['lat'].min(), 'max_lat': track['lat'].max(),
    }


def network_stats(nodes, arcs):
    n = len(nodes)
    graph = coo_matrix((np.ones(len(arcs)), (arcs['from_node'], arcs['to_node'])), shape=(n, n))
    n_weak, weak_labels = connected_components(graph, directed=True, connection='weak')
    n_strong, _ = connected_components(graph, directed=True, connection='strong')

    used = np.zeros(n, dtype=bool)
    used[arcs['from_node']] = True
    used[arcs['to_node']] = True

    # share of arcs whose reverse arc also exists (two-way roads)
    pairs = set(zip(arcs['from_node'], arcs['to_node']))
    two_way = sum((v, u) in pairs for u, v in pairs) / max(len(pairs), 1)

    return {
        'num_nodes': n,
        'num_arcs': len(arcs),
        'isolated_nodes': int((~used).sum()),
        'weak_components': int(n_weak),
        'largest_weak_component_pct': round(np.bincount(weak_labels).max() / n * 100, 2),
        'strong_components': int(n_strong),
        'pct_two_way_arcs': round(two_way * 100, 2),
        'self_loop_arcs': int((arcs['from_node'] == arcs['to_node']).sum()),
    }


def route_stats(route, arcs):
    if route is None:
        return {'route_arc_count': 0, 'route_invalid_ids': 0, 'route_breaks': 0, 'route_repeated_arcs': 0}
    route = np.array(route)
    valid = route[route < len(arcs)]
    to_nodes = arcs['to_node'].values[valid]
    from_nodes = arcs['from_node'].values[valid]
    # a break means arc i does not end where arc i+1 starts
    breaks = int((to_nodes[:-1] != from_nodes[1:]).sum())
    return {
        'route_arc_count': len(route),
        'route_invalid_ids': int(len(route) - len(valid)),
        'route_breaks': breaks,
        'route_repeated_arcs': int(len(route) - len(np.unique(route))),
    }


def run_dataset_exploration():
    os.makedirs(EDA_DIR, exist_ok=True)
    records = []

    for part_name, part_dir in PARTITIONS.items():
        trip_ids = list_trip_ids(part_dir)
        print(f"Partition [{part_name}]: {len(trip_ids)} trips")

        for tid in trip_ids:
            trip = load_trip(part_dir, tid)
            rec = {'partition': part_name, 'trip_id': tid}
            rec.update(file_info(part_dir, tid))
            rec.update(track_stats(trip['track']))
            rec.update(network_stats(trip['nodes'], trip['arcs']))
            rec.update(route_stats(trip['route'], trip['arcs']))
            records.append(rec)

    df = pd.DataFrame(records)
    out_path = EDA_DIR / "eda_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[+] Saved {out_path}")

    # overall summary per partition
    for part in PARTITIONS:
        sub = df[df['partition'] == part]
        print(f"\n--- {part} ({len(sub)} trips) ---")
        print(f"GPS observations per trip: mean={sub['num_gps_observations'].mean():.0f}, "
              f"min={sub['num_gps_observations'].min()}, max={sub['num_gps_observations'].max()}")
        print(f"Steps exactly 1 s apart: {sub['pct_exact_1s'].mean():.2f}%")
        print(f"Max time gap: {sub['max_dt_sec'].max():.0f} s | trips with gaps > {LARGE_GAP_S:.0f} s: "
              f"{(sub['num_large_gaps'] > 0).sum()}")
        print(f"NaN values: {sub['nan_count'].sum()} | invalid coords: {sub['invalid_coords'].sum()} | "
              f"speed outliers: {sub['num_speed_outliers'].sum()}")
        print(f"Road network: mean {sub['num_nodes'].mean():.0f} nodes, {sub['num_arcs'].mean():.0f} arcs, "
              f"{sub['isolated_nodes'].sum()} isolated nodes in total")
        print(f"Route arcs: mean {sub['route_arc_count'].mean():.0f} | trips with route breaks: "
              f"{(sub['route_breaks'] > 0).sum()}")
    return df


if __name__ == "__main__":
    run_dataset_exploration()
