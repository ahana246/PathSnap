# Reads the raw .track, .nodes, .arcs and .route files of a trip into pandas
import os
import numpy as np
import pandas as pd


def list_trip_ids(partition_dir):
    return sorted(d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d)))


def load_track(track_path):
    # columns: longitude, latitude, timestamp (s)
    return pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'],
                       dtype={'lon': np.float64, 'lat': np.float64, 'time': np.float64})


def load_nodes(nodes_path):
    # row index (0-based) is the node_id
    df = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'],
                     dtype={'lon': np.float64, 'lat': np.float64})
    df['node_id'] = df.index
    return df


def load_arcs(arcs_path):
    # row index (0-based) is the arc_id
    df = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['from_node', 'to_node'],
                     dtype={'from_node': np.int32, 'to_node': np.int32})
    df['arc_id'] = df.index
    return df


def load_route(route_path):
    # chronological list of reference arc_ids, None if the file is missing
    if not os.path.exists(route_path):
        return None
    df = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'], dtype={'arc_id': np.int32})
    return df['arc_id'].tolist()


def load_trip(partition_dir, trip_id):
    base = os.path.join(partition_dir, trip_id, trip_id)
    return {
        'trip_id': trip_id,
        'track': load_track(base + '.track'),
        'nodes': load_nodes(base + '.nodes'),
        'arcs': load_arcs(base + '.arcs'),
        'route': load_route(base + '.route'),
    }
