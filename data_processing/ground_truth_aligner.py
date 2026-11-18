# Builds a per-GPS-point label by aligning the track to the reference route in order
import numpy as np
from config import LABEL_MAX_DIST_M
from data_processing.geo_utils import to_meters, point_to_segment_vectorized


def align_track_to_route(trip_data, search_window=20):
    """
    Walks along the route while walking along the track: each GPS point is matched to the
    nearest route arc within the next `search_window` arcs, so labels normally only move forward.
    If no arc in the window is close (e.g. after a GPS gap), the whole route is searched.
    Points with no route arc within LABEL_MAX_DIST_M get -1 (unlabelled, route does not cover them).
    Returns a list with one reference arc_id (or -1) per GPS observation.
    """
    track, nodes, arcs, route = trip_data['track'], trip_data['nodes'], trip_data['arcs'], trip_data['route']
    if route is None or len(route) == 0:
        return None

    mean_lat = track['lat'].mean()
    node_x, node_y = to_meters(nodes['lon'].values, nodes['lat'].values, mean_lat)
    px_all, py_all = to_meters(track['lon'].values, track['lat'].values, mean_lat)

    # coordinates of each route arc's end nodes
    u = arcs['from_node'].values[route]
    v = arcs['to_node'].values[route]
    rx1, ry1, rx2, ry2 = node_x[u], node_y[u], node_x[v], node_y[v]

    curr_r = 0
    aligned = []
    for px, py in zip(px_all, py_all):
        end = min(len(route), curr_r + search_window)
        dists, _, _ = point_to_segment_vectorized(px, py, rx1[curr_r:end], ry1[curr_r:end],
                                                  rx2[curr_r:end], ry2[curr_r:end])
        if dists.min() <= LABEL_MAX_DIST_M:
            curr_r += int(np.argmin(dists))
            aligned.append(route[curr_r])
            continue

        # window failed (GPS gap or the aligner ran ahead): among all close route arcs,
        # take the one nearest in route order to the current position
        all_d, _, _ = point_to_segment_vectorized(px, py, rx1, ry1, rx2, ry2)
        close = np.flatnonzero(all_d <= LABEL_MAX_DIST_M)
        if len(close) == 0:
            aligned.append(-1)  # the route does not cover this point
            continue
        curr_r = int(close[np.argmin(np.abs(close - curr_r))])
        aligned.append(route[curr_r])
    return aligned
