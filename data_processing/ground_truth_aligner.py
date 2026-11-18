# Builds a per-GPS-point label by aligning the track to the reference route in order
import numpy as np
from data_processing.geo_utils import to_meters, point_to_segment_vectorized


def align_track_to_route(trip_data, search_window=20):
    """
    Walks along the route while walking along the track: each GPS point is matched to the
    nearest route arc within the next `search_window` arcs, so labels never move backwards.
    Returns a list with one reference arc_id per GPS observation.
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
        curr_r += int(np.argmin(dists))
        aligned.append(route[curr_r])
    return aligned
