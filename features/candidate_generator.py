# Finds candidate road arcs near each GPS point using a KD-tree and computes their features
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.geo_utils import to_meters, point_to_segment_vectorized


class FastRoadNetworkIndex:
    def __init__(self, nodes_df, arcs_df, mean_lat, step_m=35.0):
        self.mean_lat = mean_lat
        node_x, node_y = to_meters(nodes_df['lon'].values, nodes_df['lat'].values, mean_lat)

        from_nodes = arcs_df['from_node'].values
        to_nodes = arcs_df['to_node'].values
        self.x1, self.y1 = node_x[from_nodes], node_y[from_nodes]
        self.x2, self.y2 = node_x[to_nodes], node_y[to_nodes]
        self.dx, self.dy = self.x2 - self.x1, self.y2 - self.y1
        self.seg_lengths = np.hypot(self.dx, self.dy)
        self.seg_headings = np.arctan2(self.dy, self.dx)  # radians in [-pi, pi]

        # sample points every step_m metres along each arc so long arcs are still found by the tree
        n_pts = np.maximum(2, np.ceil(self.seg_lengths / step_m).astype(int))
        arc_ids = np.repeat(np.arange(len(arcs_df)), n_pts)
        pos = np.arange(len(arc_ids)) - np.repeat(np.cumsum(n_pts) - n_pts, n_pts)
        frac = pos / (n_pts[arc_ids] - 1)
        samples = np.column_stack([self.x1[arc_ids] + frac * self.dx[arc_ids],
                                   self.y1[arc_ids] + frac * self.dy[arc_ids]])

        self.tree = cKDTree(samples)
        self.sample_arc_ids = arc_ids.astype(np.int32)

    def arcs_near(self, px, py, radius):
        # unique arc ids with a sample point within radius; falls back to the nearest arcs
        idx = self.tree.query_ball_point([px, py], r=radius)
        if len(idx) == 0:
            _, idx = self.tree.query([px, py], k=min(10, len(self.sample_arc_ids)))
        return np.unique(self.sample_arc_ids[np.atleast_1d(idx)])

    def get_candidates_for_point(self, lon, lat, gps_heading, gps_speed, dt, radius=50.0, max_k=8):
        px, py = to_meters(lon, lat, self.mean_lat)
        cand_arcs = self.arcs_near(px, py, radius)

        dists, t_params, seg_lens = point_to_segment_vectorized(
            px, py, self.x1[cand_arcs], self.y1[cand_arcs], self.x2[cand_arcs], self.y2[cand_arcs])

        # keep the max_k closest arcs
        order = np.argsort(dists)[:max_k]
        arcs_sel = cand_arcs[order]
        dists = dists[order]

        # circular difference between GPS heading and arc direction
        diff = gps_heading - self.seg_headings[arcs_sel]
        heading_diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))

        return pd.DataFrame({
            'arc_id': arcs_sel,
            'dist_to_segment': dists,
            'dist_to_from': np.hypot(px - self.x1[arcs_sel], py - self.y1[arcs_sel]),
            'dist_to_to': np.hypot(px - self.x2[arcs_sel], py - self.y2[arcs_sel]),
            'proj_ratio': t_params[order],
            'segment_length': seg_lens[order],
            'heading_diff': heading_diff,
            'sin_heading_diff': np.sin(heading_diff),
            'cos_heading_diff': np.cos(heading_diff),
            'gps_speed': gps_speed,
            'dt': dt,
            'dist_rank': np.arange(len(arcs_sel)),
            'dist_minus_best': dists - dists[0],
            'num_candidates': len(arcs_sel),
        })
