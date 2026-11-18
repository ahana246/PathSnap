# Member 2: road graph of a trip and the route-vs-GPS distance used for HMM transition probabilities
import heapq
import numpy as np

from config import HMM_GAP_SPLIT_S, HMM_SEARCH_SLACK_M
from data_processing.geo_utils import to_meters


class RoadGraph:
    def __init__(self, nodes, arcs, mean_lat):
        x, y = to_meters(nodes['lon'].values, nodes['lat'].values, mean_lat)
        self.from_node = arcs['from_node'].values
        self.to_node = arcs['to_node'].values
        self.length = np.hypot(x[self.to_node] - x[self.from_node], y[self.to_node] - y[self.from_node])

        # outgoing arcs of each node, so the graph follows one-way directions
        self.adj = [[] for _ in range(len(nodes))]
        for u, v, length in zip(self.from_node, self.to_node, self.length):
            self.adj[u].append((v, length))
        self.cache = {}  # source node -> (search limit, {node: shortest distance})

    def distances_from(self, source, limit):
        # Dijkstra from one node, stopping at `limit` metres; reused for later steps when possible
        if source in self.cache and self.cache[source][0] >= limit:
            return self.cache[source][1]
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, length in self.adj[u]:
                nd = d + length
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self.cache[source] = (limit, dist)
        return dist

    def route_distance(self, a, pos_a, b, pos_b, limit):
        # driving distance from position pos_a on arc a to pos_b on arc b; None if not reachable within limit
        if a == b:
            return abs(pos_b - pos_a)
        d = self.distances_from(self.to_node[a], limit).get(self.from_node[b])
        if d is None:
            return None
        return (self.length[a] - pos_a) + d + pos_b


def step_differences(graph, prev, curr, gps_step_m, dt):
    """
    |route distance - GPS straight-line distance| for every (previous candidate, current candidate) pair.
    NaN = not reachable on the road network. Returns None after a long time gap (new segment).
    """
    if dt > HMM_GAP_SPLIT_S:
        return None
    limit = gps_step_m + HMM_SEARCH_SLACK_M
    diffs = np.full((len(prev['arc']), len(curr['arc'])), np.nan)
    for i, (a, pa) in enumerate(zip(prev['arc'], prev['pos'])):
        for j, (b, pb) in enumerate(zip(curr['arc'], curr['pos'])):
            d = graph.route_distance(a, pa, b, pb, limit)
            if d is not None:
                diffs[i, j] = abs(d - gps_step_m)
    return diffs


def transition_log(diffs, beta, unreachable_log):
    # exponential decay with the route/GPS mismatch; unreachable moves keep a small escape probability
    return np.where(np.isnan(diffs), unreachable_log, -diffs / beta)
