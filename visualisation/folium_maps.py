# Builds interactive Folium maps of GPS points, nearby roads, reference route and predicted routes
import os
import folium
import numpy as np

from config import TRAIN_DIR, STUDENT_TEST_DIR, MAPS_DIR
from data_processing.data_loader import load_trip
from data_processing.geo_utils import to_meters
from features.candidate_generator import FastRoadNetworkIndex


def arc_lines(arc_ids, arcs, nodes):
    # list of [[lat, lon], [lat, lon]] segments for the given arcs
    lat, lon = nodes['lat'].values, nodes['lon'].values
    f, t = arcs['from_node'].values[arc_ids], arcs['to_node'].values[arc_ids]
    return [[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(f, t)]


def nearby_arcs(trip, radius=100.0, step=5):
    # road arcs within `radius` metres of the GPS track (every `step`-th point)
    track = trip['track']
    mean_lat = track['lat'].mean()
    index = FastRoadNetworkIndex(trip['nodes'], trip['arcs'], mean_lat)
    xs, ys = to_meters(track['lon'].values[::step], track['lat'].values[::step], mean_lat)
    return np.unique(np.concatenate([index.arcs_near(x, y, radius) for x, y in zip(xs, ys)]))


def add_legend(m, items):
    rows = ''.join(f'<i style="background:{c};width:18px;height:4px;display:inline-block;margin-right:6px;"></i>'
                   f'{label}<br>' for label, c in items)
    html = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:10px;'
            f'border:2px solid #ccc;border-radius:8px;font-size:12px;font-family:sans-serif;">'
            f'<b>PathSnap map matching</b><br>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def create_folium_map(trip, output_html, predicted_routes=None):
    """
    predicted_routes: optional {label: (list_of_arc_ids, colour)}, e.g. classifier-only and HMM routes
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    track, nodes, arcs, route = trip['track'], trip['nodes'], trip['arcs'], trip['route']

    m = folium.Map(tiles='OpenStreetMap', control_scale=True)
    m.fit_bounds([[track['lat'].min(), track['lon'].min()], [track['lat'].max(), track['lon'].max()]])
    legend = []

    # road network near the trajectory
    net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    folium.PolyLine(arc_lines(nearby_arcs(trip), arcs, nodes), color="#90A4AE", weight=1.5, opacity=0.6).add_to(net)
    net.add_to(m)
    legend.append(("Road network", "#90A4AE"))

    # reference route
    if route:
        gt = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(arc_lines(route, arcs, nodes), color="#2E7D32", weight=5, opacity=0.8).add_to(gt)
        gt.add_to(m)
        legend.append(("Reference route", "#2E7D32"))

    # predicted routes (consecutive duplicate arcs removed)
    for label, (pred_arcs, color) in (predicted_routes or {}).items():
        pred_arcs = [a for i, a in enumerate(pred_arcs) if i == 0 or a != pred_arcs[i - 1]]
        grp = folium.FeatureGroup(name=label, show=True)
        folium.PolyLine(arc_lines(pred_arcs, arcs, nodes), color=color, weight=3, dash_array="6, 6").add_to(grp)
        grp.add_to(m)
        legend.append((label, color))

    # raw GPS observations (markers downsampled for readability)
    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    coords = track[['lat', 'lon']].values.tolist()
    folium.PolyLine(coords, color="#1E88E5", weight=1.5, opacity=0.5).add_to(gps)
    for i in range(0, len(coords), max(1, len(coords) // 300)):
        folium.CircleMarker(coords[i], radius=3, color="#1565C0", fill=True, fill_opacity=0.8,
                            popup=f"GPS #{i} (t={track['time'].iat[i]:.0f}s)").add_to(gps)
    gps.add_to(m)
    legend.append(("Raw GPS observations", "#1E88E5"))

    add_legend(m, legend)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_html)
    print(f"[+] Saved {output_html}")


if __name__ == "__main__":
    # EDA maps: one training trip and one student_test trip
    create_folium_map(load_trip(TRAIN_DIR, '00000001'), str(MAPS_DIR / "eda_train_trip_00000001.html"))
    create_folium_map(load_trip(STUDENT_TEST_DIR, '00000000'), str(MAPS_DIR / "eda_student_test_trip_00000000.html"))
