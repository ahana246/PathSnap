# Generates EDA plots: trip sizes, sampling gaps, trajectories, road network and GPS noise
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from config import EDA_DIR, EDA_PLOTS_DIR, FEATURES_DIR, TRAIN_DIR, CANDIDATE_RADIUS_M
from data_processing.data_loader import list_trip_ids, load_track, load_trip

SAMPLE_TRIP = '00000001'  # training trip used for single-trip plots


def save(name):
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / name, dpi=150)
    plt.close()
    print(f"[+] Saved {EDA_PLOTS_DIR / name}")


def plot_observations_per_trip(eda):
    plt.figure(figsize=(8, 4.5))
    for part, color in [('train', '#1976D2'), ('student_test', '#388E3C')]:
        obs = eda.loc[eda['partition'] == part, 'num_gps_observations']
        plt.hist(obs, bins=25, alpha=0.7, color=color, label=f'{part} ({len(obs)} trips, mean={obs.mean():.0f})')
    plt.title('GPS observations per trip (train + student_test)')
    plt.xlabel('Number of GPS observations')
    plt.ylabel('Number of trips')
    plt.legend()
    save("gps_observations_per_trip.png")


def plot_sampling_consistency(eda):
    parts = ['train', 'student_test']
    plt.figure(figsize=(8, 4.5))
    plt.boxplot([eda.loc[eda['partition'] == p, 'pct_exact_1s'] for p in parts], tick_labels=parts)
    plt.title('Share of steps exactly 1 s apart per trip')
    plt.ylabel('% of steps with dt = 1 s')
    save("sampling_rate_boxplot.png")


def plot_time_gaps():
    # pooled time gaps of all training trips, log scale shows the rare long dropouts
    gaps = np.concatenate([load_track(TRAIN_DIR / t / f"{t}.track")['time'].diff().dropna().values
                           for t in list_trip_ids(TRAIN_DIR)])
    plt.figure(figsize=(8, 4.5))
    plt.hist(np.clip(gaps, 0, 60), bins=60, color='#5C6BC0')
    plt.yscale('log')
    plt.title('Time gap between consecutive GPS points (train, clipped at 60 s)')
    plt.xlabel('Time gap (s)')
    plt.ylabel('Count (log scale)')
    save("time_gap_distribution.png")


def plot_network_size(eda):
    plt.figure(figsize=(8, 4.5))
    for part, color in [('train', '#0288D1'), ('student_test', '#388E3C')]:
        sub = eda[eda['partition'] == part]
        plt.scatter(sub['num_nodes'], sub['num_arcs'], c=color, alpha=0.7, edgecolors='black', label=part)
    plt.title('Road network size per trip')
    plt.xlabel('Nodes')
    plt.ylabel('Directed arcs')
    plt.legend()
    save("road_network_size.png")


def plot_connectivity(eda):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(eda['weak_components'], bins=20, color='#26A69A')
    axes[0].set_title('Weakly connected components per trip')
    axes[0].set_xlabel('Components')
    axes[0].set_ylabel('Trips')
    axes[1].hist(eda['largest_weak_component_pct'], bins=20, color='#FFA726')
    axes[1].set_title('Nodes in largest component (%)')
    axes[1].set_xlabel('% of nodes')
    save("road_network_connectivity.png")


def plot_trajectories(eda):
    # four training trips of different lengths to show trajectory shapes
    trips = eda[eda['partition'] == 'train'].sort_values('num_gps_observations')['trip_id']
    picks = trips.iloc[np.linspace(0, len(trips) - 1, 4).astype(int)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, tid in zip(axes.ravel(), picks):
        track = load_track(TRAIN_DIR / tid / f"{tid}.track")
        sc = ax.scatter(track['lon'], track['lat'], c=track['time'] - track['time'].iloc[0], s=3, cmap='viridis')
        ax.set_title(f"Trip {tid} ({len(track)} points)")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.tick_params(axis='x', labelrotation=30)  # long longitude labels overlap otherwise
        fig.colorbar(sc, ax=ax, label='Time since start (s)')
    save("gps_trajectories.png")


def plot_road_network(trip_id=SAMPLE_TRIP):
    trip = load_trip(TRAIN_DIR, trip_id)
    track, nodes, arcs = trip['track'], trip['nodes'], trip['arcs']
    pad = 0.003
    in_box = ((nodes['lon'] > track['lon'].min() - pad) & (nodes['lon'] < track['lon'].max() + pad) &
              (nodes['lat'] > track['lat'].min() - pad) & (nodes['lat'] < track['lat'].max() + pad)).values
    sub = arcs[in_box[arcs['from_node']] & in_box[arcs['to_node']]]

    def segments(a):
        f, t = a['from_node'].values, a['to_node'].values
        return np.stack([nodes[['lon', 'lat']].values[f], nodes[['lon', 'lat']].values[t]], axis=1)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(segments(sub), colors='#B0BEC5', linewidths=0.6, label='Road network'))
    ax.add_collection(LineCollection(segments(arcs.loc[trip['route']]), colors='#2E7D32', linewidths=2.5,
                                     label='Reference route'))
    ax.scatter(track['lon'], track['lat'], s=3, c='#1E88E5', label='GPS observations')
    ax.autoscale()
    ax.set_title(f"Road network, reference route and GPS (train trip {trip_id})")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    save(f"road_network_trip_{trip_id}.png")


def plot_gps_noise():
    # distance from each GPS point to its reference arc, used to justify the candidate radius
    path = FEATURES_DIR / "train_features.parquet"
    if not path.exists():
        print("Skipping GPS noise plot: run features/build_features.py first")
        return
    dist = pd.read_parquet(path, columns=['target', 'dist_to_segment']).query('target == 1')['dist_to_segment']
    plt.figure(figsize=(8, 4.5))
    plt.hist(dist.clip(upper=60), bins=60, color='#EF5350')
    plt.axvline(CANDIDATE_RADIUS_M, color='black', linestyle='--', label=f'Candidate radius ({CANDIDATE_RADIUS_M:.0f} m)')
    plt.title('Distance from GPS point to its reference arc (train)')
    plt.xlabel('Distance (m), clipped at 60')
    plt.ylabel('GPS observations')
    plt.legend()
    save("gps_noise_distance.png")


def main():
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    eda = pd.read_csv(EDA_DIR / "eda_summary.csv", dtype={'trip_id': str})
    plot_observations_per_trip(eda)
    plot_sampling_consistency(eda)
    plot_time_gaps()
    plot_network_size(eda)
    plot_connectivity(eda)
    plot_trajectories(eda)
    plot_road_network()
    plot_gps_noise()


if __name__ == "__main__":
    main()
