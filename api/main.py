# Member 4: FastAPI backend (laptop 1): runs the full PathSnap pipeline for the trip chosen in the UI
import argparse
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from config import EVALUATOR_TEST_DIR
from data_processing.data_loader import list_trip_ids
from hmm.evaluate_hmm import score_trip
from hmm.pipeline import load_artifacts, predict_trip

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pathsnap')

app = FastAPI(title='PathSnap API')
DATA_DIR = EVALUATOR_TEST_DIR  # replaced by --data-dir when started from the command line
ARTIFACTS = load_artifacts()   # model, feature list and HMM settings, loaded once at start-up


def route_lines(arcs, trip):
    # predicted arcs as [[lat, lon], [lat, lon]] segments, consecutive repeats removed
    nodes, f, t = trip['nodes'], trip['arcs']['from_node'].values, trip['arcs']['to_node'].values
    lat, lon = nodes['lat'].values, nodes['lon'].values
    unique = [a for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    return [[[lat[f[a]], lon[f[a]]], [lat[t[a]], lon[t[a]]]] for a in unique]


@app.get('/trips')
def trips():
    log.info(f"GET /trips -> {DATA_DIR.name}")
    return {'data_dir': DATA_DIR.name, 'trips': list_trip_ids(DATA_DIR)}


@app.post('/predict/{trip_id}')
def predict(trip_id: str):
    if trip_id not in list_trip_ids(DATA_DIR):
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in {DATA_DIR.name}")

    start = time.perf_counter()
    trip, preds = predict_trip(DATA_DIR, trip_id, ARTIFACTS)
    scores = score_trip(trip, preds)  # reference route stays on this laptop; only the scores are returned
    runtime = round(time.perf_counter() - start, 2)

    log.info(f"POST /predict/{trip_id} | {len(preds)} GPS points | "
             f"HMM top-1 {scores['hmm_top1']:.3f} (classifier only {scores['classifier_top1']:.3f}) | "
             f"continuity {scores['hmm_continuity']:.3f} | {runtime} s" if scores else
             f"POST /predict/{trip_id} | {len(preds)} GPS points | no reference route | {runtime} s")
    return {
        'trip_id': trip_id,
        'num_points': len(preds),
        'gps': preds[['lat', 'lon']].values.tolist(),
        'predicted_route': route_lines(preds['hmm_arc'].tolist(), trip),
        'scores': scores,
        'runtime_s': runtime,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(EVALUATOR_TEST_DIR))
    parser.add_argument('--host', default='0.0.0.0')  # reachable from the UI laptop on the same network
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir).resolve()
    log.info(f"Serving {len(list_trip_ids(DATA_DIR))} trips from {DATA_DIR}")
    uvicorn.run(app, host=args.host, port=args.port)
