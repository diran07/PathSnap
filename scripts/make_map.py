"""Offline Folium map for a student_test trip: GPS, reference route, classifier-only
and HMM predictions. Output: results/maps/<trip_id>.html

Usage: python scripts/make_map.py 00000077 [model_name]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from pathsnap import config
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip
from pathsnap.viz import arcs_to_lines, route_map


def main():
    tid = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "logreg"
    trip = load_trip(config.STUDENT_TEST_DIR, tid)
    out = predict_trip(joblib.load(config.MODELS_DIR / f"{name}.joblib"), trip)
    tr = out["prepared"].track
    nearby = out["candidates"]["arc"].unique()
    m = route_map(
        tr[["lat", "lon"]].values,
        {"Reference route": (arcs_to_lines(trip.nodes, trip.arcs, trip.route), "#2ca02c"),
         "Classifier only": (arcs_to_lines(trip.nodes, trip.arcs, out["classifier_pred"]), "#ff7f0e"),
         "HMM / Viterbi": (arcs_to_lines(trip.nodes, trip.arcs, out["hmm_pred"]), "#d62728")},
        network_lines=arcs_to_lines(trip.nodes, trip.arcs, nearby))
    path = config.RESULTS_DIR / "maps" / f"{tid}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    print("saved", path)


if __name__ == "__main__":
    main()
