"""Check a running student-test API against a saved classifier/HMM evaluation."""
import argparse
import csv
import math
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name without the .joblib extension")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    filename = "hmm_vs_classifier.csv" if args.model == "logreg" else f"{args.model}_hmm_vs_classifier.csv"
    results = Path(__file__).resolve().parents[1] / "results" / filename
    with results.open() as stream:
        reference = {(row["trip_id"], row["method"]): row for row in csv.DictReader(stream)}

    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    assert health.json() == {"status": "ok", "model": f"{args.model}.joblib"}, health.text
    trips = requests.get(f"{base}/trips", timeout=10)
    trips.raise_for_status()
    trip_ids = trips.json()["trips"]
    assert sorted(trip_ids) == sorted({tid for tid, _ in reference}), trip_ids

    for body, expected in [({"trip_id": "unknown-trip"}, 404), ({}, 422)]:
        response = requests.post(f"{base}/predict", json=body, timeout=10)
        assert response.status_code == expected, (response.status_code, response.text)

    for tid in trip_ids:
        response = requests.post(f"{base}/predict", json={"trip_id": tid}, timeout=120)
        response.raise_for_status()
        out = response.json()
        assert out["trip_id"] == tid
        assert len(out["predicted_arcs"]) == out["n_points"] == len(out["gps"])
        assert out["classifier_route"] and out["predicted_route"]
        assert min(out["predicted_arcs"]) >= 0
        for method in ("classifier", "hmm"):
            expected = reference[tid, method]
            for key in ("top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1"):
                column = "top1_acc" if key == "top1_accuracy" else key
                actual = out["metrics"][method][key]
                assert math.isclose(actual, float(expected[column]), abs_tol=1e-12), (tid, method, key, actual)
    print(f"Verified health, trip list, error responses, and predictions for {len(trip_ids)} trips.")


if __name__ == "__main__":
    main()
