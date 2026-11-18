"""Compare classifier-only vs classifier + HMM/Viterbi on the student_test trips.

Usage: python scripts/evaluate_hmm.py [model_name]   (default: models/final_model.joblib or logreg)
Output: results/hmm_vs_classifier.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from pathsnap import config
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import reference_arcs
from pathsnap.metrics import pointwise_accuracy, route_continuity, route_set_scores
from pathsnap.pipeline import predict_trip


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
    rows = []
    for tid in list_trips(config.STUDENT_TEST_DIR):
        trip = load_trip(config.STUDENT_TEST_DIR, tid)
        out = predict_trip(model, trip)
        P = out["prepared"]
        ref = reference_arcs(P)   # evaluation only, after prediction
        for kind in ["classifier_pred", "hmm_pred"]:
            pred = out[kind]
            rows.append({"trip_id": tid, "method": kind.replace("_pred", ""),
                         "top1_acc": pointwise_accuracy(pred, ref),
                         "route_continuity": route_continuity(pred, trip.arcs),
                         **route_set_scores(pred, trip.route, P.arc_len),
                         "inference_s": out["inference_s"]})
        print(tid, {r["method"]: round(r["top1_acc"], 4) for r in rows[-2:]})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "hmm_vs_classifier.csv", index=False)
    print(df.groupby("method").mean(numeric_only=True).round(4))


if __name__ == "__main__":
    main()
