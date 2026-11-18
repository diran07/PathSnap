"""PathSnap FastAPI backend (runs on Laptop 1).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Env vars:
    PATHSNAP_EVAL_DIR  folder containing the evaluation trips (default: student_test)
    PATHSNAP_MODEL     model file name in models/ (default: final_model.joblib, else logreg.joblib)

Ground truth (.route) is read ONLY here, AFTER prediction, to compute accuracy.
It is never passed to the model and never returned to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from pathsnap import config
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import reference_arcs
from pathsnap.metrics import pointwise_accuracy, route_continuity, route_set_scores
from pathsnap.pipeline import predict_trip
from pathsnap.viz import arcs_to_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
log = logging.getLogger("pathsnap")

model_name = os.environ.get("PATHSNAP_MODEL")
if not model_name:
    model_name = "final_model.joblib" if (config.MODELS_DIR / "final_model.joblib").exists() else "logreg.joblib"
MODEL = joblib.load(config.MODELS_DIR / model_name)
log.info("Loaded model %s | eval dir %s", model_name, config.EVAL_DIR)

app = FastAPI(title="PathSnap API")


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s from %s -> %s (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code,
             time.perf_counter() - t0)
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.get("/trips")
def trips():
    return {"trips": list_trips(config.EVAL_DIR)}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(config.EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    trip = load_trip(config.EVAL_DIR, req.trip_id, with_route=False)  # model never sees the route
    out = predict_trip(MODEL, trip)
    P = out["prepared"]

    # server-side evaluation against the reference route
    metrics = {}
    gt = load_trip(config.EVAL_DIR, req.trip_id, with_route=True).route
    if gt is not None:
        ref = reference_arcs(P, route=gt)
        for kind in ["classifier", "hmm"]:
            pred = out[f"{kind}_pred"]
            metrics[kind] = {"top1_accuracy": pointwise_accuracy(pred, ref),
                             "route_continuity": route_continuity(pred, trip.arcs),
                             **route_set_scores(pred, gt, P.arc_len)}
    log.info("trip %s: %d points, HMM top-1 %.4f, %.2fs", req.trip_id, len(P.pts),
             metrics.get("hmm", {}).get("top1_accuracy", float("nan")), out["inference_s"])
    # descriptive stats from GPS + predictions only (no ground truth)
    hmm_pred, clf_pred = out["hmm_pred"], out["classifier_pred"]
    t = P.track["t"].values
    seq = hmm_pred[hmm_pred >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    summary = {
        "duration_s": float(t[-1] - t[0]),
        "gps_distance_km": float(np.linalg.norm(np.diff(P.pts, axis=0), axis=1).sum() / 1000),
        "predicted_length_km": float(P.arc_len[seq].sum() / 1000),
        "predicted_segments": int(len(seq)),
        "hmm_changed_points": float((hmm_pred != clf_pred).mean()),
        "candidates_per_point": float(len(out["candidates"]) / max(len(P.pts), 1)),
    }
    return {
        "trip_id": req.trip_id,
        "model": model_name,
        "summary": summary,
        "n_points": int(len(P.pts)),
        "gps": P.track[["lat", "lon"]].values.round(6).tolist(),
        "predicted_arcs": out["hmm_pred"].tolist(),
        "predicted_route": arcs_to_lines(trip.nodes, trip.arcs, out["hmm_pred"]),
        "classifier_route": arcs_to_lines(trip.nodes, trip.arcs, out["classifier_pred"]),
        "metrics": metrics,
        "inference_seconds": out["inference_s"],
    }
