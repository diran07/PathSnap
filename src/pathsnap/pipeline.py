"""End-to-end processing of one trip: candidates -> features -> model -> HMM."""
import json
import time

import numpy as np
import pandas as pd

from . import config
from .features import FEATURES, build_features
from .hmm import viterbi_decode
from .labels import reference_arcs
from .network import ShortestPaths, generate_candidates, prepare


def trip_table(trip, with_labels=True):
    """Feature table for one trip (used for training and offline evaluation)."""
    P = prepare(trip)
    cands = generate_candidates(P)
    feats = build_features(P, cands)
    ref = reference_arcs(P) if with_labels else np.full(len(P.pts), -1)
    feats["ref_arc"] = ref[feats["pt"].values]
    feats["y"] = (feats["arc"] == feats["ref_arc"]).astype(int)
    feats["labelled"] = feats["ref_arc"] >= 0
    feats["trip_id"] = trip.trip_id
    return P, cands, feats, ref


def score_candidates(model, feats: pd.DataFrame) -> pd.DataFrame:
    df = feats.copy()
    df["p_raw"] = model.predict_proba(df[FEATURES])[:, 1]
    s = df.groupby("pt")["p_raw"].transform("sum")
    df["p"] = df["p_raw"] / s.where(s > 0, 1)
    return df


def classifier_only(P, scored):
    top = scored.sort_values(["pt", "p"], ascending=[True, False]).drop_duplicates("pt")
    out = np.full(len(P.pts), -1, dtype=np.int64)
    out[top["pt"].values] = top["arc"].values
    return out


def load_hmm_params():
    """Tuned decoder parameters from models/hmm_params.json (written by scripts/tune_hmm.py),
    or the config defaults if the file does not exist."""
    f = config.MODELS_DIR / "hmm_params.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out = {k: d[k] for k in ("beta", "unreachable", "w_emit", "backward_tol") if k in d}
    if "backward_tol" in out and out["backward_tol"] is None:
        out["backward_tol"] = float("inf")
    return out


def predict_trip(model, trip, hmm_params=None):
    """Inference only: never touches trip.route."""
    if hmm_params is None:
        hmm_params = load_hmm_params()
    t0 = time.perf_counter()
    P = prepare(trip)
    cands = generate_candidates(P)
    feats = build_features(P, cands)
    scored = score_candidates(model, feats)
    clf = classifier_only(P, scored)
    hmm = viterbi_decode(P, scored, ShortestPaths(P), **hmm_params)
    return {"prepared": P, "candidates": cands, "scored": scored,
            "classifier_pred": clf, "hmm_pred": hmm,
            "inference_s": time.perf_counter() - t0}
