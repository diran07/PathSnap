"""Tune the HMM decoder parameters on the TRAIN trips, then report on student_test.

Why out-of-fold scores: the saved classifier was fitted on all 80 train trips, so
its probabilities on those trips are over-confident. To tune the decoder on
realistic emissions, every train trip is scored by a copy of the model that was
fitted WITHOUT that trip (5-fold GroupKFold by trip_id).

Parameters searched (grid):
  beta         transition scale (m per second of gap). Small = trust network distance strongly.
  unreachable  log-prob given to transitions that are not reachable within the Dijkstra limit.
The emission weight is fixed at 1: multiplying every log-term by the same constant
does not change the Viterbi argmax, so only the ratio to the emission term matters.

Objective: observation-level Top-1 accuracy over labelled points, pooled over all
train trips. Student_test is only used to report the final comparison.

Outputs
  results/hmm_tuning_grid.csv      every configuration with train OOF metrics
  results/hmm_tuning_summary.csv   student_test: classifier-only vs HMM default vs HMM tuned
  results/plots/hmm_tuning_heatmap.png
  models/hmm_params.json           best parameters (loaded by the pipeline / API)

Usage: python scripts/tune_hmm.py [model_name]     (default: logreg)
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.hmm import build_lattice, decode
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import reference_arcs
from pathsnap.metrics import pointwise_accuracy, route_continuity, route_set_scores
from pathsnap.network import generate_candidates, prepare
from pathsnap.pipeline import classifier_only

# Search history (kept for the report):
#   run 1: beta [1..35], unreachable [-10..-50], backward_tol [inf, 5, 15, 40]
#          -> best at the grid edge (beta=1, unreachable=-50); backward_tol 5/15 clearly worse,
#             40 tied with inf. So the grid was extended towards small beta / very negative
#             unreachable, and the poor tolerances dropped.
BETAS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8]
UNREACHABLE = [-20, -30, -50, -80, -120, -200]
BACKWARD_TOL = [np.inf, 40, 80]      # inf = original model (backward moves on an arc are free)
TIE_MARGIN = 0.0005                  # within this of the best, prefer the simpler model (tol = inf)
DEFAULT = {"beta": config.HMM_BETA, "unreachable": config.HMM_UNREACHABLE_LOGP,
           "backward_tol": np.inf}


def normalise(df, p):
    df = df[["pt", "arc", "frac"]].copy()
    df["p"] = p
    s = df.groupby("pt")["p"].transform("sum")
    df["p"] = df["p"] / s.where(s > 0, 1)
    return df


def oof_scores(model):
    """Out-of-fold candidate probabilities for every train trip."""
    df = pd.read_parquet(config.PROCESSED_DIR / "train_features.parquet")
    p = np.zeros(len(df))
    for a, b in GroupKFold(n_splits=5).split(df, groups=df["trip_id"]):
        fit_rows = df.iloc[a]
        fit_rows = fit_rows[fit_rows["labelled"]]
        m = clone(model).fit(fit_rows[FEATURES], fit_rows["y"])
        p[b] = m.predict_proba(df.iloc[b][FEATURES])[:, 1]
    df["p_oof"] = p
    return {tid: g for tid, g in df.groupby("trip_id")}


def trip_lattice(split_dir, tid, scored_rows):
    """Lattice + reference labels for one trip. scored_rows must hold pt, arc, frac, p."""
    trip = load_trip(split_dir, tid)
    P = prepare(trip)
    L = build_lattice(P, scored_rows)
    ref = reference_arcs(P)
    clf = classifier_only(P, scored_rows)
    return {"tid": tid, "L": L, "ref": ref, "clf": clf, "arcs": trip.arcs,
            "route": trip.route, "arc_len": P.arc_len}


def evaluate(items, params):
    correct = total = 0
    cont = []
    per_trip = {}
    for it in items:
        pred = decode(it["L"], **params)
        m = it["ref"] >= 0
        correct += int((pred[m] == it["ref"][m]).sum())
        total += int(m.sum())
        cont.append(route_continuity(pred, it["arcs"]))
        per_trip[it["tid"]] = pred
    return correct / total, float(np.mean(cont)), per_trip


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    model = joblib.load(config.MODELS_DIR / f"{name}.joblib")

    t0 = time.perf_counter()
    oof = oof_scores(model)
    print(f"OOF scores done ({time.perf_counter() - t0:.0f}s)")

    t0 = time.perf_counter()
    train_items = Parallel(n_jobs=4)(
        delayed(trip_lattice)(config.TRAIN_DIR, tid, normalise(g, g["p_oof"].values))
        for tid, g in oof.items())
    print(f"Train lattices built ({time.perf_counter() - t0:.0f}s)")

    clf_acc = sum(int((it["clf"][it["ref"] >= 0] == it["ref"][it["ref"] >= 0]).sum()) for it in train_items) \
        / sum(int((it["ref"] >= 0).sum()) for it in train_items)

    rows = []
    for beta, unr, tol in itertools.product(BETAS, UNREACHABLE, BACKWARD_TOL):
        t0 = time.perf_counter()
        acc, cont, _ = evaluate(train_items, {"beta": beta, "unreachable": unr, "backward_tol": tol})
        rows.append({"beta": beta, "unreachable": unr, "backward_tol": tol, "train_oof_top1": acc,
                     "train_oof_continuity": cont, "decode_s": time.perf_counter() - t0})
        print(f"beta={beta:5} unreachable={unr:4} backward_tol={tol:5}  top1={acc:.4f}  continuity={cont:.4f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(config.RESULTS_DIR / "hmm_tuning_grid.csv", index=False)
    top = grid["train_oof_top1"].max()
    near = grid[grid["train_oof_top1"] >= top - TIE_MARGIN]
    simple = near[np.isinf(near["backward_tol"])]
    best = (simple if len(simple) else near).sort_values("train_oof_top1").iloc[-1]
    best_params = {"beta": float(best["beta"]), "unreachable": float(best["unreachable"]),
                   "backward_tol": float(best["backward_tol"])}
    print(f"\nTrain OOF classifier-only top1: {clf_acc:.4f}")
    print(f"Default on train OOF: top1="
          f"{grid[(grid.beta == DEFAULT['beta']) & (grid.unreachable == DEFAULT['unreachable']) & np.isinf(grid.backward_tol)].train_oof_top1.iloc[0]:.4f}")
    print(f"Best on train OOF: {best_params}  top1={best['train_oof_top1']:.4f}")
    print("\nBest top1 per backward_tol (train OOF):")
    print(grid.groupby("backward_tol")["train_oof_top1"].max().round(4))

    # heatmaps: beta x unreachable, one panel per backward tolerance
    fig, axes = plt.subplots(1, len(BACKWARD_TOL), figsize=(5 * len(BACKWARD_TOL), 3.8), sharey=True)
    vmin, vmax = grid.train_oof_top1.min(), grid.train_oof_top1.max()
    for ax, tol in zip(axes, BACKWARD_TOL):
        piv = grid[grid.backward_tol == tol].pivot(index="unreachable", columns="beta", values="train_oof_top1")
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        for (r, c), v in np.ndenumerate(piv.values):
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=6, color="w")
        ax.set_xlabel("beta")
        ax.set_title("backward moves free" if np.isinf(tol) else f"backward_tol = {tol:g} m")
    axes[0].set_ylabel("unreachable log-prob")
    fig.suptitle(f"HMM Top-1 accuracy on train trips (out-of-fold {name} scores)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(config.RESULTS_DIR / "plots" / "hmm_tuning_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # final report on student_test with the model fitted on all train trips
    test_feats = pd.read_parquet(config.PROCESSED_DIR / "student_test_features.parquet")
    test_items = [trip_lattice(config.STUDENT_TEST_DIR, tid,
                               normalise(g, model.predict_proba(g[FEATURES])[:, 1]))
                  for tid, g in test_feats.groupby("trip_id")]
    summary = []
    for label, params in [("classifier_only", None), ("hmm_default", DEFAULT), ("hmm_tuned", best_params)]:
        for it in test_items:
            pred = it["clf"] if params is None else decode(it["L"], **params)
            summary.append({"method": label, "trip_id": it["tid"],
                            "top1_acc": pointwise_accuracy(pred, it["ref"]),
                            "route_continuity": route_continuity(pred, it["arcs"]),
                            **route_set_scores(pred, it["route"], it["arc_len"])})
    summ = pd.DataFrame(summary)
    summ.to_csv(config.RESULTS_DIR / "hmm_tuning_summary.csv", index=False)
    print("\nstudent_test (mean over 10 trips):")
    print(summ.groupby("method")[["top1_acc", "route_continuity", "route_f1"]].mean().round(4))
    print("\nper trip top1:")
    print(summ.pivot(index="trip_id", columns="method", values="top1_acc").round(4))

    saved = {**best_params, "w_emit": 1.0, "tuned_on": "train out-of-fold", "model": name}
    if np.isinf(saved["backward_tol"]):
        saved["backward_tol"] = None   # JSON has no infinity; None = disabled
    json.dump(saved, open(config.MODELS_DIR / "hmm_params.json", "w"), indent=2)
    print("\nSaved models/hmm_params.json")


if __name__ == "__main__":
    main()
