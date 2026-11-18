"""Train and compare candidate-scoring models.

- 5-fold GroupKFold on the 80 training trips (groups = trip_id, so no trip is
  split across folds).
- Evaluation on the 10 student_test trips.
- Every run is appended to results/experiments.csv.

Usage:
    python scripts/train_models.py                 # all models in MODELS
    python scripts/train_models.py logreg          # one model

    python scripts/train_models.py random_forest   # train the forest only
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics

# name -> (family, factory). Factories return an unfitted sklearn-style estimator
# with predict_proba.
MODELS = {
    "logreg": ("Linear", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=config.SEED))),
    "random_forest": ("Ensemble", lambda: RandomForestClassifier(
        n_estimators=48, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", bootstrap=True, max_samples=0.25,
        n_jobs=4, random_state=config.SEED,
    )),
    # "lightgbm":      ("Boosting", lambda: LGBMClassifier(...)),
    # "knn":           ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(...))),
    # "mlp":           ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(...))),
}


def load(split):
    df = pd.read_parquet(config.PROCESSED_DIR / f"{split}_features.parquet")
    # only points with a reference label can be used for supervised training
    return df[df["labelled"]].reset_index(drop=True)


def score_observations(model, df):
    d = df[["trip_id", "pt", "y"]].copy()
    d["pt"] = d["trip_id"] + "_" + d["pt"].astype(str)
    p = model.predict_proba(df[FEATURES])[:, 1]
    s = pd.Series(p).groupby(d["pt"].values).transform("sum").values
    d["p"] = p / np.where(s > 0, s, 1)
    return d


def evaluate(model, df):
    d = score_observations(model, df)
    m = candidate_metrics(d)
    # observation-level top-1 accuracy
    top = d.loc[d.groupby("pt")["p"].idxmax()]
    npts = d["pt"].nunique()
    m["top1_acc"] = float(top["y"].sum() / npts)
    return m


def run(name):
    family, make = MODELS[name]
    tr, te = load("train"), load("student_test")
    X, y, g = tr[FEATURES], tr["y"], tr["trip_id"]

    folds = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        m = make().fit(X.iloc[a], y.iloc[a])
        folds.append(evaluate(m, tr.iloc[b]))
        print(f"{name} fold {k + 1}/5: macro F1 {folds[-1]['macro_f1']:.4f}", flush=True)
    cv = pd.DataFrame(folds)

    t0 = time.perf_counter()
    model = make().fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = evaluate(model, te)
    infer_s = time.perf_counter() - t0

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": name,
           "family": family, "config": json.dumps(model.get_params(), default=str, sort_keys=True),
           "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
           "cv_top1_mean": cv.top1_acc.mean(),
           **{f"test_{k}": v for k, v in test.items()},
           "train_s": train_s, "test_inference_s": infer_s, "seed": config.SEED}
    out = config.RESULTS_DIR / "experiments.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
    print(f"{name}: CV macroF1 {row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | "
          f"test top1 {test['top1_acc']:.4f} macroF1 {test['macro_f1']:.4f} "
          f"logloss {test['log_loss']:.4f}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or MODELS):
        run(n)
