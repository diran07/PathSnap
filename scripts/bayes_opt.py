"""Bayesian hyperparameter optimisation of the LightGBM candidate scorer (Optuna, TPE).

Why LightGBM: best CV macro-F1, best student-test top-1 and log loss of the
initial comparison, and about twice as fast to train as Random Forest
(see results/experiments.csv).

Search setup
  - Sampler   : TPE (Tree-structured Parzen Estimator), seeded.
  - Objective : mean macro-F1 over 5-fold GroupKFold on the 80 TRAIN trips
                (same folds, metric and code as scripts/train_models.py), maximised.
  - Pruning   : MedianPruner, after each fold; clearly worse trials stop early.
  - Student-test is NOT used during the search. Evaluator data is never used.

Outputs
  results/optuna_lightgbm.db           full Optuna study (resumable)
  results/bayes_opt_trials.csv         every trial: params, fold scores, state, duration
  results/bayes_opt_comparison.csv     untuned vs tuned on identical metrics
  results/plots/bayes_opt_history.png, bayes_opt_importance.png
  models/final_model.joblib            tuned model refitted on all 80 train trips
  results/experiments.csv              + one row "lightgbm_tuned"

Usage: python scripts/bayes_opt.py [n_trials]     (default 30; resumes an existing study)
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.features import FEATURES
from train_models import MODELS, evaluate, load

STUDY = "lightgbm_macro_f1"
DB = config.RESULTS_DIR / "optuna_lightgbm.db"


def make_model(p):
    return LGBMClassifier(objective="binary", bagging_freq=1, n_jobs=-1, deterministic=True,
                          force_col_wise=True, verbosity=-1, random_state=config.SEED, **p)


def suggest(trial):
    # Ranges centred on the hand-picked baseline (160 trees, lr 0.07, 31 leaves, 80 min rows):
    return {
        # more trees + smaller steps usually generalise better; cost grows linearly
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # tree complexity: 14 features, ~1M rows -> anything from shallow to fairly deep
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        # rows from one trip are correlated, so larger leaves may generalise across trips
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
        # row / feature subsampling and L2 regularisation against over-fitting
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tr, te = load("train"), load("student_test")
    X, y, g = tr[FEATURES], tr["y"], tr["trip_id"]
    folds = list(GroupKFold(n_splits=5).split(X, y, g))

    def objective(trial):
        p = suggest(trial)
        scores = []
        for k, (a, b) in enumerate(folds):
            m = make_model(p).fit(X.iloc[a], y.iloc[a])
            scores.append(evaluate(m, tr.iloc[b])["macro_f1"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                trial.set_user_attr("fold_scores", scores)
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("cv_std", float(np.std(scores, ddof=1)))
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=STUDY, storage=f"sqlite:///{DB.as_posix()}", load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # a run that was stopped mid-trial leaves that trial as RUNNING; mark it failed so the
    # search record stays honest, then continue the study up to n_trials finished trials
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            print(f"trial {t.number} was interrupted in a previous run -> marked FAIL")
    done = len([t for t in study.trials if t.state.is_finished() and t.state.name != "FAIL"])
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=max(0, n_trials - done), show_progress_bar=False,
                   callbacks=[lambda s, t: print(
                       f"trial {t.number:3d} {t.state.name:8s} value={t.value if t.value is not None else float('nan'):.4f} "
                       f"best={s.best_value:.4f}", flush=True)])
    search_s = time.perf_counter() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params", "user_attrs"))
    trials["duration_s"] = trials["duration"].dt.total_seconds()
    trials.drop(columns="duration").to_csv(config.RESULTS_DIR / "bayes_opt_trials.csv", index=False)
    best = study.best_trial
    print(f"\nBest trial {best.number}: CV macro-F1 {best.value:.4f}  params {best.params}")
    print(f"Trials: {len(study.trials)} total, "
          f"{sum(t.state.name == 'PRUNED' for t in study.trials)} pruned; search time this run {search_s / 60:.1f} min")

    # ---- untuned vs tuned on identical metrics -------------------------------------
    exp = pd.read_csv(config.RESULTS_DIR / "experiments.csv")
    base_row = exp[exp.model == "lightgbm"].iloc[-1]
    untuned = joblib.load(config.MODELS_DIR / "lightgbm.joblib")

    t0 = time.perf_counter()
    tuned = make_model(best.params).fit(X, y)
    train_s = time.perf_counter() - t0

    rows = []
    for label, model, cv_mean, cv_std in [
        ("untuned", untuned, base_row.cv_macro_f1_mean, base_row.cv_macro_f1_std),
        ("tuned", tuned, best.value, best.user_attrs["cv_std"]),
    ]:
        t0 = time.perf_counter()
        m = evaluate(model, te)
        infer_s = time.perf_counter() - t0
        rows.append({"model": f"lightgbm_{label}", "cv_macro_f1_mean": cv_mean, "cv_macro_f1_std": cv_std,
                     "test_top1_acc": m["top1_acc"], "test_macro_f1": m["macro_f1"],
                     "test_log_loss": m["log_loss"], "test_inference_s": infer_s})
    comp = pd.DataFrame(rows)
    comp.to_csv(config.RESULTS_DIR / "bayes_opt_comparison.csv", index=False)
    print("\n", comp.round(4).to_string(index=False))

    tm = comp.iloc[1]
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": "lightgbm_tuned",
           "family": "Boosting", "config": json.dumps(tuned.get_params(), default=str, sort_keys=True),
           "cv_macro_f1_mean": best.value, "cv_macro_f1_std": best.user_attrs["cv_std"],
           "cv_top1_mean": np.nan,
           **{f"test_{k}": v for k, v in evaluate(tuned, te).items()},
           "train_s": train_s, "test_inference_s": tm.test_inference_s, "seed": config.SEED}
    pd.DataFrame([row]).to_csv(config.RESULTS_DIR / "experiments.csv", mode="a", header=False, index=False)
    joblib.dump(tuned, config.MODELS_DIR / "final_model.joblib")
    print("Saved models/final_model.joblib")

    # ---- plots ----------------------------------------------------------------------
    done_t = trials[trials.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(done_t.number, done_t.value, s=18, label="completed trial", color="#1f77b4")
    pr = trials[trials.state == "PRUNED"]
    ax.scatter(pr.number, [ax.get_ylim()[0]] * len(pr), marker="x", color="#999999", label="pruned trial")
    ax.plot(done_t.number, done_t.value.cummax(), color="#d62728", label="best so far")
    ax.axhline(base_row.cv_macro_f1_mean, color="k", ls="--", lw=1, label="untuned LightGBM")
    ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1 (5-fold, train trips)")
    ax.set_title("Bayesian optimisation history (TPE)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "plots" / "bayes_opt_history.png", dpi=130)
    plt.close(fig)

    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#1f77b4")
    ax.set_xlabel("importance (fANOVA)"); ax.set_title("Hyperparameter importance")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "plots" / "bayes_opt_importance.png", dpi=130)
    plt.close(fig)
    print("Saved plots.")


if __name__ == "__main__":
    main()
