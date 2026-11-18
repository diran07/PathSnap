"""Model comparison table and plots, built only from logged experiment results.

Inputs
  results/experiments.csv                       one row per training run (latest row per model is used)
  results/<model>_hmm_vs_classifier.csv         per-trip classifier-only vs HMM results
                                                (logreg: results/hmm_vs_classifier.csv,
                                                 lightgbm_tuned: results/final_model_hmm_vs_classifier.csv)
Outputs
  results/model_comparison.csv                  consolidated table (Deliverables section 3.1)
  results/plots/model_comparison.png            CV macro-F1, test Top-1 / macro-F1, log loss, inference time
  results/plots/model_comparison_hmm.png        classifier-only vs +HMM Top-1 and route continuity

Usage: python scripts/plot_comparison.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config

ORDER = ["logreg", "random_forest", "lightgbm", "knn", "mlp", "lightgbm_tuned"]
LABEL = {"logreg": "Logistic Reg.", "random_forest": "Random Forest", "lightgbm": "LightGBM",
         "knn": "KNN", "mlp": "MLP", "lightgbm_tuned": "LightGBM (tuned)"}
HMM_FILE = {"logreg": "hmm_vs_classifier.csv", "lightgbm_tuned": "final_model_hmm_vs_classifier.csv"}
KEY_PARAMS = {
    "logreg": ["logisticregression__C"],
    "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_samples"],
    "lightgbm": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
    "lightgbm_tuned": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
    "knn": ["estimator__kneighborsclassifier__n_neighbors", "estimator__kneighborsclassifier__weights", "n_samples"],
    "mlp": ["mlpclassifier__hidden_layer_sizes", "mlpclassifier__alpha", "mlpclassifier__learning_rate_init"],
}


def key_config(name, cfg):
    try:
        p = json.loads(cfg)
    except (TypeError, ValueError):
        return ""
    out = []
    for k in KEY_PARAMS.get(name, []):
        if k in p:
            v = p[k]
            out.append(f"{k.split('__')[-1]}={round(v, 4) if isinstance(v, float) else v}")
    return ", ".join(out)


def main():
    exp = pd.read_csv(config.RESULTS_DIR / "experiments.csv")
    exp = exp.groupby("model").tail(1).set_index("model")
    models = [m for m in ORDER if m in exp.index]

    rows = []
    for m in models:
        e = exp.loc[m]
        r = {"model": LABEL[m], "family": e.family, "key_configuration": key_config(m, e.config),
             "cv_macro_f1": e.cv_macro_f1_mean, "cv_macro_f1_std": e.cv_macro_f1_std,
             "test_top1_acc": e.test_top1_acc, "test_macro_f1": e.test_macro_f1,
             "test_precision": e.test_precision, "test_recall": e.test_recall,
             "test_log_loss": e.test_log_loss, "train_s": e.train_s,
             "test_inference_s": e.test_inference_s}
        f = config.RESULTS_DIR / HMM_FILE.get(m, f"{m}_hmm_vs_classifier.csv")
        if f.exists():
            h = pd.read_csv(f).groupby("method").mean(numeric_only=True)
            r.update({"trip_top1_classifier": h.loc["classifier", "top1_acc"],
                      "trip_top1_hmm": h.loc["hmm", "top1_acc"],
                      "route_continuity_classifier": h.loc["classifier", "route_continuity"],
                      "route_continuity_hmm": h.loc["hmm", "route_continuity"],
                      "route_f1_hmm": h.loc["hmm", "route_f1"]})
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    print(tab.drop(columns=["key_configuration"]).round(4).to_string(index=False))

    x = np.arange(len(tab))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#dd8452"][:len(tab)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    a = ax[0, 0]
    a.bar(x, tab.cv_macro_f1, yerr=tab.cv_macro_f1_std, capsize=4, color=colors)
    a.set_title("Cross-validation macro-F1 (5-fold, train trips; bars = ±1 std)")
    a.set_ylim(tab.cv_macro_f1.min() - 0.06, min(1, tab.cv_macro_f1.max() + 0.04))

    a = ax[0, 1]
    w = 0.38
    a.bar(x - w / 2, tab.test_top1_acc, w, label="Top-1 accuracy", color="#4c72b0")
    a.bar(x + w / 2, tab.test_macro_f1, w, label="Macro-F1", color="#dd8452")
    a.set_title("Student-test (10 trips), classifier only")
    a.set_ylim(min(tab.test_top1_acc.min(), tab.test_macro_f1.min()) - 0.05, 1)
    a.legend()

    a = ax[1, 0]
    a.bar(x, tab.test_log_loss, color=colors)
    a.set_title("Student-test log loss (lower is better)")

    a = ax[1, 1]
    a.bar(x, tab.test_inference_s, color=colors)
    a.set_yscale("log")
    a.set_title("Inference time on student-test candidates (s, log scale)")

    for a in ax.ravel():
        a.set_xticks(x, tab.model, rotation=20, ha="right")
        for i, p in enumerate(a.patches[:len(tab)]):
            a.annotate(f"{p.get_height():.3g}", (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha="center", va="bottom", fontsize=7)
    fig.suptitle("PathSnap model comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "plots" / "model_comparison.png", dpi=130)
    plt.close(fig)

    if "trip_top1_hmm" in tab:
        t = tab.dropna(subset=["trip_top1_hmm"])
        x = np.arange(len(t))
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        for a, (c, h, title) in zip(ax, [
            ("trip_top1_classifier", "trip_top1_hmm", "Top-1 accuracy (mean over trips)"),
            ("route_continuity_classifier", "route_continuity_hmm", "Route continuity")]):
            a.bar(x - w / 2, t[c], w, label="classifier only", color="#999999")
            a.bar(x + w / 2, t[h], w, label="classifier + HMM", color="#c44e52")
            a.set_xticks(x, t.model, rotation=20, ha="right")
            a.set_ylim(min(t[c].min(), t[h].min()) - 0.05, 1)
            a.set_title(f"Student-test: {title}")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "plots" / "model_comparison_hmm.png", dpi=130)
        plt.close(fig)
    print("Saved results/model_comparison.csv and plots.")


if __name__ == "__main__":
    main()
