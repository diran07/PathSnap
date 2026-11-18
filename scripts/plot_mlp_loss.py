"""Training vs validation loss curves for the MLP (same architecture as in train_models.py).

scikit-learn's MLPClassifier only records the TRAINING loss (loss_curve_), and its
early-stopping score is accuracy on a random row split. To get an honest
validation-loss curve, this script:
  - holds out 20 % of the TRAIN trips (GroupShuffleSplit, so no trip is in both sets),
  - trains the MLP one epoch at a time with partial_fit,
  - records train and validation log loss after every epoch.
Student-test and evaluator data are not used.

Outputs: results/mlp_loss_history.csv, results/plots/mlp_loss.png
Usage:   python scripts/plot_mlp_loss.py [epochs]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap import config
from pathsnap.features import FEATURES


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df = pd.read_parquet(config.PROCESSED_DIR / "train_features.parquet")
    df = df[df["labelled"]].reset_index(drop=True)
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
                          .split(df, groups=df["trip_id"]))
    scaler = StandardScaler().fit(df.loc[tr_idx, FEATURES])
    Xtr, ytr = scaler.transform(df.loc[tr_idx, FEATURES]), df.loc[tr_idx, "y"].values
    Xva, yva = scaler.transform(df.loc[va_idx, FEATURES]), df.loc[va_idx, "y"].values
    print(f"train rows {len(tr_idx):,} ({df.loc[tr_idx, 'trip_id'].nunique()} trips) | "
          f"validation rows {len(va_idx):,} ({df.loc[va_idx, 'trip_id'].nunique()} trips)")

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                        batch_size=1024, learning_rate_init=1e-3, shuffle=True,
                        random_state=config.SEED)
    rows = []
    for ep in range(1, epochs + 1):
        mlp.partial_fit(Xtr, ytr, classes=[0, 1])
        tl = log_loss(ytr, mlp.predict_proba(Xtr)[:, 1], labels=[0, 1])
        vl = log_loss(yva, mlp.predict_proba(Xva)[:, 1], labels=[0, 1])
        rows.append({"epoch": ep, "train_log_loss": tl, "val_log_loss": vl})
        print(f"epoch {ep:2d}  train {tl:.4f}  val {vl:.4f}", flush=True)
    h = pd.DataFrame(rows)
    h.to_csv(config.RESULTS_DIR / "mlp_loss_history.csv", index=False)

    best = h.loc[h["val_log_loss"].idxmin()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h.epoch, h.train_log_loss, label="training loss", color="#1f77b4")
    ax.plot(h.epoch, h.val_log_loss, label="validation loss (held-out train trips)", color="#d62728")
    ax.axvline(best.epoch, color="grey", ls="--", lw=1, label=f"lowest validation loss (epoch {int(best.epoch)})")
    ax.set_xlabel("epoch"); ax.set_ylabel("binary log loss")
    ax.set_title("MLP (64-32) training vs validation loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "plots" / "mlp_loss.png", dpi=130)
    print(f"Best validation loss {best.val_log_loss:.4f} at epoch {int(best.epoch)}; saved plot.")


if __name__ == "__main__":
    main()
