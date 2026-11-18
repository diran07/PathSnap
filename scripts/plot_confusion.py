"""Plot student-test candidate classifications for a saved model.

Usage: python scripts/plot_confusion.py random_forest
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from train_models import config, load, score_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Saved model name, e.g. random_forest")
    name = parser.parse_args().model
    model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
    scored = score_observations(model, load("student_test"))
    winners = scored.groupby("pt")["p"].idxmax()
    predicted = scored.index.isin(winners)
    display = ConfusionMatrixDisplay.from_predictions(
        scored["y"], predicted, labels=[0, 1],
        display_labels=["Other arc", "Reference arc"],
        values_format="d", cmap="Blues", colorbar=False,
    )
    display.ax_.set_title(f"{name.replace('_', ' ').title()} on student-test candidates")
    folder = config.RESULTS_DIR / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    display.figure_.tight_layout()
    display.figure_.savefig(folder / f"{name}_confusion.png", dpi=150)
    plt.close(display.figure_)
    print(display.confusion_matrix)


if __name__ == "__main__":
    main()
