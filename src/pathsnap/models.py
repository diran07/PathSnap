"""Model helpers shared by training scripts and the API (must stay importable for joblib)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class SubsampledClassifier(BaseEstimator, ClassifierMixin):
    """Fit `estimator` on a fixed-size random subsample of the training rows.

    Used for KNN: prediction cost grows with the number of stored training rows,
    and predicting ~230k validation rows against all ~900k training rows of a
    fold is impractically slow. A seeded subsample keeps runs reproducible.
    """

    def __init__(self, estimator=None, n_samples=200_000, random_state=42):
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        if self.n_samples and n > self.n_samples:
            idx = np.random.default_rng(self.random_state).choice(n, self.n_samples, replace=False)
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))
