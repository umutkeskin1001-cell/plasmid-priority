"""Out-of-distribution detection using Mahalanobis distance.

Per-class mean and shared covariance in training feature space.
High Mahalanobis distance = likely OOD / novel backbone.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

_log = logging.getLogger(__name__)


class MahalanobisOODDetector:
    """Mahalanobis distance-based OOD detector.

    Parameters
    ----------
    epsilon : float
        Small constant added to covariance diagonal for stability.
    """

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        self.epsilon = float(epsilon)
        self._class_means: dict[int, np.ndarray] = {}
        self._precision: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisOODDetector":
        """Learn per-class means and shared precision matrix."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)

        # Per-class means
        self._class_means = {}
        for c in classes:
            self._class_means[int(c)] = X[y == c].mean(axis=0)

        # Shared covariance with Ledoit-Wolf shrinkage
        centered = np.vstack([X[y == c] - self._class_means[int(c)] for c in classes])
        cov = LedoitWolf().fit(centered).covariance_
        cov += np.eye(cov.shape[0]) * self.epsilon
        self._precision = np.linalg.inv(cov)
        self._is_fitted = True
        _log.info("Mahalanobis OOD detector fitted on %d classes", len(classes))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return minimum Mahalanobis distance to any class mean."""
        if not self._is_fitted or self._precision is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        min_dist = np.full(len(X), np.inf)
        for mean in self._class_means.values():
            diff = X - mean
            dist = np.sqrt(np.sum(diff @ self._precision * diff, axis=1))
            min_dist = np.minimum(min_dist, dist)
        return min_dist

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return -1 for OOD, class label for in-distribution."""
        dists = self.score_samples(X)
        if threshold is None:
            # Auto threshold: 95th percentile of training distances
            threshold = float(np.percentile(dists, 95))
        # If distance to nearest class is large, flag as OOD
        return np.where(dists > threshold, -1, 1)

    def compute_auroc(self, X_in: np.ndarray, X_out: np.ndarray) -> float:
        """Compute AUROC distinguishing in-distribution from OOD.

        High distance = OOD, so we negate for AUROC (higher = more in-dist).
        """
        scores_in = self.score_samples(X_in)
        scores_out = self.score_samples(X_out)
        from sklearn.metrics import roc_auc_score

        y_true = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])
        y_score = np.concatenate([-scores_in, -scores_out])
        return float(roc_auc_score(y_true, y_score))
