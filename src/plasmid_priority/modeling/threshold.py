"""F2-optimized threshold with cost-sensitive tiers.

Optimizes classification threshold on the F2-score (recall-weighted)
and provides multi-tier risk buckets.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import fbeta_score

_log = logging.getLogger(__name__)


class F2ThresholdOptimizer:
    """Find the threshold that maximizes F2-score.

    Parameters
    ----------
    beta : float
        F-beta weight (default 2.0 for recall-weighted).
    cost_fn : float
        Relative cost of false negatives vs false positives.
    """

    def __init__(self, *, beta: float = 2.0, cost_fn: float = 5.0) -> None:
        self.beta = float(beta)
        self.cost_fn = float(cost_fn)
        self._best_threshold: float = 0.5
        self._best_fbeta: float = 0.0
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> "F2ThresholdOptimizer":
        """Search thresholds and pick the one maximizing F-beta."""
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        # Cost-sensitive modification: weight FN higher
        thresholds = np.linspace(0.01, 0.99, 199)
        best_score = -1.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue
            f = fbeta_score(y_true, y_pred, beta=self.beta, zero_division=0)
            # Add cost-sensitive penalty
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            cost = fn * self.cost_fn + fp
            # Combined objective: maximize F-beta while minimizing cost
            combined = f - 0.001 * cost / len(y_true)
            if combined > best_score:
                best_score = combined
                best_t = float(t)

        self._best_threshold = best_t
        self._best_fbeta = float(
            fbeta_score(y_true, (y_score >= best_t).astype(int), beta=self.beta, zero_division=0)
        )
        self._is_fitted = True
        _log.info(
            "F2 threshold: %.3f (F%.1f=%.3f)", self._best_threshold, self.beta, self._best_fbeta
        )
        return self

    def predict(self, y_score: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return (np.asarray(y_score, dtype=float) >= self._best_threshold).astype(int)

    def tiered_risk(self, y_score: np.ndarray) -> np.ndarray:
        """Return risk tier labels: critical=3, high=2, medium=1, low=0."""
        s = np.asarray(y_score, dtype=float)
        tiers = np.zeros(len(s), dtype=int)
        tiers[s > 0.90] = 3
        tiers[(s > 0.70) & (s <= 0.90)] = 2
        tiers[(s > 0.50) & (s <= 0.70)] = 1
        return tiers

    @property
    def best_threshold(self) -> float:
        return self._best_threshold

    @property
    def best_fbeta(self) -> float:
        return self._best_fbeta
