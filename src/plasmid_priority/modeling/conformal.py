"""Conformal prediction for uncertainty quantification.

Implements split conformal prediction to produce prediction sets with
coverage guarantees. Each prediction comes with a set of possible labels
rather than a single probability.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

_log = logging.getLogger(__name__)


class SplitConformalPredictor:
    """Split conformal prediction for binary classification.

    Parameters
    ----------
    base_model : Any
        Fitted sklearn-compatible classifier with predict_proba.
    calibration_fraction : float
        Fraction of data reserved for calibration.
    alpha : float
        Miscoverage rate (1 - desired coverage). Default 0.10 for 90% coverage.
    random_state : int
    """

    def __init__(
        self,
        base_model: Any,
        *,
        calibration_fraction: float = 0.2,
        alpha: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.calibration_fraction = float(np.clip(calibration_fraction, 0.05, 0.5))
        self.alpha = float(np.clip(alpha, 0.01, 0.5))
        self.random_state = int(random_state)
        self._quantile: float | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplitConformalPredictor":
        """Fit calibration on a held-out fraction."""
        X_cal, _, y_cal, _ = train_test_split(
            X,
            y,
            test_size=1 - self.calibration_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        proba = self.base_model.predict_proba(X_cal)[:, 1]
        # Non-conformity scores: 1 - p(y_true | x)
        scores = np.where(y_cal == 1, 1.0 - proba, proba)
        # Quantile with finite-sample correction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(scores, min(q_level, 1.0)))
        self._is_fitted = True
        _log.info("Conformal quantile: %.4f (alpha=%.2f)", self._quantile, self.alpha)
        return self

    def predict_sets(self, X: np.ndarray) -> np.ndarray:
        """Return prediction sets as boolean array (n_samples, 2).

        Each row indicates which labels {0, 1} are in the prediction set.
        """
        if not self._is_fitted or self._quantile is None:
            raise RuntimeError("Call fit() first.")
        proba = self.base_model.predict_proba(X)[:, 1]
        sets = np.zeros((len(X), 2), dtype=bool)
        # Label 0 in set if non-conformity <= quantile
        sets[:, 0] = proba <= self._quantile
        # Label 1 in set if non-conformity <= quantile
        sets[:, 1] = (1.0 - proba) <= self._quantile
        return sets

    def predict_proba_intervals(self, X: np.ndarray) -> np.ndarray:
        """Return lower/upper probability bounds for the positive class.

        Returns array of shape (n_samples, 2) with [lower, upper].
        """
        sets = self.predict_sets(X)
        intervals = np.zeros((len(X), 2))
        intervals[:, 0] = np.where(sets[:, 1], 0.0, 0.5)  # conservative lower
        intervals[:, 1] = np.where(sets[:, 1], 1.0, 0.5)  # conservative upper
        return intervals

    def empirical_coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage on a test set."""
        sets = self.predict_sets(X)
        covered = sets[np.arange(len(y)), y.astype(int)]
        return float(covered.mean())
