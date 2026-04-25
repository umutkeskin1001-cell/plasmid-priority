"""Deep ensemble: multiple independent models for epistemic uncertainty.

Variance across ensemble members = epistemic uncertainty.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

_log = logging.getLogger(__name__)


class DeepEnsemble:
    """Deep ensemble of diverse base learners with variance-based uncertainty.

    Parameters
    ----------
    base_factory : Callable[[], Any]
        Factory returning an unfitted sklearn-compatible classifier.
    n_members : int
        Number of ensemble members.
    bootstrap : bool
        Whether to bootstrap-aggregate (bag) each member.
    random_state : int
    """

    def __init__(
        self,
        base_factory: Callable[[], Any] | None = None,
        *,
        n_members: int = 5,
        bootstrap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.base_factory = base_factory or self._default_factory
        self.n_members = int(n_members)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self._members: list[Any] = []
        self._is_fitted = False

    @staticmethod
    def _default_factory() -> Any:
        return HistGradientBoostingClassifier(max_iter=100, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self._members = []
        for i in range(self.n_members):
            model = clone(self.base_factory())
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                Xi, yi = X[idx], y[idx]
            else:
                Xi, yi = X, y
            model.fit(Xi, yi)
            self._members.append(model)
            _log.info("DeepEnsemble member %d/%d fitted", i + 1, self.n_members)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_proba, uncertainty) where uncertainty = std across members."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted.")
        preds = np.stack([m.predict_proba(X)[:, 1] for m in self._members], axis=1)
        mean_p = preds.mean(axis=1)
        uncertainty = preds.std(axis=1)
        return mean_p, uncertainty

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        mean_p, _ = self.predict_proba(X)
        return (mean_p >= threshold).astype(int)

    def uncertainty_gated_predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        uncertainty_threshold: float = 0.3,
    ) -> np.ndarray:
        """Return -1 for high-uncertainty samples (human review)."""
        mean_p, unc = self.predict_proba(X)
        pred = (mean_p >= threshold).astype(int)
        pred[unc > uncertainty_threshold] = -1
        return pred
