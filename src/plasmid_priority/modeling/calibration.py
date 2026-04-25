"""Beta calibration + temperature scaling for probability calibration.

BetaCalibration: [0,1] stable calibration using beta distribution CDF.
TemperatureScaling: Learnable scalar T for DL model logits.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_log = logging.getLogger(__name__)


class BetaCalibration:
    """Beta calibration for binary probabilities.

    Fits: calibrated = F_beta(a, b, m) where F_beta is the Beta CDF
    with learnable shape parameters a, b and location m.
    Simplified to a 1-parameter power transform for stability.

    Parameters
    ----------
    method : {"beta", "isotonic", "platt"}
        Calibration backend.
    """

    def __init__(self, *, method: str = "beta") -> None:
        self.method = method
        self._calibrator: Any | None = None
        self._a: float = 1.0
        self._b: float = 1.0
        self._is_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "BetaCalibration":
        probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=int)

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(probs, y_true)
        elif self.method == "platt":
            self._calibrator = LogisticRegression(max_iter=1000)
            self._calibrator.fit(probs.reshape(-1, 1), y_true)
        else:
            # Simplified beta: fit a, b minimizing Brier on power-transformed probs
            def _brier(params: np.ndarray) -> float:
                a, b = float(params[0]), float(params[1])
                # Use sigmoid-scaled power transform as Beta approximation
                transformed = expit(a * logit(probs) + b)
                return float(np.mean((transformed - y_true) ** 2))

            result = minimize_scalar(
                lambda a: _brier(np.array([a, 0.0])),
                bounds=(0.1, 10.0),
                method="bounded",
            )
            self._a = float(result.x)
            self._b = 0.0

        self._is_fitted = True
        _log.info("BetaCalibration fitted (method=%s)", self.method)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic" and self._calibrator is not None:
            return cast(np.ndarray, self._calibrator.predict(probs))
        if self.method == "platt" and self._calibrator is not None:
            return cast(np.ndarray, self._calibrator.predict_proba(probs.reshape(-1, 1))[:, 1])
        # Default beta transform
        return cast(np.ndarray, np.clip(expit(self._a * logit(probs) + self._b), 1e-6, 1 - 1e-6))


class TemperatureScaling:
    """Learnable temperature parameter for calibrating logits.

    T > 1 softens probabilities (under-confident → calibrated).
    T < 1 sharpens probabilities (over-confident → calibrated).
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._is_fitted = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """Fit temperature on validation logits."""
        logits = np.asarray(logits, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        def _nll(T: float) -> float:
            probs = expit(logits / max(T, 1e-6))
            eps = 1e-12
            return -float(
                np.mean(y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps))
            )

        result = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self._is_fitted = True
        _log.info("Temperature scaling fitted: T=%.4f", self.temperature)
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        logits = np.asarray(logits, dtype=float)
        return cast(np.ndarray, np.clip(expit(logits / self.temperature), 1e-6, 1 - 1e-6))
