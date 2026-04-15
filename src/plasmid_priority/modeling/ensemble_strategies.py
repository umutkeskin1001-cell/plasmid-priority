"""Out-of-the-box ensemble strategies for ultra-high AUC (83+)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass
class EnsembleConfig:
    """Configuration for meta-sovereign ensemble."""

    base_models: list[str]
    meta_learner_type: str = "logistic"  # logistic, ridge, or bayesian
    confidence_threshold: float = 0.7
    calibration_method: str = "isotonic"  # isotonic, platt, or temperature
    n_folds: int = 5
    use_uncertainty_gating: bool = True
    use_feature_interactions: bool = True
    interaction_degree: int = 2  # 2-way or 3-way


class MetaSovereignEnsemble:
    """Out-of-the-box adaptive ensemble for 83+ AUC.

    Strategy:
    1. Train base models with diverse feature sets
    2. Generate out-of-fold predictions (stacking)
    3. Learn meta-learner on base predictions
    4. Apply confidence-weighted calibration
    5. Uncertainty-aware gating for final prediction
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig(
            base_models=[
                "phylo_support_fusion_priority",
                "knownness_robust_priority",
                "support_synergy_priority",
                "host_transfer_synergy_priority",
                "discovery_12f_source",
                "sovereign_precision_priority",
            ]
        )
        self.base_predictions: dict[str, np.ndarray] = {}
        self.fitted_base_models: dict[str, Any] = {}
        self.meta_learner: LogisticRegression | None = None
        self.calibrators: dict[str, IsotonicRegression] = {}
        self.weights: np.ndarray | None = None
        self.uncertainty_estimates: np.ndarray | None = None

    def _predict_base_probabilities(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict probabilities from the fitted base models in config order."""
        if not self.fitted_base_models:
            raise RuntimeError("Base models not fitted. Call fit_base_models first.")
        predictions: dict[str, np.ndarray] = {}
        for model_name in self.config.base_models:
            model = self.fitted_base_models.get(model_name)
            if model is None:
                raise RuntimeError(f"Missing fitted base model: {model_name}")
            predictions[model_name] = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        return predictions

    def _calibrated_prediction_matrix(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Return calibrated base predictions in config order."""
        calibrated_columns: list[np.ndarray] = []
        for model_name in self.config.base_models:
            raw = np.asarray(predictions[model_name], dtype=float)
            if model_name in self.calibrators:
                cal = np.asarray(self.calibrators[model_name].predict(raw), dtype=float)
            else:
                cal = raw
            calibrated_columns.append(cal)
        return np.column_stack(calibrated_columns)

    def _combine_base_features(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Build meta-features from base predictions and their calibrated variants."""
        features: list[np.ndarray] = []
        calibrated_matrix = self._calibrated_prediction_matrix(predictions)

        for idx, model_name in enumerate(self.config.base_models):
            raw = np.asarray(predictions[model_name], dtype=float)
            features.append(raw)
            features.append(calibrated_matrix[:, idx])

        if self.config.use_feature_interactions:
            n_models = len(self.config.base_models)
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    features.append(calibrated_matrix[:, i] * calibrated_matrix[:, j])
            if self.config.interaction_degree >= 3 and n_models >= 3:
                for i in range(min(3, n_models)):
                    for j in range(i + 1, min(3, n_models)):
                        for k in range(j + 1, min(3, n_models)):
                            features.append(
                                calibrated_matrix[:, i]
                                * calibrated_matrix[:, j]
                                * calibrated_matrix[:, k]
                            )
        return np.column_stack(features)

    def _estimate_uncertainty_from_predictions(
        self, predictions: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Estimate uncertainty as disagreement across calibrated base models."""
        calibrated_matrix = self._calibrated_prediction_matrix(predictions)
        if calibrated_matrix.size == 0:
            return np.array([], dtype=float)
        uncertainty = np.std(calibrated_matrix, axis=1)
        return np.clip(uncertainty.astype(float), 0.0, 1.0)

    def fit_base_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_factory: Any,
    ) -> dict[str, np.ndarray]:
        """Fit all base models and collect OOF predictions."""
        oof_preds = {}
        self.fitted_base_models = {}

        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)

        for model_name in self.config.base_models:
            preds = np.zeros(len(y))

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y[train_idx]

                # Fit base model
                model = model_factory(model_name)
                model.fit(X_train, y_train)

                # OOF predictions
                preds[val_idx] = model.predict_proba(X_val)[:, 1]

            oof_preds[model_name] = preds
            self.calibrators[model_name] = IsotonicRegression(out_of_bounds="clip")
            self.calibrators[model_name].fit(preds, y)

            full_model = model_factory(model_name)
            full_model.fit(X, y)
            self.fitted_base_models[model_name] = full_model

        self.base_predictions = oof_preds
        self.uncertainty_estimates = self._estimate_uncertainty_from_predictions(oof_preds)
        return oof_preds

    def fit_meta_learner(
        self,
        X_meta: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit level-2 meta-learner on base predictions."""
        if not self.base_predictions:
            raise RuntimeError("Base predictions not fitted. Call fit_base_models first.")
        # Build meta-features: base predictions + interactions
        meta_features = self._build_meta_features(X_meta, base_predictions=self.base_predictions)

        # Fit meta-learner with strong regularization for stability
        self.meta_learner = LogisticRegression(
            C=0.1,  # Strong regularization
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        self.meta_learner.fit(meta_features, y)

        # Learn adaptive weights based on validation AUC
        self.weights = self._compute_adaptive_weights(X_meta, y)

        # Estimate uncertainty
        self.uncertainty_estimates = self._estimate_uncertainty_from_predictions(
            self.base_predictions
        )

        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def _build_meta_features(
        self,
        X: pd.DataFrame,
        *,
        base_predictions: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Build meta-features from either cached or live base predictions."""
        predictions = (
            base_predictions
            if base_predictions is not None
            else self._predict_base_probabilities(X)
        )
        return self._combine_base_features(predictions)

    def _compute_adaptive_weights(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute AUC-based adaptive weights for each base model."""
        from plasmid_priority.validation.metrics import roc_auc_score

        aucs = []
        for name, preds in self.base_predictions.items():
            try:
                auc = roc_auc_score(y, preds)
                aucs.append(max(0.5, auc))  # Floor at 0.5
            except Exception:
                aucs.append(0.5)

        # Softmax weighting
        weights = np.exp(np.array(aucs) - np.max(aucs))
        weights = weights / np.sum(weights)

        return weights

    def _estimate_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Estimate prediction uncertainty via base-model disagreement."""
        del X, y
        if not self.base_predictions:
            return np.array([], dtype=float)
        return self._estimate_uncertainty_from_predictions(self.base_predictions)

    def predict(
        self,
        X: pd.DataFrame,
        use_uncertainty_gating: bool = True,
    ) -> dict[str, Any]:
        """Predict with uncertainty-aware gating."""
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not fitted. Call fit_meta_learner first.")

        # Build meta-features
        base_predictions = self._predict_base_probabilities(X)
        meta_features = self._build_meta_features(X, base_predictions=base_predictions)

        # Meta-learner prediction
        probs = self.meta_learner.predict_proba(meta_features)[:, 1]

        # Uncertainty gating
        if use_uncertainty_gating and self.uncertainty_estimates is not None:
            # Estimate uncertainty for new data
            new_uncertainty = self._estimate_uncertainty_from_predictions(base_predictions)

            # Down-weight high-uncertainty predictions
            confidence = 1.0 / (1.0 + new_uncertainty)
            probs = probs * confidence + 0.5 * (1 - confidence)

        return {
            "probability": probs,
            "uncertainty": (
                self._estimate_uncertainty_from_predictions(base_predictions)
                if self.uncertainty_estimates is not None
                else None
            ),
            "confidence": self._compute_confidence(probs),
        }

    def _compute_confidence(self, probs: np.ndarray) -> np.ndarray:
        """Compute confidence scores."""
        # Confidence = distance from 0.5
        return np.abs(probs - 0.5) * 2


class ConformalPredictionEnsemble:
    """Conformal prediction for uncertainty quantification."""

    def __init__(self, coverage: float = 0.95):
        self.coverage = coverage
        self.q_hat: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit conformal predictor."""
        # Non-conformity scores
        scores = np.abs(y_true - y_pred)

        # Quantile for coverage
        n = len(scores)
        if n < 1:
            raise ValueError("Conformal predictor requires at least one calibration score.")
        q_level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.q_hat = np.quantile(scores, q_level, method="higher")

    def predict_set(self, y_pred: np.ndarray) -> list[tuple[float, float]]:
        """Return prediction sets with guaranteed coverage."""
        if self.q_hat is None:
            raise RuntimeError("Conformal predictor not fitted.")

        sets = []
        for pred in y_pred:
            lower = max(0.0, pred - self.q_hat)
            upper = min(1.0, pred + self.q_hat)
            sets.append((lower, upper))

        return sets


def create_meta_sovereign_model(
    base_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    method: str = "adaptive_weighted",
) -> dict[str, Any]:
    """Factory function to create optimized meta-sovereign predictions.

    Args:
        base_predictions: Dict of model_name -> prediction array
        y_true: Ground truth labels
        method: "simple_average", "weighted_average", "stacking", or "adaptive_weighted"

    Returns:
        Dict with final predictions, weights, and metadata
    """
    from plasmid_priority.validation.metrics import roc_auc_score

    if method == "simple_average":
        preds = np.mean(list(base_predictions.values()), axis=0)
        weights = np.ones(len(base_predictions)) / len(base_predictions)

    elif method == "weighted_average":
        # Weight by AUC
        aucs = []
        for name, preds in base_predictions.items():
            try:
                auc = roc_auc_score(y_true, preds)
                aucs.append(max(0.5, auc - 0.5))  # Center at 0
            except Exception:
                aucs.append(0.0)

        weights = np.array(aucs) / sum(aucs) if sum(aucs) > 0 else np.ones(len(aucs)) / len(aucs)
        pred_array = np.column_stack(list(base_predictions.values()))
        preds = np.average(pred_array, axis=1, weights=weights)

    elif method in ("stacking", "adaptive_weighted"):
        # Use MetaSovereignEnsemble
        config = EnsembleConfig(
            base_models=list(base_predictions.keys()),
            use_uncertainty_gating=(method == "adaptive_weighted"),
        )
        ensemble = MetaSovereignEnsemble(config)
        ensemble.base_predictions = base_predictions

        # Build dummy features
        X_dummy = pd.DataFrame({"pred": list(base_predictions.values())[0]})
        preds_result = ensemble.fit_meta_learner(X_dummy, y_true)
        weights = (
            ensemble.weights
            if ensemble.weights is not None
            else np.ones(len(base_predictions)) / len(base_predictions)
        )
        preds = preds_result

    else:
        raise ValueError(f"Unknown method: {method}")

    # Final calibration
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(preds, y_true)
    preds_calibrated = calibrator.predict(preds)

    return {
        "probability": preds_calibrated,
        "uncalibrated": preds,
        "weights": dict(zip(base_predictions.keys(), weights)),
        "method": method,
    }
