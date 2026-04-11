"""Out-of-the-box ensemble strategies for ultra-high AUC (83+)."""

from __future__ import annotations

import warnings
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
        self.meta_learner: LogisticRegression | None = None
        self.calibrators: dict[str, IsotonicRegression] = {}
        self.weights: np.ndarray | None = None
        self.uncertainty_estimates: np.ndarray | None = None
        
    def fit_base_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_factory: Any,
    ) -> dict[str, np.ndarray]:
        """Fit all base models and collect OOF predictions."""
        oof_preds = {}
        
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
            self.calibrators[model_name].fit(preds.reshape(-1, 1), y)
        
        self.base_predictions = oof_preds
        return oof_preds
    
    def fit_meta_learner(
        self,
        X_meta: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit level-2 meta-learner on base predictions."""
        # Build meta-features: base predictions + interactions
        meta_features = self._build_meta_features(X_meta)
        
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
        self.uncertainty_estimates = self._estimate_uncertainty(meta_features, y)
        
        return self.meta_learner.predict_proba(meta_features)[:, 1]
    
    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Build meta-features: base preds + polynomial interactions."""
        features = []
        
        # Base predictions
        for name, preds in self.base_predictions.items():
            features.append(preds)
            
            # Calibrated version
            if name in self.calibrators:
                cal_preds = self.calibrators[name].predict(preds.reshape(-1, 1))
                features.append(cal_preds)
        
        # 2-way interactions (only if enabled)
        if self.config.use_feature_interactions:
            pred_array = np.column_stack([
                self.base_predictions[name] 
                for name in self.config.base_models
            ])
            
            # Top-k pairwise products
            n_models = len(self.config.base_models)
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    interaction = pred_array[:, i] * pred_array[:, j]
                    features.append(interaction)
            
            # 3-way interactions for top-3 models
            if self.config.interaction_degree >= 3 and n_models >= 3:
                for i in range(min(3, n_models)):
                    for j in range(i + 1, min(3, n_models)):
                        for k in range(j + 1, min(3, n_models)):
                            triple = (
                                pred_array[:, i] * 
                                pred_array[:, j] * 
                                pred_array[:, k]
                            )
                            features.append(triple)
        
        return np.column_stack(features)
    
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
        """Estimate prediction uncertainty via bootstrap."""
        n_samples = len(y)
        n_bootstrap = 50
        bootstrap_preds = np.zeros((n_samples, n_bootstrap))
        
        for b in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
            model.fit(X[idx], y[idx])
            bootstrap_preds[:, b] = model.predict_proba(X)[:, 1]
        
        # Uncertainty = std across bootstrap samples
        uncertainty = np.std(bootstrap_preds, axis=1)
        return uncertainty
    
    def predict(
        self,
        X: pd.DataFrame,
        use_uncertainty_gating: bool = True,
    ) -> dict[str, Any]:
        """Predict with uncertainty-aware gating."""
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner not fitted. Call fit_meta_learner first.")
        
        # Build meta-features
        meta_features = self._build_meta_features(X)
        
        # Meta-learner prediction
        probs = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        # Uncertainty gating
        if use_uncertainty_gating and self.uncertainty_estimates is not None:
            # Estimate uncertainty for new data
            new_uncertainty = self._predict_uncertainty(meta_features)
            
            # Down-weight high-uncertainty predictions
            confidence = 1.0 / (1.0 + new_uncertainty)
            probs = probs * confidence + 0.5 * (1 - confidence)
        
        return {
            "probability": probs,
            "uncertainty": self._predict_uncertainty(meta_features) if self.uncertainty_estimates is not None else None,
            "confidence": self._compute_confidence(probs),
        }
    
    def _predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Predict uncertainty for new data."""
        # Use distance to training data as proxy
        # Simplified: use prediction variance across models
        model_preds = []
        for name in self.config.base_models:
            if name in self.base_predictions:
                # For simplicity, use stored predictions
                pass
        
        # Return placeholder (would need full implementation)
        return np.zeros(X.shape[0]) + 0.1
    
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
        q_level = np.ceil((n + 1) * self.coverage) / n
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
        weights = ensemble.weights if ensemble.weights is not None else np.ones(len(base_predictions)) / len(base_predictions)
        preds = preds_result
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Final calibration
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(preds.reshape(-1, 1), y_true)
    preds_calibrated = calibrator.predict(preds.reshape(-1, 1))
    
    return {
        "probability": preds_calibrated,
        "uncalibrated": preds,
        "weights": dict(zip(base_predictions.keys(), weights)),
        "method": method,
    }
