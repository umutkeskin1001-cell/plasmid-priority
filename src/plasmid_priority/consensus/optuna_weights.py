"""Optuna-based consensus weight optimization.

This module provides utilities for optimizing consensus model weights
using Bayesian optimization via Optuna.
"""


from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import optuna as _optuna_mod
except ImportError:
    _optuna_mod = None

_optuna: Any | None = _optuna_mod


def optimize_consensus_weights(
    y_true: np.ndarray | pd.Series,
    predictions: dict[str, np.ndarray | pd.Series],
    *,
    n_trials: int = 100,
    metric: str = "roc_auc",
    random_state: int = 42,
) -> dict[str, float]:
    """Optimize consensus model weights using Optuna.

    Args:
        y_true: True binary labels
        predictions: Dictionary mapping model names to predicted probabilities
        n_trials: Number of optimization trials
        metric: Metric to optimize (roc_auc, f1, accuracy)
        random_state: Random seed

    Returns:
        Dictionary mapping model names to optimized weights
    """
    if _optuna is None:
        raise ImportError("Optuna is required for consensus weight optimization")

    # Convert predictions to numpy arrays
    pred_arrays = {name: np.array(pred) for name, pred in predictions.items()}
    y_true_arr = np.array(y_true)
    model_names = list(predictions.keys())

    def objective(trial: Any) -> float:
        """Objective function for Optuna."""
        # Sample weights for each model
        weights = []
        for _ in model_names:
            weights.append(trial.suggest_float(f"weight_{_}", 0.0, 1.0))

        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum == 0:
            return 0.0
        normalized_weights = [w / weight_sum for w in weights]

        # Compute weighted consensus prediction
        consensus_pred = np.zeros(len(y_true_arr))
        for weight, pred in zip(normalized_weights, pred_arrays.values()):
            consensus_pred += weight * pred

        # Compute metric
        if metric == "roc_auc":
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y_true_arr, consensus_pred))
        elif metric == "f1":
            from sklearn.metrics import f1_score

            binary_pred = (consensus_pred >= 0.5).astype(int)
            return float(f1_score(y_true_arr, binary_pred))
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score

            binary_pred = (consensus_pred >= 0.5).astype(int)
            return float(accuracy_score(y_true_arr, binary_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # Create study and optimize
    study = cast(Any, _optuna).create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Get best weights
    best_trial = study.best_trial
    best_weights = {}
    for name in model_names:
        best_weights[name] = best_trial.params[f"weight_{name}"]

    # Normalize to sum to 1
    weight_sum = sum(best_weights.values())
    best_weights = {k: v / weight_sum for k, v in best_weights.items()}

    return best_weights


def compute_consensus_prediction(
    predictions: dict[str, np.ndarray | pd.Series],
    weights: dict[str, float],
) -> np.ndarray:
    """Compute weighted consensus prediction.

    Args:
        predictions: Dictionary mapping model names to predicted probabilities
        weights: Dictionary mapping model names to weights

    Returns:
        Weighted consensus prediction array
    """
    pred_arrays = {name: np.array(pred) for name, pred in predictions.items()}

    # Normalize weights
    weight_sum = sum(weights.values())
    normalized_weights = {k: v / weight_sum for k, v in weights.items()}

    # Compute weighted consensus
    consensus = np.zeros(len(next(iter(pred_arrays.values()))))
    for name, pred in pred_arrays.items():
        consensus += normalized_weights.get(name, 0.0) * pred

    return consensus
