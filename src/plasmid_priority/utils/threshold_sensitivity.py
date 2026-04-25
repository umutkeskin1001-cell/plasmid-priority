"""Threshold sensitivity analysis utilities.

This module provides functions for analyzing model performance across
different threshold values, helping to understand sensitivity to classification thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def analyze_threshold_sensitivity(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    *,
    n_thresholds: int = 100,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze model performance across different threshold values.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        n_thresholds: Number of threshold values to evaluate
        metrics: List of metrics to compute (default: all)

    Returns:
        DataFrame with performance metrics at each threshold
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "tpr", "fpr"]

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    results = []

    y_true_arr = np.array(y_true)
    y_pred_proba_arr = np.array(y_pred_proba)

    for threshold in thresholds:
        y_pred = (y_pred_proba_arr >= threshold).astype(int)

        result = {"threshold": float(threshold)}

        if "accuracy" in metrics:
            result["accuracy"] = float(accuracy_score(y_true_arr, y_pred))

        if "precision" in metrics:
            result["precision"] = float(precision_score(y_true_arr, y_pred, zero_division=0))

        if "recall" in metrics:
            result["recall"] = float(recall_score(y_true_arr, y_pred, zero_division=0))

        if "f1" in metrics:
            result["f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))

        if "tpr" in metrics:
            # True Positive Rate = Recall
            result["tpr"] = float(recall_score(y_true_arr, y_pred, zero_division=0))

        if "fpr" in metrics:
            # False Positive Rate
            tn = np.sum((y_true_arr == 0) & (y_pred == 0))
            fp = np.sum((y_true_arr == 0) & (y_pred == 1))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            result["fpr"] = float(fpr)

        results.append(result)

    return pd.DataFrame(results)


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    *,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> dict[str, Any]:
    """Find the optimal threshold for a given metric.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize (accuracy, precision, recall, f1)
        n_thresholds: Number of threshold values to evaluate

    Returns:
        Dictionary with optimal threshold and corresponding metric value
    """
    sensitivity_df = analyze_threshold_sensitivity(
        y_true,
        y_pred_proba,
        n_thresholds=n_thresholds,
        metrics=[metric],
    )

    if metric not in sensitivity_df.columns:
        raise ValueError(f"Metric '{metric}' not available")

    best_idx = sensitivity_df[metric].idxmax()
    best_threshold = pd.to_numeric(
        pd.Series([sensitivity_df.loc[best_idx, "threshold"]]),
        errors="coerce",
    ).iloc[0]
    best_value = pd.to_numeric(
        pd.Series([sensitivity_df.loc[best_idx, metric]]),
        errors="coerce",
    ).iloc[0]

    return {
        "optimal_threshold": float(best_threshold) if pd.notna(best_threshold) else float("nan"),
        f"optimal_{metric}": float(best_value) if pd.notna(best_value) else float("nan"),
    }


def compute_threshold_range_performance(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    *,
    threshold_range: tuple[float, float] = (0.3, 0.7),
    n_points: int = 10,
) -> pd.DataFrame:
    """Compute performance metrics within a specific threshold range.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold_range: (min_threshold, max_threshold)
        n_points: Number of threshold points to evaluate

    Returns:
        DataFrame with performance metrics at each threshold in range
    """
    min_thresh, max_thresh = threshold_range

    return analyze_threshold_sensitivity(
        y_true,
        y_pred_proba,
        n_thresholds=n_points,
    ).loc[lambda df: df["threshold"].between(min_thresh, max_thresh)]
