"""Nested cross-validation for unbiased model performance estimation.

Provides ``nested_cross_validate`` which wraps an outer K-Fold around
the existing inner stratified CV used by ``evaluate_model_name``.
This yields an unbiased estimate of generalization performance at the
cost of increased computation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_log = logging.getLogger(__name__)


def nested_cross_validate(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run nested cross-validation for a single model.

    The outer loop splits data into train/test folds.  The inner loop
    (delegated to ``evaluate_model_name``) handles feature selection and
    hyperparameter tuning within each outer training set.

    Args:
        scored: Scored backbone DataFrame with label and feature columns.
        model_name: Model identifier (e.g. ``"geo_spread_baseline"``).
        n_outer_splits: Number of outer CV folds.
        n_inner_splits: Number of inner CV folds.
        n_repeats: Number of outer repetitions.
        seed: Random seed for reproducibility.
        config: Optional branch configuration.

    Returns:
        Dict with keys ``outer_aucs``, ``mean_auc``, ``std_auc``, ``n_folds``.
    """
    from plasmid_priority.modeling import evaluate_model_name

    label_col = "spread_label"
    if label_col not in scored.columns:
        _log.warning("Label column %s not found; cannot run nested CV", label_col)
        return {"outer_aucs": [], "mean_auc": float("nan"), "std_auc": float("nan"), "n_folds": 0}

    labels = scored[label_col].dropna()
    if labels.nunique() < 2:
        _log.warning("Only one class present; cannot run nested CV")
        return {"outer_aucs": [], "mean_auc": float("nan"), "std_auc": float("nan"), "n_folds": 0}

    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)
    outer_aucs: list[float] = []

    valid_idx = labels.index
    X_valid = scored.loc[valid_idx]

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_valid, labels.loc[valid_idx])
    ):
        train_df = X_valid.iloc[train_idx]
        test_df = X_valid.iloc[test_idx]

        try:
            result = evaluate_model_name(
                train_df,
                model_name=model_name,
                n_splits=n_inner_splits,
                n_repeats=n_repeats,
                seed=seed + fold_idx,
            )
            if result.status == "ok" and not result.predictions.empty:
                from plasmid_priority.validation.metrics import roc_auc_score

                y_true = test_df[label_col].values
                # Align predictions with test set
                preds = result.predictions
                if model_name in preds.columns:
                    y_score = preds[model_name].values[: len(y_true)]
                    if len(y_score) == len(y_true):
                        auc = roc_auc_score(y_true, y_score)
                        outer_aucs.append(auc)
        except (ValueError, KeyError, TypeError) as exc:
            _log.warning("Nested CV fold %d failed: %s", fold_idx, exc)
            continue

    mean_auc = float(np.mean(outer_aucs)) if outer_aucs else float("nan")
    std_auc = float(np.std(outer_aucs)) if len(outer_aucs) > 1 else float("nan")

    return {
        "outer_aucs": outer_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_folds": len(outer_aucs),
    }
