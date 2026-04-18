"""SHAP-based model interpretability for Plasmid Priority.

Provides ``compute_shap_values`` and ``build_shap_summary_table`` which
wrap the ``shap`` library (optional dependency) to produce feature
importance explanations for any fitted branch model.

**FAZ 4 addition:** ``compute_shap_tree_values`` uses ``shap.TreeExplainer``
for LightGBM models — exact, fast, no subsampling needed for 989 backbones.
``compute_shap_interactions`` produces T×H×A interaction maps.

If ``shap`` is not installed, all public functions return empty results
with a ``skipped`` status instead of raising.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    *,
    max_display: int = 20,
    sample_size: int | None = 100,
) -> dict[str, Any]:
    """Compute SHAP values for a fitted model.

    Args:
        model: Any sklearn-compatible model with a ``predict_proba`` method.
        X: Feature matrix to explain.
        max_display: Maximum number of features in the summary plot.
        sample_size: If set, subsample *X* for speed (SHAP is expensive).

    Returns:
        Dict with keys ``shap_values``, ``feature_names``, ``base_value``,
        ``status``.  If shap is unavailable, ``status`` is ``"skipped"``.
    """
    if not SHAP_AVAILABLE:
        _log.info("shap not installed; skipping SHAP explanation")
        return {"status": "skipped", "reason": "shap_not_installed"}

    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample, max_display=max_display)
        return {
            "status": "ok",
            "shap_values": shap_values.values,
            "feature_names": list(X_sample.columns),
            "base_value": float(shap_values.base_values.mean())
            if hasattr(shap_values, "base_values")
            else 0.0,
        }
    except (ValueError, TypeError, AttributeError) as exc:
        _log.warning("SHAP computation failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def compute_shap_tree_values(
    model: Any,
    X: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute exact SHAP values using TreeExplainer for tree-based models.

    LightGBM models support ``shap.TreeExplainer`` which provides exact
    (not approximate) SHAP values with no subsampling needed — suitable
    for the full 989-backbone dataset.

    Args:
        model: A fitted LightGBM or other tree-based model.
        X: Feature matrix to explain (all rows, no subsampling).
        feature_names: Optional feature name list; defaults to X.columns.

    Returns:
        Dict with keys ``shap_values``, ``feature_names``, ``base_value``,
        ``explainer``, ``status``.  If shap is unavailable, ``status`` is
        ``"skipped"``.
    """
    if not SHAP_AVAILABLE:
        _log.info("shap not installed; skipping TreeExplainer SHAP")
        return {"status": "skipped", "reason": "shap_not_installed"}

    try:
        explainer = shap.TreeExplainer(model)
        X_arr = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "LightGBM binary classifier with TreeExplainer shap values output "
                    "has changed to a list of ndarray"
                ),
                category=UserWarning,
            )
            shap_values = explainer.shap_values(X_arr)

        # For binary classification, LightGBM returns a list with
        # two arrays (negative class, positive class).  We want the
        # positive-class SHAP values.
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = np.asarray(shap_values[1], dtype=float)
        else:
            shap_values = np.asarray(shap_values, dtype=float)

        # Handle 3-D arrays (rare but possible)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        names = feature_names if feature_names is not None else list(X.columns)
        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[-1])

        return {
            "status": "ok",
            "shap_values": shap_values,
            "feature_names": names,
            "base_value": base_value,
            "explainer": explainer,
        }
    except (ValueError, TypeError, AttributeError) as exc:
        _log.warning("TreeExplainer SHAP computation failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def compute_shap_interactions(
    model: Any,
    X: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute SHAP interaction values using TreeExplainer.

    Produces an n_samples × n_features × n_features tensor of interaction
    values.  Useful for T×H×A interaction maps.

    Args:
        model: A fitted LightGBM model (TreeExplainer required).
        X: Feature matrix to explain.
        feature_names: Optional feature name list.

    Returns:
        Dict with keys ``interaction_values``, ``feature_names``, ``status``.
    """
    if not SHAP_AVAILABLE:
        _log.info("shap not installed; skipping SHAP interactions")
        return {"status": "skipped", "reason": "shap_not_installed"}

    try:
        explainer = shap.TreeExplainer(model)
        X_arr = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "LightGBM binary classifier with TreeExplainer shap values output "
                    "has changed to a list of ndarray"
                ),
                category=UserWarning,
            )
            interaction_values = explainer.shap_interaction_values(X_arr)

        # For binary classification, take the positive-class slice
        if isinstance(interaction_values, list) and len(interaction_values) == 2:
            interaction_values = np.asarray(interaction_values[1], dtype=float)
        else:
            interaction_values = np.asarray(interaction_values, dtype=float)

        names = feature_names if feature_names is not None else list(X.columns)

        return {
            "status": "ok",
            "interaction_values": interaction_values,
            "feature_names": names,
        }
    except (ValueError, TypeError, AttributeError) as exc:
        _log.warning("SHAP interaction computation failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def build_global_feature_importance(
    shap_result: dict[str, Any],
) -> pd.DataFrame:
    """Build a global feature importance ranking from TreeExplainer SHAP values.

    Args:
        shap_result: Output of ``compute_shap_tree_values``.

    Returns:
        DataFrame with ``feature``, ``mean_abs_shap``, ``std_shap``,
        ``direction`` (mean signed SHAP).  Sorted by mean_abs_shap descending.
    """
    if shap_result.get("status") != "ok":
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "std_shap", "direction"])

    values = np.asarray(shap_result["shap_values"])
    feature_names = shap_result["feature_names"]

    # Handle 3-D arrays (classification: classes × samples × features)
    if values.ndim == 3:
        values = values[:, :, -1]  # positive-class SHAP

    mean_abs = np.abs(values).mean(axis=0)
    std = np.abs(values).std(axis=0)
    direction = values.mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "std_shap": std,
                "direction": direction,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def fit_global_lightgbm_model(
    eligible: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Fit a single LightGBM model on all eligible backbones for SHAP explanation.

    Per the FAZ 4 spec (Yöntem A): OOF cross-validation produces 5 separate
    models, but SHAP expects a single model.  We fit a *global* LightGBM on
    the full 989-backbone dataset **solely for interpretability** — OOF
    predictions remain the official evaluation scores.

    Args:
        eligible: DataFrame with feature columns and ``spread_label``.
        columns: Feature column names to use.
        fit_kwargs: Optional LightGBM hyperparameters.

    Returns:
        Dict with keys ``model``, ``X``, ``feature_names``, ``status``.
    """
    try:
        import lightgbm as _lgb  # noqa: F401 — guard only

        _HAS_LIGHTGBM = True
    except ImportError:
        _HAS_LIGHTGBM = False

    if not _HAS_LIGHTGBM:
        _log.info("lightgbm not installed; skipping global model fit")
        return {"status": "skipped", "reason": "lightgbm_not_installed"}

    fit_kwargs = fit_kwargs or {}
    y = eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    X = eligible[columns].fillna(0.0).to_numpy(dtype=float)

    if len(y) < 8 or len(np.unique(y)) < 2:
        return {"status": "skipped", "reason": "insufficient_data"}

    try:
        from plasmid_priority.modeling.module_a import (
            _fit_lightgbm_classifier,
        )

        model = _fit_lightgbm_classifier(X, y, fit_kwargs=fit_kwargs)
        return {
            "status": "ok",
            "model": model,
            "X": pd.DataFrame(X, columns=columns),
            "feature_names": columns,
        }
    except (ImportError, ValueError, TypeError, AttributeError) as exc:
        _log.warning("Global LightGBM fit failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def build_shap_dependence_data(
    shap_result: dict[str, Any],
    X: pd.DataFrame,
    *,
    top_features: int = 5,
) -> list[dict[str, Any]]:
    """Build SHAP dependence plot data for the top features.

    For each top feature, returns the feature values, SHAP values, and
    the most-correlated interaction feature — ready for plotting.

    Args:
        shap_result: Output of ``compute_shap_tree_values``.
        X: Feature matrix used for SHAP (same rows as shap_result).
        top_features: Number of top features by mean |SHAP| to include.

    Returns:
        List of dicts, each with ``feature``, "feature_values", "shap_values",
        ``interaction_feature``, ``interaction_values``.  Empty list if SHAP
        was skipped or errored.
    """
    if shap_result.get("status") != "ok":
        return []

    values = np.asarray(shap_result["shap_values"])
    feature_names = shap_result["feature_names"]

    if values.ndim == 3:
        values = values[:, :, -1]

    # Rank features by mean |SHAP|
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_features]

    # Compute full correlation matrix once (vectorised)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(values.T)
    np.fill_diagonal(corr_matrix, 0.0)
    corr_matrix = np.abs(np.nan_to_num(corr_matrix))

    results: list[dict[str, Any]] = []
    for idx in top_idx:
        fname = feature_names[idx]
        feat_vals = np.asarray(X[fname], dtype=float) if fname in X.columns else np.zeros(len(X))
        shap_vals = values[:, idx]

        # Find most-correlated interaction feature (excluding self)
        best_j = int(np.argmax(corr_matrix[idx]))
        if corr_matrix[idx, best_j] > 0:
            interaction_feature = feature_names[best_j]
            interaction_vals = (
                np.asarray(X[interaction_feature], dtype=float)
                if interaction_feature in X.columns
                else np.zeros(len(X))
            )
        else:
            interaction_feature = ""
            interaction_vals = np.zeros(len(X))

        results.append(
            {
                "feature": fname,
                "feature_values": feat_vals,
                "shap_values": shap_vals,
                "interaction_feature": interaction_feature,
                "interaction_values": interaction_vals,
            }
        )

    return results


def build_shap_summary_table(
    shap_result: dict[str, Any],
) -> pd.DataFrame:
    """Convert SHAP values to a feature-importance summary table.

    Args:
        shap_result: Output of ``compute_shap_values``.

    Returns:
        DataFrame with columns ``feature``, ``mean_abs_shap``, ``std_shap``.
        Empty if shap was skipped or errored.
    """
    if shap_result.get("status") != "ok":
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "std_shap"])

    values = np.asarray(shap_result["shap_values"])
    feature_names = shap_result["feature_names"]

    # Handle 3-D arrays (classification: classes × samples × features)
    if values.ndim == 3:
        values = values[:, :, -1]  # positive-class SHAP

    mean_abs = np.abs(values).mean(axis=0)
    std = np.abs(values).std(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "std_shap": std,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
