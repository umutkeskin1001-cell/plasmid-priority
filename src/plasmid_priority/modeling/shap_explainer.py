"""SHAP-based model interpretability for Plasmid Priority.

Provides ``compute_shap_values`` and ``build_shap_summary_table`` which
wrap the ``shap`` library (optional dependency) to produce feature
importance explanations for any fitted branch model.

If ``shap`` is not installed, all public functions return empty results
with a ``skipped`` status instead of raising.
"""

from __future__ import annotations

import logging
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

    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "std_shap": std,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
