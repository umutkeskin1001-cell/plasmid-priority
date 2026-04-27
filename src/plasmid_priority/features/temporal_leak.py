"""Temporal leak detection utilities for feature validation.

This module provides functions for detecting temporal leakage in features,
which occurs when future information is inadvertently used in training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def detect_temporal_leak(
    scored: pd.DataFrame,
    *,
    year_column: str = "year",
    label_column: str = "spread_label",
) -> dict[str, Any]:
    """Detect temporal leakage in features.

    Temporal leakage occurs when features contain information from future
    time points that would not be available at prediction time.

    Args:
        scored: Input dataframe with features and temporal information
        year_column: Column name containing year information
        label_column: Column name containing the target label

    Returns:
        Dictionary with leak detection results:
        - has_leak: Boolean indicating if temporal leak was detected
        - leak_features: List of features with potential temporal leak
        - leak_severity: Severity score (0.0-1.0)
    """
    if year_column not in scored.columns:
        return {
            "has_leak": False,
            "leak_features": [],
            "leak_severity": 0.0,
            "error": f"Year column '{year_column}' not found",
        }

    if label_column not in scored.columns:
        return {
            "has_leak": False,
            "leak_features": [],
            "leak_severity": 0.0,
            "error": f"Label column '{label_column}' not found",
        }

    working = scored.copy()
    working = working.loc[working[label_column].notna()]

    if working.empty:
        return {
            "has_leak": False,
            "leak_features": [],
            "leak_severity": 0.0,
            "error": "No valid labeled data",
        }

    # Check for features that correlate perfectly with year
    numeric_cols = working.select_dtypes(include=[np.number]).columns
    leak_features = []

    for col in numeric_cols:
        if col in (year_column, label_column):
            continue

        try:
            col_values = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
            year_values = pd.to_numeric(working[year_column], errors="coerce").fillna(0.0)

            # Check if feature is perfectly correlated with year
            correlation = col_values.corr(year_values)

            if abs(correlation) > 0.95:
                leak_features.append(
                    {
                        "feature": col,
                        "correlation": float(correlation),
                        "severity": "high" if abs(correlation) > 0.99 else "medium",
                    }
                )
        except Exception as exc:
            LOGGER.warning(
                "Caught suppressed exception: %s",
                exc,
                exc_info=True,
            )
            continue

    has_leak = len(leak_features) > 0
    leak_severity = 0.0
    for record in leak_features:
        correlation = float(
            pd.to_numeric(
                pd.Series([record.get("correlation", 0.0)]),
                errors="coerce",
            ).iloc[0],
        )
        if np.isfinite(correlation):
            leak_severity = max(leak_severity, abs(correlation))

    return {
        "has_leak": has_leak,
        "leak_features": leak_features,
        "leak_severity": leak_severity,
    }


def audit_feature_temporal_leak(
    scored: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    year_column: str = "year",
    label_column: str = "spread_label",
) -> pd.DataFrame:
    """Audit specific features for temporal leakage.

    Args:
        scored: Input dataframe with features
        feature_columns: Specific features to audit (None = all numeric features)
        year_column: Column name containing year information
        label_column: Column name containing the target label

    Returns:
        DataFrame with audit results for each feature
    """
    working = scored.copy()

    if feature_columns is None:
        feature_columns = working.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in feature_columns:
        if col in (year_column, label_column):
            continue

        if col not in working.columns:
            results.append(
                {
                    "feature": col,
                    "status": "missing",
                    "correlation": float("nan"),
                    "leak_detected": False,
                }
            )
            continue

        try:
            col_values = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
            year_values = pd.to_numeric(working[year_column], errors="coerce").fillna(0.0)

            correlation = col_values.corr(year_values)
            leak_detected = abs(correlation) > 0.95

            results.append(
                {
                    "feature": col,
                    "status": "ok",
                    "correlation": float(correlation),
                    "leak_detected": leak_detected,
                }
            )
        except Exception as e:
            results.append(
                {
                    "feature": col,
                    "status": f"error: {str(e)}",
                    "correlation": float("nan"),
                    "leak_detected": False,
                }
            )

    return pd.DataFrame(results)
