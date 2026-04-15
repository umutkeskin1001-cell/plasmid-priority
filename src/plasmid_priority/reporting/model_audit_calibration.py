"""Calibration audit tables for model_audit.

Extracted from model_audit.py for maintainability.
Contains calibration metric tables, blocked holdout calibration,
and internal calibration helpers.
"""

from __future__ import annotations

# Re-export from the canonical module until full migration is complete.
# After migration, the implementations below will be moved here and
# model_audit.py will import from this module instead.
from plasmid_priority.reporting.model_audit import (  # noqa: F401
    _apply_calibration_transform,
    _calibration_feature_matrix,
    _calibration_metrics_from_arrays,
    _clip_probability_array,
    _fit_calibration_transform,
    _nested_calibrated_predictions,
    build_blocked_holdout_calibration_summary,
    build_blocked_holdout_calibration_table,
    build_calibration_metric_table,
)

__all__ = [
    "_apply_calibration_transform",
    "_calibration_feature_matrix",
    "_calibration_metrics_from_arrays",
    "_clip_probability_array",
    "_fit_calibration_transform",
    "_nested_calibrated_predictions",
    "build_blocked_holdout_calibration_summary",
    "build_blocked_holdout_calibration_table",
    "build_calibration_metric_table",
]
