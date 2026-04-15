"""Robustness and stability audit tables for model_audit.

Extracted from model_audit.py for maintainability.
Contains bootstrap stability, permutation nulls, negative controls,
future sentinels, temporal drift, and group holdout audits.
"""

from __future__ import annotations

from plasmid_priority.reporting.model_audit import (  # noqa: F401
    build_blocked_holdout_summary,
    build_future_sentinel_audit,
    build_group_holdout_performance,
    build_logistic_implementation_audit,
    build_magic_number_sensitivity_table,
    build_model_simplicity_summary,
    build_negative_control_audit,
    build_permutation_null_tables,
    build_priority_bootstrap_stability_table,
    build_selection_adjusted_permutation_null,
    build_sleeper_threat_table,
    build_source_balance_resampling_table,
    build_temporal_drift_summary,
    build_temporal_rank_stability_table,
    build_variant_rank_consistency_table,
)

__all__ = [
    "build_blocked_holdout_summary",
    "build_future_sentinel_audit",
    "build_group_holdout_performance",
    "build_logistic_implementation_audit",
    "build_magic_number_sensitivity_table",
    "build_model_simplicity_summary",
    "build_negative_control_audit",
    "build_permutation_null_tables",
    "build_priority_bootstrap_stability_table",
    "build_selection_adjusted_permutation_null",
    "build_sleeper_threat_table",
    "build_source_balance_resampling_table",
    "build_temporal_drift_summary",
    "build_temporal_rank_stability_table",
    "build_variant_rank_consistency_table",
]
