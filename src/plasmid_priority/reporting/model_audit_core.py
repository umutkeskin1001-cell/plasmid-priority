"""Core model audit helpers and shared utilities.

Extracted from model_audit.py for maintainability.
Contains shared TypedDicts, helper functions, model family summary,
model comparison, and model subgroup performance tables.
"""

from __future__ import annotations

from plasmid_priority.reporting.model_audit import (  # noqa: F401
    _active_model_metrics,
    _BootstrapCandidateStats,
    _guardrail_loss_from_scorecard_row,
    _guardrail_loss_series,
    _model_track_summary,
    _rank_percentile_series,
    _resolve_parallel_jobs,
    _safe_model_track,
    _safe_spearman,
    _select_governance_scorecard_row,
    _stable_unit_interval,
    build_model_comparison_table,
    build_model_family_summary,
    build_model_subgroup_performance,
    sanitize_adaptive_gated_predictions,
)

__all__ = [
    "_BootstrapCandidateStats",
    "_active_model_metrics",
    "_guardrail_loss_from_scorecard_row",
    "_guardrail_loss_series",
    "_model_track_summary",
    "_rank_percentile_series",
    "_resolve_parallel_jobs",
    "_safe_model_track",
    "_safe_spearman",
    "_select_governance_scorecard_row",
    "_stable_unit_interval",
    "build_model_comparison_table",
    "build_model_family_summary",
    "build_model_subgroup_performance",
    "sanitize_adaptive_gated_predictions",
]
