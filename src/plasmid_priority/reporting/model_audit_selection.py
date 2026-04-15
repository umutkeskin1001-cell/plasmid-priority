"""Model selection audit tables for model_audit.

Extracted from model_audit.py for maintainability.
Contains candidate universe, primary model selection, benchmark protocol,
scorecard, and single-model finalist/decision tables.
"""

from __future__ import annotations

from plasmid_priority.reporting.model_audit import (  # noqa: F401
    build_benchmark_protocol_table,
    build_candidate_universe_table,
    build_consensus_candidate_ranking,
    build_frozen_scientific_acceptance_audit,
    build_model_selection_scorecard,
    build_official_benchmark_panel,
    build_primary_model_selection_summary,
    build_single_model_finalist_audit,
    build_single_model_official_decision,
    build_single_model_pareto_finalists,
    build_single_model_selection_adjusted_permutation_null,
)

__all__ = [
    "build_benchmark_protocol_table",
    "build_candidate_universe_table",
    "build_consensus_candidate_ranking",
    "build_frozen_scientific_acceptance_audit",
    "build_model_selection_scorecard",
    "build_official_benchmark_panel",
    "build_primary_model_selection_summary",
    "build_single_model_finalist_audit",
    "build_single_model_official_decision",
    "build_single_model_pareto_finalists",
    "build_single_model_selection_adjusted_permutation_null",
]
