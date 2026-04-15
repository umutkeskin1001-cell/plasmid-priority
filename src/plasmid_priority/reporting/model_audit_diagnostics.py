"""Diagnostic audit tables for model_audit.

Extracted from model_audit.py for maintainability.
Contains feature diagnostics, knownness audits, novelty margins,
gate consistency, score distributions, and component floor checks.
"""

from __future__ import annotations

from plasmid_priority.reporting.model_audit import (  # noqa: F401
    build_amrfinder_coverage_table,
    build_component_floor_diagnostics,
    build_gate_consistency_audit,
    build_h_feature_diagnostics,
    build_knownness_audit_tables,
    build_novelty_margin_summary,
    build_score_axis_summary,
    build_score_distribution_diagnostics,
)

__all__ = [
    "build_amrfinder_coverage_table",
    "build_component_floor_diagnostics",
    "build_gate_consistency_audit",
    "build_h_feature_diagnostics",
    "build_knownness_audit_tables",
    "build_novelty_margin_summary",
    "build_score_axis_summary",
    "build_score_distribution_diagnostics",
]
