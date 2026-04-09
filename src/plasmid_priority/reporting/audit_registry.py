"""Named schema contracts for validation and benchmark audit artifacts."""

from __future__ import annotations

import pandas as pd

from plasmid_priority.reporting.artifact_contracts import (
    ReportArtifactContract,
    validate_report_artifact_contract,
)

SELECTION_ADJUSTED_PERMUTATION_DETAIL = "selection_adjusted_permutation_detail"
SELECTION_ADJUSTED_PERMUTATION_SUMMARY = "selection_adjusted_permutation_summary"
BENCHMARK_PROTOCOL = "benchmark_protocol"
OFFICIAL_BENCHMARK_PANEL = "official_benchmark_panel"
FROZEN_SCIENTIFIC_ACCEPTANCE_AUDIT = "frozen_scientific_acceptance_audit"


AUDIT_ARTIFACT_CONTRACTS: dict[str, ReportArtifactContract] = {
    SELECTION_ADJUSTED_PERMUTATION_DETAIL: ReportArtifactContract(
        required_columns=(
            "permutation_index",
            "selection_scope",
            "benchmark_surface_hash",
            "n_models_in_scope",
            "selected_model_name",
            "selected_null_roc_auc",
            "selected_null_average_precision",
            "effective_permutations",
            "stopping_reason",
            "selection_adjusted_p_max",
        ),
    ),
    SELECTION_ADJUSTED_PERMUTATION_SUMMARY: ReportArtifactContract(
        required_columns=(
            "model_name",
            "null_protocol",
            "selection_scope",
            "selection_reference_model",
            "benchmark_surface_hash",
            "n_models_in_scope",
            "n_permutations",
            "effective_permutations",
            "stopping_reason",
            "selection_adjusted_p_max",
            "observed_roc_auc",
            "observed_average_precision",
            "null_roc_auc_mean",
            "null_roc_auc_std",
            "null_roc_auc_q975",
            "p_ci_lower",
            "p_ci_upper",
            "selection_adjusted_empirical_p_roc_auc",
            "null_average_precision_mean",
            "null_average_precision_std",
            "null_average_precision_q975",
            "selection_adjusted_empirical_p_average_precision",
            "modal_selected_model_name",
            "modal_selected_model_share",
        ),
        unique_key="model_name",
    ),
    BENCHMARK_PROTOCOL: ReportArtifactContract(
        required_columns=(
            "model_name",
            "benchmark_role",
            "benchmark_status",
            "benchmark_track",
            "model_family",
            "roc_auc",
            "average_precision",
            "selection_rationale",
            "benchmark_guardrail_status",
        ),
    ),
    OFFICIAL_BENCHMARK_PANEL: ReportArtifactContract(
        required_columns=(
            "model_name",
            "benchmark_role",
            "benchmark_status",
            "benchmark_track",
            "model_family",
            "roc_auc",
            "average_precision",
            "selection_rationale",
            "benchmark_guardrail_status",
        ),
    ),
    FROZEN_SCIENTIFIC_ACCEPTANCE_AUDIT: ReportArtifactContract(
        required_columns=(
            "model_name",
            "model_track",
            "selection_rank",
            "roc_auc",
            "average_precision",
            "matched_knownness_weighted_roc_auc",
            "knownness_matched_gap",
            "source_holdout_weighted_roc_auc",
            "source_holdout_gap",
            "spatial_holdout_roc_auc",
            "spatial_holdout_gap",
            "ece",
            "selection_adjusted_empirical_p_roc_auc",
            "matched_knownness_gate_pass",
            "source_holdout_gate_pass",
            "spatial_holdout_gate_pass",
            "calibration_gate_pass",
            "selection_adjusted_gate_pass",
            "leakage_review_gate_pass",
            "scientific_acceptance_scored",
            "scientific_acceptance_flag",
            "scientific_acceptance_status",
            "scientific_acceptance_failed_criteria",
            "matched_knownness_gap_min",
            "source_holdout_gap_min",
            "spatial_holdout_gap_min",
            "ece_max",
            "selection_adjusted_p_max",
        ),
        unique_key="model_name",
    ),
}


def validate_audit_artifact(frame: pd.DataFrame, *, artifact_name: str) -> None:
    try:
        contract = AUDIT_ARTIFACT_CONTRACTS[artifact_name]
    except KeyError as exc:
        raise KeyError(f"Unknown audit artifact contract: {artifact_name}") from exc
    validate_report_artifact_contract(frame, artifact_name=artifact_name, contract=contract)
