"""Shared helpers for report pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from plasmid_priority.utils.files import ensure_directory


class TableRouter:
    """Route canonical tables to core or diagnostics directories."""

    def __init__(self, core_dir: Path, diag_dir: Path, analysis_dir: Path) -> None:
        self.core_dir = core_dir
        self.diag_dir = diag_dir
        self.analysis_dir = analysis_dir
        self.core_files = {
            "model_metrics.tsv",
            "model_selection_scorecard.tsv",
            "benchmark_protocol.tsv",
            "official_benchmark_panel.tsv",
            "model_comparison_summary.tsv",
            "model_selection_summary.tsv",
            "decision_yield_summary.tsv",
            "threshold_utility_summary.tsv",
            "report_overview.tsv",
            "candidate_evidence_matrix.tsv",
            "candidate_threshold_flip.tsv",
            "candidate_universe.tsv",
            "candidate_portfolio.tsv",
            "candidate_case_studies.tsv",
            "confirmatory_cohort_summary.tsv",
            "consensus_shortlist.tsv",
            "core_model_coefficients.tsv",
            "frozen_scientific_acceptance_audit.tsv",
            "top_primary_candidates.tsv",
            "operational_risk_watchlist.tsv",
            "blocked_holdout_summary.tsv",
            "spatial_holdout_summary.tsv",
            "headline_validation_summary.tsv",
            "single_model_official_decision.tsv",
        }

    def __truediv__(self, name: str) -> Path:
        if name in self.core_files:
            return self.core_dir / name
        return self.diag_dir / name


DEFAULT_ROUTED_OUTPUT_FILES = (
    "top_primary_candidates.tsv",
    "consensus_shortlist.tsv",
    "candidate_evidence_matrix.tsv",
    "candidate_threshold_flip.tsv",
    "decision_yield_summary.tsv",
    "threshold_utility_summary.tsv",
    "report_overview.tsv",
    "decision_budget_curve.tsv",
    "false_negative_audit.tsv",
    "confirmatory_cohort_summary.tsv",
    "model_selection_scorecard.tsv",
    "frozen_scientific_acceptance_audit.tsv",
    "model_selection_summary.tsv",
    "benchmark_protocol.tsv",
    "official_benchmark_panel.tsv",
    "module_b_amr_class_comparison.tsv",
    "model_metrics.tsv",
    "sensitivity_summary.tsv",
    "source_stratified_consistency.tsv",
    "calibration_table.tsv",
    "model_family_summary.tsv",
    "model_subgroup_performance.tsv",
    "model_comparison_summary.tsv",
    "calibration_metrics.tsv",
    "blocked_holdout_calibration_summary.tsv",
    "primary_model_coefficients.tsv",
    "primary_model_coefficient_stability.tsv",
    "coefficient_stability_cv.tsv",
    "feature_dropout_importance.tsv",
    "source_balance_resampling.tsv",
    "group_holdout_performance.tsv",
    "blocked_holdout_summary.tsv",
    "single_model_pareto_screen.tsv",
    "single_model_pareto_finalists.tsv",
    "single_model_official_decision.tsv",
    "permutation_null_distribution.tsv",
    "permutation_null_summary.tsv",
    "selection_adjusted_permutation_null_distribution.tsv",
    "selection_adjusted_permutation_null_summary.tsv",
    "rolling_temporal_validation.tsv",
    "rolling_assignment_diagnostics.tsv",
    "candidate_rank_stability.tsv",
    "candidate_variant_consistency.tsv",
    "candidate_multiverse_stability.tsv",
    "model_simplicity_summary.tsv",
    "knownness_audit_summary.tsv",
    "knownness_stratified_performance.tsv",
    "negative_control_audit.tsv",
    "future_sentinel_audit.tsv",
    "logistic_implementation_audit.tsv",
    "logistic_convergence_audit.tsv",
    "outcome_robustness_grid.tsv",
    "threshold_sensitivity_summary.tsv",
    "l2_sensitivity_summary.tsv",
    "weighting_sensitivity_summary.tsv",
    "h_feature_diagnostics.tsv",
    "score_axis_summary.tsv",
    "score_distribution_diagnostics.tsv",
    "component_floor_diagnostics.tsv",
    "temporal_drift_summary.tsv",
    "country_quality_summary.tsv",
    "backbone_purity_atlas.tsv",
    "assignment_confidence_summary.tsv",
    "incremental_value_over_baseline.tsv",
    "novelty_specialist_metrics.tsv",
    "novelty_specialist_predictions.tsv",
    "adaptive_gated_metrics.tsv",
    "adaptive_gated_predictions.tsv",
    "gate_consistency_audit.tsv",
    "knownness_matched_validation.tsv",
    "matched_stratum_propensity_audit.tsv",
    "nonlinear_deconfounding_audit.tsv",
    "country_upload_propensity.tsv",
    "macro_region_jump_outcome.tsv",
    "secondary_outcome_performance.tsv",
    "weighted_country_outcome_audit.tsv",
    "new_country_count_audit.tsv",
    "metadata_quality_summary.tsv",
    "event_timing_outcomes.tsv",
    "exposure_adjusted_event_outcomes.tsv",
    "exposure_adjusted_outcome_audit.tsv",
    "ordinal_outcome_audit.tsv",
    "country_missingness_bounds.tsv",
    "country_missingness_sensitivity.tsv",
    "geographic_jump_distance_outcome.tsv",
    "duplicate_completeness_change_audit.tsv",
    "amr_uncertainty_summary.tsv",
    "mash_similarity_graph.tsv",
    "counterfactual_shortlist_comparison.tsv",
    "annual_candidate_freeze_summary.tsv",
    "pathogen_detection_group_comparison.tsv",
    "prospective_candidate_freeze.tsv",
    "candidate_dossiers.tsv",
    "candidate_risk_flags.tsv",
    "novelty_watchlist.tsv",
    "novelty_margin_summary.tsv",
    "candidate_portfolio.tsv",
    "candidate_case_studies.tsv",
    "operational_risk_watchlist.tsv",
    "operational_risk_dictionary_full.tsv",
    "module_f_backbone_identity.tsv",
    "module_f_enrichment.tsv",
    "module_f_top_hits.tsv",
)

DEFAULT_CORE_OUTPUT_FILES = ("core_model_coefficients.tsv", "spatial_holdout_summary.tsv")


def build_table_router(reports_dir: Path, analysis_dir: Path) -> tuple[Path, Path, TableRouter]:
    core_dir = reports_dir / "core_tables"
    diag_dir = reports_dir / "diagnostic_tables"
    ensure_directory(core_dir)
    ensure_directory(diag_dir)
    return core_dir, diag_dir, TableRouter(core_dir, diag_dir, analysis_dir)


def register_default_report_outputs(
    run: object,
    *,
    final_tables_dir: TableRouter,
    core_dir: Path,
    jury_brief_path: Path,
    turkish_summary_path: Path,
    executive_summary_path: Path,
    pitch_notes_path: Path,
    headline_summary_path: Path,
    family_summary_path: Path,
) -> None:
    record_output = getattr(run, "record_output")
    for path in (
        jury_brief_path,
        turkish_summary_path,
        executive_summary_path,
        pitch_notes_path,
        headline_summary_path,
        family_summary_path,
    ):
        record_output(path)
    for file_name in DEFAULT_ROUTED_OUTPUT_FILES:
        record_output(final_tables_dir / file_name)
    for file_name in DEFAULT_CORE_OUTPUT_FILES:
        record_output(core_dir / file_name)
