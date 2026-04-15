"""Typed specs for the clinical hazard branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasmid_priority.shared.specs import (
    BranchBenchmarkSpec,
    BranchConfig,
    BranchFitConfig,
    load_branch_config,
    resolve_branch_fit_config,
    resolve_branch_model_names,
)

CLINICAL_HAZARD_BENCHMARK_NAME = "clinical_hazard_v1"
CLINICAL_HAZARD_PRIMARY_MODEL_NAME = "clinical_hazard_governance"
CLINICAL_HAZARD_CONSERVATIVE_MODEL_NAME = "clinical_hazard_baseline"
CLINICAL_HAZARD_GOVERNANCE_MODEL_NAME = "clinical_hazard_governance"
CLINICAL_HAZARD_RESEARCH_MODEL_NAME = "clinical_hazard_research"

DEFAULT_CLINICAL_HAZARD_BENCHMARK = BranchBenchmarkSpec(
    name=CLINICAL_HAZARD_BENCHMARK_NAME,
    split_year=2015,
    horizon_years=5,
    assignment_mode="training_only",
    entity_id_column="backbone_id",
    label_column="clinical_hazard_label",
    outcome_column="clinical_fraction_future",
    required_columns=(
        "backbone_id",
        "clinical_hazard_label",
        "clinical_fraction_future",
        "last_resort_fraction_future",
        "mdr_proxy_fraction_future",
        "pd_clinical_support_future",
        "split_year",
        "backbone_assignment_mode",
        "max_resolved_year_train",
        "min_resolved_year_test",
        "training_only_future_unseen_backbone_flag",
    ),
    temporal_metadata_columns=(
        "split_year",
        "backbone_assignment_mode",
        "max_resolved_year_train",
        "min_resolved_year_test",
    ),
    future_columns=(
        "clinical_fraction_future",
        "last_resort_fraction_future",
        "mdr_proxy_fraction_future",
        "pd_clinical_support_future",
    ),
    positive_threshold=0.15,
    min_positive_conditions=2,
)

CLINICAL_HAZARD_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "clinical_hazard_baseline": (
        "A_eff_norm",
        "A_raw_norm",
        "coherence_score",
    ),
    "clinical_hazard_parsimonious": (
        "A_eff_norm",
        "A_raw_norm",
        "A_recurrence_norm",
        "coherence_score",
        "amr_gene_count_norm",
        "amr_class_count_norm",
        "clinical_context_fraction_norm",
        "last_resort_convergence_norm",
    ),
    "clinical_hazard_enriched": (
        "A_eff_norm",
        "A_raw_norm",
        "A_recurrence_norm",
        "coherence_score",
        "amr_gene_count_norm",
        "amr_class_count_norm",
        "amr_mechanism_diversity_norm",
        "clinical_context_fraction_norm",
        "pathogenic_context_fraction_norm",
        "mdr_proxy_fraction_norm",
        "xdr_proxy_fraction_norm",
        "pd_clinical_support_norm",
        "vfdb_cargo_burden_norm",
        "stress_response_burden_norm",
        "last_resort_convergence_norm",
        "amr_clinical_escalation_norm",
    ),
    "clinical_hazard_governance": (
        "A_eff_norm",
        "A_raw_norm",
        "A_recurrence_norm",
        "coherence_score",
        "amr_gene_count_norm",
        "amr_class_count_norm",
        "amr_mechanism_diversity_norm",
        "clinical_context_fraction_norm",
        "pathogenic_context_fraction_norm",
        "mdr_proxy_fraction_norm",
        "xdr_proxy_fraction_norm",
        "pd_clinical_support_norm",
        "vfdb_cargo_burden_norm",
        "stress_response_burden_norm",
        "last_resort_convergence_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "amr_support_norm",
        "amr_support_norm_residual",
        "metadata_support_depth_norm",
        "context_support_guard_norm",
        "H_external_host_range_norm",
        "A_clinical_context_synergy_norm",
        "A_host_range_synergy_norm",
        "A_novelty_synergy_norm",
        "amr_load_density_norm",
        "amr_clinical_escalation_norm",
        "amr_mdr_proxy_norm",
        "amr_xdr_proxy_norm",
        "silent_carrier_risk_norm",
        "evolutionary_jump_score_norm",
    ),
    "clinical_hazard_research": (
        "A_eff_norm",
        "A_raw_norm",
        "A_recurrence_norm",
        "coherence_score",
        "amr_gene_count_norm",
        "amr_class_count_norm",
        "amr_mechanism_diversity_norm",
        "clinical_context_fraction_norm",
        "pathogenic_context_fraction_norm",
        "mdr_proxy_fraction_norm",
        "xdr_proxy_fraction_norm",
        "pd_clinical_support_norm",
        "vfdb_cargo_burden_norm",
        "stress_response_burden_norm",
        "last_resort_convergence_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "amr_support_norm",
        "amr_support_norm_residual",
        "metadata_support_depth_norm",
        "context_support_guard_norm",
        "H_external_host_range_norm",
        "A_clinical_context_synergy_norm",
        "A_host_range_synergy_norm",
        "A_novelty_synergy_norm",
        "amr_load_density_norm",
        "amr_clinical_escalation_norm",
        "amr_mdr_proxy_norm",
        "amr_xdr_proxy_norm",
        "silent_carrier_risk_norm",
        "evolutionary_jump_score_norm",
    ),
}

DEFAULT_CLINICAL_HAZARD_FIT_CONFIG_PAYLOAD: dict[str, dict[str, Any]] = {
    "clinical_hazard_baseline": {
        "l2": 1.5,
        "sample_weight_mode": None,
        "max_iter": 250,
        "calibration": "none",
    },
    "clinical_hazard_parsimonious": {
        "l2": 2.0,
        "sample_weight_mode": "source_balanced",
        "max_iter": 400,
        "calibration": "isotonic",
    },
    "clinical_hazard_enriched": {
        "l2": 2.5,
        "sample_weight_mode": "source_balanced",
        "max_iter": 500,
        "calibration": "isotonic",
    },
    "clinical_hazard_governance": {
        "l2": 3.0,
        "sample_weight_mode": "source_balanced+knownness_balanced",
        "max_iter": 600,
        "calibration": "isotonic",
        "agreement_review_threshold": 0.75,
        "support_knownness_threshold": 0.30,
        "ood_knownness_threshold": 0.10,
        "confidence_review_threshold": 0.45,
    },
    "clinical_hazard_research": {
        "l2": 3.0,
        "sample_weight_mode": "class_balanced+knownness_balanced",
        "max_iter": 500,
        "calibration": "platt",
    },
}

CLINICAL_HAZARD_CORE_MODEL_NAMES: tuple[str, ...] = (
    "clinical_hazard_baseline",
    "clinical_hazard_parsimonious",
    "clinical_hazard_enriched",
    "clinical_hazard_governance",
)
CLINICAL_HAZARD_RESEARCH_MODEL_NAMES: tuple[str, ...] = ("clinical_hazard_research",)


def load_clinical_hazard_config(config: Mapping[str, Any] | None = None) -> ClinicalHazardConfig:
    return load_branch_config(
        "clinical_hazard",
        config,
        benchmark_defaults=DEFAULT_CLINICAL_HAZARD_BENCHMARK,
        primary_model_name=CLINICAL_HAZARD_PRIMARY_MODEL_NAME,
        conservative_model_name=CLINICAL_HAZARD_CONSERVATIVE_MODEL_NAME,
        fallback_model_name=CLINICAL_HAZARD_GOVERNANCE_MODEL_NAME,
        research_model_name=CLINICAL_HAZARD_RESEARCH_MODEL_NAME,
        core_model_names=CLINICAL_HAZARD_CORE_MODEL_NAMES,
        research_model_names=CLINICAL_HAZARD_RESEARCH_MODEL_NAMES,
        ablation_model_names=(),
        feature_sets=CLINICAL_HAZARD_FEATURE_SETS,
        fit_config=DEFAULT_CLINICAL_HAZARD_FIT_CONFIG_PAYLOAD,
    )


ClinicalHazardBenchmarkSpec = BranchBenchmarkSpec
ClinicalHazardConfig = BranchConfig
ClinicalHazardFitConfig = BranchFitConfig


def resolve_clinical_hazard_model_names(
    config: Mapping[str, Any] | ClinicalHazardConfig | None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    clinical_config = config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    return resolve_branch_model_names(
        clinical_config,
        include_research=include_research,
        include_ablation=include_ablation,
    )


def resolve_clinical_hazard_fit_config(
    config: Mapping[str, Any] | ClinicalHazardConfig | None,
    model_name: str,
) -> ClinicalHazardFitConfig:
    clinical_config = config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    return resolve_branch_fit_config(clinical_config, model_name)
