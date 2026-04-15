"""Typed specs for the bio transfer branch."""

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

BIO_TRANSFER_BENCHMARK_NAME = "bio_transfer_v1"
BIO_TRANSFER_PRIMARY_MODEL_NAME = "bio_transfer_governance"
BIO_TRANSFER_CONSERVATIVE_MODEL_NAME = "bio_transfer_baseline"
BIO_TRANSFER_GOVERNANCE_MODEL_NAME = "bio_transfer_governance"
BIO_TRANSFER_RESEARCH_MODEL_NAME = "bio_transfer_research"

DEFAULT_BIO_TRANSFER_BENCHMARK = BranchBenchmarkSpec(
    name=BIO_TRANSFER_BENCHMARK_NAME,
    split_year=2015,
    horizon_years=5,
    assignment_mode="training_only",
    entity_id_column="backbone_id",
    label_column="bio_transfer_label",
    outcome_column="future_new_host_genera_count",
    required_columns=(
        "backbone_id",
        "bio_transfer_label",
        "future_new_host_genera_count",
        "future_new_host_families_count",
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
        "n_new_host_genera_future",
        "n_new_host_families_future",
        "host_phylo_dispersion_gain_future",
    ),
    positive_threshold=2.0,
    min_positive_conditions=1,
)

BIO_TRANSFER_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "bio_transfer_baseline": (
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "orit_support",
    ),
    "bio_transfer_parsimonious": (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "mobility_support_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
    ),
    "bio_transfer_enriched": (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "H_phylogenetic_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "mobility_support_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "host_phylogenetic_dispersion_norm",
        "host_taxon_evenness_norm",
        "ecology_context_diversity_norm",
        "H_external_host_range_norm",
        "host_breadth_mobility_synergy_norm",
        "T_H_obs_synergy_norm",
        "T_A_synergy_norm",
    ),
    "bio_transfer_governance": (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "H_phylogenetic_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "mobility_support_norm",
        "mob_suite_support_norm",
        "relaxase_support_norm",
        "conjugation_support_norm",
        "replicon_complexity_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "host_phylogenetic_dispersion_norm",
        "host_taxon_evenness_norm",
        "ecology_context_diversity_norm",
        "mash_graph_novelty_score",
        "mash_graph_bridge_fraction",
        "H_external_host_range_norm",
        "host_breadth_mobility_synergy_norm",
        "mobility_host_synergy_norm",
        "T_H_obs_synergy_norm",
        "T_A_synergy_norm",
    ),
    "bio_transfer_research": (
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "T_eff_norm",
        "H_obs_specialization_norm",
        "H_phylogenetic_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "orit_support",
        "mobility_support_norm",
        "mob_suite_support_norm",
        "relaxase_support_norm",
        "conjugation_support_norm",
        "replicon_complexity_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "host_phylogenetic_dispersion_norm",
        "host_taxon_evenness_norm",
        "ecology_context_diversity_norm",
        "H_external_host_range_norm",
        "host_breadth_mobility_synergy_norm",
        "mobility_host_synergy_norm",
        "mash_graph_novelty_score",
        "mash_graph_bridge_fraction",
        "T_H_obs_synergy_norm",
        "T_A_synergy_norm",
    ),
}

DEFAULT_BIO_TRANSFER_FIT_CONFIG_PAYLOAD: dict[str, dict[str, Any]] = {
    "bio_transfer_baseline": {
        "l2": 1.5,
        "sample_weight_mode": None,
        "max_iter": 250,
        "calibration": "none",
    },
    "bio_transfer_parsimonious": {
        "l2": 2.0,
        "sample_weight_mode": "source_balanced",
        "max_iter": 400,
        "calibration": "isotonic",
    },
    "bio_transfer_enriched": {
        "l2": 2.5,
        "sample_weight_mode": "source_balanced",
        "max_iter": 500,
        "calibration": "isotonic",
    },
    "bio_transfer_governance": {
        "l2": 3.0,
        "sample_weight_mode": "source_balanced+knownness_balanced",
        "max_iter": 600,
        "calibration": "isotonic",
        "agreement_review_threshold": 0.75,
        "support_knownness_threshold": 0.30,
        "ood_knownness_threshold": 0.10,
        "confidence_review_threshold": 0.45,
    },
    "bio_transfer_research": {
        "l2": 3.0,
        "sample_weight_mode": "class_balanced+knownness_balanced",
        "max_iter": 500,
        "calibration": "platt",
    },
}

BIO_TRANSFER_CORE_MODEL_NAMES: tuple[str, ...] = (
    "bio_transfer_baseline",
    "bio_transfer_parsimonious",
    "bio_transfer_enriched",
    "bio_transfer_governance",
)
BIO_TRANSFER_RESEARCH_MODEL_NAMES: tuple[str, ...] = ("bio_transfer_research",)


def load_bio_transfer_config(
    config: Mapping[str, Any] | BioTransferConfig | None = None,
) -> BioTransferConfig:
    return load_branch_config(
        "bio_transfer",
        config,
        benchmark_defaults=DEFAULT_BIO_TRANSFER_BENCHMARK,
        primary_model_name=BIO_TRANSFER_PRIMARY_MODEL_NAME,
        conservative_model_name=BIO_TRANSFER_CONSERVATIVE_MODEL_NAME,
        fallback_model_name=BIO_TRANSFER_GOVERNANCE_MODEL_NAME,
        research_model_name=BIO_TRANSFER_RESEARCH_MODEL_NAME,
        core_model_names=BIO_TRANSFER_CORE_MODEL_NAMES,
        research_model_names=BIO_TRANSFER_RESEARCH_MODEL_NAMES,
        ablation_model_names=(),
        feature_sets=BIO_TRANSFER_FEATURE_SETS,
        fit_config=DEFAULT_BIO_TRANSFER_FIT_CONFIG_PAYLOAD,
    )


BioTransferBenchmarkSpec = BranchBenchmarkSpec
BioTransferConfig = BranchConfig
BioTransferFitConfig = BranchFitConfig


def resolve_bio_transfer_model_names(
    config: Mapping[str, Any] | BioTransferConfig | None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    return resolve_branch_model_names(
        bio_config,
        include_research=include_research,
        include_ablation=include_ablation,
    )


def resolve_bio_transfer_fit_config(
    config: Mapping[str, Any] | BioTransferConfig | None,
    model_name: str,
) -> BioTransferFitConfig:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    return resolve_branch_fit_config(bio_config, model_name)
