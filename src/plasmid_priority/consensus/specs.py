"""Typed specs for the consensus branch."""

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

CONSENSUS_BENCHMARK_NAME = "consensus_v1"
CONSENSUS_PRIMARY_MODEL_NAME = "operational_consensus"
CONSENSUS_CONSERVATIVE_MODEL_NAME = "operational_consensus"
CONSENSUS_GOVERNANCE_MODEL_NAME = "operational_consensus"
CONSENSUS_RESEARCH_MODEL_NAME = "research_consensus"

DEFAULT_CONSENSUS_BENCHMARK = BranchBenchmarkSpec(
    name=CONSENSUS_BENCHMARK_NAME,
    split_year=2015,
    horizon_years=5,
    assignment_mode="training_only",
    entity_id_column="backbone_id",
    label_column="spread_label",
    outcome_column="n_new_countries_future",
    required_columns=(
        "backbone_id",
        "spread_label",
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
    future_columns=(),
    positive_threshold=3.0,
    min_positive_conditions=1,
)

CONSENSUS_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "operational_consensus": (
        "p_geo",
        "p_bio_transfer",
        "p_clinical_hazard",
    ),
    "research_consensus": (
        "p_geo",
        "p_bio_transfer",
        "p_clinical_hazard",
        "confidence_geo",
        "confidence_bio_transfer",
        "confidence_clinical_hazard",
        "ood_geo",
        "ood_bio_transfer",
        "ood_clinical_hazard",
        "branch_agreement_score",
    ),
}

DEFAULT_CONSENSUS_FIT_CONFIG_PAYLOAD: dict[str, dict[str, Any]] = {
    "operational_consensus": {
        "l2": 0.0,
        "sample_weight_mode": None,
        "max_iter": 1,
        "calibration": "none",
    },
    "research_consensus": {
        "l2": 0.0,
        "sample_weight_mode": None,
        "max_iter": 1,
        "calibration": "none",
    },
}

CONSENSUS_CORE_MODEL_NAMES: tuple[str, ...] = ("operational_consensus",)
CONSENSUS_RESEARCH_MODEL_NAMES: tuple[str, ...] = ("research_consensus",)


def load_consensus_config(
    config: Mapping[str, Any] | ConsensusConfig | None = None,
) -> ConsensusConfig:
    return load_branch_config(
        "consensus",
        config,
        benchmark_defaults=DEFAULT_CONSENSUS_BENCHMARK,
        primary_model_name=CONSENSUS_PRIMARY_MODEL_NAME,
        conservative_model_name=CONSENSUS_CONSERVATIVE_MODEL_NAME,
        fallback_model_name=CONSENSUS_GOVERNANCE_MODEL_NAME,
        research_model_name=CONSENSUS_RESEARCH_MODEL_NAME,
        core_model_names=CONSENSUS_CORE_MODEL_NAMES,
        research_model_names=CONSENSUS_RESEARCH_MODEL_NAMES,
        ablation_model_names=(),
        feature_sets=CONSENSUS_FEATURE_SETS,
        fit_config=DEFAULT_CONSENSUS_FIT_CONFIG_PAYLOAD,
    )


ConsensusBenchmarkSpec = BranchBenchmarkSpec
ConsensusConfig = BranchConfig
ConsensusFitConfig = BranchFitConfig


def resolve_consensus_model_names(
    config: Mapping[str, Any] | ConsensusConfig | None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    consensus_config = (
        config if isinstance(config, ConsensusConfig) else load_consensus_config(config)
    )
    return resolve_branch_model_names(
        consensus_config,
        include_research=include_research,
        include_ablation=include_ablation,
    )


def resolve_consensus_fit_config(
    config: Mapping[str, Any] | ConsensusConfig | None,
    model_name: str,
) -> ConsensusFitConfig:
    consensus_config = (
        config if isinstance(config, ConsensusConfig) else load_consensus_config(config)
    )
    return resolve_branch_fit_config(consensus_config, model_name)
