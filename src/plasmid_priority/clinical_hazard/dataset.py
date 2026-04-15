"""Dataset assembly helpers for the clinical hazard branch."""

from __future__ import annotations

from typing import Any

import pandas as pd

from plasmid_priority.clinical_hazard.contracts import (
    ClinicalHazardInputContract,
    build_clinical_hazard_input_contract,
    validate_clinical_hazard_input_contract,
)
from plasmid_priority.clinical_hazard.features import build_clinical_hazard_features
from plasmid_priority.clinical_hazard.specs import (
    ClinicalHazardConfig,
    load_clinical_hazard_config,
)
from plasmid_priority.shared.branching import (
    BranchDataset,
    build_branch_dataset_from_prepared,
    prepare_branch_scored_table,
)
from plasmid_priority.shared.labels import build_clinical_hazard_labels

ClinicalHazardDataset = BranchDataset


def _build_label_frame(
    scored_frame: pd.DataFrame,
    records: pd.DataFrame | None,
    pd_metadata: pd.DataFrame | None,
    split_year: int,
    horizon_years: int,
) -> pd.DataFrame:
    label_frame = build_clinical_hazard_labels(
        records if records is not None else scored_frame,
        pd_metadata=pd_metadata,
        split_year=split_year,
        horizon_years=horizon_years,
    )
    if label_frame.empty or "training_only_future_unseen_backbone_flag" not in scored_frame.columns:
        return label_frame
    unseen_backbones = (
        scored_frame.loc[
            scored_frame["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool),
            "backbone_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    if not unseen_backbones:
        return label_frame
    masked = label_frame.copy()
    masked.loc[
        masked["backbone_id"].astype(str).isin(unseen_backbones), "clinical_hazard_label"
    ] = pd.NA
    masked.loc[
        masked["backbone_id"].astype(str).isin(unseen_backbones),
        "clinical_hazard_label_reason",
    ] = "future_unseen_or_missing_future_window"
    return masked


def prepare_clinical_hazard_scored_table(
    scored: pd.DataFrame,
    *,
    config: ClinicalHazardConfig | dict[str, Any] | None = None,
    contract: ClinicalHazardInputContract | None = None,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label: str = "clinical hazard scored table",
) -> pd.DataFrame:
    clinical_config = (
        config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    )
    clinical_contract = contract or build_clinical_hazard_input_contract()
    prepared = prepare_branch_scored_table(
        scored,
        config=clinical_config,
        contract=clinical_contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=lambda scored_frame, _records, split_year, horizon_years: _build_label_frame(
            scored_frame,
            records,
            pd_metadata,
            split_year,
            horizon_years,
        ),
        feature_builder=build_clinical_hazard_features,
        branch_label_column="clinical_hazard_label",
    )
    validate_clinical_hazard_input_contract(prepared, contract=clinical_contract, label=label)
    prepared["clinical_hazard_primary_model_name"] = clinical_config.primary_model_name
    prepared["clinical_hazard_conservative_model_name"] = clinical_config.conservative_model_name
    return prepared


def prepare_clinical_hazard_dataset(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: ClinicalHazardConfig | dict[str, Any] | None = None,
    contract: ClinicalHazardInputContract | None = None,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label: str = "clinical hazard scored table",
) -> ClinicalHazardDataset:
    clinical_config = (
        config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    )
    clinical_contract = contract or build_clinical_hazard_input_contract()
    prepared = prepare_clinical_hazard_scored_table(
        scored,
        config=clinical_config,
        contract=clinical_contract,
        records=records,
        pd_metadata=pd_metadata,
        label=label,
    )
    return build_branch_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=clinical_config,
        contract=clinical_contract,
        label_column="clinical_hazard_label",
    )


def build_clinical_hazard_dataset_from_prepared(
    prepared: pd.DataFrame,
    *,
    model_name: str,
    config: ClinicalHazardConfig | dict[str, Any] | None = None,
    contract: ClinicalHazardInputContract | None = None,
) -> ClinicalHazardDataset:
    clinical_config = (
        config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    )
    clinical_contract = contract or build_clinical_hazard_input_contract()
    return build_branch_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=clinical_config,
        contract=clinical_contract,
        label_column="clinical_hazard_label",
    )


def resolve_clinical_hazard_dataset_model_names(
    config: dict[str, Any] | ClinicalHazardConfig | None = None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    clinical_config = (
        config if isinstance(config, ClinicalHazardConfig) else load_clinical_hazard_config(config)
    )
    names = list(clinical_config.core_model_names)
    names.extend([clinical_config.primary_model_name, clinical_config.conservative_model_name])
    if clinical_config.fallback_model_name:
        names.append(clinical_config.fallback_model_name)
    if include_research:
        names.extend(clinical_config.research_model_names)
        if clinical_config.research_model_name:
            names.append(clinical_config.research_model_name)
    if include_ablation:
        names.extend(clinical_config.ablation_model_names)
    return tuple(dict.fromkeys(str(name) for name in names if str(name).strip()))
