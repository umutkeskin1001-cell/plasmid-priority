"""Dataset assembly helpers for the bio transfer branch."""

from __future__ import annotations

from typing import Any

import pandas as pd

from plasmid_priority.bio_transfer.contracts import (
    BioTransferInputContract,
    build_bio_transfer_input_contract,
    validate_bio_transfer_input_contract,
)
from plasmid_priority.bio_transfer.features import build_bio_transfer_features
from plasmid_priority.bio_transfer.specs import (
    BioTransferConfig,
    load_bio_transfer_config,
)
from plasmid_priority.shared.branching import (
    BranchDataset,
    build_branch_dataset_from_prepared,
    prepare_branch_scored_table,
)
from plasmid_priority.shared.labels import build_bio_transfer_labels

BioTransferDataset = BranchDataset


def _build_label_frame(
    scored_frame: pd.DataFrame,
    records: pd.DataFrame | None,
    split_year: int,
    horizon_years: int,
) -> pd.DataFrame:
    label_frame = build_bio_transfer_labels(
        records if records is not None else scored_frame,
        split_year,
        horizon_years,
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
    masked.loc[masked["backbone_id"].astype(str).isin(unseen_backbones), "bio_transfer_label"] = (
        pd.NA
    )
    masked.loc[
        masked["backbone_id"].astype(str).isin(unseen_backbones),
        "bio_transfer_label_reason",
    ] = "future_unseen_or_missing_future_window"
    return masked


def prepare_bio_transfer_scored_table(
    scored: pd.DataFrame,
    *,
    config: BioTransferConfig | dict[str, Any] | None = None,
    contract: BioTransferInputContract | None = None,
    records: pd.DataFrame | None = None,
    label: str = "bio transfer scored table",
) -> pd.DataFrame:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    bio_contract = contract or build_bio_transfer_input_contract()
    prepared = prepare_branch_scored_table(
        scored,
        config=bio_config,
        contract=bio_contract,
        records=records,
        label_builder=_build_label_frame,
        feature_builder=build_bio_transfer_features,
        branch_label_column="bio_transfer_label",
    )
    validate_bio_transfer_input_contract(prepared, contract=bio_contract, label=label)
    prepared["bio_transfer_primary_model_name"] = bio_config.primary_model_name
    prepared["bio_transfer_conservative_model_name"] = bio_config.conservative_model_name
    return prepared


def prepare_bio_transfer_dataset(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: BioTransferConfig | dict[str, Any] | None = None,
    contract: BioTransferInputContract | None = None,
    records: pd.DataFrame | None = None,
    label: str = "bio transfer scored table",
) -> BioTransferDataset:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    bio_contract = contract or build_bio_transfer_input_contract()
    prepared = prepare_bio_transfer_scored_table(
        scored,
        config=bio_config,
        contract=bio_contract,
        records=records,
        label=label,
    )
    return build_branch_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=bio_config,
        contract=bio_contract,
        label_column="bio_transfer_label",
    )


def build_bio_transfer_dataset_from_prepared(
    prepared: pd.DataFrame,
    *,
    model_name: str,
    config: BioTransferConfig | dict[str, Any] | None = None,
    contract: BioTransferInputContract | None = None,
) -> BioTransferDataset:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    bio_contract = contract or build_bio_transfer_input_contract()
    return build_branch_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=bio_config,
        contract=bio_contract,
        label_column="bio_transfer_label",
    )


def resolve_bio_transfer_dataset_model_names(
    config: dict[str, Any] | BioTransferConfig | None = None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    bio_config = (
        config if isinstance(config, BioTransferConfig) else load_bio_transfer_config(config)
    )
    names = list(bio_config.core_model_names)
    names.extend([bio_config.primary_model_name, bio_config.conservative_model_name])
    if bio_config.fallback_model_name:
        names.append(bio_config.fallback_model_name)
    if include_research:
        names.extend(bio_config.research_model_names)
        if bio_config.research_model_name:
            names.append(bio_config.research_model_name)
    if include_ablation:
        names.extend(bio_config.ablation_model_names)
    return tuple(dict.fromkeys(str(name) for name in names if str(name).strip()))
