"""Discovery-track input contracts and fail-fast leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from plasmid_priority.modeling.module_a import get_feature_track, get_model_track
from plasmid_priority.modeling.module_a_support import MODULE_A_FEATURE_SETS


@dataclass(frozen=True)
class DiscoveryInputContract:
    """Minimal structural contract for discovery-safe scored inputs."""

    split_year: int
    required_assignment_mode: str = "training_only"
    require_training_only_assignment: bool = True
    require_temporal_metadata: bool = True


def build_discovery_input_contract(split_year: int) -> DiscoveryInputContract:
    """Construct the default discovery contract for a given split year."""
    return DiscoveryInputContract(split_year=int(split_year))


def discovery_model_names(model_names: Iterable[str]) -> list[str]:
    """Return the subset of model names that belong to the discovery track."""
    names: list[str] = []
    for model_name in model_names:
        if str(get_model_track(str(model_name))) == "discovery":
            names.append(str(model_name))
    return names


def validate_discovery_input_contract(
    scored: pd.DataFrame,
    *,
    model_names: Iterable[str],
    contract: DiscoveryInputContract,
    label: str = "discovery input",
) -> None:
    """Fail fast when a discovery evaluation surface is not discovery-safe."""
    discovery_models = discovery_model_names(model_names)
    if not discovery_models:
        return
    if scored.empty:
        raise ValueError(f"{label} is empty; discovery models cannot run on an empty score table.")

    missing_columns = [
        column
        for column in ["split_year", "backbone_assignment_mode"]
        if column not in scored.columns
    ]
    if contract.require_temporal_metadata:
        missing_columns.extend(
            [
                column
                for column in ["max_resolved_year_train", "min_resolved_year_test"]
                if column not in scored.columns
            ]
        )
    if missing_columns:
        missing_text = ", ".join(f"`{column}`" for column in sorted(set(missing_columns)))
        raise ValueError(
            f"{label} is missing discovery-contract metadata columns: {missing_text}."
        )

    split_year_values = (
        pd.to_numeric(scored["split_year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if split_year_values != [int(contract.split_year)]:
        raise ValueError(
            f"{label} has split_year metadata {split_year_values}, "
            f"expected only {int(contract.split_year)}."
        )

    if contract.require_training_only_assignment:
        assignment_modes = (
            scored["backbone_assignment_mode"].fillna("").astype(str).str.strip().replace("", pd.NA)
        )
        invalid_modes = sorted(
            {
                mode
                for mode in assignment_modes.dropna().unique().tolist()
                if mode != contract.required_assignment_mode
            }
        )
        if invalid_modes:
            invalid_text = ", ".join(f"`{mode}`" for mode in invalid_modes)
            raise ValueError(
                f"{label} contains non-discovery backbone assignment modes: {invalid_text}. "
                f"Discovery models require `{contract.required_assignment_mode}` assignment."
            )

    max_train_year = pd.to_numeric(scored["max_resolved_year_train"], errors="coerce").dropna()
    if not max_train_year.empty and not max_train_year.le(int(contract.split_year)).all():
        raise ValueError(
            f"{label} contains training-year metadata beyond split_year={int(contract.split_year)}."
        )

    min_test_year = pd.to_numeric(scored["min_resolved_year_test"], errors="coerce").dropna()
    if not min_test_year.empty and not min_test_year.gt(int(contract.split_year)).all():
        raise ValueError(
            f"{label} contains test-year metadata at or before "
            f"split_year={int(contract.split_year)}."
        )

    if (
        "training_only_future_unseen_backbone_flag" in scored.columns
        and "spread_label" in scored.columns
    ):
        unseen_labeled = scored.loc[
            scored["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool)
            & scored["spread_label"].notna()
        ]
        if not unseen_labeled.empty:
            raise ValueError(
                f"{label} assigns outcome labels to {len(unseen_labeled)} "
                "future-unseen training-only backbones."
            )

    violating_features: dict[str, list[str]] = {}
    for model_name in discovery_models:
        feature_names = MODULE_A_FEATURE_SETS[str(model_name)]
        invalid_features = [
            feature_name
            for feature_name in feature_names
            if str(get_feature_track(str(feature_name))) != "discovery"
        ]
        if invalid_features:
            violating_features[str(model_name)] = invalid_features
    if violating_features:
        detail = "; ".join(
            f"{model_name}: {', '.join(sorted(features))}"
            for model_name, features in sorted(violating_features.items())
        )
        raise ValueError(
            f"{label} includes discovery models with non-discovery features: {detail}."
        )
