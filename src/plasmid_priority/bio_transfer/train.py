"""Training helpers for the bio transfer branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from plasmid_priority.bio_transfer.features import build_bio_transfer_features
from plasmid_priority.bio_transfer.contracts import build_bio_transfer_input_contract
from plasmid_priority.bio_transfer.dataset import prepare_bio_transfer_scored_table
from plasmid_priority.shared.labels import build_bio_transfer_labels
from plasmid_priority.bio_transfer.specs import load_bio_transfer_config
from plasmid_priority.shared.branching import fit_branch, fit_branch_model, fit_branch_model_predictions


def _bio_transfer_label_builder(
    records: pd.DataFrame | None,
):
    return lambda _scored, _records, split_year, horizon_years: build_bio_transfer_labels(
        records if records is not None else _scored,
        split_year,
        horizon_years,
    )


def fit_bio_transfer_model(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int = 3,
    n_repeats: int = 2,
    seed: int = 42,
    config: Mapping[str, Any] | Any | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
) -> Any:
    bio_config = load_bio_transfer_config(config)
    bio_contract = build_bio_transfer_input_contract(config)
    return fit_branch_model(
        scored,
        model_name=model_name,
        config=bio_config,
        contract=bio_contract,
        records=records,
        label_builder=_bio_transfer_label_builder(records),
        feature_builder=build_bio_transfer_features,
        branch_label_column="bio_transfer_label",
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        include_ci=include_ci,
    )


def fit_bio_transfer_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: Mapping[str, Any] | Any | None = None,
    records: pd.DataFrame | None = None,
    include_posterior_uncertainty: bool = True,
) -> pd.DataFrame:
    bio_config = load_bio_transfer_config(config)
    bio_contract = build_bio_transfer_input_contract(config)
    return fit_branch_model_predictions(
        scored,
        model_name=model_name,
        config=bio_config,
        contract=bio_contract,
        records=records,
        label_builder=_bio_transfer_label_builder(records),
        feature_builder=build_bio_transfer_features,
        branch_label_column="bio_transfer_label",
        include_posterior_uncertainty=include_posterior_uncertainty,
    )


def fit_bio_transfer_branch(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    include_research: bool = False,
    include_ablation: bool = False,
    n_splits: int = 3,
    n_repeats: int = 2,
    seed: int = 42,
    n_jobs: int | None = 1,
    config: Mapping[str, Any] | Any | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
    prepared_scored: pd.DataFrame | None = None,
) -> dict[str, Any]:
    bio_config = load_bio_transfer_config(config)
    bio_contract = build_bio_transfer_input_contract(config)
    prepared = (
        prepared_scored
        if prepared_scored is not None
        else prepare_bio_transfer_scored_table(
            scored,
            config=bio_config,
            contract=bio_contract,
            records=records,
        )
    )
    return fit_branch(
        scored,
        model_names=model_names,
        include_research=include_research,
        include_ablation=include_ablation,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        n_jobs=n_jobs,
        config=bio_config,
        contract=bio_contract,
        records=records,
        label_builder=_bio_transfer_label_builder(records),
        feature_builder=build_bio_transfer_features,
        branch_label_column="bio_transfer_label",
        include_ci=include_ci,
        prepared_scored=prepared,
    )
