"""Training helpers for the clinical hazard branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from plasmid_priority.clinical_hazard.contracts import build_clinical_hazard_input_contract
from plasmid_priority.clinical_hazard.dataset import prepare_clinical_hazard_scored_table
from plasmid_priority.clinical_hazard.features import build_clinical_hazard_features
from plasmid_priority.clinical_hazard.specs import load_clinical_hazard_config
from plasmid_priority.shared.branching import fit_branch, fit_branch_model, fit_branch_model_predictions
from plasmid_priority.shared.labels import build_clinical_hazard_labels


def fit_clinical_hazard_model(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int = 3,
    n_repeats: int = 2,
    seed: int = 42,
    config: Mapping[str, Any] | Any | None = None,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    include_ci: bool = True,
) -> Any:
    clinical_config = load_clinical_hazard_config(config)
    clinical_contract = build_clinical_hazard_input_contract(config)
    return fit_branch_model(
        scored,
        model_name=model_name,
        config=clinical_config,
        contract=clinical_contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=lambda _scored, _records, split_year, horizon_years: build_clinical_hazard_labels(
            records if records is not None else _scored,
            pd_metadata,
            split_year,
            horizon_years,
        ),
        feature_builder=build_clinical_hazard_features,
        branch_label_column="clinical_hazard_label",
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        include_ci=include_ci,
    )


def fit_clinical_hazard_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: Mapping[str, Any] | Any | None = None,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    include_posterior_uncertainty: bool = True,
) -> pd.DataFrame:
    clinical_config = load_clinical_hazard_config(config)
    clinical_contract = build_clinical_hazard_input_contract(config)
    return fit_branch_model_predictions(
        scored,
        model_name=model_name,
        config=clinical_config,
        contract=clinical_contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=lambda _scored, _records, split_year, horizon_years: build_clinical_hazard_labels(
            records if records is not None else _scored,
            pd_metadata,
            split_year,
            horizon_years,
        ),
        feature_builder=build_clinical_hazard_features,
        branch_label_column="clinical_hazard_label",
        include_posterior_uncertainty=include_posterior_uncertainty,
    )


def fit_clinical_hazard_branch(
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
    pd_metadata: pd.DataFrame | None = None,
    include_ci: bool = True,
    prepared_scored: pd.DataFrame | None = None,
) -> dict[str, Any]:
    clinical_config = load_clinical_hazard_config(config)
    clinical_contract = build_clinical_hazard_input_contract(config)
    prepared = (
        prepared_scored
        if prepared_scored is not None
        else prepare_clinical_hazard_scored_table(
            scored,
            config=clinical_config,
            contract=clinical_contract,
            records=records,
            pd_metadata=pd_metadata,
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
        config=clinical_config,
        contract=clinical_contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=lambda _scored, _records, split_year, horizon_years: build_clinical_hazard_labels(
            records if records is not None else _scored,
            pd_metadata,
            split_year,
            horizon_years,
        ),
        feature_builder=build_clinical_hazard_features,
        branch_label_column="clinical_hazard_label",
        include_ci=include_ci,
        prepared_scored=prepared,
    )
