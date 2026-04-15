"""Generic branch dataset and evaluation helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from plasmid_priority.exceptions import ModelFitError
from plasmid_priority.modeling import evaluate_feature_columns, fit_feature_columns_predictions
from plasmid_priority.modeling.module_a import (
    ModelResult,
    annotate_knownness_metadata,
    assert_feature_columns_present,
)
from plasmid_priority.modeling.module_a_support import build_failed_model_result
from plasmid_priority.shared.contracts import (
    ensure_branch_label_alias,
    validate_branch_input_contract,
)
from plasmid_priority.shared.specs import (
    BranchBenchmarkSpec,
    BranchConfig,
    resolve_branch_model_names,
)
from plasmid_priority.utils.parallel import limit_native_threads

LabelBuilder = Callable[..., pd.DataFrame]
FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(slots=True)
class BranchDataset:
    """Prepared branch dataset for a single model evaluation."""

    scored: pd.DataFrame
    eligible: pd.DataFrame
    model_name: str
    feature_columns: tuple[str, ...]
    benchmark: BranchBenchmarkSpec
    config: BranchConfig
    fit_config: dict[str, Any]
    label_column: str

    @property
    def y(self) -> pd.Series:
        return self.eligible[self.label_column].astype(int)


def prepare_branch_scored_table(
    scored: pd.DataFrame,
    *,
    config: BranchConfig,
    contract: Any,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label_builder: Callable[..., pd.DataFrame] | None = None,
    feature_builder: FeatureBuilder | None = None,
    branch_label_column: str,
    pd_metadata_merge_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate and enrich a branch scored surface."""
    working = scored.copy(deep=False)
    if label_builder is not None and branch_label_column not in working.columns:
        label_frame = label_builder(
            working,
            records,
            config.benchmark.split_year,
            config.benchmark.horizon_years,
        )
        if not label_frame.empty:
            merge_columns = ["backbone_id"]
            merge_columns.extend(
                column
                for column in label_frame.columns
                if column != "backbone_id" and column not in working.columns
            )
            working = working.merge(label_frame.loc[:, merge_columns], on="backbone_id", how="left")
    validate_branch_input_contract(
        working, contract=contract, label=f"{config.benchmark.name} scored table"
    )
    working = ensure_branch_label_alias(working, branch_label_column)
    if feature_builder is not None:
        working = feature_builder(working)
    working = annotate_knownness_metadata(working)
    working[f"{config.benchmark.name}_primary_model_name"] = config.primary_model_name
    working[f"{config.benchmark.name}_conservative_model_name"] = config.conservative_model_name
    if (
        pd_metadata_merge_columns
        and pd_metadata is not None
        and not pd_metadata.empty
        and "backbone_id" in pd_metadata.columns
    ):
        merge_columns = [
            column
            for column in dict.fromkeys(["backbone_id", *pd_metadata_merge_columns])
            if column in pd_metadata.columns
            and (column == "backbone_id" or column not in working.columns)
        ]
        if len(merge_columns) > 1:
            meta = pd_metadata.loc[:, merge_columns].copy(deep=False)
            meta["backbone_id"] = meta["backbone_id"].astype(str)
            working = working.merge(
                meta.drop_duplicates(subset=["backbone_id"]), on="backbone_id", how="left"
            )
    return working


def build_branch_dataset_from_prepared(
    prepared: pd.DataFrame,
    *,
    model_name: str,
    config: BranchConfig,
    contract: Any,
    label_column: str,
) -> BranchDataset:
    """Build a branch dataset from a prepared table."""
    if str(model_name) not in config.feature_sets:
        raise KeyError(f"Unknown branch model: {model_name}")
    feature_columns = tuple(config.feature_sets[str(model_name)])
    assert_feature_columns_present(
        prepared,
        feature_columns,
        label=f"Branch model `{model_name}` input",
    )
    if label_column not in prepared.columns:
        raise KeyError(f"Prepared branch table is missing `{label_column}`.")
    eligible = prepared.loc[prepared[label_column].notna()].copy(deep=False)
    eligible[label_column] = eligible[label_column].astype(int)
    fit_config = config.fit_config[str(model_name)].model_dump(mode="python")
    return BranchDataset(
        scored=prepared,
        eligible=eligible,
        model_name=str(model_name),
        feature_columns=feature_columns,
        benchmark=contract.benchmark,
        config=config,
        fit_config=fit_config,
        label_column=label_column,
    )


def prepare_branch_dataset(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: BranchConfig,
    contract: Any,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label_builder: Callable[..., pd.DataFrame] | None = None,
    feature_builder: FeatureBuilder | None = None,
    branch_label_column: str,
    prepared_scored: pd.DataFrame | None = None,
) -> BranchDataset:
    prepared = (
        prepared_scored
        if prepared_scored is not None
        else prepare_branch_scored_table(
            scored,
            config=config,
            contract=contract,
            records=records,
            pd_metadata=pd_metadata,
            label_builder=label_builder,
            feature_builder=feature_builder,
            branch_label_column=branch_label_column,
        )
    )
    return build_branch_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=config,
        contract=contract,
        label_column=branch_label_column,
    )


def _fit_branch_model_on_dataset(
    dataset: BranchDataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    include_ci: bool = True,
) -> ModelResult:
    return evaluate_feature_columns(
        dataset.scored,
        columns=list(dataset.feature_columns),
        label=dataset.model_name,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_config=dataset.fit_config,
        include_ci=include_ci,
    )


def fit_branch_model(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: BranchConfig,
    contract: Any,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label_builder: Callable[..., pd.DataFrame] | None = None,
    feature_builder: FeatureBuilder | None = None,
    branch_label_column: str,
    prepared_scored: pd.DataFrame | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    include_ci: bool = True,
) -> ModelResult:
    dataset = prepare_branch_dataset(
        scored,
        model_name=model_name,
        config=config,
        contract=contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=label_builder,
        feature_builder=feature_builder,
        branch_label_column=branch_label_column,
        prepared_scored=prepared_scored,
    )
    return _fit_branch_model_on_dataset(
        dataset,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        include_ci=include_ci,
    )


def fit_branch_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: BranchConfig,
    contract: Any,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label_builder: Callable[..., pd.DataFrame] | None = None,
    feature_builder: FeatureBuilder | None = None,
    branch_label_column: str,
    prepared_scored: pd.DataFrame | None = None,
    include_posterior_uncertainty: bool = True,
) -> pd.DataFrame:
    dataset = prepare_branch_dataset(
        scored,
        model_name=model_name,
        config=config,
        contract=contract,
        records=records,
        pd_metadata=pd_metadata,
        label_builder=label_builder,
        feature_builder=feature_builder,
        branch_label_column=branch_label_column,
        prepared_scored=prepared_scored,
    )
    train = dataset.eligible
    all_rows = dataset.scored
    predictions = fit_feature_columns_predictions(
        train,
        all_rows,
        columns=list(dataset.feature_columns),
        fit_config=dataset.fit_config,
    )
    if predictions.empty or include_posterior_uncertainty:
        return predictions
    return predictions.loc[
        :, [column for column in predictions.columns if column in {"backbone_id", "prediction"}]
    ].copy()


def fit_branch(
    scored: pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
    include_research: bool = False,
    include_ablation: bool = False,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    n_jobs: int | None = 1,
    config: BranchConfig,
    contract: Any,
    records: pd.DataFrame | None = None,
    pd_metadata: pd.DataFrame | None = None,
    label_builder: Callable[..., pd.DataFrame] | None = None,
    feature_builder: FeatureBuilder | None = None,
    branch_label_column: str,
    include_ci: bool = True,
    prepared_scored: pd.DataFrame | None = None,
) -> dict[str, ModelResult]:
    selected_model_names = (
        list(
            resolve_branch_model_names(
                config, include_research=include_research, include_ablation=include_ablation
            )
        )
        if model_names is None
        else [str(name) for name in model_names]
    )
    missing = sorted(set(selected_model_names) - set(config.feature_sets))
    if missing:
        raise KeyError(f"Unknown branch model(s): {', '.join(missing)}")
    prepared = (
        prepared_scored
        if prepared_scored is not None
        else prepare_branch_scored_table(
            scored,
            config=config,
            contract=contract,
            records=records,
            pd_metadata=pd_metadata,
            label_builder=label_builder,
            feature_builder=feature_builder,
            branch_label_column=branch_label_column,
        )
    )

    _log = logging.getLogger(__name__)

    def _fit_one(name: str) -> tuple[str, ModelResult]:
        try:
            dataset = prepare_branch_dataset(
                scored,
                model_name=name,
                config=config,
                contract=contract,
                records=records,
                pd_metadata=pd_metadata,
                label_builder=label_builder,
                feature_builder=feature_builder,
                branch_label_column=branch_label_column,
                prepared_scored=prepared,
            )
            result = _fit_branch_model_on_dataset(
                dataset,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                include_ci=include_ci,
            )
        except (ValueError, KeyError, TypeError, RuntimeError, ModelFitError) as exc:
            _log.warning("branch model %s failed: %s", name, exc)
            result = build_failed_model_result(str(name), str(exc))
        return name, result

    jobs = max(1, min(int(n_jobs or 1), len(selected_model_names))) if selected_model_names else 1
    if jobs > 1 and len(selected_model_names) > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                results = dict(executor.map(_fit_one, selected_model_names))
    else:
        results = dict(_fit_one(name) for name in selected_model_names)
    return results
