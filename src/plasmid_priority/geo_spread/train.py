"""Training helpers for the geo spread branch."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from plasmid_priority.exceptions import ModelFitError
from plasmid_priority.geo_spread.dataset import (
    build_geo_spread_dataset_from_prepared,
    prepare_geo_spread_dataset,
    prepare_geo_spread_scored_table,
    resolve_geo_spread_dataset_model_names,
)
from plasmid_priority.geo_spread.specs import GeoSpreadConfig, load_geo_spread_config
from plasmid_priority.modeling import evaluate_feature_columns, fit_feature_columns_predictions
from plasmid_priority.modeling.module_a_support import ModelResult, build_failed_model_result
from plasmid_priority.utils.parallel import limit_native_threads


def _fit_geo_spread_model_on_dataset(
    dataset: Any,
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


def fit_geo_spread_model(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
) -> ModelResult:
    """Fit and evaluate one geo spread model on the scored branch surface."""
    geo_dataset = prepare_geo_spread_dataset(
        scored,
        model_name=model_name,
        config=load_geo_spread_config(config),
        records=records,
    )
    return _fit_geo_spread_model_on_dataset(
        geo_dataset,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        include_ci=include_ci,
    )


def fit_geo_spread_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
    records: pd.DataFrame | None = None,
    include_posterior_uncertainty: bool = True,
) -> pd.DataFrame:
    """Fit one geo spread model on eligible rows and score the full table."""
    geo_dataset = prepare_geo_spread_dataset(
        scored,
        model_name=model_name,
        config=load_geo_spread_config(config),
        records=records,
    )
    train = geo_dataset.eligible
    all_rows = geo_dataset.scored
    predictions = fit_feature_columns_predictions(
        train,
        all_rows,
        columns=list(geo_dataset.feature_columns),
        fit_config=geo_dataset.fit_config,
    )
    if predictions.empty or include_posterior_uncertainty:
        return predictions
    return predictions.loc[
        :, [column for column in predictions.columns if column in {"backbone_id", "prediction"}]
    ].copy()


def fit_geo_spread_branch(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    include_research: bool = False,
    include_ablation: bool = False,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    n_jobs: int | None = 1,
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
    prepared_scored: pd.DataFrame | None = None,
) -> dict[str, ModelResult]:
    """Fit the branch model surface and return typed evaluation results."""
    geo_config = load_geo_spread_config(config)
    prepared_scored = (
        prepared_scored
        if prepared_scored is not None
        else prepare_geo_spread_scored_table(scored, config=geo_config, records=records)
    )
    selected_model_names = (
        list(
            resolve_geo_spread_dataset_model_names(
                config,
                include_research=include_research,
                include_ablation=include_ablation,
            )
        )
        if model_names is None
        else [str(name) for name in model_names]
    )
    missing = sorted(set(selected_model_names) - set(geo_config.feature_sets))
    if missing:
        raise KeyError(f"Unknown geo spread model(s): {', '.join(missing)}")

    _log = logging.getLogger(__name__)

    def _fit_one(name: str) -> tuple[str, ModelResult]:
        try:
            dataset = build_geo_spread_dataset_from_prepared(
                prepared_scored,
                model_name=name,
                config=geo_config,
                contract=None,
            )
            result = _fit_geo_spread_model_on_dataset(
                dataset,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                include_ci=include_ci,
            )
        except (ValueError, KeyError, TypeError, RuntimeError, ModelFitError) as exc:
            _log.warning("geo_spread model %s failed: %s", name, exc)
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
