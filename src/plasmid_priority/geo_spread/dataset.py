"""Dataset assembly helpers for the geo spread branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from plasmid_priority.geo_spread.contracts import (
    GeoSpreadInputContract,
    build_geo_spread_input_contract,
    validate_geo_spread_input_contract,
)
from plasmid_priority.geo_spread.enrichment import enrich_geo_spread_scored_table
from plasmid_priority.geo_spread.specs import (
    GeoSpreadBenchmarkSpec,
    GeoSpreadConfig,
    load_geo_spread_config,
    resolve_geo_spread_model_names,
)
from plasmid_priority.modeling.module_a import (
    annotate_knownness_metadata,
    assert_feature_columns_present,
)


@dataclass(slots=True)
class GeoSpreadDataset:
    """Prepared geo spread dataset for a single model evaluation."""

    scored: pd.DataFrame
    eligible: pd.DataFrame
    model_name: str
    feature_columns: tuple[str, ...]
    benchmark: GeoSpreadBenchmarkSpec
    config: GeoSpreadConfig
    contract: GeoSpreadInputContract
    fit_config: dict[str, Any]

    @property
    def label_column(self) -> str:
        return self.benchmark.label_column

    @property
    def y(self) -> pd.Series:
        return self.eligible[self.label_column].astype(int)


def prepare_geo_spread_scored_table(
    scored: pd.DataFrame,
    *,
    config: GeoSpreadConfig | dict[str, Any] | None = None,
    contract: GeoSpreadInputContract | None = None,
    records: pd.DataFrame | None = None,
    label: str = "geo spread scored table",
) -> pd.DataFrame:
    """Validate and enrich the scored surface for geo spread modeling."""
    geo_config = config if isinstance(config, GeoSpreadConfig) else load_geo_spread_config(config)
    geo_contract = contract or build_geo_spread_input_contract()
    validate_geo_spread_input_contract(scored, contract=geo_contract, label=label)
    working = enrich_geo_spread_scored_table(
        scored,
        split_year=geo_config.benchmark.split_year,
        records=records,
    )
    working = annotate_knownness_metadata(working)
    working["geo_spread_primary_model_name"] = geo_config.primary_model_name
    working["geo_spread_conservative_model_name"] = geo_config.conservative_model_name
    return working


def prepare_geo_spread_dataset(
    scored: pd.DataFrame,
    *,
    model_name: str,
    config: GeoSpreadConfig | dict[str, Any] | None = None,
    contract: GeoSpreadInputContract | None = None,
    records: pd.DataFrame | None = None,
    label: str = "geo spread scored table",
) -> GeoSpreadDataset:
    """Build a model-specific geo spread dataset."""
    geo_config = config if isinstance(config, GeoSpreadConfig) else load_geo_spread_config(config)
    if str(model_name) not in geo_config.feature_sets:
        raise KeyError(f"Unknown geo spread model: {model_name}")
    geo_contract = contract or build_geo_spread_input_contract()
    prepared = prepare_geo_spread_scored_table(
        scored,
        config=geo_config,
        contract=geo_contract,
        records=records,
        label=label,
    )
    return build_geo_spread_dataset_from_prepared(
        prepared,
        model_name=model_name,
        config=geo_config,
        contract=geo_contract,
    )


def build_geo_spread_dataset_from_prepared(
    prepared: pd.DataFrame,
    *,
    model_name: str,
    config: GeoSpreadConfig | dict[str, Any] | None = None,
    contract: GeoSpreadInputContract | None = None,
) -> GeoSpreadDataset:
    """Build a geo spread dataset from a table that was already validated and annotated."""
    geo_config = config if isinstance(config, GeoSpreadConfig) else load_geo_spread_config(config)
    if str(model_name) not in geo_config.feature_sets:
        raise KeyError(f"Unknown geo spread model: {model_name}")
    geo_contract = contract or build_geo_spread_input_contract()
    feature_columns = tuple(geo_config.feature_sets[str(model_name)])
    assert_feature_columns_present(
        prepared,
        feature_columns,
        label=f"Geo spread model `{model_name}` input",
    )
    eligible = prepared.loc[prepared[geo_contract.benchmark.label_column].notna()].copy()
    eligible[geo_contract.benchmark.label_column] = eligible[
        geo_contract.benchmark.label_column
    ].astype(int)
    fit_config = geo_config.fit_config[str(model_name)].model_dump(mode="python")
    return GeoSpreadDataset(
        scored=prepared,
        eligible=eligible,
        model_name=str(model_name),
        feature_columns=feature_columns,
        benchmark=geo_contract.benchmark,
        config=geo_config,
        contract=geo_contract,
        fit_config=fit_config,
    )


def resolve_geo_spread_dataset_model_names(
    config: dict[str, Any] | None = None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    """Resolve the ordered model surface for dataset-driven evaluations."""
    return resolve_geo_spread_model_names(
        config,
        include_research=include_research,
        include_ablation=include_ablation,
    )
