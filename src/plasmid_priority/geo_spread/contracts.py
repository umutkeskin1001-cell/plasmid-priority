"""Branch-level contract checks for the geo spread task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from plasmid_priority.geo_spread.specs import GeoSpreadBenchmarkSpec, load_geo_spread_config


class GeoSpreadInputContract(BaseModel):
    """Structural contract for geo spread backbone tables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: GeoSpreadBenchmarkSpec = Field(default_factory=GeoSpreadBenchmarkSpec)
    require_temporal_metadata: bool = True
    require_assignment_mode: bool = True
    require_label_consistency: bool = True

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.benchmark.required_columns


def build_geo_spread_input_contract(
    config: Mapping[str, Any] | None = None,
    *,
    split_year: int | None = None,
) -> GeoSpreadInputContract:
    """Build the geo spread input contract from project config defaults."""
    geo_config = load_geo_spread_config(config)
    benchmark = geo_config.benchmark
    if split_year is not None and int(split_year) != int(benchmark.split_year):
        benchmark = benchmark.model_copy(update={"split_year": int(split_year)})
    return GeoSpreadInputContract(benchmark=benchmark)


def _missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def validate_geo_spread_input_contract(
    scored: pd.DataFrame,
    *,
    contract: GeoSpreadInputContract | None = None,
    label: str = "geo spread input",
) -> None:
    """Fail fast when the geo spread branch input violates structural rules."""
    contract = contract or build_geo_spread_input_contract()
    benchmark = contract.benchmark

    missing = _missing_columns(scored, contract.required_columns)
    if contract.require_temporal_metadata:
        missing.extend(_missing_columns(scored, benchmark.temporal_metadata_columns))
    if missing:
        missing_text = ", ".join(f"`{column}`" for column in sorted(set(missing)))
        raise ValueError(f"{label} is missing required columns: {missing_text}.")

    split_year_values = (
        pd.to_numeric(scored["split_year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if split_year_values != [int(benchmark.split_year)]:
        raise ValueError(
            f"{label} has split_year metadata {split_year_values}, expected only {int(benchmark.split_year)}."
        )

    if contract.require_assignment_mode:
        assignment_modes = (
            scored["backbone_assignment_mode"].fillna("").astype(str).str.strip().replace("", np.nan)
        )
        invalid_modes = sorted(
            {
                mode
                for mode in assignment_modes.dropna().unique().tolist()
                if mode != benchmark.assignment_mode
            }
        )
        if invalid_modes:
            invalid_text = ", ".join(f"`{mode}`" for mode in invalid_modes)
            raise ValueError(
                f"{label} contains non-discovery backbone assignment modes: {invalid_text}. "
                f"Geo spread requires `{benchmark.assignment_mode}` assignment."
            )

    label_values = pd.to_numeric(scored[benchmark.label_column], errors="coerce")
    if not label_values.dropna().isin([0.0, 1.0]).all():
        raise ValueError(f"{label} contains invalid label values; only 0, 1, and NA are allowed.")

    if benchmark.outcome_column in scored.columns:
        outcome = pd.to_numeric(scored[benchmark.outcome_column], errors="coerce")
        valid_outcome = outcome.dropna()
        if not valid_outcome.empty and (valid_outcome < 0).any():
            raise ValueError(f"{label} contains negative values in `{benchmark.outcome_column}`.")
        if contract.require_label_consistency:
            eligible = label_values.notna() & outcome.notna()
            if eligible.any():
                subset = scored.loc[eligible]
                expected = (
                    outcome.loc[eligible] >= float(benchmark.min_new_countries_for_spread)
                ).astype(float)
                observed = label_values.loc[eligible].astype(float)
                mismatch_mask = observed.to_numpy(dtype=float) != expected.to_numpy(dtype=float)
                mismatched = subset.loc[mismatch_mask]
                if not mismatched.empty:
                    raise ValueError(
                        f"{label} has {len(mismatched)} row(s) whose `{benchmark.label_column}` "
                        f"does not match `{benchmark.outcome_column}` thresholding."
                    )

    if "training_only_future_unseen_backbone_flag" in scored.columns:
        unseen_labeled = scored.loc[
            scored["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool)
            & scored[benchmark.label_column].notna()
        ]
        if not unseen_labeled.empty:
            raise ValueError(
                f"{label} assigns labels to {len(unseen_labeled)} future-unseen backbones."
            )
