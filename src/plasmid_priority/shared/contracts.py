"""Generic branch input contracts and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from plasmid_priority.shared.specs import BranchBenchmarkSpec


class BranchInputContract(BaseModel):
    """Structural contract for a branch backbone table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BranchBenchmarkSpec = Field(default_factory=BranchBenchmarkSpec)
    require_temporal_metadata: bool = True
    require_assignment_mode: bool = True
    require_label_consistency: bool = True
    allow_future_feature_names: bool = False

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.benchmark.required_columns


def build_branch_input_contract(
    benchmark: BranchBenchmarkSpec | Mapping[str, Any] | None = None,
    *,
    split_year: int | None = None,
) -> BranchInputContract:
    """Build a branch input contract from the resolved benchmark."""
    if benchmark is None:
        resolved = BranchBenchmarkSpec()
    elif isinstance(benchmark, BranchBenchmarkSpec):
        resolved = benchmark
    else:
        resolved = BranchBenchmarkSpec.model_validate(dict(benchmark))
    if split_year is not None and int(split_year) != int(resolved.split_year):
        resolved = resolved.model_copy(update={"split_year": int(split_year)})
    return BranchInputContract(benchmark=resolved)


def _missing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def ensure_branch_label_alias(
    frame: pd.DataFrame,
    label_column: str,
    *,
    alias: str = "spread_label",
) -> pd.DataFrame:
    """Mirror the branch label into the shared Module A label alias."""
    working = frame.copy()
    if label_column in working.columns:
        working[alias] = working[label_column]
    elif alias in working.columns:
        working[label_column] = working[alias]
    return working


def validate_branch_feature_set(
    features: Sequence[str], *, label: str = "branch feature set"
) -> None:
    """Reject feature sets that expose future-derived columns."""
    normalized = [str(feature).strip() for feature in features if str(feature).strip()]
    if not normalized:
        raise ValueError(f"{label} must contain at least one feature")
    future_columns = sorted(feature for feature in normalized if feature.endswith("_future"))
    if future_columns:
        joined = ", ".join(f"`{feature}`" for feature in future_columns)
        raise ValueError(f"{label} contains future-derived columns: {joined}")


def validate_branch_input_contract(
    scored: pd.DataFrame,
    *,
    contract: BranchInputContract | None = None,
    label: str = "branch input",
) -> None:
    """Fail fast when branch input violates the split-safe structural contract."""
    contract = contract or build_branch_input_contract()
    benchmark = contract.benchmark

    missing = _missing_columns(scored, contract.required_columns)
    if contract.require_temporal_metadata:
        missing.extend(_missing_columns(scored, benchmark.temporal_metadata_columns))
    if missing:
        missing_text = ", ".join(f"`{column}`" for column in sorted(set(missing)))
        raise ValueError(f"{label} is missing required columns: {missing_text}.")

    split_year_values = (
        pd.to_numeric(scored["split_year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if split_year_values != [int(benchmark.split_year)]:
        raise ValueError(
            f"{label} has split_year metadata {split_year_values}, expected only "
            f"{int(benchmark.split_year)}."
        )

    if contract.require_assignment_mode:
        assignment_modes = (
            scored["backbone_assignment_mode"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", np.nan)
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
                f"Branch requires `{benchmark.assignment_mode}` assignment."
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
                expected = (outcome.loc[eligible] >= float(benchmark.positive_threshold)).astype(
                    float
                )
                observed = label_values.loc[eligible].astype(float)
                mismatch_mask = observed.to_numpy(dtype=float) != expected.to_numpy(dtype=float)
                if mismatch_mask.any():
                    raise ValueError(
                        f"{label} has {int(mismatch_mask.sum())} row(s) whose "
                        f"`{benchmark.label_column}` does not match "
                        f"`{benchmark.outcome_column}` thresholding."
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
