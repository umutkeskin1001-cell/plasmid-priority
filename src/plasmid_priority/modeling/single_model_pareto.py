"""Deterministic Pareto utilities for single-model selection."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from plasmid_priority.protocol import (
    DEFAULT_OFFICIAL_ACCEPTANCE_THRESHOLDS,
    DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS,
)

_WEIGHTED_SCORE_COLUMN = "weighted_objective_score"
_FAILURE_SEVERITY_COLUMN = "failure_severity"
_MODEL_NAME_COLUMN = "model_name"

_OBJECTIVE_COLUMN_BY_WEIGHT_NAME = {
    "reliability": "reliability_score",
    "predictive_power": "predictive_power_score",
    "compute_efficiency": "compute_efficiency_score",
}

_FAILURE_CUTOFF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("knownness_matched_gap", "matched_knownness_gap_min"),
    ("source_holdout_gap", "source_holdout_gap_min"),
    ("spatial_holdout_gap", "spatial_holdout_gap_min"),
    ("blocked_holdout_raw_ece", "ece_max"),
    ("blocked_holdout_ece", "ece_max"),
    ("ece", "ece_max"),
    ("selection_adjusted_empirical_p_roc_auc", "selection_adjusted_p_max"),
)


def _required_columns_exist(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _extract_failed_criteria(row: pd.Series) -> tuple[str, ...]:
    if str(row.get("scientific_acceptance_status", "")).strip().lower() == "pass":
        return ()
    raw = row.get("scientific_acceptance_failed_criteria")
    if raw is None or raw == "":
        raw = row.get("failed_criteria")
    if raw is None or raw == "":
        return ("fail",)
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized.startswith("fail:"):
            normalized = normalized.removeprefix("fail:")
        tokens = [
            token.strip()
            for token in normalized.replace(";", ",").split(",")
            if token.strip()
        ]
        return tuple(tokens) or ("fail",)
    if isinstance(raw, Iterable):
        tokens = [str(token).strip() for token in raw if str(token).strip()]
        return tuple(tokens) or ("fail",)
    token = str(raw).strip()
    return (token,) if token else ("fail",)


def _failure_shortfall_total(row: pd.Series) -> float:
    total = 0.0
    for column, threshold_name in _FAILURE_CUTOFF_COLUMNS:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        numeric_value = float(value)
        threshold = float(DEFAULT_OFFICIAL_ACCEPTANCE_THRESHOLDS[threshold_name])
        if threshold_name == "ece_max":
            total += max(0.0, numeric_value - threshold)
        elif threshold_name == "selection_adjusted_p_max":
            total += max(0.0, numeric_value - threshold)
        else:
            total += max(0.0, threshold - numeric_value)
    return total


def _compute_failure_severity(row: pd.Series) -> float:
    failed_criteria = _extract_failed_criteria(row)
    criterion_count = len(failed_criteria)
    guardrail_penalty = float(criterion_count) + (0.5 * max(criterion_count - 1, 0))
    shortfall_penalty = 12.0 * _failure_shortfall_total(row)
    return guardrail_penalty + shortfall_penalty


def add_weighted_objective(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the protocol-weighted objective score."""

    required_columns = tuple(_OBJECTIVE_COLUMN_BY_WEIGHT_NAME.values())
    _required_columns_exist(frame, required_columns)

    working = frame.copy()
    objective = pd.Series(0.0, index=working.index, dtype=float)
    for weight_name, column_name in _OBJECTIVE_COLUMN_BY_WEIGHT_NAME.items():
        objective = objective + (
            DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS[weight_name]
            * _numeric_series(working, column_name)
        )
    working[_WEIGHTED_SCORE_COLUMN] = objective.astype(float)
    return working


def add_failure_severity(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic failure-severity score."""

    working = scorecard.copy()
    working[_FAILURE_SEVERITY_COLUMN] = working.apply(_compute_failure_severity, axis=1).astype(
        float
    )
    return working


def _ensure_objective_and_severity(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if _WEIGHTED_SCORE_COLUMN not in working.columns:
        working = add_weighted_objective(working)
    if _FAILURE_SEVERITY_COLUMN not in working.columns:
        working = add_failure_severity(working)
    return working


def rank_single_model_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by failure severity, weighted objective, then name."""

    if candidates.empty:
        return candidates.copy()

    working = _ensure_objective_and_severity(candidates)
    _required_columns_exist(working, (_MODEL_NAME_COLUMN,))
    ranked = working.sort_values(
        by=[_FAILURE_SEVERITY_COLUMN, _WEIGHTED_SCORE_COLUMN, _MODEL_NAME_COLUMN],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ranked.reset_index(drop=True)


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    left_severity = float(left[_FAILURE_SEVERITY_COLUMN])
    right_severity = float(right[_FAILURE_SEVERITY_COLUMN])
    left_score = float(left[_WEIGHTED_SCORE_COLUMN])
    right_score = float(right[_WEIGHTED_SCORE_COLUMN])
    return (
        left_severity <= right_severity
        and left_score >= right_score
        and (left_severity < right_severity or left_score > right_score)
    )


def build_pareto_shortlist(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the deterministic non-dominated shortlist."""

    if candidates.empty:
        return candidates.copy()

    working = rank_single_model_candidates(candidates)
    shortlist_rows: list[int] = []
    for idx, row in working.iterrows():
        dominated = False
        for other_idx, other in working.iterrows():
            if idx == other_idx:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            shortlist_rows.append(idx)
    shortlist = working.loc[shortlist_rows].copy()
    return shortlist.sort_values(
        by=[_FAILURE_SEVERITY_COLUMN, _WEIGHTED_SCORE_COLUMN, _MODEL_NAME_COLUMN],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = [
    "add_failure_severity",
    "add_weighted_objective",
    "build_pareto_shortlist",
    "rank_single_model_candidates",
]
