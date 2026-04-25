from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

HARD_GATE_COLUMNS: tuple[str, ...] = (
    "feature_safety_pass",
    "label_quality_pass",
    "temporal_gate_pass",
    "source_holdout_gate_pass",
    "knownness_gate_pass",
    "calibration_gate_pass",
    "abstention_gate_pass",
)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def annotate_gate_first_scorecard(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return scorecard.copy()

    working = scorecard.copy()
    for column in HARD_GATE_COLUMNS:
        working[column] = _bool_series(working, column)

    gate_matrix = working.loc[:, list(HARD_GATE_COLUMNS)]
    working["failed_gate_count"] = (~gate_matrix).sum(axis=1).astype(int)
    working["all_hard_gates_pass"] = working["failed_gate_count"].eq(0)

    decision_utility = pd.to_numeric(
        working.get("decision_utility", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    brier = pd.to_numeric(
        working.get("brier_score", pd.Series(1.0, index=working.index)),
        errors="coerce",
    ).fillna(1.0)
    ece = pd.to_numeric(
        working.get("expected_calibration_error", pd.Series(1.0, index=working.index)),
        errors="coerce",
    ).fillna(1.0)
    feature_count = pd.to_numeric(
        working.get("feature_count", pd.Series(999.0, index=working.index)),
        errors="coerce",
    ).fillna(999.0)
    auc = pd.to_numeric(
        working.get("roc_auc", pd.Series(0.5, index=working.index)),
        errors="coerce",
    ).fillna(0.5)
    average_precision = pd.to_numeric(
        working.get("average_precision", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)

    working["gate_first_selection_score"] = (
        decision_utility
        + (1.0 - brier.clip(lower=0.0, upper=1.0))
        + (1.0 - ece.clip(lower=0.0, upper=1.0))
        + auc
        + average_precision
        - (feature_count.clip(lower=0.0) * 0.001)
    )
    return working


def select_gate_first_model(scorecard: pd.DataFrame) -> dict[str, Any]:
    annotated = annotate_gate_first_scorecard(scorecard)
    if annotated.empty:
        raise ValueError("No candidate models were provided")

    eligible = annotated.loc[annotated["all_hard_gates_pass"]].copy()
    if eligible.empty:
        raise ValueError("No candidate model passed all hard gates")

    sort_columns = [
        "gate_first_selection_score",
        "decision_utility",
        "brier_score",
        "expected_calibration_error",
        "feature_count",
        "roc_auc",
        "average_precision",
    ]
    ascending = [False, False, True, True, True, False, False]
    for column in sort_columns:
        if column not in eligible.columns:
            eligible[column] = np.nan

    winner = cast(
        dict[str, Any],
        eligible.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .iloc[0]
        .to_dict(),
    )
    winner["selection_status"] = "selected"
    return winner
