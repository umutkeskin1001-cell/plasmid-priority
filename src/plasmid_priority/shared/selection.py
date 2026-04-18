"""Branch model selection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd


def _safe_metric_series(frame: pd.DataFrame, columns: list[str], *, ascending: bool) -> pd.Series:
    if not columns:
        return pd.Series(0.5, index=frame.index, dtype=float)
    numeric = frame.loc[:, [column for column in columns if column in frame.columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.empty:
        return pd.Series(0.5, index=frame.index, dtype=float)
    values = numeric.mean(axis=1, skipna=True)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=frame.index, dtype=float)
    ranked = values.rank(method="average", pct=True)
    if ascending:
        ranked = 1.0 - ranked
    return cast(pd.Series, ranked.fillna(0.5))


def _numeric_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_branch_selection_scorecard(
    results: Mapping[str, Any],
    *,
    calibration_summary: pd.DataFrame | None = None,
    selection_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build a bounded selection scorecard over candidate models."""
    calibration_frame = calibration_summary if calibration_summary is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    selection_weights = selection_weights or {
        "roc_auc": 0.40,
        "average_precision": 0.25,
        "calibrated_expected_calibration_error": -0.20,
        "brier_score": -0.10,
        "low_knownness_roc_auc": 0.05,
        "low_knownness_average_precision": 0.05,
        "ood_rate": -0.10,
        "review_fraction": -0.05,
    }
    for model_name, result in results.items():
        row = {
            "model_name": str(model_name),
            "status": getattr(result, "status", "ok"),
            "error_message": getattr(result, "error_message", None),
        }
        row.update(getattr(result, "metrics", {}))
        if not calibration_frame.empty:
            match = calibration_frame.loc[
                calibration_frame.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            ]
            if not match.empty:
                for column in (
                    "calibration_method",
                    "abstain_rate",
                    "mean_confidence",
                    "ood_rate",
                    "calibrated_expected_calibration_error",
                    "calibrated_brier_score",
                ):
                    if column in match.columns:
                        row[column] = match.iloc[0].get(column)
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard

    oriented_components: list[pd.Series] = []
    total_weight = 0.0
    for metric_name, weight in selection_weights.items():
        abs_weight = abs(float(weight))
        if abs_weight <= 0.0:
            continue
        oriented = _safe_metric_series(
            scorecard,
            [str(metric_name)],
            ascending=float(weight) < 0.0,
        )
        oriented_components.append(oriented * abs_weight)
        total_weight += abs_weight
    if oriented_components and total_weight > 0.0:
        stacked = pd.concat(oriented_components, axis=1)
        scorecard["selection_score"] = stacked.sum(axis=1) / total_weight
    else:
        scorecard["selection_score"] = 0.5
    scorecard["selection_rank"] = (
        scorecard["selection_score"].rank(method="dense", ascending=False).astype(int)
    )
    scorecard = scorecard.sort_values(
        ["selection_score", "model_name"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scorecard["selection_rank"] = np.arange(1, len(scorecard) + 1)
    discrimination_ok = _numeric_metric(scorecard, "roc_auc") >= 0.6
    calibration_metric = _numeric_metric(scorecard, "calibrated_expected_calibration_error")
    calibration_metric = calibration_metric.where(
        calibration_metric.notna(),
        _numeric_metric(scorecard, "expected_calibration_error"),
    )
    calibration_ok = calibration_metric <= 0.2
    ood_ok = _numeric_metric(scorecard, "ood_rate") <= 0.2
    review_metric = _numeric_metric(scorecard, "review_fraction")
    review_metric = review_metric.where(
        review_metric.notna(),
        _numeric_metric(scorecard, "abstain_rate"),
    )
    review_ok = review_metric <= 0.35

    rationale_parts = [
        ("strong_discrimination", discrimination_ok),
        ("good_calibration", calibration_ok),
        ("low_ood", ood_ok),
        ("low_review", review_ok),
    ]
    rationale_frame = pd.DataFrame(index=scorecard.index)
    for token, mask in rationale_parts:
        rationale_frame[token] = mask.fillna(False).astype(bool)
    rationale_labels = [
        ", ".join(
            column
            for column, active in zip(rationale_frame.columns, row, strict=False)
            if bool(active)
        )
        or "balanced_selection"
        for row in rationale_frame.to_numpy(dtype=bool, copy=False)
    ]
    scorecard["selection_rationale"] = rationale_labels
    return scorecard


def select_branch_primary_model(
    results: Mapping[str, Any],
    *,
    calibration_summary: pd.DataFrame | None = None,
    selection_weights: Mapping[str, float] | None = None,
) -> tuple[str, pd.DataFrame]:
    """Select the top branch model using the scorecard."""
    scorecard = build_branch_selection_scorecard(
        results,
        calibration_summary=calibration_summary,
        selection_weights=selection_weights,
    )
    if scorecard.empty:
        return "", scorecard
    return str(scorecard.iloc[0]["model_name"]), scorecard
