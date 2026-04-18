"""Explanation and review helpers for branch predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_branch_review_reason_table(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prediction_calibrated",
    confidence_column: str = "confidence_score",
    ood_column: str = "ood_flag",
    review_column: str = "review_flag",
) -> pd.DataFrame:
    """Build a compact review watchlist table from branch predictions."""
    if predictions.empty:
        return pd.DataFrame(
            columns=["backbone_id", "review_flag", "review_reason", "confidence_score", "ood_flag"]
        )
    working = predictions.copy()
    if score_column not in working.columns and "prediction" in working.columns:
        working[score_column] = working["prediction"]
    score = pd.to_numeric(
        working.get(score_column, pd.Series(index=working.index, dtype=float)),
        errors="coerce",
    )
    confidence = pd.to_numeric(
        working.get(confidence_column, pd.Series(index=working.index, dtype=float)),
        errors="coerce",
    )
    ood_flag = (
        working.get(ood_column, pd.Series(False, index=working.index)).fillna(False).astype(bool)
    )
    review_flag = (
        working.get(review_column, pd.Series(False, index=working.index)).fillna(False).astype(bool)
    )
    review_reason = np.where(
        ood_flag,
        "ood",
        np.where(review_flag, "review", np.where(confidence < 0.45, "low_confidence", "clear")),
    )
    return pd.DataFrame(
        {
            "backbone_id": working.get("backbone_id", pd.Series(dtype=str)).astype(str),
            "prediction_calibrated": score,
            "confidence_score": confidence,
            "ood_flag": ood_flag,
            "review_flag": review_flag,
            "review_reason": review_reason,
        }
    )


def build_branch_explanation_table(
    predictions: pd.DataFrame,
    *,
    branch_name: str,
) -> pd.DataFrame:
    """Produce a compact explanation matrix for a branch."""
    if predictions.empty:
        return pd.DataFrame(columns=["branch_name", "backbone_id", "explanation", "review_reason"])
    review_table = build_branch_review_reason_table(predictions)
    review_table.insert(0, "branch_name", str(branch_name))
    review_table["explanation"] = np.where(
        review_table["review_flag"].astype(bool),
        "review_required",
        np.where(review_table["ood_flag"].astype(bool), "ood_attention", "clear"),
    )
    return review_table
