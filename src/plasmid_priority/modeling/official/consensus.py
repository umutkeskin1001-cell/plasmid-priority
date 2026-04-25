from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

EVIDENCE_TIER_FACTORS: Mapping[str, float] = {
    "high": 1.00,
    "moderate": 0.90,
    "low": 0.72,
    "insufficient": 0.55,
}
UNCERTAINTY_TIER_FACTORS: Mapping[str, float] = {
    "low": 1.00,
    "moderate": 0.90,
    "high": 0.72,
    "extreme": 0.55,
}


def _score_matrix(frame: pd.DataFrame, score_columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in score_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing consensus score columns: {missing}")
    scores = frame.loc[:, list(score_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return cast(pd.DataFrame, scores.clip(lower=0.0, upper=1.0))


def _tier_factor(values: pd.Series, factors: Mapping[str, float], *, default: float) -> pd.Series:
    normalized = values.fillna("").astype(str).str.lower().str.strip()
    return normalized.map(factors).fillna(default).astype("float64")


def conservative_evidence_consensus(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    evidence_tier_column: str = "evidence_tier",
    uncertainty_tier_column: str = "uncertainty_tier",
    review_agreement_threshold: float = 0.55,
) -> pd.DataFrame:
    """Fuse official model scores with explicit evidence and uncertainty penalties."""
    if not score_columns:
        raise ValueError("At least one score column is required for consensus")
    scores = _score_matrix(frame, score_columns)
    median_score = scores.median(axis=1).astype("float64")
    score_std = scores.std(axis=1).fillna(0.0).astype("float64")
    agreement = pd.Series(
        np.clip(1.0 - (2.0 * score_std.to_numpy(dtype=float)), 0.0, 1.0),
        index=frame.index,
        dtype="float64",
    )

    evidence_tier = (
        frame[evidence_tier_column]
        if evidence_tier_column in frame.columns
        else pd.Series("insufficient", index=frame.index)
    )
    uncertainty_tier = (
        frame[uncertainty_tier_column]
        if uncertainty_tier_column in frame.columns
        else pd.Series("high", index=frame.index)
    )
    evidence_factor = _tier_factor(evidence_tier, EVIDENCE_TIER_FACTORS, default=0.55)
    uncertainty_factor = _tier_factor(uncertainty_tier, UNCERTAINTY_TIER_FACTORS, default=0.72)

    consensus_score = (
        median_score
        * evidence_factor
        * uncertainty_factor
        * (0.85 + 0.15 * agreement)
    )
    consensus_score = pd.Series(
        np.clip(consensus_score.to_numpy(dtype=float), 0.0, 1.0),
        index=frame.index,
        dtype="float64",
    )
    review_flag = (
        agreement.lt(review_agreement_threshold)
        | evidence_factor.lt(0.80)
        | uncertainty_factor.lt(0.80)
    )

    return pd.DataFrame(
        {
            "conservative_consensus_score": consensus_score,
            "model_agreement": agreement,
            "model_score_median": median_score,
            "model_score_std": score_std,
            "evidence_factor": evidence_factor,
            "uncertainty_factor": uncertainty_factor,
            "consensus_review_flag": review_flag.astype(bool),
        },
        index=frame.index,
    )
