"""Consensus fusion logic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def _score_columns(frame: pd.DataFrame) -> pd.Series:
    for candidate in ("prediction_calibrated", "calibrated_prediction", "prediction", "oof_prediction"):
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _numeric_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _confidence_columns(frame: pd.DataFrame) -> pd.Series:
    for candidate in ("confidence_score", "confidence", "prediction_confidence"):
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _ood_columns(frame: pd.DataFrame) -> pd.Series:
    for candidate in ("ood_flag", "ood", "out_of_distribution_flag"):
        if candidate in frame.columns:
            return frame[candidate].fillna(False).astype(bool)
    return pd.Series(False, index=frame.index, dtype=bool)


def merge_branch_predictions(
    geo_predictions: pd.DataFrame,
    bio_predictions: pd.DataFrame,
    clinical_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge calibrated branch predictions into a common consensus table."""
    frames: list[pd.DataFrame] = []
    spread_label_added = False
    for frame, prefix in (
        (geo_predictions, "geo"),
        (bio_predictions, "bio_transfer"),
        (clinical_predictions, "clinical_hazard"),
    ):
        if frame.empty:
            continue
        renamed = frame.loc[:, ~frame.columns.duplicated()].copy()
        renamed["backbone_id"] = renamed["backbone_id"].astype(str)
        renamed[f"p_{prefix}"] = _score_columns(renamed)
        renamed[f"confidence_{prefix}"] = _confidence_columns(renamed)
        renamed[f"ood_{prefix}"] = _ood_columns(renamed)
        keep_columns = ["backbone_id", f"p_{prefix}", f"confidence_{prefix}", f"ood_{prefix}"]
        if not spread_label_added and "spread_label" in renamed.columns:
            keep_columns.insert(1, "spread_label")
            spread_label_added = True
        frames.append(renamed.loc[:, keep_columns].drop_duplicates(subset=["backbone_id"]))
    if not frames:
        return pd.DataFrame()
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.merge(frame, on="backbone_id", how="outer")
    return merged


def _default_weights() -> dict[str, float]:
    return {
        "p_geo": 0.50,
        "p_bio_transfer": 0.25,
        "p_clinical_hazard": 0.25,
    }


def _resolve_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = _default_weights()
    if weights:
        for key, value in weights.items():
            if key in resolved:
                resolved[key] = float(value)
    total = sum(max(value, 0.0) for value in resolved.values())
    if total <= 0.0:
        return _default_weights()
    return {key: max(value, 0.0) / total for key, value in resolved.items()}


def build_operational_consensus_frame(
    merged: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    confidence_floor: float = 0.25,
    ood_threshold: float = 0.20,
    review_threshold: float = 0.20,
) -> pd.DataFrame:
    """Build the bounded operational consensus surface."""
    if merged.empty:
        return pd.DataFrame(columns=["backbone_id", "consensus_score"])
    resolved_weights = _resolve_weights(weights)
    working = merged.copy()
    for column in ("p_geo", "p_bio_transfer", "p_clinical_hazard"):
        working[column] = _numeric_or_default(working, column, default=0.5).fillna(0.5)
    for column in ("confidence_geo", "confidence_bio_transfer", "confidence_clinical_hazard"):
        working[column] = _numeric_or_default(working, column, default=confidence_floor).fillna(confidence_floor)
    for column in ("ood_geo", "ood_bio_transfer", "ood_clinical_hazard"):
        working[column] = working.get(column, pd.Series(False, index=working.index)).fillna(False).astype(bool)
    consensus_raw = (
        resolved_weights["p_geo"] * working["p_geo"]
        + resolved_weights["p_bio_transfer"] * working["p_bio_transfer"]
        + resolved_weights["p_clinical_hazard"] * working["p_clinical_hazard"]
    )
    branch_confidence = (
        0.50 * working["confidence_geo"]
        + 0.25 * working["confidence_bio_transfer"]
        + 0.25 * working["confidence_clinical_hazard"]
    )
    ood_mean = (
        working["ood_geo"].astype(float)
        + working["ood_bio_transfer"].astype(float)
        + working["ood_clinical_hazard"].astype(float)
    ) / 3.0
    branch_matrix = working.loc[:, ["p_geo", "p_bio_transfer", "p_clinical_hazard"]].to_numpy(dtype=float)
    agreement_score = pd.Series(
        1.0 - np.abs(branch_matrix - branch_matrix.mean(axis=1, keepdims=True)).mean(axis=1),
        index=working.index,
        dtype=float,
    ).clip(lower=0.0, upper=1.0)
    confidence_attenuation = np.clip(0.50 + 0.50 * branch_confidence.fillna(confidence_floor), 0.25, 1.0)
    ood_attenuation = np.clip(1.0 - 0.35 * ood_mean.fillna(0.0), 0.40, 1.0)
    consensus_score = np.clip(consensus_raw * confidence_attenuation * ood_attenuation, 0.0, 1.0)
    consensus_review_flag = (
        (branch_confidence < review_threshold)
        | (ood_mean > ood_threshold)
        | (agreement_score < 0.55)
    )
    consensus_priority_tier = pd.Series("low", index=working.index, dtype=object)
    consensus_priority_tier[consensus_score >= 0.75] = "high"
    consensus_priority_tier[(consensus_score >= 0.50) & (consensus_score < 0.75)] = "medium"
    consensus_priority_tier[consensus_review_flag] = "review"
    out = working.copy()
    out["consensus_raw"] = consensus_raw
    out["consensus_score"] = consensus_score
    out["branch_agreement_score"] = agreement_score
    out["consensus_review_flag"] = consensus_review_flag
    out["consensus_priority_tier"] = consensus_priority_tier
    out["branch_contribution_geo"] = resolved_weights["p_geo"] * working["p_geo"] * confidence_attenuation * ood_attenuation
    out["branch_contribution_bio_transfer"] = resolved_weights["p_bio_transfer"] * working["p_bio_transfer"] * confidence_attenuation * ood_attenuation
    out["branch_contribution_clinical_hazard"] = resolved_weights["p_clinical_hazard"] * working["p_clinical_hazard"] * confidence_attenuation * ood_attenuation
    return out


def build_research_consensus_frame(
    merged: pd.DataFrame,
    *,
    label_column: str = "spread_label",
) -> pd.DataFrame:
    """Search bounded weights on the labeled consensus surface when available."""
    if merged.empty:
        return pd.DataFrame(columns=["backbone_id", "consensus_score"])
    working = merged.copy()
    if label_column not in working.columns:
        return build_operational_consensus_frame(working)
    y = pd.to_numeric(working[label_column], errors="coerce")
    valid = y.notna()
    if valid.sum() < 4 or y.loc[valid].nunique() < 2:
        return build_operational_consensus_frame(working)
    candidate_weights = (
        {"p_geo": 0.40, "p_bio_transfer": 0.30, "p_clinical_hazard": 0.30},
        {"p_geo": 0.45, "p_bio_transfer": 0.275, "p_clinical_hazard": 0.275},
        {"p_geo": 0.50, "p_bio_transfer": 0.25, "p_clinical_hazard": 0.25},
        {"p_geo": 0.55, "p_bio_transfer": 0.225, "p_clinical_hazard": 0.225},
    )
    best_frame = build_operational_consensus_frame(working, weights=candidate_weights[2])
    best_score = float("-inf")
    for weights in candidate_weights:
        frame = build_operational_consensus_frame(working, weights=weights)
        score = pd.to_numeric(frame["consensus_score"], errors="coerce")
        if score.loc[valid].nunique() < 2:
            continue
        auc = roc_auc_score(y.loc[valid].astype(int), score.loc[valid])
        ap = average_precision(y.loc[valid].astype(int), score.loc[valid])
        combined = float(0.6 * auc + 0.4 * ap)
        if combined > best_score:
            best_score = combined
            best_frame = frame.copy()
            best_frame["research_weight_geo"] = weights["p_geo"]
            best_frame["research_weight_bio_transfer"] = weights["p_bio_transfer"]
            best_frame["research_weight_clinical_hazard"] = weights["p_clinical_hazard"]
            best_frame["research_score"] = combined
    return best_frame
