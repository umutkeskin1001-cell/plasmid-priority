"""Consensus fusion logic."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def _score_columns(frame: pd.DataFrame) -> pd.Series:
    for candidate in (
        "prediction_calibrated",
        "calibrated_prediction",
        "prediction",
        "oof_prediction",
    ):
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


def _branch_agreement_score(branch_matrix: np.ndarray) -> pd.Series:
    if branch_matrix.size == 0:
        return pd.Series(dtype=float)
    if branch_matrix.shape[1] == 1:
        return pd.Series(1.0, index=range(branch_matrix.shape[0]), dtype=float)
    centered = branch_matrix - branch_matrix.mean(axis=1, keepdims=True)
    dispersion = np.sqrt(np.square(centered).mean(axis=1))
    pairwise_scale = float(np.sqrt(3.0) / 2.0)
    score = 1.0 - np.clip(dispersion / pairwise_scale, 0.0, 1.0)
    return pd.Series(score, dtype=float)


def _resolve_attenuation_params(
    params: Mapping[str, float] | None,
) -> dict[str, float]:
    resolved = {
        "confidence_floor": 0.65,
        "confidence_scale": 0.35,
        "ood_floor": 0.65,
        "ood_scale": 0.35,
        "agreement_floor": 0.55,
        "agreement_scale": 0.45,
        "review_agreement_threshold": 0.55,
    }
    if params:
        for key, value in params.items():
            if key in resolved:
                resolved[key] = float(value)
    return resolved


def _apply_consensus_attenuation(
    consensus_raw: pd.Series,
    branch_confidence: pd.Series,
    ood_mean: pd.Series,
    agreement_score: pd.Series,
    *,
    confidence_floor: float,
    confidence_scale: float,
    ood_floor: float,
    ood_scale: float,
    agreement_floor: float,
    agreement_scale: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    index = consensus_raw.index
    confidence_attenuation = pd.Series(
        np.clip(
            confidence_floor + confidence_scale * branch_confidence.fillna(0.0),
            0.0,
            1.0,
        ),
        index=index,
        dtype=float,
    )
    ood_attenuation = pd.Series(
        np.clip(1.0 - ood_scale * ood_mean.fillna(0.0), ood_floor, 1.0),
        index=index,
        dtype=float,
    )
    agreement_attenuation = pd.Series(
        np.clip(
            agreement_floor + agreement_scale * agreement_score.fillna(0.0),
            0.0,
            1.0,
        ),
        index=index,
        dtype=float,
    )
    consensus_attenuation = pd.Series(
        np.minimum.reduce(
            [
                confidence_attenuation.to_numpy(),
                ood_attenuation.to_numpy(),
                agreement_attenuation.to_numpy(),
            ]
        ),
        index=index,
        dtype=float,
    )
    consensus_score = pd.Series(
        np.clip(consensus_raw * consensus_attenuation, 0.0, 1.0),
        index=index,
        dtype=float,
    )
    consensus_uncertainty = pd.Series(
        np.clip(
            (1.0 - agreement_score.fillna(0.0)) * 0.45
            + (1.0 - branch_confidence.fillna(0.0)) * 0.35
            + ood_mean.fillna(0.0) * 0.20,
            0.0,
            1.0,
        ),
        index=index,
        dtype=float,
    )
    score_lower = pd.Series(
        np.clip(consensus_score - consensus_uncertainty * 0.5, 0.0, 1.0),
        index=index,
        dtype=float,
    )
    score_upper = pd.Series(
        np.clip(consensus_score + consensus_uncertainty * 0.5, 0.0, 1.0),
        index=index,
        dtype=float,
    )
    return consensus_attenuation, consensus_score, consensus_uncertainty, score_lower, score_upper


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
    attenuation_params: Mapping[str, float] | None = None,
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
        working[column] = _numeric_or_default(
            working,
            column,
            default=confidence_floor,
        ).fillna(confidence_floor)
    for column in ("ood_geo", "ood_bio_transfer", "ood_clinical_hazard"):
        working[column] = (
            working.get(column, pd.Series(False, index=working.index)).fillna(False).astype(bool)
        )
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
    branch_matrix = working.loc[:, ["p_geo", "p_bio_transfer", "p_clinical_hazard"]].to_numpy(
        dtype=float
    )
    agreement_score = _branch_agreement_score(branch_matrix).reindex(working.index, fill_value=1.0)
    params = _resolve_attenuation_params(attenuation_params)
    consensus_attenuation, consensus_score, consensus_uncertainty, score_lower, score_upper = (
        _apply_consensus_attenuation(
            consensus_raw,
            branch_confidence,
            ood_mean,
            agreement_score,
            confidence_floor=params["confidence_floor"],
            confidence_scale=params["confidence_scale"],
            ood_floor=params["ood_floor"],
            ood_scale=params["ood_scale"],
            agreement_floor=params["agreement_floor"],
            agreement_scale=params["agreement_scale"],
        )
    )
    consensus_review_flag = (
        (branch_confidence < review_threshold)
        | (ood_mean > ood_threshold)
        | (agreement_score < params["review_agreement_threshold"])
    )
    consensus_priority_tier = pd.Series("low", index=working.index, dtype=object)
    consensus_priority_tier[consensus_score >= 0.75] = "high"
    consensus_priority_tier[(consensus_score >= 0.50) & (consensus_score < 0.75)] = "medium"
    consensus_priority_tier[consensus_review_flag] = "review"
    out = working.copy()
    out["consensus_raw"] = consensus_raw
    out["consensus_score"] = consensus_score
    out["branch_agreement_score"] = agreement_score
    out["consensus_attenuation"] = consensus_attenuation
    out["consensus_uncertainty"] = consensus_uncertainty
    out["consensus_score_lower"] = score_lower
    out["consensus_score_upper"] = score_upper
    out["consensus_review_flag"] = consensus_review_flag
    out["consensus_priority_tier"] = consensus_priority_tier
    out["branch_contribution_geo"] = (
        resolved_weights["p_geo"] * working["p_geo"] * consensus_attenuation
    )
    out["branch_contribution_bio_transfer"] = (
        resolved_weights["p_bio_transfer"] * working["p_bio_transfer"] * consensus_attenuation
    )
    out["branch_contribution_clinical_hazard"] = (
        resolved_weights["p_clinical_hazard"] * working["p_clinical_hazard"] * consensus_attenuation
    )
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
    geo_grid = np.linspace(0.35, 0.65, 13)
    candidate_weights = tuple(
        {
            "p_geo": float(geo_weight),
            "p_bio_transfer": float((1.0 - geo_weight) / 2.0),
            "p_clinical_hazard": float((1.0 - geo_weight) / 2.0),
        }
        for geo_weight in geo_grid
    )
    best_weights = candidate_weights[0]
    best_weight_score = float("-inf")
    valid_index = valid.index[valid]
    valid_y = y.loc[valid_index].astype(int)
    for weights in candidate_weights:
        frame = build_operational_consensus_frame(working, weights=weights)
        score = pd.to_numeric(frame["consensus_score"], errors="coerce").loc[valid_index]
        if score.nunique() < 2:
            continue
        try:
            combined = float(
                0.6 * roc_auc_score(valid_y, score) + 0.4 * average_precision(valid_y, score)
            )
        except ValueError:
            continue
        if combined > best_weight_score:
            best_weight_score = combined
            best_weights = weights

    candidate_attenuation_params = tuple(
        {
            "confidence_floor": confidence_floor,
            "confidence_scale": confidence_scale,
            "ood_floor": ood_floor,
            "ood_scale": ood_scale,
            "agreement_floor": agreement_floor,
            "agreement_scale": agreement_scale,
            "review_agreement_threshold": review_agreement_threshold,
        }
        for (
            confidence_floor,
            confidence_scale,
            ood_floor,
            ood_scale,
            agreement_floor,
            agreement_scale,
            review_agreement_threshold,
        ) in product(
            (0.55, 0.65),
            (0.25, 0.35),
            (0.60, 0.65),
            (0.25, 0.35),
            (0.45, 0.55),
            (0.35, 0.50),
            (0.45, 0.55, 0.65),
        )
    )

    frame_for_tuning = working.loc[valid_index].copy()
    y_for_tuning = valid_y
    if len(y_for_tuning) >= 6 and y_for_tuning.nunique() == 2:
        n_splits = min(5, int(y_for_tuning.value_counts().min()))
        n_splits = max(n_splits, 2)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = list(splitter.split(np.zeros(len(y_for_tuning)), y_for_tuning))
    else:
        fold_indices = [(np.arange(len(y_for_tuning)), np.arange(len(y_for_tuning)))]

    best_score = float("-inf")
    best_attenuation_params: dict[str, float] = {}
    best_frame = build_operational_consensus_frame(working, weights=best_weights)
    for attenuation_params in candidate_attenuation_params:
        candidate_frame = build_operational_consensus_frame(
            frame_for_tuning,
            weights=best_weights,
            attenuation_params=attenuation_params,
        )
        score = pd.to_numeric(candidate_frame["consensus_score"], errors="coerce").reset_index(
            drop=True
        )
        fold_scores: list[float] = []
        for _, fold_valid in fold_indices:
            fold_score = score.iloc[fold_valid]
            fold_y = y_for_tuning.reset_index(drop=True).iloc[fold_valid]
            # Guard against NaN — sklearn metrics crash on NaN input
            if fold_score.isna().any() or fold_y.isna().any():
                valid_mask = fold_score.notna() & fold_y.notna()
                fold_score = fold_score[valid_mask]
                fold_y = fold_y[valid_mask]
                if len(fold_score) < 4:
                    continue
            if fold_score.nunique() < 2 or fold_y.nunique() < 2:
                continue
            try:
                fold_scores.append(
                    float(
                        0.6 * roc_auc_score(fold_y.astype(int), fold_score)
                        + 0.4 * average_precision(fold_y.astype(int), fold_score)
                    )
                )
            except ValueError:
                continue
        if not fold_scores:
            continue
        combined = float(np.mean(fold_scores))
        if combined > best_score:
            best_score = combined
            best_attenuation_params = attenuation_params
            best_frame = build_operational_consensus_frame(
                working,
                weights=best_weights,
                attenuation_params=attenuation_params,
            )

    best_frame = best_frame.copy()
    best_frame["research_weight_geo"] = best_weights["p_geo"]
    best_frame["research_weight_bio_transfer"] = best_weights["p_bio_transfer"]
    best_frame["research_weight_clinical_hazard"] = best_weights["p_clinical_hazard"]
    if best_attenuation_params:
        best_frame["research_confidence_floor"] = best_attenuation_params["confidence_floor"]
        best_frame["research_confidence_scale"] = best_attenuation_params["confidence_scale"]
        best_frame["research_ood_floor"] = best_attenuation_params["ood_floor"]
        best_frame["research_ood_scale"] = best_attenuation_params["ood_scale"]
        best_frame["research_agreement_floor"] = best_attenuation_params["agreement_floor"]
        best_frame["research_agreement_scale"] = best_attenuation_params["agreement_scale"]
        best_frame["research_agreement_threshold"] = best_attenuation_params[
            "review_agreement_threshold"
        ]
    best_frame["research_score"] = best_score if np.isfinite(best_score) else float("nan")
    return best_frame
