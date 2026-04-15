"""Generic calibration and uncertainty helpers for branch prediction surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from plasmid_priority.shared.specs import BranchConfig, BranchFitConfig
from plasmid_priority.validation.metrics import (
    brier_score,
    calibration_curve_data,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
)


def _normalize_method(value: object | None) -> str:
    normalized = str(value or "none").strip().lower()
    if normalized in {"", "none"}:
        return "none"
    if normalized not in {"platt", "isotonic"}:
        raise ValueError("calibration method must be one of: none, platt, isotonic")
    return normalized


def _safe_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _float_option(payload: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _fit_platt_calibrator(y_true: np.ndarray, y_score: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    if np.unique(y_true).size < 2 or len(y_true) < 2:
        return lambda values: _safe_probability(values)
    model = LogisticRegression(
        C=1e6,
        fit_intercept=True,
        max_iter=1000,
        random_state=0,
        solver="lbfgs",
    )
    model.fit(np.asarray(y_score, dtype=float).reshape(-1, 1), np.asarray(y_true, dtype=int))
    return lambda values: _safe_probability(model.predict_proba(np.asarray(values, dtype=float).reshape(-1, 1))[:, 1])


def _fit_isotonic_calibrator(y_true: np.ndarray, y_score: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    if np.unique(y_true).size < 2 or len(y_true) < 2:
        return lambda values: _safe_probability(values)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(np.asarray(y_score, dtype=float), np.asarray(y_true, dtype=float))
    return lambda values: _safe_probability(model.predict(np.asarray(values, dtype=float)))


@dataclass(slots=True)
class BranchCalibrationResult:
    """Calibrated prediction surface for one branch model."""

    model_name: str
    calibration_method: str
    predictions: pd.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]


def _prediction_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    frame = getattr(value, "predictions", None)
    if isinstance(frame, pd.DataFrame):
        return frame
    raise TypeError("Expected a dataframe or an object with a dataframe `predictions` attribute")


def _series_or_default(frame: pd.DataFrame, column: str, default: float = float("nan")) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def calibrate_branch_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    fit_config: Mapping[str, Any] | BranchFitConfig | BranchConfig | Any | None = None,
    scored: pd.DataFrame | None = None,
    label_column: str = "spread_label",
    score_column: str = "oof_prediction",
    knownness_column: str = "knownness_score",
    entity_id_column: str = "backbone_id",
) -> BranchCalibrationResult:
    """Calibrate one model's branch predictions and derive uncertainty flags."""
    if predictions.empty:
        empty = pd.DataFrame(
            columns=[
                entity_id_column,
                score_column,
                "prediction_calibrated",
                "prediction_uncertainty",
                "confidence_score",
                "confidence_band",
                "support_sufficient_flag",
                "ood_flag",
                "review_flag",
                "review_reason",
            ]
        )
        return BranchCalibrationResult(
            model_name=str(model_name),
            calibration_method="none",
            predictions=empty,
            summary={
                "model_name": str(model_name),
                "calibration_method": "none",
                "n_rows": 0,
                "n_labeled": 0,
                "n_review": 0,
            },
            provenance={},
        )

    working = _prediction_frame(predictions).copy()
    if label_column not in working.columns and scored is not None and not scored.empty:
        if entity_id_column in working.columns and label_column in scored.columns:
            working = working.merge(
                scored.loc[:, [entity_id_column, label_column]].drop_duplicates(subset=[entity_id_column]),
                on=entity_id_column,
                how="left",
            )
    if label_column not in working.columns:
        raise KeyError(f"Branch predictions require `{label_column}` for calibration.")

    fit_payload: dict[str, Any]
    if isinstance(fit_config, BranchConfig):
        fit_payload = dict(fit_config.fit_config.get(str(model_name), BranchFitConfig()).model_dump(mode="python"))
    elif isinstance(fit_config, BranchFitConfig):
        fit_payload = dict(fit_config.model_dump(mode="python"))
    elif hasattr(fit_config, "model_dump"):
        fit_payload = dict(fit_config.model_dump(mode="python"))
    else:
        fit_payload = dict(fit_config or {})

    method = _normalize_method(fit_payload.get("calibration", "none"))
    support_threshold = float(np.clip(_float_option(fit_payload, "support_knownness_threshold", 0.25), 0.0, 1.0))
    ood_knownness_threshold = float(np.clip(_float_option(fit_payload, "ood_knownness_threshold", 0.10), 0.0, 1.0))
    confidence_review_threshold = float(np.clip(_float_option(fit_payload, "confidence_review_threshold", 0.45), 0.0, 1.0))

    y_true = pd.to_numeric(working[label_column], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    raw_scores = pd.to_numeric(working[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    labeled_mask = y_true >= 0

    if method == "platt":
        calibrator = _fit_platt_calibrator(y_true[labeled_mask], raw_scores[labeled_mask])
    elif method == "isotonic":
        calibrator = _fit_isotonic_calibrator(y_true[labeled_mask], raw_scores[labeled_mask])
    else:
        calibrator = lambda values: _safe_probability(values)

    calibrated = calibrator(raw_scores)
    knownness_numeric = _series_or_default(working, knownness_column).fillna(np.nan).to_numpy(dtype=float)
    support_sufficient = np.where(
        np.isnan(knownness_numeric),
        True,
        knownness_numeric >= support_threshold,
    )
    finite_labeled_scores = raw_scores[labeled_mask & np.isfinite(raw_scores)]
    if finite_labeled_scores.size:
        lower_support = float(np.nanquantile(finite_labeled_scores, 0.02))
        upper_support = float(np.nanquantile(finite_labeled_scores, 0.98))
    else:
        lower_support = 0.0
        upper_support = 1.0
    ood_flag = (
        np.where(np.isnan(knownness_numeric), False, knownness_numeric < ood_knownness_threshold)
        | (raw_scores < (lower_support - 0.05))
        | (raw_scores > (upper_support + 0.05))
    )
    score_uncertainty = 4.0 * raw_scores * (1.0 - raw_scores)
    calibration_gap = np.abs(calibrated - raw_scores)
    support_uncertainty = np.where(np.isnan(knownness_numeric), 0.35, 1.0 - np.clip(knownness_numeric, 0.0, 1.0))
    prediction_uncertainty = np.clip(
        0.55 * score_uncertainty + 0.30 * calibration_gap + 0.15 * support_uncertainty,
        0.0,
        1.0,
    )
    confidence = np.clip(1.0 - prediction_uncertainty, 0.0, 1.0)
    review_flag = (~support_sufficient) | ood_flag | (confidence < confidence_review_threshold)
    review_reason = np.where(
        ood_flag,
        "ood",
        np.where(
            ~support_sufficient,
            "low_support",
            np.where(confidence < confidence_review_threshold, "low_confidence", "none"),
        ),
    )

    calibrated_frame = working.copy()
    calibrated_frame["prediction_raw"] = raw_scores
    calibrated_frame["prediction_calibrated"] = calibrated
    calibrated_frame["prediction_uncertainty"] = prediction_uncertainty
    calibrated_frame["confidence_score"] = confidence
    calibrated_frame["confidence_band"] = pd.cut(
        confidence,
        bins=[-0.01, 0.55, 0.75, 1.01],
        labels=["low", "medium", "high"],
    ).astype(object)
    calibrated_frame["support_sufficient_flag"] = support_sufficient
    calibrated_frame["ood_flag"] = ood_flag
    calibrated_frame["review_flag"] = review_flag
    calibrated_frame["review_reason"] = review_reason
    calibrated_frame["calibration_method"] = method

    labeled_mask_series = pd.Series(labeled_mask, index=working.index)
    summary = {
        "model_name": str(model_name),
        "calibration_method": method,
        "n_rows": int(len(working)),
        "n_labeled": int(labeled_mask.sum()),
        "n_review": int(review_flag.sum()),
        "mean_confidence": float(np.nanmean(confidence)) if len(confidence) else float("nan"),
        "ood_rate": float(np.mean(ood_flag)) if len(ood_flag) else float("nan"),
        "abstain_rate": float(np.mean(review_flag)) if len(review_flag) else float("nan"),
    }
    if labeled_mask.any():
        summary.update(
            {
                "raw_brier_score": float(brier_score(y_true[labeled_mask], raw_scores[labeled_mask])),
                "calibrated_brier_score": float(brier_score(y_true[labeled_mask], calibrated[labeled_mask])),
                "raw_expected_calibration_error": float(
                    expected_calibration_error(y_true[labeled_mask], raw_scores[labeled_mask])
                ),
                "calibrated_expected_calibration_error": float(
                    expected_calibration_error(y_true[labeled_mask], calibrated[labeled_mask])
                ),
                "raw_max_calibration_error": float(
                    max_calibration_error(y_true[labeled_mask], raw_scores[labeled_mask])
                ),
                "calibrated_max_calibration_error": float(
                    max_calibration_error(y_true[labeled_mask], calibrated[labeled_mask])
                ),
                "raw_log_loss": float(log_loss(y_true[labeled_mask], raw_scores[labeled_mask])),
                "calibrated_log_loss": float(log_loss(y_true[labeled_mask], calibrated[labeled_mask])),
                "calibration_curve": calibration_curve_data(
                    y_true[labeled_mask], calibrated[labeled_mask]
                ),
            }
        )
    provenance = {
        "model_name": str(model_name),
        "calibration_method": method,
        "support_threshold": support_threshold,
        "ood_knownness_threshold": ood_knownness_threshold,
        "confidence_review_threshold": confidence_review_threshold,
    }
    return BranchCalibrationResult(
        model_name=str(model_name),
        calibration_method=method,
        predictions=calibrated_frame,
        summary=summary,
        provenance=provenance,
    )


def build_branch_calibration_summary(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    fit_config: Mapping[str, Any] | BranchFitConfig | BranchConfig | None = None,
    label_column: str = "spread_label",
) -> pd.DataFrame:
    """Summarize calibration outcomes for a branch model surface."""
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        if getattr(result, "predictions", pd.DataFrame()).empty:
            rows.append({"model_name": str(model_name), "calibration_method": "none", "n_rows": 0})
            continue
        calibration = calibrate_branch_predictions(
            result.predictions,
            model_name=model_name,
            fit_config=fit_config,
            scored=scored,
            label_column=label_column,
        )
        row = {"model_name": str(model_name)}
        row.update(calibration.summary)
        rows.append(row)
    return pd.DataFrame(rows)


def build_branch_calibrated_prediction_table(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    fit_config: Mapping[str, Any] | BranchFitConfig | BranchConfig | None = None,
    label_column: str = "spread_label",
) -> pd.DataFrame:
    """Combine calibrated predictions into one branch table."""
    frames: list[pd.DataFrame] = []
    for model_name, result in results.items():
        predictions = getattr(result, "predictions", pd.DataFrame())
        if predictions.empty:
            continue
        calibration = calibrate_branch_predictions(
            predictions,
            model_name=model_name,
            fit_config=fit_config,
            scored=scored,
            label_column=label_column,
        )
        frame = calibration.predictions.copy()
        frame["model_name"] = str(model_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
