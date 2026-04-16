"""Calibration, uncertainty, and abstention helpers for the geo spread branch.

Re-exports shared calibration primitives from ``plasmid_priority.shared.calibration``
instead of duplicating them locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from plasmid_priority.geo_spread.dataset import prepare_geo_spread_scored_table
from plasmid_priority.geo_spread.select import (
    GEO_SPREAD_ADAPTIVE_PRIORITY,
    GEO_SPREAD_META_PRIORITY,
    GEO_SPREAD_RELIABILITY_BLEND,
)
from plasmid_priority.geo_spread.specs import GeoSpreadConfig, load_geo_spread_config
from plasmid_priority.shared.calibration import (  # noqa: F401 – re-exported for backward compatibility
    _fit_isotonic_calibrator,
    _fit_platt_calibrator,
    _float_option,
    _identity_calibrator,
    _normalize_method,
    _prediction_frame,
    _safe_probability,
)
from plasmid_priority.validation.metrics import (
    brier_score,
    calibration_curve_data,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
)

_DERIVED_MODEL_CALIBRATION: dict[str, str] = {
    GEO_SPREAD_RELIABILITY_BLEND: "isotonic",
    GEO_SPREAD_ADAPTIVE_PRIORITY: "isotonic",
    GEO_SPREAD_META_PRIORITY: "isotonic",
}


def _prepare_knownness_surface(scored: pd.DataFrame | None) -> pd.DataFrame | None:
    if scored is None:
        return None
    if "knownness_score" in scored.columns:
        return scored
    return prepare_geo_spread_scored_table(scored)


def _confidence_band(confidence: np.ndarray, *, index: pd.Index | None = None) -> pd.Series:
    labels = np.full(len(confidence), "low", dtype=object)
    labels[confidence >= 0.75] = "high"
    labels[(confidence >= 0.55) & (confidence < 0.75)] = "medium"
    return pd.Series(labels, index=index, dtype=object)


def _review_reason(
    *,
    support_sufficient: np.ndarray,
    ood_flag: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
) -> pd.Series:
    reasons = np.full(len(confidence), "none", dtype=object)
    reasons[ood_flag] = "ood"
    reasons[(~ood_flag) & (~support_sufficient)] = "low_support"
    reasons[(~ood_flag) & support_sufficient & (confidence < float(confidence_threshold))] = (
        "low_confidence"
    )
    return pd.Series(reasons, dtype=object)


@dataclass(slots=True)
class GeoSpreadCalibrationResult:
    """Calibrated geo spread prediction surface for one model."""

    model_name: str
    calibration_method: str
    predictions: pd.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]


def calibrate_geo_spread_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    fit_config: Mapping[str, Any] | GeoSpreadConfig | Any | None = None,
    scored: pd.DataFrame | None = None,
    label_column: str = "spread_label",
    score_column: str = "oof_prediction",
    knownness_column: str = "knownness_score",
    backbone_id_column: str = "backbone_id",
) -> GeoSpreadCalibrationResult:
    """Calibrate one model's geo spread predictions and derive uncertainty flags."""
    if predictions.empty:
        empty = pd.DataFrame(
            columns=[
                backbone_id_column,
                score_column,
                "calibrated_prediction",
                "prediction_uncertainty",
                "confidence",
                "confidence_band",
                "support_sufficiency_flag",
                "ood_flag",
                "abstain_or_review_flag",
                "review_reason",
            ]
        )
        return GeoSpreadCalibrationResult(
            model_name=str(model_name),
            calibration_method="none",
            predictions=empty,
            summary={
                "model_name": str(model_name),
                "calibration_method": "none",
                "n_rows": 0,
                "n_labeled": 0,
                "n_abstain": 0,
            },
            provenance={},
        )

    working = predictions.copy()
    prepared_scored = _prepare_knownness_surface(scored)
    if (
        prepared_scored is not None
        and knownness_column not in working.columns
        and backbone_id_column in working.columns
    ):
        knownness_frame = prepared_scored.loc[
            :, [c for c in (backbone_id_column, knownness_column) if c in prepared_scored.columns]
        ].copy()
        if not knownness_frame.empty and backbone_id_column in knownness_frame.columns:
            working = working.merge(knownness_frame, on=backbone_id_column, how="left")
    if label_column not in working.columns:
        if (
            prepared_scored is None
            or label_column not in prepared_scored.columns
            or backbone_id_column not in working.columns
        ):
            raise KeyError(f"Geo spread predictions require `{label_column}` for calibration.")
        working = working.merge(
            prepared_scored.loc[:, [backbone_id_column, label_column]],
            on=backbone_id_column,
            how="left",
        )
    working = working.reset_index(drop=True)

    if isinstance(fit_config, GeoSpreadConfig):
        default_fit = fit_config.fit_config.get(str(model_name))
        fit_payload = dict(default_fit.model_dump(mode="python")) if default_fit else {}
    elif fit_config is not None and hasattr(fit_config, "model_dump"):
        fit_payload = dict(fit_config.model_dump(mode="python"))
    else:
        fit_payload = dict(fit_config or {})
    if "calibration" not in fit_payload and str(model_name) in _DERIVED_MODEL_CALIBRATION:
        fit_payload["calibration"] = _DERIVED_MODEL_CALIBRATION[str(model_name)]
    method = _normalize_method(fit_payload.get("calibration", "none"))
    support_threshold = float(
        np.clip(_float_option(fit_payload, "support_knownness_threshold", 0.25), 0.0, 1.0)
    )
    ood_knownness_threshold = float(
        np.clip(_float_option(fit_payload, "ood_knownness_threshold", 0.10), 0.0, 1.0)
    )
    confidence_review_threshold = float(
        np.clip(_float_option(fit_payload, "confidence_review_threshold", 0.45), 0.0, 1.0)
    )

    y_true = (
        pd.to_numeric(working[label_column], errors="coerce")
        .fillna(-1)
        .astype(int)
        .to_numpy(dtype=int)
    )
    raw_scores = (
        pd.to_numeric(working[score_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    labeled_mask = y_true >= 0

    if method == "platt":
        calibrator = _fit_platt_calibrator(y_true[labeled_mask], raw_scores[labeled_mask])
    elif method == "isotonic":
        calibrator = _fit_isotonic_calibrator(y_true[labeled_mask], raw_scores[labeled_mask])
    else:
        calibrator = _identity_calibrator

    calibrated = calibrator(raw_scores)

    knownness_values = (
        pd.to_numeric(working[knownness_column], errors="coerce")
        if knownness_column in working.columns
        else pd.Series(np.nan, index=working.index, dtype=float)
    )
    knownness_numeric = knownness_values.fillna(np.nan).to_numpy(dtype=float)
    support_sufficient = np.where(
        np.isnan(knownness_numeric),
        True,
        knownness_numeric >= support_threshold,
    )

    labeled_scores = raw_scores[labeled_mask]
    finite_labeled_scores = labeled_scores[np.isfinite(labeled_scores)]
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
    support_uncertainty = np.where(
        np.isnan(knownness_numeric), 0.35, 1.0 - np.clip(knownness_numeric, 0.0, 1.0)
    )
    prediction_uncertainty = np.clip(
        0.55 * score_uncertainty + 0.30 * calibration_gap + 0.15 * support_uncertainty,
        0.0,
        1.0,
    )
    confidence = np.clip(1.0 - prediction_uncertainty, 0.0, 1.0)
    confidence_band = _confidence_band(confidence, index=working.index)
    review_reason = _review_reason(
        support_sufficient=support_sufficient,
        ood_flag=ood_flag,
        confidence=confidence,
        confidence_threshold=confidence_review_threshold,
    ).set_axis(working.index)
    abstain_or_review_flag = (
        (confidence < confidence_review_threshold) | (~support_sufficient) | ood_flag
    )

    calibrated_frame = working.loc[
        :,
        [
            c
            for c in working.columns
            if c in {backbone_id_column, score_column, label_column, knownness_column}
        ],
    ].copy()
    calibrated_frame["calibrated_prediction"] = calibrated
    calibrated_frame["prediction_uncertainty"] = prediction_uncertainty
    calibrated_frame["confidence"] = confidence
    calibrated_frame["confidence_band"] = confidence_band
    calibrated_frame["support_sufficiency_flag"] = support_sufficient
    calibrated_frame["ood_flag"] = ood_flag
    calibrated_frame["abstain_or_review_flag"] = abstain_or_review_flag
    calibrated_frame["review_reason"] = review_reason
    calibrated_frame["calibration_method"] = method
    calibrated_frame["model_name"] = str(model_name)
    calibrated_frame["priority_score"] = calibrated * (1.0 - 0.5 * prediction_uncertainty)
    calibrated_frame["raw_score_support_lower"] = lower_support
    calibrated_frame["raw_score_support_upper"] = upper_support

    summary = {
        "model_name": str(model_name),
        "calibration_method": method,
        "support_knownness_threshold": support_threshold,
        "confidence_review_threshold": confidence_review_threshold,
        "n_rows": int(len(working)),
        "n_labeled": int(labeled_mask.sum()),
        "n_abstain": int(np.sum(abstain_or_review_flag)),
        "abstain_rate": float(np.mean(abstain_or_review_flag)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_prediction_uncertainty": float(np.mean(prediction_uncertainty)),
        "support_sufficiency_rate": float(np.mean(support_sufficient.astype(float))),
        "ood_rate": float(np.mean(ood_flag.astype(float))),
        "raw_brier_score": float(brier_score(y_true[labeled_mask], raw_scores[labeled_mask]))
        if labeled_mask.any()
        else float("nan"),
        "calibrated_brier_score": float(brier_score(y_true[labeled_mask], calibrated[labeled_mask]))
        if labeled_mask.any()
        else float("nan"),
        "raw_log_loss": float(log_loss(y_true[labeled_mask], raw_scores[labeled_mask]))
        if labeled_mask.any()
        else float("nan"),
        "calibrated_log_loss": float(log_loss(y_true[labeled_mask], calibrated[labeled_mask]))
        if labeled_mask.any()
        else float("nan"),
        "raw_expected_calibration_error": float(
            expected_calibration_error(y_true[labeled_mask], raw_scores[labeled_mask])
        )
        if labeled_mask.any()
        else float("nan"),
        "calibrated_expected_calibration_error": float(
            expected_calibration_error(y_true[labeled_mask], calibrated[labeled_mask])
        )
        if labeled_mask.any()
        else float("nan"),
        "raw_max_calibration_error": float(
            max_calibration_error(y_true[labeled_mask], raw_scores[labeled_mask])
        )
        if labeled_mask.any()
        else float("nan"),
        "calibrated_max_calibration_error": float(
            max_calibration_error(y_true[labeled_mask], calibrated[labeled_mask])
        )
        if labeled_mask.any()
        else float("nan"),
    }
    summary["calibration_curve"] = (
        calibration_curve_data(
            y_true[labeled_mask],
            calibrated[labeled_mask] if labeled_mask.any() else np.array([], dtype=float),
        )
        if labeled_mask.any()
        else []
    )

    provenance = {
        "model_name": str(model_name),
        "calibration_method": method,
        "score_column": score_column,
        "label_column": label_column,
        "knownness_column": knownness_column,
        "support_bounds": {
            "lower": lower_support,
            "upper": upper_support,
        },
    }
    return GeoSpreadCalibrationResult(
        model_name=str(model_name),
        calibration_method=method,
        predictions=calibrated_frame,
        summary=summary,
        provenance=provenance,
    )


def build_geo_spread_calibration_summary(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
) -> pd.DataFrame:
    """Summarize calibrated geo spread predictions for one branch run."""
    rows: list[dict[str, Any]] = []
    geo_config = load_geo_spread_config(config)
    for model_name, predictions in results.items():
        predictions_frame = _prediction_frame(predictions)
        fit_config = geo_config.fit_config.get(str(model_name))
        calibration = calibrate_geo_spread_predictions(
            predictions_frame,
            model_name=str(model_name),
            fit_config=fit_config,
            scored=scored,
        )
        row = {"model_name": str(model_name)}
        row.update(calibration.summary)
        rows.append(row)
    return pd.DataFrame(rows)


def build_geo_spread_calibrated_prediction_table(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
) -> pd.DataFrame:
    """Combine calibrated predictions for every geo spread model into one table."""
    frames: list[pd.DataFrame] = []
    geo_config = load_geo_spread_config(config)
    for model_name, predictions in results.items():
        predictions_frame = _prediction_frame(predictions)
        fit_config = geo_config.fit_config.get(str(model_name))
        calibration = calibrate_geo_spread_predictions(
            predictions_frame,
            model_name=str(model_name),
            fit_config=fit_config,
            scored=scored,
        )
        frame = calibration.predictions.copy()
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
