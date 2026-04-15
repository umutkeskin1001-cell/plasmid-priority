"""Calibration, uncertainty, and abstention helpers for the clinical hazard branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from plasmid_priority.clinical_hazard.specs import ClinicalHazardConfig
from plasmid_priority.shared.calibration import (
    BranchCalibrationResult,
    build_branch_calibrated_prediction_table,
    build_branch_calibration_summary,
    calibrate_branch_predictions,
)

ClinicalHazardCalibrationResult = BranchCalibrationResult


def calibrate_clinical_hazard_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    fit_config: Mapping[str, Any] | ClinicalHazardConfig | Any | None = None,
    scored: pd.DataFrame | None = None,
    label_column: str = "clinical_hazard_label",
    score_column: str = "oof_prediction",
    knownness_column: str = "knownness_score",
    backbone_id_column: str = "backbone_id",
) -> BranchCalibrationResult:
    return calibrate_branch_predictions(
        predictions,
        model_name=model_name,
        fit_config=fit_config,
        scored=scored,
        label_column=label_column,
        score_column=score_column,
        knownness_column=knownness_column,
        entity_id_column=backbone_id_column,
    )


def build_clinical_hazard_calibration_summary(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    config: Mapping[str, Any] | ClinicalHazardConfig | None = None,
) -> pd.DataFrame:
    return build_branch_calibration_summary(
        results, scored=scored, fit_config=config, label_column="clinical_hazard_label"
    )


def build_clinical_hazard_calibrated_prediction_table(
    results: Mapping[str, Any],
    *,
    scored: pd.DataFrame | None = None,
    config: Mapping[str, Any] | ClinicalHazardConfig | None = None,
) -> pd.DataFrame:
    return build_branch_calibrated_prediction_table(
        results, scored=scored, fit_config=config, label_column="clinical_hazard_label"
    )
