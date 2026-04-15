"""Deterministic provenance helpers for the clinical hazard branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.clinical_hazard.specs import ClinicalHazardConfig, load_clinical_hazard_config
from plasmid_priority.shared.provenance import build_branch_run_provenance


def build_clinical_hazard_run_provenance(
    scored: pd.DataFrame,
    *,
    model_names: Sequence[str],
    config: Mapping[str, Any] | ClinicalHazardConfig | None = None,
    script_name: str = "run_clinical_hazard_branch",
    source_paths: Sequence[str | Path] | None = None,
    recommended_primary_model_name: str | None = None,
    calibration_summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    clinical_config = load_clinical_hazard_config(config)
    feature_surface = {
        name: list(clinical_config.feature_sets.get(name, ())) for name in model_names
    }
    return build_branch_run_provenance(
        scored,
        branch_name="clinical_hazard",
        benchmark_name=clinical_config.benchmark.name,
        split_year=clinical_config.benchmark.split_year,
        primary_model_name=clinical_config.primary_model_name,
        model_names=model_names,
        config_payload=clinical_config.model_dump(mode="json"),
        feature_surface=feature_surface,
        script_name=script_name,
        source_paths=source_paths,
        recommended_primary_model_name=recommended_primary_model_name,
        calibration_summary=calibration_summary,
        predictions=predictions,
        calibrated_predictions=calibrated_predictions,
        label_column="clinical_hazard_label",
    )
