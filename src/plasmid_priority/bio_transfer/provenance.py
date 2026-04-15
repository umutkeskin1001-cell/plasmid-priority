"""Deterministic provenance helpers for the bio transfer branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.bio_transfer.specs import BioTransferConfig, load_bio_transfer_config
from plasmid_priority.shared.provenance import build_branch_run_provenance, content_hash, dataframe_content_hash, stable_json_dumps


def build_bio_transfer_run_provenance(
    scored: pd.DataFrame,
    *,
    model_names: Sequence[str],
    config: Mapping[str, Any] | BioTransferConfig | None = None,
    script_name: str = "run_bio_transfer_branch",
    source_paths: Sequence[str | Path] | None = None,
    recommended_primary_model_name: str | None = None,
    calibration_summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    bio_config = load_bio_transfer_config(config)
    feature_surface = {name: list(bio_config.feature_sets.get(name, ())) for name in model_names}
    return build_branch_run_provenance(
        scored,
        branch_name="bio_transfer",
        benchmark_name=bio_config.benchmark.name,
        split_year=bio_config.benchmark.split_year,
        primary_model_name=bio_config.primary_model_name,
        model_names=model_names,
        config_payload=bio_config.model_dump(mode="json"),
        feature_surface=feature_surface,
        script_name=script_name,
        source_paths=source_paths,
        recommended_primary_model_name=recommended_primary_model_name,
        calibration_summary=calibration_summary,
        predictions=predictions,
        calibrated_predictions=calibrated_predictions,
        label_column="bio_transfer_label",
    )
