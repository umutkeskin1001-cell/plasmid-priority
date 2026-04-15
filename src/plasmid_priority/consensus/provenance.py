"""Deterministic provenance helpers for the consensus branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.consensus.specs import ConsensusConfig, load_consensus_config
from plasmid_priority.shared.provenance import build_branch_run_provenance


def build_consensus_run_provenance(
    scored: pd.DataFrame,
    *,
    model_names: Sequence[str],
    config: Mapping[str, Any] | ConsensusConfig | None = None,
    script_name: str = "run_consensus_branch",
    source_paths: Sequence[str | Path] | None = None,
    recommended_primary_model_name: str | None = None,
    calibration_summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    consensus_config = load_consensus_config(config)
    feature_surface = {
        name: list(consensus_config.feature_sets.get(name, ())) for name in model_names
    }
    return build_branch_run_provenance(
        scored,
        branch_name="consensus",
        benchmark_name=consensus_config.benchmark.name,
        split_year=consensus_config.benchmark.split_year,
        primary_model_name=consensus_config.primary_model_name,
        model_names=model_names,
        config_payload=consensus_config.model_dump(mode="json"),
        feature_surface=feature_surface,
        script_name=script_name,
        source_paths=source_paths,
        recommended_primary_model_name=recommended_primary_model_name,
        calibration_summary=calibration_summary,
        predictions=predictions,
        calibrated_predictions=calibrated_predictions,
        label_column="spread_label",
    )
