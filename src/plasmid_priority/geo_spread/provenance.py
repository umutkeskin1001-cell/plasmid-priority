"""Deterministic provenance helpers for the geo spread branch.

Re-exports provenance primitives from ``plasmid_priority.shared.provenance``
instead of duplicating them locally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.geo_spread.select import (
    GEO_SPREAD_ADAPTIVE_HIGH_KNOWNNESS_MODEL,
    GEO_SPREAD_ADAPTIVE_KNOWNNESS_CENTER,
    GEO_SPREAD_ADAPTIVE_KNOWNNESS_SHARPNESS,
    GEO_SPREAD_ADAPTIVE_PRIORITY,
    GEO_SPREAD_ADAPTIVE_SUPPORT_MODEL,
    GEO_SPREAD_BLEND_COMPONENTS,
    GEO_SPREAD_META_PRIORITY,
    GEO_SPREAD_RELIABILITY_BLEND,
)
from plasmid_priority.geo_spread.specs import GeoSpreadConfig, load_geo_spread_config
from plasmid_priority.shared.provenance import (  # noqa: F401 – re-exported for backward compatibility
    content_hash,
    dataframe_content_hash,
    stable_json_dumps,
)
from plasmid_priority.utils.files import path_signature_with_hash


@dataclass(slots=True)
class GeoSpreadRunProvenance:
    """Canonical run signature for a geo spread branch execution."""

    script_name: str
    benchmark_name: str
    split_year: int
    model_names: tuple[str, ...]
    primary_model_name: str
    config_hash: str
    input_hash: str
    feature_surface_hash: str
    n_rows: int
    n_positive: int
    recommended_primary_model_name: str | None = None
    calibration_summary_hash: str | None = None
    predictions_hash: str | None = None
    calibration_table_hash: str | None = None
    source_signatures: list[dict[str, Any]] | None = None
    input_signature: dict[str, Any] | None = None
    config_signature: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "script_name": self.script_name,
            "benchmark_name": self.benchmark_name,
            "split_year": self.split_year,
            "model_names": list(self.model_names),
            "primary_model_name": self.primary_model_name,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "feature_surface_hash": self.feature_surface_hash,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
        }
        if self.recommended_primary_model_name:
            payload["recommended_primary_model_name"] = self.recommended_primary_model_name
        if self.calibration_summary_hash is not None:
            payload["calibration_summary_hash"] = self.calibration_summary_hash
        if self.predictions_hash is not None:
            payload["predictions_hash"] = self.predictions_hash
        if self.calibration_table_hash is not None:
            payload["calibration_table_hash"] = self.calibration_table_hash
        if self.source_signatures is not None:
            payload["source_signatures"] = self.source_signatures
        if self.input_signature is not None:
            payload["input_signature"] = self.input_signature
        if self.config_signature is not None:
            payload["config_signature"] = self.config_signature
        payload["run_signature"] = content_hash(payload)
        return payload


def build_geo_spread_run_provenance(
    scored: pd.DataFrame,
    *,
    model_names: Sequence[str],
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
    script_name: str = "run_geo_spread_branch",
    source_paths: Sequence[str | Path] | None = None,
    recommended_primary_model_name: str | None = None,
    calibration_summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance record for a geo spread run."""
    geo_config = load_geo_spread_config(config)
    feature_surface: dict[str, Any] = {}
    for model_name in model_names:
        normalized_name = str(model_name)
        if normalized_name == GEO_SPREAD_RELIABILITY_BLEND:
            feature_surface[normalized_name] = {
                "derived_from": [
                    {"model_name": str(component_name), "weight": float(weight)}
                    for component_name, weight in GEO_SPREAD_BLEND_COMPONENTS
                ]
            }
            continue
        if normalized_name == GEO_SPREAD_ADAPTIVE_PRIORITY:
            feature_surface[normalized_name] = {
                "support_model_name": GEO_SPREAD_ADAPTIVE_SUPPORT_MODEL,
                "high_knownness_model_name": GEO_SPREAD_ADAPTIVE_HIGH_KNOWNNESS_MODEL,
                "knownness_center": float(GEO_SPREAD_ADAPTIVE_KNOWNNESS_CENTER),
                "knownness_sharpness": float(GEO_SPREAD_ADAPTIVE_KNOWNNESS_SHARPNESS),
            }
            continue
        if normalized_name == GEO_SPREAD_META_PRIORITY:
            feature_surface[normalized_name] = {
                "meta_components": [
                    GEO_SPREAD_ADAPTIVE_PRIORITY,
                    GEO_SPREAD_RELIABILITY_BLEND,
                    "geo_context_hybrid_priority",
                    "geo_support_light_priority",
                    "geo_phylo_ecology_priority",
                ],
                "knownness_aware": True,
            }
            continue
        feature_surface[normalized_name] = list(geo_config.feature_sets.get(normalized_name, ()))
    config_payload = geo_config.model_dump(mode="json")
    config_hash = content_hash(config_payload)
    input_columns = [
        column
        for column in (
            "backbone_id",
            "spread_label",
            "n_new_countries",
            "split_year",
            "backbone_assignment_mode",
            "max_resolved_year_train",
            "min_resolved_year_test",
            "training_only_future_unseen_backbone_flag",
        )
        if column in scored.columns
    ]
    input_hash = dataframe_content_hash(scored, columns=input_columns, sort_by="backbone_id")
    feature_surface_hash = content_hash(feature_surface)
    source_signatures = None
    if source_paths:
        source_signatures = [
            path_signature_with_hash(Path(path), include_file_hash=True) for path in source_paths
        ]
    input_signature = {
        "row_count": int(len(scored)),
        "column_count": int(len(scored.columns)),
        "columns": list(scored.columns),
        "hash": input_hash,
    }
    config_signature = {
        "name": geo_config.benchmark.name,
        "split_year": geo_config.benchmark.split_year,
        "selection": geo_config.selection.model_dump(mode="json"),
        "hash": config_hash,
    }
    provenance = GeoSpreadRunProvenance(
        script_name=script_name,
        benchmark_name=geo_config.benchmark.name,
        split_year=geo_config.benchmark.split_year,
        model_names=tuple(str(name) for name in model_names),
        primary_model_name=geo_config.primary_model_name,
        recommended_primary_model_name=(
            str(recommended_primary_model_name).strip() if recommended_primary_model_name else None
        ),
        config_hash=config_hash,
        input_hash=input_hash,
        feature_surface_hash=feature_surface_hash,
        n_rows=int(len(scored)),
        n_positive=int(
            pd.to_numeric(scored.get("spread_label"), errors="coerce").fillna(0).astype(int).sum()
        )
        if "spread_label" in scored.columns
        else 0,
        calibration_summary_hash=dataframe_content_hash(calibration_summary, sort_by="model_name")
        if calibration_summary is not None and not calibration_summary.empty
        else None,
        predictions_hash=dataframe_content_hash(predictions, sort_by=["model_name", "backbone_id"])
        if predictions is not None and not predictions.empty
        else None,
        calibration_table_hash=dataframe_content_hash(
            calibrated_predictions, sort_by=["model_name", "backbone_id"]
        )
        if calibrated_predictions is not None and not calibrated_predictions.empty
        else None,
        source_signatures=source_signatures,
        input_signature=input_signature,
        config_signature=config_signature,
    )
    return provenance.to_dict()
