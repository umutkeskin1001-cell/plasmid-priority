"""Deterministic provenance helpers for branch runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.utils.files import path_signature_with_hash


def _stable_json_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_json_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_json_payload(item) for item in value]
    if isinstance(value, set):
        return [_stable_json_payload(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return _stable_json_payload(value.to_list())
    if isinstance(value, pd.Index):
        return _stable_json_payload(value.to_list())
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def stable_json_dumps(value: Any) -> str:
    """Dump a Python object to canonical JSON for hashing."""
    return json.dumps(_stable_json_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(payload: Any) -> str:
    """Return a SHA-256 hash for any JSON-serializable payload."""
    digest = hashlib.sha256()
    digest.update(stable_json_dumps(payload).encode("utf-8"))
    return digest.hexdigest()


def dataframe_content_hash(
    frame: pd.DataFrame | None,
    *,
    columns: Sequence[str] | None = None,
    sort_by: Sequence[str] | str | None = None,
) -> str:
    """Return a deterministic hash of a dataframe's content and schema."""
    if frame is None:
        return content_hash(None)
    working = frame.copy()
    if columns is not None:
        selected = [str(column) for column in columns if str(column) in working.columns]
        working = working.loc[:, selected]
    if sort_by is not None and not working.empty:
        sort_columns = [str(sort_by)] if isinstance(sort_by, str) else [str(column) for column in sort_by]
        present = [column for column in sort_columns if column in working.columns]
        if present:
            working = working.sort_values(present, kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("\0".join(map(str, working.columns.tolist())).encode("utf-8"))
    digest.update("\0".join(map(str, working.dtypes.astype(str).tolist())).encode("utf-8"))
    if working.empty:
        digest.update(b"<empty>")
        return digest.hexdigest()
    normalized = working.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: stable_json_dumps(value) if not isinstance(value, (str, int, bool)) else value
        )
    series_hash = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64", copy=False)
    digest.update(series_hash.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class BranchRunProvenance:
    """Canonical run signature for a branch execution."""

    branch_name: str
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
    calibrated_predictions_hash: str | None = None
    source_signatures: list[dict[str, Any]] | None = None
    input_signature: dict[str, Any] | None = None
    config_signature: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "branch_name": self.branch_name,
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
        if self.calibrated_predictions_hash is not None:
            payload["calibrated_predictions_hash"] = self.calibrated_predictions_hash
        if self.source_signatures is not None:
            payload["source_signatures"] = self.source_signatures
        if self.input_signature is not None:
            payload["input_signature"] = self.input_signature
        if self.config_signature is not None:
            payload["config_signature"] = self.config_signature
        payload["run_signature"] = content_hash(payload)
        return payload


def build_branch_run_provenance(
    scored: pd.DataFrame,
    *,
    branch_name: str,
    benchmark_name: str,
    split_year: int,
    primary_model_name: str,
    model_names: Sequence[str],
    config_payload: Mapping[str, Any],
    feature_surface: Mapping[str, Any],
    script_name: str,
    source_paths: Sequence[str | Path] | None = None,
    recommended_primary_model_name: str | None = None,
    calibration_summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
    label_column: str = "spread_label",
) -> dict[str, Any]:
    """Build a deterministic provenance record for a branch run."""
    config_hash = content_hash(config_payload)
    input_columns = [
        column
        for column in (
            "backbone_id",
            label_column,
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
            path_signature_with_hash(Path(path), include_file_hash=True)
            for path in source_paths
        ]
    input_signature = {
        "row_count": int(len(scored)),
        "column_count": int(len(scored.columns)),
        "columns": list(scored.columns),
        "hash": input_hash,
    }
    config_signature = {
        "name": benchmark_name,
        "split_year": int(split_year),
        "hash": config_hash,
    }
    provenance = BranchRunProvenance(
        branch_name=str(branch_name),
        script_name=script_name,
        benchmark_name=benchmark_name,
        split_year=int(split_year),
        model_names=tuple(str(name) for name in model_names),
        primary_model_name=primary_model_name,
        recommended_primary_model_name=(
            str(recommended_primary_model_name).strip() if recommended_primary_model_name else None
        ),
        config_hash=config_hash,
        input_hash=input_hash,
        feature_surface_hash=feature_surface_hash,
        n_rows=int(len(scored)),
        n_positive=int(pd.to_numeric(scored.get(label_column), errors="coerce").fillna(0).astype(int).sum())
        if label_column in scored.columns
        else 0,
        calibration_summary_hash=dataframe_content_hash(calibration_summary, sort_by="model_name")
        if calibration_summary is not None and not calibration_summary.empty
        else None,
        predictions_hash=dataframe_content_hash(predictions, sort_by=["model_name", "backbone_id"])
        if predictions is not None and not predictions.empty
        else None,
        calibrated_predictions_hash=dataframe_content_hash(
            calibrated_predictions, sort_by=["model_name", "backbone_id"]
        )
        if calibrated_predictions is not None and not calibrated_predictions.empty
        else None,
        source_signatures=source_signatures,
        input_signature=input_signature,
        config_signature=config_signature,
    )
    return provenance.to_dict()
