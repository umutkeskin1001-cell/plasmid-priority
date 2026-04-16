"""Pipeline boundary validation helpers.

These helpers enforce two invariants at script boundaries:
- outputs must be present and fresh relative to the inputs recorded in the run summary
- tabular outputs that represent core pipeline surfaces must satisfy schema checks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from plasmid_priority.utils.files import path_signature_with_hash
from plasmid_priority.validation.schemas import (
    validate_backbone_table,
    validate_deduplicated_plasmids,
    validate_harmonized_plasmids,
    validate_scored_backbones,
)


def _validate_consensus_predictions(frame: pd.DataFrame, *, label: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": "fail",
            "table": label,
            "n_rows": 0,
            "errors": [{"message": "Consensus table is empty."}],
        }
    if "backbone_id" not in frame.columns:
        return {
            "status": "fail",
            "table": label,
            "n_rows": int(len(frame)),
            "errors": [{"message": "Consensus table is missing `backbone_id`."}],
        }
    score_column = None
    for candidate in (
        "consensus_score",
        "prediction_calibrated",
        "calibrated_prediction",
        "prediction",
        "oof_prediction",
    ):
        if candidate in frame.columns:
            score_column = candidate
            break
    if score_column is None:
        return {
            "status": "fail",
            "table": label,
            "n_rows": int(len(frame)),
            "errors": [
                {"message": "Consensus table is missing a usable score column."},
            ],
        }
    score = pd.to_numeric(frame[score_column], errors="coerce")
    if score.isna().all():
        return {
            "status": "fail",
            "table": label,
            "n_rows": int(len(frame)),
            "errors": [
                {"message": f"Consensus score column `{score_column}` contains no numeric data."},
            ],
        }
    invalid = frame.loc[score.notna() & ((score < 0.0) | (score > 1.0))]
    if not invalid.empty:
        return {
            "status": "fail",
            "table": label,
            "n_rows": int(len(frame)),
            "errors": [
                {
                    "message": (
                        f"Consensus score column `{score_column}` contains "
                        f"{len(invalid)} out-of-range values."
                    )
                }
            ],
        }
    optional_bounds = (
        "consensus_uncertainty",
        "consensus_score_lower",
        "consensus_score_upper",
        "consensus_attenuation",
        "branch_agreement_score",
        "branch_contribution_geo",
        "branch_contribution_bio_transfer",
        "branch_contribution_clinical_hazard",
    )
    for column in optional_bounds:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any() and ((numeric < 0.0) | (numeric > 1.0)).any():
            return {
                "status": "fail",
                "table": label,
                "n_rows": int(len(frame)),
                "errors": [
                    {
                        "message": (
                            f"Consensus table column `{column}` contains values outside [0, 1]."
                        )
                    }
                ],
            }
    return {
        "status": "pass",
        "table": label,
        "n_rows": int(len(frame)),
        "errors": [],
    }


TABULAR_VALIDATORS: dict[str, Callable[[pd.DataFrame, str], dict[str, Any]]] = {
    "harmonized_plasmids.tsv": lambda frame, label: validate_harmonized_plasmids(frame),
    "backbone_table.tsv": lambda frame, label: validate_backbone_table(frame),
    "backbone_scored.tsv": lambda frame, label: validate_scored_backbones(frame),
    "deduplicated_plasmids.tsv": lambda frame, label: validate_deduplicated_plasmids(frame),
    "consensus_predictions.tsv": _validate_consensus_predictions,
    "consensus_calibrated_predictions.tsv": _validate_consensus_predictions,
    "geo_spread_calibrated_predictions.tsv": _validate_consensus_predictions,
    "bio_transfer_calibrated_predictions.tsv": _validate_consensus_predictions,
    "clinical_hazard_calibrated_predictions.tsv": _validate_consensus_predictions,
}


def _normalize_output_path(value: str | Path, *, project_root: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if project_root is None:
        return path.resolve()
    return (project_root / path).resolve()


def _validate_tsv_artifact(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as exc:
        return {
            "status": "error",
            "table": path.name,
            "n_rows": 0,
            "errors": [{"message": f"Failed to parse TSV: {exc}"}],
        }
    validator = TABULAR_VALIDATORS.get(path.name)
    if validator is not None:
        return validator(frame, path.name)
    return {"status": "pass", "table": path.name, "n_rows": int(len(frame)), "errors": []}


def _validate_json_artifact(path: Path) -> dict[str, Any]:
    try:
        import json

        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception as exc:
        return {
            "status": "error",
            "table": path.name,
            "n_rows": 0,
            "errors": [{"message": f"Failed to parse JSON: {exc}"}],
        }
    return {"status": "pass", "table": path.name, "n_rows": 0, "errors": []}


def validate_output_artifact(
    path: str | Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a single output artifact by type and filename."""
    resolved = _normalize_output_path(path, project_root=project_root)
    if not resolved.exists():
        return {
            "status": "fail",
            "table": resolved.name,
            "n_rows": 0,
            "errors": [{"message": f"Missing output artifact: {resolved}"}],
        }
    if resolved.suffix.lower() == ".tsv":
        return _validate_tsv_artifact(resolved)
    if resolved.suffix.lower() == ".json":
        return _validate_json_artifact(resolved)
    try:
        path_signature_with_hash(resolved, include_file_hash=True)
    except Exception as exc:
        return {
            "status": "error",
            "table": resolved.name,
            "n_rows": 0,
            "errors": [{"message": f"Unable to inspect output artifact: {exc}"}],
        }
    return {"status": "pass", "table": resolved.name, "n_rows": 0, "errors": []}


def validate_script_boundary(
    summary: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate outputs and freshness from a ManagedScriptRun summary."""
    output_paths = summary.get("output_files_written", [])
    if not isinstance(output_paths, list):
        return {
            "status": "fail",
            "errors": [{"message": "Script summary is missing output path metadata."}],
            "artifacts": [],
        }
    input_manifest = summary.get("input_manifest", {})
    max_input_mtime_ns = 0
    if isinstance(input_manifest, dict):
        for entry in input_manifest.values():
            if not isinstance(entry, dict):
                continue
            mtime_ns = entry.get("mtime_ns")
            if isinstance(mtime_ns, int):
                max_input_mtime_ns = max(max_input_mtime_ns, mtime_ns)
    artifacts: list[dict[str, Any]] = []
    issues: list[str] = []
    for output_path in output_paths:
        artifact = validate_output_artifact(output_path, project_root=project_root)
        artifacts.append(artifact)
        if artifact.get("status") != "pass":
            issues.extend(err.get("message", str(err)) for err in artifact.get("errors", []))
        resolved = _normalize_output_path(output_path, project_root=project_root)
        if resolved.exists() and max_input_mtime_ns > 0:
            try:
                signature = path_signature_with_hash(resolved, include_file_hash=True)
            except Exception as exc:
                issues.append(f"Failed to inspect output freshness for {resolved}: {exc}")
                continue
            mtime_ns = signature.get("mtime_ns")
            if isinstance(mtime_ns, int) and mtime_ns < max_input_mtime_ns:
                issues.append(
                    (
                        f"Output {resolved} is older than its recorded inputs;"
                        " boundary freshness failed."
                    )
                )
    return {
        "status": "pass" if not issues else "fail",
        "errors": [{"message": message} for message in issues],
        "artifacts": artifacts,
    }
