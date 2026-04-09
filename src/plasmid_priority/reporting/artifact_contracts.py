"""Lightweight schema checks for report artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

PROVENANCE_REQUIRED_FIELDS = (
    "artifact_family",
    "generated_at",
    "protocol_id",
    "protocol_hash",
    "code_hash",
    "input_data_hash",
)


@dataclass(frozen=True)
class ReportArtifactContract:
    required_columns: tuple[str, ...] = ()
    unique_key: str | None = None
    probability_columns: tuple[str, ...] = ()


def validate_required_columns(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    required_columns: Iterable[str],
) -> None:
    required = {str(column) for column in required_columns}
    missing = sorted(required.difference(frame.columns.astype(str)))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{artifact_name} missing required columns: {joined}")


def validate_unique_key(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    key: str,
) -> None:
    if frame.empty or key not in frame.columns:
        return
    duplicated = frame[key].astype(str).duplicated(keep=False)
    if duplicated.any():
        values = ", ".join(sorted(frame.loc[duplicated, key].astype(str).unique().tolist()))
        raise ValueError(f"{artifact_name} has duplicate `{key}` values: {values}")


def validate_probability_columns(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    columns: Iterable[str],
) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (~values.between(0.0, 1.0, inclusive="both"))
        if invalid.any():
            raise ValueError(
                f"{artifact_name} column `{column}` contains values outside [0, 1]"
            )


def validate_report_artifact(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    required_columns: Iterable[str] = (),
    unique_key: str | None = None,
    probability_columns: Iterable[str] = (),
) -> None:
    validate_required_columns(
        frame,
        artifact_name=artifact_name,
        required_columns=required_columns,
    )
    if unique_key is not None:
        validate_unique_key(frame, artifact_name=artifact_name, key=unique_key)
    validate_probability_columns(
        frame,
        artifact_name=artifact_name,
        columns=probability_columns,
    )


def validate_report_artifact_contract(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    contract: ReportArtifactContract,
) -> None:
    validate_report_artifact(
        frame,
        artifact_name=artifact_name,
        required_columns=contract.required_columns,
        unique_key=contract.unique_key,
        probability_columns=contract.probability_columns,
    )


def validate_provenance_record(record: object, *, artifact_name: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{artifact_name} must be a JSON object")
    missing = [field for field in PROVENANCE_REQUIRED_FIELDS if field not in record]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{artifact_name} missing required provenance fields: {joined}")

    for field in PROVENANCE_REQUIRED_FIELDS:
        value = record.get(field)
        if not str(value).strip():
            raise ValueError(f"{artifact_name} field `{field}` cannot be empty")

    generated_at = str(record.get("generated_at", "")).strip()
    try:
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{artifact_name} field `generated_at` must be ISO-8601") from exc
