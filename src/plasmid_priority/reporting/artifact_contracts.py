"""Lightweight schema checks for report artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


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
