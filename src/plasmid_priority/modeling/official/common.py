from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

DEFAULT_MIN_LABELED_ROWS = 12
DEFAULT_MIN_CLASS_COUNT = 4


@dataclass(frozen=True)
class OfficialSupervisedReadiness:
    status: str
    label_column: str | None
    requested_features: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    labeled_count: int
    positive_count: int
    negative_count: int
    min_labeled_rows: int = DEFAULT_MIN_LABELED_ROWS
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT

    @property
    def requested_feature_count(self) -> int:
        return len(self.requested_features)

    @property
    def available_feature_count(self) -> int:
        return len(self.available_features)

    @property
    def missing_feature_count(self) -> int:
        return len(self.missing_features)


def dedupe_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(column) for column in feature_columns if str(column)))


def numeric_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    error_label: str,
) -> pd.DataFrame:
    selected_features = dedupe_feature_columns(feature_columns)
    if not selected_features:
        raise ValueError("At least one feature column is required")
    missing = [column for column in selected_features if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {error_label} features: {missing}")
    matrix = frame.loc[:, list(selected_features)].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, matrix)


def valid_binary_labels(
    frame: pd.DataFrame,
    label_column: str,
    *,
    error_label: str,
) -> pd.Series:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid_labels = labels.dropna()
    unique_values = {float(value) for value in valid_labels.unique()}
    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(f"{error_label} expects a binary 0/1 label")
    valid_binary = valid_labels.astype(int)
    if int(valid_binary.nunique()) != 2:
        raise ValueError(
            f"{error_label} requires at least one positive and one negative label",
        )
    return valid_binary


def assess_supervised_readiness(
    frame: pd.DataFrame,
    *,
    label_column: str | None,
    requested_features: Sequence[str],
    min_labeled_rows: int = DEFAULT_MIN_LABELED_ROWS,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> OfficialSupervisedReadiness:
    selected_features = dedupe_feature_columns(requested_features)
    available_features = tuple(
        column for column in selected_features if column in frame.columns
    )
    missing_features = tuple(
        column for column in selected_features if column not in frame.columns
    )
    if not selected_features:
        return OfficialSupervisedReadiness(
            status="not_fit_no_requested_features",
            label_column=label_column,
            requested_features=selected_features,
            available_features=available_features,
            missing_features=missing_features,
            labeled_count=0,
            positive_count=0,
            negative_count=0,
            min_labeled_rows=min_labeled_rows,
            min_class_count=min_class_count,
        )
    if label_column is None or label_column not in frame.columns:
        return OfficialSupervisedReadiness(
            status="not_fit_label_unavailable",
            label_column=label_column,
            requested_features=selected_features,
            available_features=available_features,
            missing_features=missing_features,
            labeled_count=0,
            positive_count=0,
            negative_count=0,
            min_labeled_rows=min_labeled_rows,
            min_class_count=min_class_count,
        )
    if missing_features:
        return OfficialSupervisedReadiness(
            status="not_fit_missing_supervised_features",
            label_column=label_column,
            requested_features=selected_features,
            available_features=available_features,
            missing_features=missing_features,
            labeled_count=0,
            positive_count=0,
            negative_count=0,
            min_labeled_rows=min_labeled_rows,
            min_class_count=min_class_count,
        )

    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid_labels = labels.dropna()
    if valid_labels.empty:
        return OfficialSupervisedReadiness(
            status="not_fit_no_labeled_rows",
            label_column=label_column,
            requested_features=selected_features,
            available_features=available_features,
            missing_features=missing_features,
            labeled_count=0,
            positive_count=0,
            negative_count=0,
            min_labeled_rows=min_labeled_rows,
            min_class_count=min_class_count,
        )

    unique_values = {float(value) for value in valid_labels.unique()}
    if not unique_values.issubset({0.0, 1.0}):
        return OfficialSupervisedReadiness(
            status="not_fit_label_not_binary",
            label_column=label_column,
            requested_features=selected_features,
            available_features=available_features,
            missing_features=missing_features,
            labeled_count=int(len(valid_labels)),
            positive_count=0,
            negative_count=0,
            min_labeled_rows=min_labeled_rows,
            min_class_count=min_class_count,
        )

    binary_labels = valid_labels.astype(int)
    positive_count = int(binary_labels.eq(1).sum())
    negative_count = int(binary_labels.eq(0).sum())
    labeled_count = int(len(binary_labels))
    if labeled_count < min_labeled_rows:
        status = "not_fit_too_few_labeled_rows"
    elif positive_count < min_class_count or negative_count < min_class_count:
        status = "not_fit_class_too_small"
    elif int(binary_labels.nunique()) != 2:
        status = "not_fit_label_single_class"
    else:
        status = "fit"

    return OfficialSupervisedReadiness(
        status=status,
        label_column=label_column,
        requested_features=selected_features,
        available_features=available_features,
        missing_features=missing_features,
        labeled_count=labeled_count,
        positive_count=positive_count,
        negative_count=negative_count,
        min_labeled_rows=min_labeled_rows,
        min_class_count=min_class_count,
    )
