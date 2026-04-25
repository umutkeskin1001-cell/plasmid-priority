from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


@dataclass
class SparseCalibratedLogistic:
    model_name: str
    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    center: np.ndarray
    scale: np.ndarray
    estimator: Any
    calibration_method: str = "native_logistic"

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_columns)
        imputed = self.imputer.transform(matrix)
        return cast(np.ndarray, (imputed - self.center) / self.scale)

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        probabilities = self.estimator.predict_proba(self._transform(frame))[:, 1]
        clipped = np.clip(probabilities.astype(float), 0.0, 1.0)
        return pd.Series(clipped, index=frame.index, dtype="float64")


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing official model features: {missing}")
    matrix = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, matrix)


def _valid_binary_labels(frame: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid_labels = labels.dropna().astype(int)
    if not set(valid_labels.unique()).issubset({0, 1}):
        raise ValueError("Official sparse logistic expects a binary 0/1 label")
    if int(valid_labels.nunique()) != 2:
        raise ValueError(
            "Official sparse logistic requires at least one positive and one negative label",
        )
    return valid_labels


def _select_sparse_features(
    frame: pd.DataFrame,
    labels: pd.Series,
    feature_columns: Sequence[str],
    max_features: int,
) -> tuple[str, ...]:
    if max_features <= 0:
        raise ValueError("max_features must be positive")
    candidates = tuple(dict.fromkeys(str(column) for column in feature_columns))
    if not candidates:
        raise ValueError("At least one feature column is required")
    if len(candidates) <= max_features:
        return candidates

    scores: list[tuple[float, int, str]] = []
    for order, column in enumerate(candidates):
        values = pd.to_numeric(frame.loc[labels.index, column], errors="coerce")
        combined = pd.DataFrame({"x": values, "y": labels}).dropna()
        if combined["x"].nunique(dropna=True) <= 1:
            score = 0.0
        else:
            correlation = combined["x"].corr(combined["y"])
            score = (
                abs(float(correlation))
                if correlation is not None and np.isfinite(correlation)
                else 0.0
            )
        scores.append((score, -order, column))
    scores.sort(reverse=True)
    selected = {column for _, _, column in scores[:max_features]}
    return tuple(column for column in candidates if column in selected)


def fit_sparse_calibrated_logistic(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Sequence[str],
    max_features: int = 8,
    regularization_c: float = 1.0,
) -> SparseCalibratedLogistic:
    labels = _valid_binary_labels(frame, label_column)
    selected_features = _select_sparse_features(frame, labels, feature_columns, max_features)
    matrix = _feature_matrix(frame.loc[labels.index], selected_features)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(matrix)
    center = np.nanmedian(imputed, axis=0)
    q75 = np.nanpercentile(imputed, 75, axis=0)
    q25 = np.nanpercentile(imputed, 25, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    transformed = (imputed - center) / scale

    estimator = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    estimator.fit(transformed, labels.to_numpy(dtype=int))

    return SparseCalibratedLogistic(
        model_name="sparse_calibrated_logistic",
        feature_columns=selected_features,
        imputer=imputer,
        center=center.astype(float),
        scale=scale.astype(float),
        estimator=estimator,
    )
