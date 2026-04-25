from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


@dataclass
class BoundedTreeChallenger:
    model_name: str
    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    estimator: Any
    max_depth: int

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        matrix = _feature_matrix(frame, self.feature_columns)
        imputed = self.imputer.transform(matrix)
        probabilities = self.estimator.predict_proba(imputed)[:, 1]
        clipped = np.clip(probabilities.astype(float), 0.0, 1.0)
        return pd.Series(clipped, index=frame.index, dtype="float64")


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing bounded tree features: {missing}")
    matrix = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, matrix)


def _binary_labels(frame: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    labels = pd.to_numeric(frame[label_column], errors="coerce").dropna().astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Official bounded tree expects a binary 0/1 label")
    if int(labels.nunique()) != 2:
        raise ValueError(
            "Official bounded tree requires at least one positive and one negative label",
        )
    return labels


def fit_bounded_tree_challenger(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Sequence[str],
    max_depth: int = 3,
    min_samples_leaf: int = 2,
) -> BoundedTreeChallenger:
    selected_features = tuple(dict.fromkeys(str(column) for column in feature_columns))
    if not selected_features:
        raise ValueError("At least one feature column is required")
    labels = _binary_labels(frame, label_column)
    matrix = _feature_matrix(frame.loc[labels.index], selected_features)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(matrix)
    estimator = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    estimator.fit(imputed, labels.to_numpy(dtype=int))

    return BoundedTreeChallenger(
        model_name="bounded_monotonic_tree",
        feature_columns=selected_features,
        imputer=imputer,
        estimator=estimator,
        max_depth=max_depth,
    )
