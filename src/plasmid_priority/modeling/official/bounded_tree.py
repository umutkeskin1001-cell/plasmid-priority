from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from plasmid_priority.modeling.official.common import (
    dedupe_feature_columns,
    numeric_feature_matrix,
    valid_binary_labels,
)


@dataclass
class BoundedTreeChallenger:
    model_name: str
    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    estimator: Any
    max_depth: int

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        matrix = numeric_feature_matrix(
            frame,
            self.feature_columns,
            error_label="bounded tree",
        )
        imputed = self.imputer.transform(matrix)
        probabilities = self.estimator.predict_proba(imputed)[:, 1]
        clipped = np.clip(probabilities.astype(float), 0.0, 1.0)
        return pd.Series(clipped, index=frame.index, dtype="float64")


def fit_bounded_tree_challenger(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Sequence[str],
    max_depth: int = 3,
    min_samples_leaf: int = 2,
) -> BoundedTreeChallenger:
    selected_features = dedupe_feature_columns(feature_columns)
    labels = valid_binary_labels(
        frame,
        label_column,
        error_label="Official bounded tree",
    )
    matrix = numeric_feature_matrix(
        frame.loc[labels.index],
        selected_features,
        error_label="bounded tree",
    )

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
        feature_columns=dedupe_feature_columns(selected_features),
        imputer=imputer,
        estimator=estimator,
        max_depth=max_depth,
    )
