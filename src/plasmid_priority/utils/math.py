"""Shared mathematical utilities used across scoring and feature modules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def geometric_mean(values: list[float]) -> float:
    """Compute the geometric mean; returns 0.0 if any value is non-positive."""
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def geometric_mean_frame(frame: pd.DataFrame) -> pd.Series:
    """Row-wise geometric mean across DataFrame columns; zero when any column <= 0."""
    array = frame.astype(float).to_numpy()
    positive_mask = np.all(array > 0.0, axis=1)
    result = np.zeros(len(frame), dtype=float)
    if positive_mask.any():
        result[positive_mask] = np.exp(np.log(array[positive_mask]).mean(axis=1))
    return pd.Series(result, index=frame.index, dtype=float)
