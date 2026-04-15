"""Temporal helpers used by branch label factories and contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def resolve_window_bounds(split_year: int, horizon_years: int) -> tuple[int, int]:
    """Return the inclusive temporal bounds for the future window."""
    split = int(split_year)
    horizon = int(horizon_years)
    if horizon < 0:
        raise ValueError("horizon_years must be non-negative")
    return split + 1, split + horizon


def _coerce_year_series(values: pd.Series | Sequence[Any]) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce")


def split_year_window_mask(
    years: pd.Series | Sequence[Any],
    *,
    split_year: int,
    horizon_years: int | None = None,
) -> pd.Series:
    """Return a mask for rows within the post-split future window."""
    numeric = _coerce_year_series(years)
    if horizon_years is None:
        return numeric > float(split_year)
    lower, upper = resolve_window_bounds(split_year, horizon_years)
    return numeric.between(lower, upper, inclusive="both")


def future_window_mask(
    years: pd.Series | Sequence[Any],
    *,
    split_year: int,
    horizon_years: int,
) -> pd.Series:
    """Return a mask for rows inside the horizon-limited future window."""
    return split_year_window_mask(years, split_year=split_year, horizon_years=horizon_years)


def pre_split_mask(
    years: pd.Series | Sequence[Any],
    *,
    split_year: int,
) -> pd.Series:
    """Return a mask for rows at or before the split year."""
    numeric = _coerce_year_series(years)
    return numeric <= float(split_year)
