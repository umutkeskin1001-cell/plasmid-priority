"""Shared numeric and dataframe coercion helpers.

Centralizing these operations keeps coercion semantics consistent and avoids
repeating verbose pandas conversion chains throughout the codebase.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar, cast

import pandas as pd


def to_numeric_series(
    values: object,
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.Series:
    """Coerce arbitrary values to a numeric series.

    The ``errors`` keyword is intentionally exposed for backward compatibility
    with older call sites that explicitly passed pandas conversion policy.
    """
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors=errors)
    if isinstance(values, pd.DataFrame):
        if values.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(values.iloc[:, 0], errors=errors)
    if isinstance(values, (list, tuple)):
        return pd.to_numeric(pd.Series(values), errors=errors)
    return pd.to_numeric(pd.Series(values), errors=errors)


def to_numeric_float(values: object, *, default: float = 0.0) -> pd.Series:
    """Coerce to float series and fill missing values with a numeric default."""
    return to_numeric_series(values).fillna(float(default)).astype(float)


def to_numeric_int(values: object, *, default: int = 0) -> pd.Series:
    """Coerce to integer series with explicit missing-value handling."""
    return to_numeric_series(values).fillna(int(default)).astype(int)


_SeriesOrFrameT = TypeVar("_SeriesOrFrameT", pd.Series, pd.DataFrame)


def fill0(values: _SeriesOrFrameT, *, default: float = 0.0) -> _SeriesOrFrameT:
    """Fill NA values in a series with a float default."""
    return values.fillna(float(default))


def int0(values: _SeriesOrFrameT, *, default: int = 0) -> _SeriesOrFrameT:
    """Fill NA values and coerce to integer dtype."""
    return values.fillna(int(default)).astype(int)


def scalar_float(value: object, *, default: float = float("nan")) -> float:
    """Coerce a scalar-like value to float with fallback."""
    numeric = to_numeric_series([value]).iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def scalar_int(value: object, *, default: int = 0) -> int:
    """Coerce a scalar-like value to int with fallback."""
    numeric = to_numeric_series([value]).iloc[0]
    return int(numeric) if pd.notna(numeric) else int(default)


class _Copyable(Protocol):
    def copy(self) -> object: ...


_T = TypeVar("_T", bound=_Copyable)


def copy_frame(frame: _T) -> _T:
    """Generic copy wrapper for pandas objects used in table pipelines."""
    return cast(_T, frame.copy())


def copy_series(series: pd.Series) -> pd.Series:
    """Explicit series copy wrapper used to reduce ad-hoc copy call sites."""
    return series.copy()


def is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


def is_present(value: Any) -> bool:
    return bool(pd.notna(value))
