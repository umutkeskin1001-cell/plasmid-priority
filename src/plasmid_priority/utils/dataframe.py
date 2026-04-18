"""Shared pandas dataframe manipulation utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd

try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - fallback for environments without duckdb
    _duckdb = None  # type: ignore[assignment]

duckdb: ModuleType | None = _duckdb


@lru_cache(maxsize=512)
def _read_tsv_cached(path_str: str, size: int, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_str, sep="\t", low_memory=False)


def clear_read_tsv_cache() -> None:
    _read_tsv_cached.cache_clear()


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def read_tsv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a TSV with parser settings that are stable for mixed metadata tables."""
    if kwargs:
        read_kwargs = {"sep": "\t", **kwargs}
        read_kwargs.setdefault("low_memory", False)
        try:
            return cast(pd.DataFrame, pd.read_csv(path, **read_kwargs))
        except IndexError:
            if "usecols" not in read_kwargs or read_kwargs.get("engine") == "python":
                raise
            retry_kwargs = dict(read_kwargs)
            retry_kwargs.pop("low_memory", None)
            retry_kwargs["engine"] = "python"
            return cast(pd.DataFrame, pd.read_csv(path, **retry_kwargs))
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return _read_tsv_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns)).copy()
    except IndexError:
        raise


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file, preferring DuckDB when available."""
    resolved = Path(path).resolve()
    if duckdb is None:
        return pd.read_parquet(resolved)
    query = f"SELECT * FROM read_parquet({_sql_literal(resolved)})"
    with duckdb.connect(database=":memory:") as connection:
        return cast(pd.DataFrame, connection.execute(query).fetchdf())


def coalescing_left_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
) -> pd.DataFrame:
    """Left-merge while coalescing duplicate columns instead of emitting `_x`/`_y` noise."""
    if left.empty or right.empty:
        return left
    overlap = [column for column in right.columns if column != on and column in left.columns]
    if right[on].is_unique:
        keyed = right.drop_duplicates(subset=[on], keep="first").set_index(on, drop=True)
        merged = left.copy()
        key_values = merged[on]
        for column in right.columns:
            if column == on:
                continue
            incoming = (
                key_values.map(keyed[column])
                if column in keyed.columns
                else pd.Series(index=merged.index, dtype=keyed.dtypes.get(column, "object"))
            )
            if column in merged.columns:
                merged[column] = merged[column].where(merged[column].notna(), incoming)
            else:
                merged[column] = incoming
        return merged

    if not overlap:
        return left.merge(right, on=on, how="left")

    merged = left.merge(right, on=on, how="left", suffixes=("", "__incoming"))
    for column in overlap:
        incoming_column = f"{column}__incoming"
        if incoming_column not in merged.columns:
            continue
        incoming_values = merged[incoming_column]
        merged[column] = merged[column].where(merged[column].notna(), incoming_values)
        merged = merged.drop(columns=[incoming_column])
    return merged


def clean_text_series(series: pd.Series) -> pd.Series:
    """Clean a pandas Series by filling NA values and stripping whitespace.

    This is the canonical implementation used across the project for consistent
    text cleaning in dataframes.
    """
    return series.fillna("").astype(str).str.strip()


def dominant_share(series: pd.Series) -> float:
    """Compute the dominant value share (0.0-1.0) for a pandas Series.

    This is the canonical implementation used across the project for computing
    the share of the most frequent non-empty value in a series.

    Args:
        series: A pandas Series of values (typically strings)

    Returns:
        The share of the most frequent non-empty value, or 0.0 if empty
    """
    cleaned = clean_text_series(series)
    cleaned = cleaned.loc[cleaned.ne("")]
    if cleaned.empty:
        return 0.0
    return float(cleaned.value_counts(normalize=True).iloc[0])
