"""Shared pandas dataframe manipulation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_tsv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a TSV with parser settings that are stable for mixed metadata tables."""
    read_kwargs = {"sep": "\t", **kwargs}
    read_kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **read_kwargs)
    except IndexError:
        if "usecols" not in read_kwargs or read_kwargs.get("engine") == "python":
            raise
        retry_kwargs = dict(read_kwargs)
        retry_kwargs.pop("low_memory", None)
        retry_kwargs["engine"] = "python"
        return pd.read_csv(path, **retry_kwargs)


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
    if not overlap:
        return left.merge(right, on=on, how="left")

    merged = left.merge(right, on=on, how="left", suffixes=("", "__incoming"))
    for column in overlap:
        incoming = f"{column}__incoming"
        if incoming not in merged.columns:
            continue
        merged[column] = merged[column].where(merged[column].notna(), merged[incoming])
        merged = merged.drop(columns=incoming)
    return merged
