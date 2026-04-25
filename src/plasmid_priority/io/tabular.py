"""Header-level readers for TSV, CSV, and NCBI assembly summary files."""

from __future__ import annotations

import csv
import gzip
from functools import lru_cache
from pathlib import Path
from typing import TextIO


def _open_text(path: Path, encoding: str = "utf-8") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding=encoding)
    return path.open("r", encoding=encoding)


@lru_cache(maxsize=256)
def _peek_table_columns_cached(path: Path, delimiter: str) -> tuple[str, ...]:
    """Read the first non-empty header line from a delimited text file."""
    try:
        with _open_text(path, encoding="utf-8") as handle:
            return _read_first_row(handle, delimiter)
    except UnicodeDecodeError:
        # Fallback to latin-1 for non-UTF8 files (common in some legacy/Turkish datasets)
        with _open_text(path, encoding="latin-1") as handle:
            return _read_first_row(handle, delimiter)


def _read_first_row(handle: TextIO, delimiter: str) -> tuple[str, ...]:
    reader = csv.reader(handle, delimiter=delimiter)
    for row in reader:
        if row:
            return tuple(column.strip() for column in row)
    raise ValueError("Could not read a header row from file.")


def peek_table_columns(path: Path, *, delimiter: str) -> list[str]:
    return list(_peek_table_columns_cached(path, delimiter))


@lru_cache(maxsize=256)
def peek_parquet_columns(path: Path) -> tuple[str, ...]:
    """Read column names from a Parquet file without loading the data."""
    import pyarrow.parquet as pq
    return tuple(pq.read_schema(path).names)


@lru_cache(maxsize=256)
def _read_ncbi_assembly_summary_columns_cached(path: Path) -> tuple[str, ...]:
    """Read the canonical column names from an NCBI assembly summary file."""
    try:
        return _read_ncbi_header(path, encoding="utf-8")
    except UnicodeDecodeError:
        return _read_ncbi_header(path, encoding="latin-1")


def _read_ncbi_header(path: Path, encoding: str) -> tuple[str, ...]:
    with path.open("r", encoding=encoding) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            if line.startswith("#"):
                line = line[1:]
            return tuple(column.strip() for column in line.split("\t"))
    raise ValueError(f"Could not locate the assembly summary header in {path}")


def read_ncbi_assembly_summary_columns(path: Path) -> list[str]:
    return list(_read_ncbi_assembly_summary_columns_cached(path))
