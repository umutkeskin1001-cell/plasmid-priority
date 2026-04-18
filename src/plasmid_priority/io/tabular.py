"""Header-level readers for TSV, CSV, and NCBI assembly summary files."""

from __future__ import annotations

import csv
import gzip
from functools import lru_cache
from pathlib import Path
from typing import TextIO


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


@lru_cache(maxsize=256)
def _peek_table_columns_cached(path: Path, delimiter: str) -> tuple[str, ...]:
    """Read the first non-empty header line from a delimited text file."""
    with _open_text(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if row:
                return tuple(column.strip() for column in row)
    raise ValueError(f"Could not read a header row from {path}")


def peek_table_columns(path: Path, *, delimiter: str) -> list[str]:
    return list(_peek_table_columns_cached(path, delimiter))


@lru_cache(maxsize=256)
def _read_ncbi_assembly_summary_columns_cached(path: Path) -> tuple[str, ...]:
    """Read the canonical column names from an NCBI assembly summary file."""
    with path.open("r", encoding="utf-8") as handle:
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
