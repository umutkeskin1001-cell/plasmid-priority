"""Parquet and DuckDB utilities for data layer optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

_duckdb: Any
try:
    import duckdb as _duckdb
except ImportError:
    _duckdb = None

duckdb: Any = _duckdb


def _sql_string_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal for DuckDB statements."""
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def convert_tsv_to_parquet(
    tsv_path: str | Path,
    parquet_path: str | Path,
    *,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd",
    overwrite: bool = False,
) -> Path:
    """Convert a TSV file to Parquet format using DuckDB for performance.

    Args:
        tsv_path: Path to input TSV file
        parquet_path: Path to output Parquet file
        compression: Compression algorithm (default: zstd)
        overwrite: Overwrite existing Parquet file

    Returns:
        Path to output Parquet file
    """
    tsv_path = Path(tsv_path)
    parquet_path = Path(parquet_path)

    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"Parquet file already exists: {parquet_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if duckdb is not None:
        # Use DuckDB for efficient conversion
        con = duckdb.connect(database=":memory:")
        tsv_literal = _sql_string_literal(tsv_path)
        parquet_literal = _sql_string_literal(parquet_path)
        compression_literal = _sql_string_literal(str(compression).upper())
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto({tsv_literal}, sep='\t', header=true)
            ) TO {parquet_literal} (FORMAT PARQUET, COMPRESSION {compression_literal});
        """)
    else:
        # Fallback to pandas
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        df.to_parquet(parquet_path, compression=compression, index=False)

    return parquet_path


def read_with_duckdb(
    path: str | Path,
    query: str | None = None,
) -> pd.DataFrame:
    """Read data file using DuckDB for optimized performance.

    Args:
        path: Path to data file (TSV, CSV, or Parquet)
        query: Optional DuckDB SQL query. If None, reads entire file.

    Returns:
        DataFrame with query results
    """
    path = Path(path)

    if duckdb is None:
        raise ImportError("DuckDB is required for this operation")

    con = duckdb.connect(database=":memory:")

    if query is None:
        if path.suffix == ".parquet":
            query = f"SELECT * FROM read_parquet({_sql_string_literal(path)})"
        elif path.suffix in (".tsv", ".csv"):
            sep = "\t" if path.suffix == ".tsv" else ","
            query = (
                f"SELECT * FROM read_csv_auto({_sql_string_literal(path)}, "
                f"sep={_sql_string_literal(sep)}, header=true)"
            )
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    return cast(pd.DataFrame, con.execute(query).fetchdf())
