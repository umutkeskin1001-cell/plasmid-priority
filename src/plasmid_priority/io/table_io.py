"""Unified tabular IO helpers with Parquet-first defaults."""


import math
from collections.abc import Collection, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias, cast

import pandas as pd

_duckdb: ModuleType | None
try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - optional dependency
    _duckdb = None

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


# Global switch — can be overridden via environment or config
USE_POLARS = True
duckdb: ModuleType | None = _duckdb

FilterOperator = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not in"]
FilterClause: TypeAlias = tuple[str, FilterOperator, Any]
SUPPORTED_FILTER_OPERATORS: frozenset[str] = frozenset(
    {"==", "!=", ">", ">=", "<", "<=", "in", "not in"},
)


def _validate_filter_operator(operator: str) -> FilterOperator:
    if operator not in SUPPORTED_FILTER_OPERATORS:
        raise ValueError(f"Unsupported filter operator: {operator}")
    return cast(FilterOperator, operator)


def _require_collection_value(value: Any, *, operator: FilterOperator) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ValueError(
            f"Operator '{operator}' requires a non-string collection value, "
            f"got {type(value).__name__}",
        )
    return list(value)


def _quote_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("Column identifier must be a non-empty string")
    return '"' + identifier.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float values are not supported in SQL filters")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_filter_clause(column: str, operator: FilterOperator, value: Any) -> str:
    column_sql = _quote_identifier(column)
    if operator == "==":
        return f"{column_sql} IS NULL" if value is None else f"{column_sql} = {_sql_literal(value)}"
    if operator == "!=":
        if value is None:
            return f"{column_sql} IS NOT NULL"
        return f"{column_sql} <> {_sql_literal(value)}"
    if operator in {">", ">=", "<", "<="}:
        if value is None:
            raise ValueError(f"Operator '{operator}' does not support None as a value")
        return f"{column_sql} {operator} {_sql_literal(value)}"
    values = _require_collection_value(value, operator=operator)
    if not values:
        return "FALSE" if operator == "in" else "TRUE"
    keyword = "IN" if operator == "in" else "NOT IN"
    values_sql = ", ".join(_sql_literal(item) for item in values)
    return f"{column_sql} {keyword} ({values_sql})"


def _duckdb_select_query(
    path: Path,
    columns: Sequence[str] | None = None,
    filters: Sequence[FilterClause] | None = None,
) -> str:
    suffix = path.suffix.lower()
    path_sql = _sql_literal(str(path))
    if suffix == ".parquet":
        source_sql = f"read_parquet({path_sql})"
    elif suffix == ".csv":
        source_sql = f"read_csv_auto({path_sql}, header=true)"
    elif suffix == ".tsv":
        source_sql = f"read_csv_auto({path_sql}, header=true, delim='\\t')"
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")

    select_sql = "*" if not columns else ", ".join(_quote_identifier(column) for column in columns)
    where_clauses: list[str] = []
    if filters:
        for column, operator, value in filters:
            op = _validate_filter_operator(operator)
            where_clauses.append(_duckdb_filter_clause(column, op, value))

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return f"SELECT {select_sql} FROM {source_sql}{where_sql}"


def _apply_filters_lazy_frame(
    lazy_frame: Any,
    filters: Sequence[FilterClause] | None,
) -> Any:
    if not filters:
        return lazy_frame
    for column, operator, value in filters:
        op = _validate_filter_operator(operator)
        expr = pl.col(column)
        if op == "==":
            lazy_frame = lazy_frame.filter(expr == value)
        elif op == "!=":
            lazy_frame = lazy_frame.filter(expr != value)
        elif op == ">":
            lazy_frame = lazy_frame.filter(expr > value)
        elif op == ">=":
            lazy_frame = lazy_frame.filter(expr >= value)
        elif op == "<":
            lazy_frame = lazy_frame.filter(expr < value)
        elif op == "<=":
            lazy_frame = lazy_frame.filter(expr <= value)
        elif op == "in":
            values = _require_collection_value(value, operator=op)
            lazy_frame = lazy_frame.filter(expr.is_in(values))
        elif op == "not in":
            values = _require_collection_value(value, operator=op)
            lazy_frame = lazy_frame.filter(~expr.is_in(values))
    return lazy_frame


def _apply_filters_frame(
    frame: pd.DataFrame,
    filters: Sequence[FilterClause] | None,
) -> pd.DataFrame:
    if not filters:
        return frame

    filtered = frame
    for column, operator, value in filters:
        op = _validate_filter_operator(operator)
        series = filtered[column]
        if op == "==":
            mask = series.isna() if value is None else series == value
        elif op == "!=":
            mask = series.notna() if value is None else series != value
        elif op == ">":
            mask = series > value
        elif op == ">=":
            mask = series >= value
        elif op == "<":
            mask = series < value
        elif op == "<=":
            mask = series <= value
        elif op == "in":
            values = _require_collection_value(value, operator=op)
            mask = series.isin(values)
        elif op == "not in":
            values = _require_collection_value(value, operator=op)
            mask = ~series.isin(values)
        filtered = filtered.loc[mask]
    return filtered


def read_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    filters: Sequence[FilterClause] | None = None,
) -> pd.DataFrame:
    """Read a table with optional projection and filtering.

    Prioritizes Parquet + Polars for 5-50x speedup over pandas TSV.
    """
    resolved = Path(path).resolve()
    parquet_path = resolved.with_suffix(".parquet")

    # High performance path: Polars + Parquet
    if USE_POLARS and pl is not None and parquet_path.exists():
        try:
            lazy_frame = pl.scan_parquet(str(parquet_path))
            if columns:
                lazy_frame = lazy_frame.select(list(columns))
            lazy_frame = _apply_filters_lazy_frame(lazy_frame, filters)
            return cast(pd.DataFrame, lazy_frame.collect().to_pandas())
        except Exception:
            # Fallback if parquet is corrupted or incompatible
            pass

    # Middle path: DuckDB for filtered CSV/TSV
    if duckdb is not None and (columns or filters):
        try:
            query = _duckdb_select_query(path=resolved, columns=columns, filters=filters)
            with duckdb.connect(database=":memory:") as connection:
                result = connection.execute(query).fetchdf()
                return cast(pd.DataFrame, result)
        except Exception:
            pass

    # Fallback path: standard pandas
    if resolved.suffix == ".parquet":
        frame = pd.read_parquet(resolved, columns=list(columns) if columns else None)
        return _apply_filters_frame(frame, filters)

    if resolved.suffix in (".tsv", ".csv"):
        sep = "\t" if resolved.suffix == ".tsv" else ","
        frame = pd.read_csv(
            resolved,
            sep=sep,
            low_memory=False,
            usecols=list(columns) if columns else None,
        )
        return _apply_filters_frame(frame, filters)

    raise ValueError(f"Unsupported table format: {resolved.suffix}")


def scan_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    filters: Sequence[FilterClause] | None = None,
) -> Any:
    """Return a lazy table object when available."""
    resolved = Path(path).resolve()
    parquet_path = resolved.with_suffix(".parquet")

    if pl is not None:
        actual_path = parquet_path if parquet_path.exists() else resolved
        if actual_path.suffix == ".parquet":
            lazy_frame = pl.scan_parquet(str(actual_path))
        elif actual_path.suffix in (".tsv", ".csv"):
            separator = "\t" if actual_path.suffix == ".tsv" else ","
            lazy_frame = pl.scan_csv(str(actual_path), separator=separator, has_header=True)
        else:
            raise ValueError(f"Unsupported table format: {actual_path.suffix}")

        if columns:
            lazy_frame = lazy_frame.select(list(columns))
        lazy_frame = _apply_filters_lazy_frame(lazy_frame, filters)
        return lazy_frame

    if duckdb is not None:
        query = _duckdb_select_query(path=resolved, columns=columns, filters=filters)
        connection = duckdb.connect(database=":memory:")
        return connection.sql(query)

    return read_table(resolved, columns=columns, filters=filters)


def write_table(df: pd.DataFrame, path: str | Path, *, format: str = "parquet") -> Path:
    """Write table in the selected format.

    Always writes Parquet alongside TSV for high performance transition.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet for speed (zstd compression)
    parquet_target = target.with_suffix(".parquet")
    df.to_parquet(parquet_target, compression="zstd", index=False)

    # Also write TSV for backward compatibility during transition
    if target.suffix == ".tsv" or format == "tsv":
        df.to_csv(target, sep="\t", index=False)
    elif target.suffix == ".csv" or format == "csv":
        df.to_csv(target, index=False)
    elif format == "parquet" and target.suffix != ".parquet":
        # If path didn't have .parquet but format was parquet, we already wrote it above
        pass

    return target
