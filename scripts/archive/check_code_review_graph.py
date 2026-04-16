#!/usr/bin/env python3
"""Fail-fast guard for code-review-graph false-zero states."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DB = PROJECT_ROOT / ".code-review-graph" / "graph.db"


def _count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    cursor = conn.cursor()
    try:
        row = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    except sqlite3.Error as exc:  # pragma: no cover - exercised via CLI failures
        raise RuntimeError(f"Failed to read table '{table_name}': {exc}") from exc
    if row is None:
        raise RuntimeError(f"Could not read row count for table '{table_name}'.")
    return int(row[0])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_GRAPH_DB,
        help="Path to code-review-graph SQLite DB (default: .code-review-graph/graph.db)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = args.db_path.expanduser()
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()
    if not db_path.exists():
        raise RuntimeError(f"Graph DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        nodes = _count_rows(conn, "nodes")
        edges = _count_rows(conn, "edges")
        files = _count_rows(conn, "files")

    print(f"code-review-graph counts -> nodes={nodes}, edges={edges}, files={files}")
    if nodes == 0 and edges == 0 and files == 0:
        raise RuntimeError(
            "Detected false-zero graph state (nodes=0, edges=0, files=0). Treating as hard failure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
