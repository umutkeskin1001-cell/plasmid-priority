from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.check_code_review_graph import main


def _create_graph_db(path: Path, *, nodes: int, edges: int, files: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
        for _ in range(nodes):
            conn.execute("INSERT INTO nodes DEFAULT VALUES")
        for _ in range(edges):
            conn.execute("INSERT INTO edges DEFAULT VALUES")
        for _ in range(files):
            conn.execute("INSERT INTO files DEFAULT VALUES")
        conn.commit()


def test_graph_guard_fails_on_false_zero_state(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    _create_graph_db(db_path, nodes=0, edges=0, files=0)
    with pytest.raises(RuntimeError, match="false-zero graph state"):
        main(["--db-path", str(db_path)])


def test_graph_guard_passes_when_graph_has_content(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    _create_graph_db(db_path, nodes=1, edges=1, files=1)
    assert main(["--db-path", str(db_path)]) == 0
