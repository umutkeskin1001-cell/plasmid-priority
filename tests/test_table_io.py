from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plasmid_priority.io import table_io


def test_read_table_pandas_fallback_projection_and_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "backbone_id": ["bb1", "bb2", "bb3"],
            "priority_index": [0.1, 0.8, 0.4],
            "cluster": ["A", "B", "A"],
        },
    )
    table_path = tmp_path / "scores.tsv"
    frame.to_csv(table_path, sep="\t", index=False)

    monkeypatch.setattr(table_io, "pl", None)
    monkeypatch.setattr(table_io, "duckdb", None)

    result = table_io.read_table(
        table_path,
        columns=["backbone_id", "priority_index"],
        filters=[("priority_index", ">", 0.2)],
    )

    assert result.columns.tolist() == ["backbone_id", "priority_index"]
    assert result["backbone_id"].tolist() == ["bb2", "bb3"]


def test_apply_filters_frame_supports_membership_operators() -> None:
    frame = pd.DataFrame(
        {
            "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
            "priority_index": [0.1, 0.8, 0.4, 0.6],
            "cluster": ["A", "B", "A", "C"],
        },
    )
    result = table_io._apply_filters_frame(
        frame,
        filters=[
            ("priority_index", ">=", 0.4),
            ("cluster", "in", ["A", "B"]),
            ("backbone_id", "not in", ["bb2"]),
        ],
    )
    assert result["backbone_id"].tolist() == ["bb3"]


def test_apply_filters_frame_rejects_unsupported_operator() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported filter operator: between"):
        table_io._apply_filters_frame(
            frame,
            filters=[("x", "between", (1, 2))],  # type: ignore[list-item]
        )


def test_duckdb_select_query_builds_safe_query() -> None:
    query = table_io._duckdb_select_query(
        Path("/tmp/data.csv"),
        columns=["backbone_id", "priority_index"],
        filters=[
            ("backbone_id", "==", "bb'1"),
            ("priority_index", ">=", 0.5),
            ("cluster", "in", ["A", "B"]),
        ],
    )
    assert 'SELECT "backbone_id", "priority_index" FROM read_csv_auto(\'/tmp/data.csv\'' in query
    assert "\"backbone_id\" = 'bb''1'" in query
    assert "\"priority_index\" >= 0.5" in query
    assert "\"cluster\" IN ('A', 'B')" in query


def test_duckdb_select_query_rejects_bad_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported filter operator: like"):
        table_io._duckdb_select_query(
            Path("/tmp/data.tsv"),
            filters=[("backbone_id", "like", "bb%")],  # type: ignore[list-item]
        )


def test_read_table_rejects_path_without_supported_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = tmp_path / "scores"
    unsupported.write_text("backbone_id\tpriority_index\nbb1\t0.1\n", encoding="utf-8")

    monkeypatch.setattr(table_io, "pl", None)
    monkeypatch.setattr(table_io, "duckdb", None)

    with pytest.raises(ValueError, match="Unsupported table format"):
        table_io.read_table(unsupported)
