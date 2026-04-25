"""Tests for validation.boundaries module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from plasmid_priority.validation.boundaries import (
    TABULAR_VALIDATORS,
    _normalize_output_path,
    _validate_consensus_predictions,
    _validate_json_artifact,
    _validate_tsv_artifact,
    validate_output_artifact,
    validate_script_boundary,
)


def test_validate_consensus_predictions_empty() -> None:
    """Test validation with empty dataframe."""
    df = pd.DataFrame()
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert result["n_rows"] == 0
    assert "empty" in result["errors"][0]["message"].lower()


def test_validate_consensus_predictions_missing_backbone_id() -> None:
    """Test validation without backbone_id column."""
    df = pd.DataFrame({"score": [0.5, 0.6]})
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert "backbone_id" in result["errors"][0]["message"]


def test_validate_consensus_predictions_missing_score_column() -> None:
    """Test validation without any score column."""
    df = pd.DataFrame({"backbone_id": ["A", "B"]})
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert "score column" in result["errors"][0]["message"]


def test_validate_consensus_predictions_all_na_scores() -> None:
    """Test validation with all NA scores."""
    df = pd.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "consensus_score": [None, None],
        }
    )
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert "no numeric data" in result["errors"][0]["message"]


def test_validate_consensus_predictions_out_of_range() -> None:
    """Test validation with out-of-range scores."""
    df = pd.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "consensus_score": [0.5, 1.5],
        }
    )
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert "out-of-range" in result["errors"][0]["message"]


def test_validate_consensus_predictions_valid() -> None:
    """Test validation with valid data."""
    df = pd.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "consensus_score": [0.5, 0.6],
        }
    )
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "pass"
    assert result["n_rows"] == 2
    assert len(result["errors"]) == 0


def test_validate_consensus_predictions_optional_bounds_invalid() -> None:
    """Test validation with invalid optional bounds."""
    df = pd.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "consensus_score": [0.5, 0.6],
            "consensus_uncertainty": [0.5, 1.5],  # Out of range
        }
    )
    result = _validate_consensus_predictions(df, label="test")
    assert result["status"] == "fail"
    assert "consensus_uncertainty" in result["errors"][0]["message"]


def test_normalize_output_path_absolute() -> None:
    """Test path normalization with absolute path."""
    path = "/tmp/test.tsv"
    result = _normalize_output_path(path)
    assert result == Path(path)


def test_normalize_output_path_relative() -> None:
    """Test path normalization with relative path."""
    path = "data/test.tsv"
    result = _normalize_output_path(path)
    assert result.is_absolute()


def test_normalize_output_path_relative_with_project_root() -> None:
    """Test path normalization with project root."""
    path = "data/test.tsv"
    project_root = Path("/tmp/project")
    result = _normalize_output_path(path, project_root=project_root)
    assert result == (project_root / path).resolve()


def test_validate_tsv_artifact_valid() -> None:
    """Test TSV validation with valid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.tsv"
        path.write_text("col1\tcol2\n1\t2\n", encoding="utf-8")
        result = _validate_tsv_artifact(path)
        assert result["status"] == "pass"
        assert result["n_rows"] == 1


def test_validate_tsv_artifact_invalid() -> None:
    """Test TSV validation with invalid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("not a tsv", encoding="utf-8")
        result = _validate_tsv_artifact(path)
        # Should fail to parse as TSV
        assert result["status"] in ("error", "pass")


def test_validate_json_artifact_valid() -> None:
    """Test JSON validation with valid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        result = _validate_json_artifact(path)
        assert result["status"] == "pass"


def test_validate_json_artifact_invalid() -> None:
    """Test JSON validation with invalid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        path.write_text("not json", encoding="utf-8")
        result = _validate_json_artifact(path)
        assert result["status"] == "error"
        assert "Failed to parse JSON" in result["errors"][0]["message"]


def test_validate_output_artifact_missing() -> None:
    """Test validation with missing artifact."""
    result = validate_output_artifact("/nonexistent/file.tsv")
    assert result["status"] == "fail"
    assert "Missing output artifact" in result["errors"][0]["message"]


def test_validate_output_artifact_valid_tsv() -> None:
    """Test validation with valid TSV artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.tsv"
        path.write_text("col1\tcol2\n1\t2\n", encoding="utf-8")
        result = validate_output_artifact(path)
        assert result["status"] == "pass"


def test_validate_output_artifact_valid_json() -> None:
    """Test validation with valid JSON artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        result = validate_output_artifact(path)
        assert result["status"] == "pass"


def test_validate_script_boundary_missing_output_paths() -> None:
    """Test script boundary validation with missing output paths."""
    summary = {"output_files_written": []}  # Empty list instead of missing key
    result = validate_script_boundary(summary)
    assert result["status"] == "pass"  # Empty list should pass


def test_validate_script_boundary_invalid_output_paths_type() -> None:
    """Test script boundary validation with invalid output paths type."""
    summary = {"output_files_written": "not a list"}
    result = validate_script_boundary(summary)
    assert result["status"] == "fail"
    assert "output path metadata" in result["errors"][0]["message"]


def test_validate_script_boundary_valid() -> None:
    """Test script boundary validation with valid summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")

        summary = {
            "output_files_written": [str(path)],
            "input_manifest": {},
        }
        result = validate_script_boundary(summary)
        assert result["status"] == "pass"


def test_tabular_validators_dict() -> None:
    """Test that TABULAR_VALIDATORS has expected keys."""
    expected_keys = [
        "harmonized_plasmids.tsv",
        "backbone_table.tsv",
        "backbone_scored.tsv",
        "deduplicated_plasmids.tsv",
        "consensus_predictions.tsv",
    ]
    for key in expected_keys:
        assert key in TABULAR_VALIDATORS
