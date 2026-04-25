"""Tests for validation.schemas module."""

from __future__ import annotations

import pandas as pd

from plasmid_priority.validation.schemas import (
    BACKBONE_TABLE_SCHEMA,
    DEDUPLICATED_PLASMID_SCHEMA,
    HARMONIZED_PLASMID_SCHEMA,
    MAX_YEAR,
    MIN_YEAR,
    PANDERA_AVAILABLE,
    SCORED_BACKBONE_SCHEMA,
    _build_schema,
    _validate_table,
    format_validation_report,
    run_all_validations,
    validate_backbone_table,
    validate_deduplicated_plasmids,
    validate_harmonized_plasmids,
    validate_scored_backbones,
)


def test_schema_constants() -> None:
    """Test schema constants."""
    assert MIN_YEAR == 1950
    assert MAX_YEAR == 2050


def test_build_schema_without_pandera() -> None:
    """Test _build_schema when pandera is not available."""
    # This test assumes PANDERA_AVAILABLE might be False in some environments
    # We can't easily mock it, but we can test the logic
    if not PANDERA_AVAILABLE:
        schema = _build_schema({}, description="test")
        assert schema is None


def test_build_schema_with_pandera() -> None:
    """Test _build_schema when pandera is available."""
    if PANDERA_AVAILABLE:
        from pandera.pandas import Column

        columns = {
            "col1": Column(str, nullable=False),
            "col2": Column(float, nullable=True),
        }
        schema = _build_schema(columns, description="test schema")
        assert schema is not None
        assert schema.description == "test schema"


def test_harmonized_plasmid_schema_structure() -> None:
    """Test that HARMONIZED_PLASMID_SCHEMA has expected structure."""
    if PANDERA_AVAILABLE:
        from plasmid_priority.validation.schemas import HARMONIZED_PLASMID_SCHEMA

        if HARMONIZED_PLASMID_SCHEMA is not None:
            # Schema should be defined
            assert HARMONIZED_PLASMID_SCHEMA.columns is not None


def test_validate_table_without_pandera() -> None:
    """Test _validate_table when pandera is not available."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    result = _validate_table(df, None, "test")
    if not PANDERA_AVAILABLE:
        assert result["status"] == "pass"


def test_validate_table_with_pandera() -> None:
    """Test _validate_table when pandera is available."""
    if PANDERA_AVAILABLE:
        from pandera.pandas import Column, DataFrameSchema

        schema = DataFrameSchema(
            {
                "col1": Column(int, nullable=False),
            }
        )
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = _validate_table(df, schema, "test")
        assert result["status"] == "pass"


def test_validate_harmonized_plasmids_valid() -> None:
    """Test validation of harmonized plasmids with valid data."""
    df = pd.DataFrame(
        {
            "sequence_accession": ["ACC1", "ACC2"],
            "backbone_id": ["BB1", "BB2"],
            "resolved_year": [2015.0, 2016.0],
            "country": ["USA", "UK"],
            "gc_content": [50.0, 52.0],
            "size": [5000.0, 6000.0],
        }
    )
    result = validate_harmonized_plasmids(df)
    assert result["status"] == "pass"


def test_validate_harmonized_plasmids_invalid_year() -> None:
    """Test validation of harmonized plasmids with invalid year."""
    df = pd.DataFrame(
        {
            "sequence_accession": ["ACC1", "ACC2"],
            "backbone_id": ["BB1", "BB2"],
            "resolved_year": [1800.0, 2016.0],  # Invalid year
            "country": ["USA", "UK"],
            "gc_content": [50.0, 52.0],
            "size": [5000.0, 6000.0],
        }
    )
    result = validate_harmonized_plasmids(df)
    # Should fail if pandera is available
    if PANDERA_AVAILABLE:
        assert result["status"] in ("fail", "error")


def test_validate_backbone_table_valid() -> None:
    """Test validation of backbone table with valid data."""
    df = pd.DataFrame(
        {
            "backbone_id": ["BB1", "BB2"],
            "primary_cluster_id": ["cluster1", "cluster2"],
        }
    )
    result = validate_backbone_table(df)
    assert result["status"] == "pass"


def test_validate_backbone_table_missing_required() -> None:
    """Test validation of backbone table missing required column."""
    df = pd.DataFrame(
        {
            "primary_cluster_id": ["cluster1", "cluster2"],
        }
    )
    result = validate_backbone_table(df)
    if PANDERA_AVAILABLE:
        assert result["status"] in ("fail", "error")


def test_validate_scored_backbones_valid() -> None:
    """Test validation of scored backbones with valid data."""
    df = pd.DataFrame(
        {
            "backbone_id": ["BB1", "BB2"],
            "spread_label": [0.0, 1.0],
            "priority_index": [0.5, 0.8],
            "T_eff_norm": [0.3, 0.6],
            "H_eff_norm": [0.4, 0.7],
            "A_eff_norm": [0.2, 0.5],
        }
    )
    result = validate_scored_backbones(df)
    assert result["status"] == "pass"


def test_validate_scored_backbones_invalid_range() -> None:
    """Test validation of scored backbones with invalid range."""
    df = pd.DataFrame(
        {
            "backbone_id": ["BB1", "BB2"],
            "spread_label": [0.0, 2.0],  # Invalid: should be 0 or 1
            "priority_index": [0.5, 0.8],
        }
    )
    result = validate_scored_backbones(df)
    if PANDERA_AVAILABLE:
        assert result["status"] in ("fail", "error")


def test_validate_deduplicated_plasmids_valid() -> None:
    """Test validation of deduplicated plasmids with valid data."""
    df = pd.DataFrame(
        {
            "sequence_accession": ["ACC1", "ACC2"],
            "backbone_id": ["BB1", "BB2"],
        }
    )
    result = validate_deduplicated_plasmids(df)
    assert result["status"] == "pass"


def test_schema_definitions_exist() -> None:
    """Test that all schema definitions exist."""
    if PANDERA_AVAILABLE:
        assert HARMONIZED_PLASMID_SCHEMA is not None
        assert BACKBONE_TABLE_SCHEMA is not None
        assert SCORED_BACKBONE_SCHEMA is not None
        assert DEDUPLICATED_PLASMID_SCHEMA is not None


def test_year_bounds() -> None:
    """Test year bounds are sensible."""
    assert MIN_YEAR < MAX_YEAR
    assert MIN_YEAR >= 1900


def test_run_all_validations_none() -> None:
    """Test run_all_validations with all None inputs."""
    result = run_all_validations()
    assert result is not None
    assert isinstance(result, dict)


def test_run_all_validations_with_data() -> None:
    """Test run_all_validations with valid data."""
    df = pd.DataFrame(
        {
            "sequence_accession": ["ACC1", "ACC2"],
            "backbone_id": ["BB1", "BB2"],
        }
    )
    result = run_all_validations(
        harmonized=df,
        backbones=df,
        scored=df,
        deduplicated=df,
    )
    assert result is not None
    assert isinstance(result, dict)


def test_format_validation_report() -> None:
    """Test format_validation_report function."""
    df = pd.DataFrame(
        {
            "sequence_accession": ["ACC1", "ACC2"],
            "backbone_id": ["BB1", "BB2"],
        }
    )
    result = run_all_validations(harmonized=df)
    report = format_validation_report(result)
    assert report is not None
    assert isinstance(report, str)
    assert len(report) > 0
    if PANDERA_AVAILABLE:
        assert "PASS" in report or "FAIL" in report


def test_format_validation_report_empty() -> None:
    """Test format_validation_report with empty results."""
    result = run_all_validations()
    report = format_validation_report(result)
    assert report is not None
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_validation_report_with_failures() -> None:
    """Test format_validation_report with validation failures."""
    df = pd.DataFrame(
        {
            "backbone_id": ["BB1", "BB2"],
            # Missing required columns for harmonized schema
        }
    )
    result = run_all_validations(backbones=df)
    report = format_validation_report(result)
    assert report is not None
    assert isinstance(report, str)
    assert MAX_YEAR <= 2100
