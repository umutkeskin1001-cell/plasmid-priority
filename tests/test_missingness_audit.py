"""Tests for missingness audit functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.validation.missingness import (
    audit_backbone_tables,
    audit_missingness,
    format_missingness_report,
    print_backbone_audit_report,
)


class TestAuditMissingness:
    """Test the core missingness audit function."""

    def test_empty_dataframe(self):
        """Audit should handle empty DataFrames gracefully."""
        df = pd.DataFrame()
        result = audit_missingness(df, "empty_test")

        assert result["table_name"] == "empty_test"
        assert result["n_rows"] == 0
        assert result["status"] == "empty_table"
        assert result["columns"] == []

    def test_no_missing_values(self):
        """Audit should report no missing values for complete data."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )
        result = audit_missingness(df, "complete_test")

        assert result["status"] == "ok"
        assert result["n_rows"] == 3
        assert result["high_missingness_count"] == 0

        for col in result["columns"]:
            assert col["missing_count"] == 0
            assert col["missing_fraction"] == 0.0
            assert col["high_missingness_flag"] is False

    def test_high_missingness_detection(self):
        """Audit should flag columns with high missingness."""
        df = pd.DataFrame(
            {
                "complete": [1, 2, 3, 4],
                "mostly_missing": [1, np.nan, np.nan, np.nan],
            }
        )
        result = audit_missingness(df, "partial_test", high_missingness_threshold=0.5)

        assert result["status"] == "concern"
        assert result["high_missingness_count"] == 1

        mostly_missing_col = next(c for c in result["columns"] if c["column"] == "mostly_missing")
        assert mostly_missing_col["missing_count"] == 3
        assert mostly_missing_col["missing_fraction"] == 0.75
        assert mostly_missing_col["high_missingness_flag"] is True

    def test_sorted_by_missingness(self):
        """Results should be sorted by missing fraction descending."""
        df = pd.DataFrame(
            {
                "some_missing": [1, 2, np.nan],
                "all_present": [1, 2, 3],
                "all_missing": [np.nan, np.nan, np.nan],
            }
        )
        result = audit_missingness(df, "sorted_test")

        fractions = [c["missing_fraction"] for c in result["columns"]]
        assert fractions == sorted(fractions, reverse=True)
        assert result["columns"][0]["column"] == "all_missing"
        assert result["columns"][0]["missing_fraction"] == 1.0

    def test_split_column_analysis(self):
        """Audit should break down missingness by split column."""
        df = pd.DataFrame(
            {
                "value": [1, np.nan, 3, np.nan],
                "split": ["train", "train", "test", "test"],
            }
        )
        result = audit_missingness(df, "split_test", split_column="split")

        value_col = next(c for c in result["columns"] if c["column"] == "value")
        assert "by_split" in value_col
        assert value_col["by_split"]["train"]["missing_count"] == 1
        assert value_col["by_split"]["test"]["missing_count"] == 1


class TestFormatMissingnessReport:
    """Test the report formatting function."""

    def test_report_contains_key_info(self):
        """Report should contain table name and status."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan],
                "col2": ["a", "b", "c"],
            }
        )
        audit = audit_missingness(df, "test_table")
        report = format_missingness_report(audit)

        assert "test_table" in report
        assert "col1" in report
        assert "col2" in report
        assert "Missingness Audit" in report


class TestAuditBackboneTables:
    """Test the backbone-specific audit entry point."""

    def test_backbone_table_only(self):
        """Audit should work with just backbone table."""
        backbone = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C"],
                "member_count_train": [1, 2, np.nan],
            }
        )
        result = audit_backbone_tables(backbone_table=backbone)

        assert "backbone_table" in result
        assert "scored_backbone_table" not in result
        assert "backbone_table" in result["tables_audited"]

    def test_scored_table_only(self):
        """Audit should work with just scored backbone table."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C"],
                "spread_label": [1.0, 0.0, np.nan],
                "priority_index": [0.5, np.nan, 0.3],
            }
        )
        result = audit_backbone_tables(scored_backbone_table=scored)

        assert "scored_backbone_table" in result
        assert "backbone_table" not in result

    def test_both_tables(self):
        """Audit should work with both tables."""
        backbone = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C"],
                "member_count_train": [1, 2, 3],
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C"],
                "spread_label": [1.0, 0.0, np.nan],
            }
        )
        result = audit_backbone_tables(
            backbone_table=backbone,
            scored_backbone_table=scored,
        )

        assert "backbone_table" in result
        assert "scored_backbone_table" in result
        assert len(result["tables_audited"]) == 2

    def test_eligibility_split_for_scored(self):
        """Scored table should automatically get eligibility split."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C", "D"],
                "spread_label": [1.0, 0.0, np.nan, np.nan],
                "priority_index": [0.5, 0.6, 0.3, np.nan],
            }
        )
        result = audit_backbone_tables(scored_backbone_table=scored)

        scored_audit = result["scored_backbone_table"]
        priority_col = next(c for c in scored_audit["columns"] if c["column"] == "priority_index")

        # Should have by_split info for eligible vs ineligible
        assert "by_split" in priority_col
        assert "eligible" in priority_col["by_split"]
        assert "ineligible" in priority_col["by_split"]

    def test_concern_status_aggregation(self):
        """Overall status should be concern if any table has concern."""
        backbone = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "mostly_missing": [np.nan, np.nan],
            }
        )
        result = audit_backbone_tables(
            backbone_table=backbone,
            high_missingness_threshold=0.5,
        )

        assert result["overall_status"] == "concern"
        assert result["high_missingness_columns_total"] > 0


class TestPrintBackboneAuditReport:
    """Test the report printing function (smoke tests)."""

    def test_print_does_not_crash(self, capsys):
        """Printing should not crash with valid results."""
        backbone = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "value": [1, 2],
            }
        )
        result = audit_backbone_tables(backbone_table=backbone)

        # Function uses logging, not print - just verify it doesn't crash
        print_backbone_audit_report(result)
