"""Tests for Pandera schema validation functionality.

This test module ensures:
1. Schema validation works correctly when Pandera is available
2. Fallback behavior is explicit and documented when Pandera is unavailable
3. The "skipped" status is clearly distinguished from "passed"
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from plasmid_priority.validation import (
    PANDERA_AVAILABLE,
    run_all_validations,
    validate_backbone_table,
    validate_deduplicated_plasmids,
    validate_harmonized_plasmids,
    validate_scored_backbones,
)


class TestSchemaValidationAvailability:
    """Test that schema validation availability is correctly reported."""

    def test_pandera_availability_flag(self):
        """PANDERA_AVAILABLE flag should accurately reflect import status."""
        has_pandera = importlib.util.find_spec("pandera.pandas") is not None
        assert PANDERA_AVAILABLE is has_pandera

    def test_validation_returns_expected_status_format(self):
        """All validation functions should return consistent status format."""
        df = pd.DataFrame({"test": [1, 2, 3]})

        # Test one validation function for format consistency
        result = validate_backbone_table(df)

        # All results must have these keys
        assert "status" in result
        assert "table" in result
        assert "n_rows" in result
        assert "errors" in result

        # Status must be one of the expected values
        assert result["status"] in ("pass", "fail", "error", "skipped")


class TestSchemaValidationWithValidData:
    """Test validation with valid data (only runs if Pandera is available)."""

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_harmonized_plasmids_valid_data(self):
        """Valid harmonized plasmid data should pass validation."""
        df = pd.DataFrame(
            {
                "sequence_accession": ["ACC001", "ACC002", "ACC003"],
                "backbone_id": ["BB001", "BB002", "BB003"],
                "resolved_year": [2020.0, 2021.0, 2022.0],
                "country": ["USA", "UK", None],
                "predicted_mobility": ["conjugative", "mobilizable", None],
                "gc_content": [50.0, 55.0, 60.0],
                "size": [10000.0, 20000.0, 30000.0],
            }
        )

        result = validate_harmonized_plasmids(df)

        assert result["status"] == "pass"
        assert result["table"] == "harmonized_plasmids"
        assert result["n_rows"] == 3
        assert result["errors"] == []

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_backbone_table_valid_data(self):
        """Valid backbone table data should pass validation."""
        df = pd.DataFrame(
            {
                "backbone_id": ["BB001", "BB002", "BB003"],
                "primary_cluster_id": ["C001", "C002", None],
                "predicted_mobility": ["conjugative", None, "mobilizable"],
                "mpf_type": ["MPF_F", None, None],
                "primary_replicon": ["IncF", None, "IncI"],
                "backbone_assignment_rule": ["primary_cluster_id", None, None],
                "backbone_seen_in_training": [True, False, None],
            }
        )

        result = validate_backbone_table(df)

        assert result["status"] == "pass"
        assert result["table"] == "backbone_table"
        assert result["n_rows"] == 3
        assert result["errors"] == []

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_scored_backbones_valid_data(self):
        """Valid scored backbone data should pass validation."""
        df = pd.DataFrame(
            {
                "backbone_id": ["BB001", "BB002", "BB003"],
                "spread_label": [1.0, 0.0, np.nan],
                "priority_index": [0.8, 0.3, 0.5],
                "bio_priority_index": [0.7, 0.4, 0.6],
                "T_eff_norm": [0.6, 0.2, 0.5],
                "H_eff_norm": [0.5, 0.3, 0.4],
                "A_eff_norm": [0.4, 0.1, 0.3],
                "log1p_member_count_train": [2.0, 1.0, 0.0],
                "log1p_n_countries_train": [1.5, 0.5, 0.0],
                "refseq_share_train": [0.8, 0.2, 0.5],
                "coherence_score": [0.9, 0.7, 0.8],
            }
        )

        result = validate_scored_backbones(df)

        assert result["status"] == "pass"
        assert result["table"] == "scored_backbones"
        assert result["n_rows"] == 3
        assert result["errors"] == []

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_deduplicated_plasmids_valid_data(self):
        """Valid deduplicated plasmid data should pass validation."""
        df = pd.DataFrame(
            {
                "sequence_accession": ["ACC001", "ACC002", "ACC003"],
                "backbone_id": ["BB001", "BB001", "BB002"],
                "is_canonical_representative": [True, False, True],
                "dedup_representative_group": ["G001", "G001", "G002"],
            }
        )

        result = validate_deduplicated_plasmids(df)

        assert result["status"] == "pass"
        assert result["table"] == "deduplicated_plasmids"
        assert result["n_rows"] == 3
        assert result["errors"] == []


class TestSchemaValidationFailureCases:
    """Test validation with invalid data (only runs if Pandera is available)."""

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_backbone_table_missing_required_column(self):
        """Missing required column should cause validation to fail."""
        df = pd.DataFrame(
            {
                # Missing "backbone_id" which is required
                "primary_cluster_id": ["C001", "C002"],
            }
        )

        result = validate_backbone_table(df)

        assert result["status"] in ("fail", "error")
        assert result["n_rows"] == 2
        # Should have error details
        assert len(result["errors"]) > 0

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_harmonized_plasmids_null_accession(self):
        """Null sequence_accession should fail validation."""
        df = pd.DataFrame(
            {
                "sequence_accession": [None, "ACC002"],
                "backbone_id": ["BB001", "BB002"],
            }
        )

        result = validate_harmonized_plasmids(df)

        assert result["status"] in ("fail", "error")

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_validate_scored_backbones_out_of_range(self):
        """Values outside [0, 1] for normalized columns should fail."""
        df = pd.DataFrame(
            {
                "backbone_id": ["BB001"],
                "priority_index": [1.5],  # Should be in [0, 1]
            }
        )

        result = validate_scored_backbones(df)

        assert result["status"] in ("fail", "error")


class TestRunAllValidations:
    """Test the batch validation runner."""

    @pytest.mark.skipif(not PANDERA_AVAILABLE, reason="Pandera not installed")
    def test_run_all_with_valid_data(self):
        """Running all validations with valid data should pass."""
        harmonized = pd.DataFrame(
            {
                "sequence_accession": ["ACC001"],
                "backbone_id": ["BB001"],
                "resolved_year": [2020.0],
            }
        )
        backbones = pd.DataFrame(
            {
                "backbone_id": ["BB001"],
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": ["BB001"],
                "spread_label": [1.0],
                "priority_index": [0.8],
            }
        )

        results = run_all_validations(
            harmonized=harmonized,
            backbones=backbones,
            scored=scored,
        )

        # Should have summary
        assert "_summary" in results
        summary = results["_summary"]
        assert summary["total_tables"] == 3
        assert summary["passed"] == 3
        assert summary["failed"] == 0
        assert summary["overall_status"] == "pass"


class TestFallbackBehaviorDocumentation:
    """Document and verify fallback behavior when Pandera is unavailable.

    These tests pass regardless of Pandera availability and document
    that the fallback behavior is explicit and clear.
    """

    def test_fallback_status_is_skipped(self):
        """When Pandera is unavailable, status should be 'skipped'."""
        df = pd.DataFrame({"backbone_id": ["BB001"]})
        result = validate_backbone_table(df)

        if not PANDERA_AVAILABLE:
            assert result["status"] == "skipped"
            assert result.get("reason") == "pandera_not_installed"

    def test_fallback_does_not_claim_validation_passed(self):
        """Fallback must not return 'pass' status when validation didn't run."""
        df = pd.DataFrame({"backbone_id": ["BB001"]})
        result = validate_backbone_table(df)

        if not PANDERA_AVAILABLE:
            # Critical: must NOT claim pass when we didn't actually validate
            assert result["status"] != "pass"
            # Must be explicit about why
            assert result["status"] == "skipped"

    def test_run_all_validations_skipped_summary(self):
        """Batch runner should report skipped correctly when Pandera unavailable."""
        df = pd.DataFrame({"backbone_id": ["BB001"]})
        results = run_all_validations(backbones=df)

        if not PANDERA_AVAILABLE:
            summary = results["_summary"]
            # All tables should be skipped, not passed
            assert summary["skipped"] == 1
            assert summary["passed"] == 0
            assert summary["failed"] == 0
            assert summary["overall_status"] == "skipped"

    def test_fallback_returns_empty_errors_list(self):
        """Fallback should return empty errors list (no validation = no errors)."""
        df = pd.DataFrame({"backbone_id": ["BB001"]})
        result = validate_backbone_table(df)

        if not PANDERA_AVAILABLE:
            assert result["errors"] == []
            assert result["n_rows"] == 1  # Still reports row count

    def test_explicit_fallback_documentation(self):
        """Document that fallback behavior is intentional and explicit."""
        # This test documents the design decision:
        # - When Pandera is unavailable, we return "skipped" status
        # - This is different from "pass" which would imply validation ran
        # - The behavior is honest: we report we didn't validate
        # - This keeps the repo usable in lightweight environments
        assert True  # Documentation test
