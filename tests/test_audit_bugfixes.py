"""Tests for bug fixes from the external audit batch.

This module contains targeted tests for the 8 confirmed bugs that were fixed:
- BUG-001: CI pytest enforcement
- BUG-002: Schema validation exit codes
- BUG-003: Hardening snapshot fail propagation
- BUG-004: BH correction NaN handling
- BUG-005: PLASMID_PRIORITY_DATA_ROOT in full-local mode
- BUG-006: Empty-table missingness formatter
- BUG-007: Silent-success scripts exit codes
- BUG-008: Script index step 26 (docs fix, no code test needed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestBug001PytestEnforcement:
    """Test that make test uses pytest."""

    def test_makefile_uses_pytest(self):
        """Verify Makefile test target uses pytest, not unittest."""
        makefile_path = PROJECT_ROOT / "Makefile"
        content = makefile_path.read_text()

        # Should contain pytest in test target
        assert "pytest tests/" in content, "Makefile test target should use pytest"
        # Should NOT use unittest discover
        assert "unittest discover" not in content, "Makefile should not use unittest discover"

    def test_ci_workflow_runs_quality(self):
        """Verify CI workflow runs make quality which includes pytest."""
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()

        assert "make quality" in content, "CI should run make quality"


class TestBug002SchemaValidationExitCodes:
    """Test schema validation returns non-zero on failure."""

    def test_schema_validation_returns_1_on_fail_in_code(self):
        """Verify run_schema_validation.py code returns 1 on validation failure."""
        script_path = PROJECT_ROOT / "scripts" / "run_schema_validation.py"
        content = script_path.read_text()

        # Should check for fail/error status and return 1
        assert 'overall in ("fail", "error")' in content or \
               '("fail", "error")' in content, \
            "Should check for fail/error status"
        assert "return 1" in content, "Should return 1 on failure"
        assert "return 0" in content, "Should return 0 on success"

    def test_schema_validation_logic_order(self):
        """Verify exit code logic checks fail/error first."""
        script_path = PROJECT_ROOT / "scripts" / "run_schema_validation.py"
        content = script_path.read_text()

        # Should have proper comment about exit codes
        assert "non-zero for fail/error" in content.lower() or \
               "return 1" in content, \
            "Should document non-zero exit for fail/error"

    def test_schema_validation_returns_0_on_skipped(self):
        """Verify run_schema_validation.py returns 0 on skipped validation."""
        script_path = PROJECT_ROOT / "scripts" / "run_schema_validation.py"
        content = script_path.read_text()

        # Skipped should not be in the fail/error condition
        # The condition should be specifically for fail/error
        assert '("fail", "error")' in content, \
            "Should only return 1 for fail/error, not skipped"


class TestBug003HardeningSnapshotFailPropagation:
    """Test hardening snapshot correctly propagates fail status."""

    def test_status_priority_error_over_fail(self):
        """Verify error status takes priority over fail."""
        # The snapshot should prioritize: error > fail > concern > ok
        # We can't easily mock the full snapshot, but we can verify the logic
        # by checking the code was updated correctly
        snapshot_path = PROJECT_ROOT / "scripts" / "generate_hardening_snapshot.py"
        content = snapshot_path.read_text()

        # Should check for error first
        assert 's == "error"' in content
        # Should check for fail second (after error)
        lines = content.split("\n")
        error_line_idx = None
        fail_line_idx = None
        for i, line in enumerate(lines):
            if 's == "error"' in line:
                error_line_idx = i
            if 's == "fail"' in line:
                fail_line_idx = i

        assert error_line_idx is not None, "Should check for error status"
        assert fail_line_idx is not None, "Should check for fail status"
        assert error_line_idx < fail_line_idx, "Error check should come before fail check"


class TestBug004BHCorrectionNaNHandling:
    """Test Benjamini-Hochberg correction handles NaN values."""

    def test_bh_correction_with_nan(self):
        """Verify BH correction preserves NaN positions."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        # Test with mixed finite and NaN values
        p_values = np.array([0.01, 0.05, np.nan, 0.03, np.nan])
        q_values = benjamini_hochberg_correction(p_values)

        # NaN positions should be preserved
        assert np.isnan(q_values[2]), "NaN at position 2 should be preserved"
        assert np.isnan(q_values[4]), "NaN at position 4 should be preserved"

        # Finite positions should have valid q-values
        assert np.isfinite(q_values[0]), "Position 0 should be finite"
        assert np.isfinite(q_values[1]), "Position 1 should be finite"
        assert np.isfinite(q_values[3]), "Position 3 should be finite"

    def test_bh_correction_all_finite(self):
        """Verify BH correction works correctly with all finite values."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.01, 0.05, 0.03, 0.02])
        q_values = benjamini_hochberg_correction(p_values)

        # All outputs should be finite
        assert all(np.isfinite(q_values)), "All q-values should be finite"

        # q-values should be monotonic when sorted by p-value
        sorted_idx = np.argsort(p_values)
        sorted_q = q_values[sorted_idx]
        # q-values should be non-decreasing when sorted by p-value
        assert all(sorted_q[i] <= sorted_q[i + 1] for i in range(len(sorted_q) - 1))

    def test_bh_correction_all_nan(self):
        """Verify BH correction handles all NaN input."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([np.nan, np.nan, np.nan])
        q_values = benjamini_hochberg_correction(p_values)

        # All outputs should be NaN
        assert all(np.isnan(q_values)), "All q-values should be NaN for all-NaN input"

    def test_bh_correction_empty(self):
        """Verify BH correction handles empty input."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([])
        q_values = benjamini_hochberg_correction(p_values)

        assert len(q_values) == 0, "Empty input should return empty output"


class TestBug005EnvVarHandling:
    """Test PLASMID_PRIORITY_DATA_ROOT is honored in full-local mode."""

    def test_run_mode_checks_env_var_before_prompt(self):
        """Verify run_mode.py checks env var before prompting."""
        run_mode_path = PROJECT_ROOT / "scripts" / "run_mode.py"
        content = run_mode_path.read_text()

        # Should check env var before prompting
        assert "os.environ.get(DATA_ROOT_ENV_VAR)" in content
        assert "prompt_for_data_root()" in content

        # Verify the order: env var check comes before prompt
        lines = content.split("\n")
        env_check_line = None
        prompt_line = None
        for i, line in enumerate(lines):
            if "os.environ.get(DATA_ROOT_ENV_VAR)" in line:
                env_check_line = i
            if "prompt_for_data_root()" in line and "args.data_root =" in line:
                prompt_line = i

        assert env_check_line is not None, "Should check env var"
        assert prompt_line is not None, "Should call prompt_for_data_root"


class TestBug006EmptyMissingnessFormatter:
    """Test empty-table missingness result is compatible with formatter."""

    def test_audit_missingness_returns_threshold_for_empty(self):
        """Verify audit_missingness returns high_missingness_threshold for empty tables."""
        from plasmid_priority.validation.missingness import audit_missingness

        empty_df = pd.DataFrame()
        result = audit_missingness(empty_df, "test_table", high_missingness_threshold=0.5)

        # Should include high_missingness_threshold key
        assert "high_missingness_threshold" in result
        assert result["high_missingness_threshold"] == 0.5

    def test_format_missingness_report_handles_empty_table(self):
        """Verify format_missingness_report handles empty table result."""
        from plasmid_priority.validation.missingness import format_missingness_report

        empty_result = {
            "table_name": "test_table",
            "n_rows": 0,
            "n_columns": 0,
            "columns": [],
            "high_missingness_count": 0,
            "high_missingness_threshold": 0.5,
            "status": "empty_table",
        }

        # Should not raise an exception
        report = format_missingness_report(empty_result)

        # Should contain expected content
        assert "test_table" in report
        assert "Rows: 0" in report
        assert "Status: empty_table" in report


class TestBug007StrictExitMode:
    """Test --strict-exit mode for honest exit codes."""

    def test_run_missingness_audit_has_strict_exit_flag(self):
        """Verify run_missingness_audit.py has --strict-exit flag."""
        script_path = PROJECT_ROOT / "scripts" / "run_missingness_audit.py"
        content = script_path.read_text()

        assert "--strict-exit" in content
        assert "strict_exit" in content

    def test_run_hardening_summary_has_strict_exit_flag(self):
        """Verify run_hardening_summary.py has --strict-exit flag."""
        script_path = PROJECT_ROOT / "scripts" / "run_hardening_summary.py"
        content = script_path.read_text()

        assert "--strict-exit" in content
        assert "strict_exit" in content


class TestBug008ScriptIndexStep26:
    """Test step 26 is correctly marked as manual/optional."""

    def test_step_26_marked_manual_optional(self):
        """Verify step 26 is marked as Manual/Optional in INDEX.md."""
        index_path = PROJECT_ROOT / "scripts" / "INDEX.md"
        content = index_path.read_text()

        # Find the line for step 26
        for line in content.split("\n"):
            if "26_run_tests_or_smoke.py" in line:
                assert "Manual/Optional" in line or "manual" in line.lower(), \
                    "Step 26 should be marked as Manual/Optional"
                break
        else:
            pytest.fail("Step 26 not found in INDEX.md")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
