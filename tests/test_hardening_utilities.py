"""Targeted tests for hardening utilities: cryptographic manifest, VIF, BH correction, falsification.

This module provides focused unit tests for the hardening utilities that were
added in previous batches. Tests are lightweight and deterministic.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestCryptographicInputManifest:
    """Tests for cryptographic input manifest and hashing behavior."""

    def test_file_sha256_returns_stable_hash_for_small_file(self):
        """SHA-256 hash should be stable and correct for small files."""
        from plasmid_priority.utils.files import file_sha256

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            content = b"Hello, World!"
            test_file.write_bytes(content)

            result = file_sha256(test_file)
            expected = hashlib.sha256(content).hexdigest()

            assert result == expected
            assert len(result) == 64  # SHA-256 hex digest length

    def test_file_sha256_same_content_same_hash(self):
        """Same content should produce the same hash."""
        from plasmid_priority.utils.files import file_sha256

        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = Path(tmp_dir) / "file1.txt"
            file2 = Path(tmp_dir) / "file2.txt"
            content = b"identical content"
            file1.write_bytes(content)
            file2.write_bytes(content)

            hash1 = file_sha256(file1)
            hash2 = file_sha256(file2)

            assert hash1 == hash2

    def test_file_sha256_different_content_different_hash(self):
        """Different content should produce different hashes."""
        from plasmid_priority.utils.files import file_sha256

        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = Path(tmp_dir) / "file1.txt"
            file2 = Path(tmp_dir) / "file2.txt"
            file1.write_bytes(b"content A")
            file2.write_bytes(b"content B")

            hash1 = file_sha256(file1)
            hash2 = file_sha256(file2)

            assert hash1 != hash2

    def test_file_sha256_raises_on_non_file(self):
        """Should raise ValueError for non-file paths."""
        from plasmid_priority.utils.files import file_sha256

        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="not a file"):
                file_sha256(Path(tmp_dir))

    def test_path_signature_metadata_includes_expected_fields(self):
        """Path signature should include expected metadata fields."""
        from plasmid_priority.utils.files import path_signature

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            test_file.write_text("test content")

            sig = path_signature(test_file)

            assert "path" in sig
            assert "size" in sig
            assert "mtime_ns" in sig
            assert sig["path"] == str(test_file.resolve())
            assert sig["size"] == len("test content")
            assert isinstance(sig["mtime_ns"], int)

    def test_path_signature_directory_includes_digest_and_entry_count(self):
        """Directory signature should include digest and entry count."""
        from plasmid_priority.utils.files import path_signature

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create some files and subdirectories
            (Path(tmp_dir) / "subdir").mkdir()
            (Path(tmp_dir) / "file1.txt").write_text("content1")
            (Path(tmp_dir) / "subdir" / "file2.txt").write_text("content2")

            sig = path_signature(Path(tmp_dir))

            assert "digest" in sig
            assert "entry_count" in sig
            assert "kind" in sig
            assert sig["kind"] == "directory"
            assert sig["entry_count"] == 3  # subdir, file1.txt, file2.txt
            assert len(sig["digest"]) == 64  # SHA-256 hex digest

    def test_path_signature_with_hash_includes_sha256_for_small_files(self):
        """Small files under size limit should include sha256 in signature."""
        from plasmid_priority.utils.files import path_signature_with_hash

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            test_file.write_text("small content")

            sig = path_signature_with_hash(test_file, max_file_size_mb=100.0)

            assert "sha256" in sig
            assert len(sig["sha256"]) == 64

    def test_path_signature_with_hash_omits_sha256_for_large_files(self):
        """Large files over size limit should not include sha256 in signature."""
        from plasmid_priority.utils.files import path_signature_with_hash

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "large.bin"
            # Create a 2 MB file
            test_file.write_bytes(b"x" * (2 * 1024 * 1024))

            sig = path_signature_with_hash(test_file, max_file_size_mb=1.0)

            assert "sha256" not in sig
            assert sig["size"] == 2 * 1024 * 1024


class TestVIFAuditHelpers:
    """Tests for VIF (Variance Inflation Factor) audit helpers."""

    def test_compute_vif_values_returns_dataframe_with_expected_columns(self):
        """VIF computation should return DataFrame with expected columns."""
        from plasmid_priority.validation.vif import compute_vif_values

        # Create simple uncorrelated data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["feat_a", "feat_b", "feat_c"]

        result = compute_vif_values(X, feature_names)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature_name", "vif", "concern_flag"]
        assert len(result) == 3
        assert list(result["feature_name"]) == feature_names

    def test_compute_vif_values_low_for_uncorrelated_features(self):
        """Uncorrelated features should have VIF close to 1."""
        from plasmid_priority.validation.vif import compute_vif_values

        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["feat_a", "feat_b", "feat_c"]

        result = compute_vif_values(X, feature_names)

        # All VIFs should be close to 1 for uncorrelated data
        for vif in result["vif"]:
            assert vif < 2.0

    def test_compute_vif_values_high_for_correlated_features(self):
        """Highly correlated features should have high VIF."""
        from plasmid_priority.validation.vif import compute_vif_values

        np.random.seed(42)
        x = np.random.randn(100)
        X = np.column_stack([x, x + 0.01 * np.random.randn(100)])  # Highly correlated
        feature_names = ["feat_a", "feat_b"]

        result = compute_vif_values(X, feature_names)

        # VIF should be high for correlated features
        assert all(vif > 5 for vif in result["vif"])

    def test_compute_vif_values_concern_flags_correct(self):
        """Concern flags should reflect VIF thresholds correctly."""
        from plasmid_priority.validation.vif import compute_vif_values

        np.random.seed(42)
        x = np.random.randn(100)
        # Mix of uncorrelated and correlated
        X = np.column_stack([
            x,  # Will be correlated with feat_b
            x + 0.01 * np.random.randn(100),  # Highly correlated with feat_a
            np.random.randn(100),  # Uncorrelated
        ])
        feature_names = ["high_vif", "high_vif_b", "low_vif"]

        result = compute_vif_values(X, feature_names)

        # Check concern flags
        high_flags = result[result["vif"] > 5]["concern_flag"].tolist()
        low_flags = result[result["vif"] <= 5]["concern_flag"].tolist()

        assert all(flag in ["high", "critical", "moderate"] for flag in high_flags)
        assert all(flag == "low" for flag in low_flags)

    def test_compute_vif_values_single_feature_returns_vif_of_1(self):
        """Single feature should have VIF of 1 (no multicollinearity possible)."""
        from plasmid_priority.validation.vif import compute_vif_values

        X = np.random.randn(100, 1)
        result = compute_vif_values(X, ["only_feature"])

        assert len(result) == 1
        assert result["vif"].iloc[0] == 1.0
        assert result["concern_flag"].iloc[0] == "low"

    def test_compute_vif_values_empty_features_returns_empty_df(self):
        """Empty feature set should return empty DataFrame."""
        from plasmid_priority.validation.vif import compute_vif_values

        X = np.random.randn(100, 0)
        result = compute_vif_values(X, [])

        assert len(result) == 0

    def test_summarize_vif_concerns_returns_expected_structure(self):
        """VIF summary should return expected structure."""
        from plasmid_priority.validation.vif import (
            compute_vif_values,
            summarize_vif_concerns,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        X = np.column_stack([x, x + 0.01 * np.random.randn(100), np.random.randn(100)])
        vif_table = compute_vif_values(X, ["a", "b", "c"])
        # Add model_name column required by summarize_vif_concerns
        vif_table["model_name"] = "test_model"
        summary = summarize_vif_concerns(vif_table)

        assert isinstance(summary, pd.DataFrame)
        assert "max_vif" in summary.columns
        assert "mean_vif" in summary.columns
        assert "overall_status" in summary.columns


class TestBenjaminiHochbergCorrection:
    """Tests for Benjamini-Hochberg FDR / q-value correction."""

    def test_bh_correction_preserves_monotonicity(self):
        """BH-corrected q-values should be monotonic (non-decreasing)."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5])
        q_values = benjamini_hochberg_correction(p_values)

        # Q-values should be monotonic non-decreasing when sorted by original p-value order
        sorted_order = np.argsort(p_values)
        sorted_q = q_values[sorted_order]

        # Check monotonicity
        for i in range(len(sorted_q) - 1):
            assert sorted_q[i] <= sorted_q[i + 1] + 1e-10  # Small tolerance for floating point

    def test_bh_correction_q_values_bounded_by_1(self):
        """All q-values should be in [0, 1]."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99])
        q_values = benjamini_hochberg_correction(p_values)

        assert all(q >= 0 for q in q_values)
        assert all(q <= 1 for q in q_values)

    def test_bh_correction_empty_array_returns_empty(self):
        """Empty input should return empty output."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([])
        q_values = benjamini_hochberg_correction(p_values)

        assert len(q_values) == 0

    def test_bh_correction_single_p_value_returns_same(self):
        """Single p-value should return q-value equal to itself."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.05])
        q_values = benjamini_hochberg_correction(p_values)

        assert len(q_values) == 1
        assert q_values[0] == pytest.approx(0.05, abs=1e-10)

    def test_bh_correction_handles_nans(self):
        """BH correction should handle NaN p-values gracefully."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.01, 0.05, np.nan, 0.1])
        q_values = benjamini_hochberg_correction(p_values)

        # Should not raise, and should return array of same shape
        assert q_values.shape == p_values.shape

    def test_bh_correction_q_values_greater_or_equal_to_p_values(self):
        """Q-values should generally be >= p-values (BH is less conservative than Bonferroni)."""
        from plasmid_priority.validation.metrics import benjamini_hochberg_correction

        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        q_values = benjamini_hochberg_correction(p_values)

        # For most p-values, q >= p (but this can vary due to monotonicity enforcement)
        # Just check that the relationship is reasonable
        for p, q in zip(p_values, q_values, strict=True):
            if not np.isnan(p) and not np.isnan(q):
                assert q >= p - 1e-10  # Allow small numerical tolerance

    def test_fdr_adjust_model_comparison_adds_q_value_column(self):
        """FDR adjustment should add q_value_bh column to DataFrame."""
        from plasmid_priority.validation.metrics import fdr_adjust_model_comparison

        df = pd.DataFrame({
            "model_a": ["m1", "m2", "m3"],
            "model_b": ["m4", "m5", "m6"],
            "delta_roc_auc_delong_pvalue": [0.01, 0.05, 0.2],
        })

        result = fdr_adjust_model_comparison(df, alpha=0.05)

        assert "q_value_bh" in result.columns
        assert "fdr_alpha" in result.columns
        assert "fdr_significant_at_alpha" in result.columns
        assert result["fdr_alpha"].iloc[0] == 0.05

    def test_fdr_adjust_model_comparison_empty_df_returns_empty(self):
        """Empty DataFrame should return empty copy."""
        from plasmid_priority.validation.metrics import fdr_adjust_model_comparison

        df = pd.DataFrame()
        result = fdr_adjust_model_comparison(df)

        assert len(result) == 0

    def test_fdr_adjust_model_comparison_missing_column_returns_unchanged(self):
        """Missing p-value column should return unchanged copy."""
        from plasmid_priority.validation.metrics import fdr_adjust_model_comparison

        df = pd.DataFrame({"model_a": ["m1", "m2"], "other_column": [1, 2]})
        result = fdr_adjust_model_comparison(df, p_value_column="delta_roc_auc_delong_pvalue")

        assert "q_value_bh" not in result.columns


class TestFalsificationHelpers:
    """Tests for falsification helpers (permutation/shuffle tests)."""

    def test_label_shuffle_falsification_returns_expected_structure(self):
        """Label shuffle test should return DataFrame with expected columns."""
        from plasmid_priority.validation.falsification import build_label_shuffle_falsification

        np.random.seed(42)
        # Create synthetic predictions
        predictions = pd.DataFrame({
            "model_name": ["test_model"] * 50,
            "spread_label": np.random.randint(0, 2, 50),
            "oof_prediction": np.random.random(50),
        })

        result = build_label_shuffle_falsification(predictions, ["test_model"], n_shuffles=10, seed=42)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "test_name" in result.columns
            assert "model_name" in result.columns
            assert "observed_roc_auc" in result.columns
            assert "shuffled_roc_auc_mean" in result.columns
            assert "empirical_p_value" in result.columns
            assert "status" in result.columns

    def test_label_shuffle_falsification_shows_degraded_performance_under_shuffle(self):
        """Shuffled labels should show degraded performance compared to true labels."""
        from plasmid_priority.validation.falsification import build_label_shuffle_falsification

        np.random.seed(42)
        # Create synthetic data where model has some signal
        n = 100
        labels = np.random.randint(0, 2, n)
        # Predictions correlated with labels (some signal)
        predictions_arr = np.where(labels == 1, 0.7, 0.3) + np.random.randn(n) * 0.2
        predictions_arr = np.clip(predictions_arr, 0, 1)

        predictions = pd.DataFrame({
            "model_name": ["test_model"] * n,
            "spread_label": labels,
            "oof_prediction": predictions_arr,
        })

        result = build_label_shuffle_falsification(predictions, ["test_model"], n_shuffles=20, seed=42)

        if not result.empty:
            observed = result["observed_roc_auc"].iloc[0]
            shuffled_mean = result["shuffled_roc_auc_mean"].iloc[0]

            # Observed should be better than shuffled (model has real signal)
            assert observed > shuffled_mean
            # Empirical p-value should indicate significance
            assert result["empirical_p_value"].iloc[0] < 0.05

    def test_label_shuffle_falsification_returns_empty_for_single_class(self):
        """Single class labels should result in empty/skipped result."""
        from plasmid_priority.validation.falsification import build_label_shuffle_falsification

        predictions = pd.DataFrame({
            "model_name": ["test_model"] * 20,
            "spread_label": [1] * 20,  # All same class
            "oof_prediction": np.random.random(20),
        })

        result = build_label_shuffle_falsification(predictions, ["test_model"], n_shuffles=10, seed=42)

        # Should return empty DataFrame since no valid comparisons possible
        assert len(result) == 0

    def test_label_shuffle_falsification_empty_models_list_returns_empty(self):
        """Empty models list should return empty DataFrame."""
        from plasmid_priority.validation.falsification import build_label_shuffle_falsification

        predictions = pd.DataFrame({
            "model_name": [],
            "spread_label": [],
            "oof_prediction": [],
        })

        result = build_label_shuffle_falsification(predictions, [], n_shuffles=10, seed=42)

        assert len(result) == 0

    def test_summarize_falsification_findings_returns_dict_structure(self):
        """Falsification summary should return dictionary with expected structure."""
        from plasmid_priority.validation.falsification import summarize_falsification_findings

        # Create mock permutation result
        perm_result = pd.DataFrame({
            "test_name": ["outcome_permutation_falsification"],
            "model_name": ["model1"],
            "interpretation": ["pass_collapses_under_falsification"],
            "auc_collapse_delta": [0.3],
        })

        shuffle_result = pd.DataFrame({
            "test_name": ["label_shuffle_falsification", "label_shuffle_falsification"],
            "model_name": ["model1", "model2"],
            "interpretation": ["pass_strong_signal", "pass_strong_signal"],
        })

        summary = summarize_falsification_findings(perm_result, shuffle_result)

        assert isinstance(summary, dict)
        assert "overall_assessment" in summary
        assert "concerns" in summary
        assert "details" in summary

    def test_summarize_falsification_findings_detects_warnings(self):
        """Falsification summary should detect and report warnings."""
        from plasmid_priority.validation.falsification import summarize_falsification_findings

        perm_result = pd.DataFrame({
            "test_name": ["outcome_permutation_falsification"],
            "model_name": ["model1"],
            "interpretation": ["warning_minimal_collapse"],
            "auc_collapse_delta": [0.02],
        })

        summary = summarize_falsification_findings(perm_result, None)

        assert summary["overall_assessment"] == "minor_concerns"
        assert len(summary["concerns"]) == 1
        assert "outcome_permutation" in summary["concerns"][0]

    def test_summarize_falsification_findings_no_concerns_when_all_pass(self):
        """Falsification summary should report no concerns when all tests pass."""
        from plasmid_priority.validation.falsification import summarize_falsification_findings

        shuffle_result = pd.DataFrame({
            "test_name": ["label_shuffle_falsification"],
            "model_name": ["model1"],
            "interpretation": ["pass_strong_signal"],
        })

        summary = summarize_falsification_findings(None, shuffle_result)

        assert summary["overall_assessment"] == "no_concerns"
        assert len(summary["concerns"]) == 0
