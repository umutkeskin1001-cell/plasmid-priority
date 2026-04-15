"""Tests for lead-time bias audit functionality."""

from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.reporting.lead_time_bias_audit import (
    build_lead_time_bias_audit,
    build_visibility_decile_table,
    compute_lead_time_bias_metrics,
    summarize_lead_time_bias_findings,
)


class LeadTimeBiasAuditTests(unittest.TestCase):
    """Tests for lead-time bias audit."""

    def test_compute_metrics_returns_expected_keys(self) -> None:
        """Lead-time bias metrics returns dictionary with expected keys."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(20)),
                "spread_label": [1] * 5 + [0] * 15,
                "log1p_member_count_train": list(range(20)),
            }
        )

        result = compute_lead_time_bias_metrics(scored)

        self.assertEqual(result["visibility_column"], "log1p_member_count_train")
        self.assertEqual(result["outcome_column"], "spread_label")
        self.assertEqual(result["status"], "ok")
        self.assertIn("visibility_outcome_spearman", result)
        self.assertIn("lead_time_bias_concern", result)

    def test_compute_metrics_handles_missing_spread_label(self) -> None:
        """Handles missing spread_label gracefully."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "log1p_member_count_train": [1.0, 2.0],
            }
        )

        result = compute_lead_time_bias_metrics(scored)
        self.assertEqual(result["status"], "missing_outcome_column")

    def test_compute_metrics_handles_insufficient_data(self) -> None:
        """Handles insufficient data gracefully."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "spread_label": [1, 0],
                "log1p_member_count_train": [1.0, 2.0],
            }
        )

        result = compute_lead_time_bias_metrics(scored)
        self.assertEqual(result["status"], "insufficient_data")

    def test_high_correlation_triggers_high_concern(self) -> None:
        """Spearman > 0.3 triggers high concern."""
        # Create data with strong positive correlation
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(100)),
                "log1p_member_count_train": list(range(100)),  # 0-99
                "spread_label": [1 if i > 70 else 0 for i in range(100)],  # spread at high counts
            }
        )

        result = compute_lead_time_bias_metrics(scored)

        # Should have high correlation and high concern
        self.assertIsNotNone(result["visibility_outcome_spearman"])
        if result["visibility_outcome_spearman"] is not None:
            self.assertGreater(result["visibility_outcome_spearman"], 0.3)
            self.assertEqual(result["lead_time_bias_concern"], "high")

    def test_build_decile_table_returns_expected_structure(self) -> None:
        """Decile table has expected structure."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(100)),
                "spread_label": [1] * 30 + [0] * 70,
                "log1p_member_count_train": list(range(100)),
            }
        )

        result = build_visibility_decile_table(scored)

        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertIn("quantile_bin", result.columns)
            self.assertIn("n_backbones", result.columns)
            self.assertIn("spread_rate", result.columns)

    def test_build_audit_returns_two_dataframes(self) -> None:
        """build_lead_time_bias_audit returns summary and decile tables."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(50)),
                "spread_label": [1] * 15 + [0] * 35,
                "log1p_member_count_train": list(range(50)),
                "log1p_n_countries_train": list(range(50)),
            }
        )

        summary, deciles = build_lead_time_bias_audit(scored)

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIsInstance(deciles, pd.DataFrame)

    def test_summarize_handles_empty_table(self) -> None:
        """Summary handles empty input gracefully."""
        empty_summary = pd.DataFrame()
        empty_deciles = pd.DataFrame()

        result = summarize_lead_time_bias_findings(empty_summary, empty_deciles)

        self.assertEqual(result["overall_concern_level"], "unknown")

    def test_summarize_counts_concerns_correctly(self) -> None:
        """Summary correctly counts concern levels."""
        summary_table = pd.DataFrame(
            {
                "visibility_column": ["v1", "v2", "v3"],
                "status": ["ok", "ok", "ok"],
                "lead_time_bias_concern": ["high", "moderate", "low"],
            }
        )
        deciles = pd.DataFrame()  # Empty is fine for this test

        result = summarize_lead_time_bias_findings(summary_table, deciles)

        self.assertEqual(result["overall_concern_level"], "high")
        self.assertEqual(result["n_metrics_evaluated"], 3)
        self.assertEqual(result["n_high_concern"], 1)
        self.assertEqual(result["n_moderate_concern"], 1)

    def test_build_decile_with_missing_visibility(self) -> None:
        """Handles missing visibility column."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "spread_label": [1, 0],
            }
        )

        result = build_visibility_decile_table(scored, visibility_column="nonexistent")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
