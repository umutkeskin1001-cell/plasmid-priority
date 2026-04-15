"""Tests for EPV audit functionality."""

from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.reporting.epv_audit import (
    build_epv_audit_table,
    compute_epv_for_model,
    summarize_epv_concerns,
)


class EPVAuditTests(unittest.TestCase):
    """Tests for EPV (events-per-variable) audit."""

    def test_compute_epv_returns_expected_keys(self) -> None:
        """EPV computation returns dictionary with expected keys."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C", "D"],
                "spread_label": [1, 0, 1, 0],
                "feature1": [0.1, 0.2, 0.3, 0.4],
            }
        )

        result = compute_epv_for_model(
            scored,
            "test_model",
            feature_set_override=["feature1", "feature2"],
        )

        self.assertEqual(result["model_name"], "test_model")
        self.assertEqual(result["n_features"], 2)
        self.assertEqual(result["n_positive_events"], 2)
        self.assertEqual(result["epv"], 1.0)  # 2 positives / 2 features
        self.assertEqual(result["epv_status"], "ok")

    def test_compute_epv_handles_no_spread_label(self) -> None:
        """EPV computation handles missing spread_label gracefully."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B"],
                "feature1": [0.1, 0.2],
            }
        )

        result = compute_epv_for_model(scored, "test_model")
        self.assertEqual(result["epv_status"], "missing_spread_label")

    def test_compute_epv_handles_no_positive_events(self) -> None:
        """EPV computation handles zero positive events."""
        scored = pd.DataFrame(
            {
                "backbone_id": ["A", "B", "C"],
                "spread_label": [0, 0, 0],
                "feature1": [0.1, 0.2, 0.3],
            }
        )

        result = compute_epv_for_model(
            scored,
            "test_model",
            feature_set_override=["feature1"],
        )

        self.assertEqual(result["n_positive_events"], 0)
        self.assertEqual(result["epv_status"], "no_positive_events")

    def test_epv_interpretation_very_low(self) -> None:
        """EPV < 5 gets very-high-risk interpretation."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(100)),
                "spread_label": [1] * 10 + [0] * 90,  # 10 positives
                "f1": range(100),
            }
        )

        result = compute_epv_for_model(
            scored,
            "test_model",
            feature_set_override=["f1", "f2", "f3"],  # 3 features -> EPV = 3.33
        )

        self.assertEqual(result["epv_interpretation"], "very_high_risk_of_overfitting")

    def test_epv_interpretation_good(self) -> None:
        """EPV > 20 gets 'good' interpretation."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(100)),
                "spread_label": [1] * 50 + [0] * 50,  # 50 positives
                "f1": range(100),
            }
        )

        result = compute_epv_for_model(
            scored,
            "test_model",
            feature_set_override=["f1", "f2"],  # 2 features -> EPV = 25
        )

        self.assertEqual(result["epv_interpretation"], "low_risk")

    def test_summarize_epv_concerns_counts_correctly(self) -> None:
        """EPV summary correctly counts concerns."""
        epv_table = pd.DataFrame(
            {
                "model_name": ["m1", "m2", "m3", "m4"],
                "epv_status": ["ok", "ok", "ok", "no_features"],
                "epv": [3.0, 8.0, 25.0, float("nan")],
                "epv_interpretation": ["very_low", "low", "good", "cannot_compute"],
            }
        )

        summary = summarize_epv_concerns(epv_table)

        self.assertEqual(summary["n_models_evaluated"], 3)  # Only 'ok' rows
        self.assertEqual(summary["n_models_low_epv"], 2)  # EPV < 10
        self.assertEqual(summary["n_models_very_low_epv"], 1)  # EPV < 5
        self.assertIn("m1", summary["models_requiring_review"])
        self.assertIn("m2", summary["models_requiring_review"])

    def test_build_epv_audit_table_returns_dataframe(self) -> None:
        """build_epv_audit_table returns a DataFrame."""
        scored = pd.DataFrame(
            {
                "backbone_id": list(range(20)),
                "spread_label": [1] * 5 + [0] * 15,
            }
        )

        result = build_epv_audit_table(
            scored,
            model_names=["baseline_both"],  # Use a known model if available
            include_official=False,
            include_challengers=False,
        )

        # May be empty if baseline_both not in MODULE_A_FEATURE_SETS
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
