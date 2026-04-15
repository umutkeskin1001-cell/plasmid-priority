from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.geo_spread import build_geo_spread_report_card, format_geo_spread_report_markdown
from plasmid_priority.modeling.module_a import ModelResult


class GeoSpreadReportTests(unittest.TestCase):
    def test_build_geo_spread_report_card_and_markdown(self) -> None:
        results = {
            "geo_counts_baseline": ModelResult(
                name="geo_counts_baseline",
                metrics={
                    "roc_auc": 0.84,
                    "average_precision": 0.61,
                    "brier_score": 0.19,
                    "expected_calibration_error": 0.05,
                    "precision_at_top_10": 0.7,
                    "recall_at_top_10": 0.6,
                    "decision_utility_score": 0.12,
                    "n_backbones": 12,
                    "n_positive": 6,
                },
                predictions=pd.DataFrame({"backbone_id": ["bb1"], "oof_prediction": [0.8]}),
            ),
            "geo_parsimonious_priority": ModelResult(
                name="geo_parsimonious_priority",
                metrics={
                    "roc_auc": 0.91,
                    "average_precision": 0.73,
                    "brier_score": 0.14,
                    "expected_calibration_error": 0.03,
                    "precision_at_top_10": 0.8,
                    "recall_at_top_10": 0.7,
                    "decision_utility_score": 0.22,
                    "n_backbones": 12,
                    "n_positive": 6,
                },
                predictions=pd.DataFrame({"backbone_id": ["bb1"], "oof_prediction": [0.9]}),
            ),
        }
        calibration_summary = pd.DataFrame(
            {
                "model_name": ["geo_counts_baseline", "geo_parsimonious_priority"],
                "calibration_method": ["none", "isotonic"],
                "abstain_rate": [0.1, 0.2],
                "mean_confidence": [0.8, 0.7],
                "ood_rate": [0.0, 0.05],
                "calibrated_expected_calibration_error": [0.04, 0.02],
                "calibrated_brier_score": [0.18, 0.12],
            }
        )
        provenance = {
            "benchmark_name": "geo_spread_v1",
            "split_year": 2015,
            "run_signature": "abc123",
            "primary_model_name": "geo_parsimonious_priority",
            "config_hash": "cfg",
            "input_hash": "inp",
            "feature_surface_hash": "feat",
        }
        report = build_geo_spread_report_card(
            results,
            calibration_summary=calibration_summary,
            provenance=provenance,
        )
        markdown = format_geo_spread_report_markdown(report, provenance=provenance)

        self.assertEqual(report.iloc[0]["model_name"], "geo_parsimonious_priority")
        self.assertIn("run_signature", report.columns)
        self.assertIn("Geo spread report", markdown)
        self.assertIn("geo_parsimonious_priority", markdown)
        self.assertIn("best_predictive_model", markdown)


if __name__ == "__main__":
    unittest.main()
