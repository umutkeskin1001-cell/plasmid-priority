from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.geo_spread import (
    build_geo_spread_calibrated_prediction_table,
    build_geo_spread_calibration_summary,
    calibrate_geo_spread_predictions,
)


class GeoSpreadCalibrationTests(unittest.TestCase):
    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "oof_prediction": [0.05, 0.15, 0.82, 0.93],
                "spread_label": [0, 0, 1, 1],
            }
        )

    def _scored(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "knownness_score": [0.9, 0.8, 0.4, 0.2],
            }
        )

    def test_calibrate_geo_spread_predictions_adds_uncertainty_surface(self) -> None:
        result = calibrate_geo_spread_predictions(
            self._predictions(),
            model_name="geo_parsimonious_priority",
            fit_config={"calibration": "isotonic"},
            scored=self._scored(),
        )
        self.assertEqual(result.model_name, "geo_parsimonious_priority")
        self.assertEqual(result.calibration_method, "isotonic")
        self.assertIn("calibrated_prediction", result.predictions.columns)
        self.assertIn("confidence_band", result.predictions.columns)
        self.assertIn("abstain_or_review_flag", result.predictions.columns)
        self.assertEqual(len(result.predictions), 4)
        self.assertIn("calibrated_expected_calibration_error", result.summary)

    def test_build_calibration_tables_are_deterministic(self) -> None:
        results = {"geo_parsimonious_priority": self._predictions()}
        summary_1 = build_geo_spread_calibration_summary(
            results,
            scored=self._scored(),
            config={
                "geo_spread": {
                    "fit_config": {
                        "geo_parsimonious_priority": {"calibration": "isotonic"}
                    }
                }
            },
        )
        summary_2 = build_geo_spread_calibration_summary(
            results,
            scored=self._scored(),
            config={
                "geo_spread": {
                    "fit_config": {
                        "geo_parsimonious_priority": {"calibration": "isotonic"}
                    }
                }
            },
        )
        table = build_geo_spread_calibrated_prediction_table(
            results,
            scored=self._scored(),
            config={
                "geo_spread": {
                    "fit_config": {
                        "geo_parsimonious_priority": {"calibration": "isotonic"}
                    }
                }
            },
        )
        self.assertTrue(summary_1.equals(summary_2))
        self.assertIn("priority_score", table.columns)
        self.assertEqual(set(table["model_name"]), {"geo_parsimonious_priority"})


if __name__ == "__main__":
    unittest.main()
