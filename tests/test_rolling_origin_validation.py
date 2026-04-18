from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.validation.rolling_origin import run_rolling_origin_validation


class RollingOriginValidationTests(unittest.TestCase):
    def test_rolling_origin_validation_computes_stability_metric(self) -> None:
        rows = []
        for split_year, auc in [(2012, 0.70), (2013, 0.80), (2014, 0.90)]:
            for idx in range(24):
                rows.append(
                    {
                        "split_year": split_year,
                        "test_year_end": split_year + 5,
                        "backbone_assignment_mode": "training_only",
                        "spread_label": 1 if idx % 2 == 0 else 0,
                        "backbone_id": f"{split_year}_{idx}",
                        "prediction": 0.5,
                        "roc_auc": auc,
                    }
                )
        scored = pd.DataFrame(rows)

        def _fake_model_evaluator(window: pd.DataFrame, _model_name: str) -> dict[str, float]:
            split_year = int(window["split_year"].iloc[0])
            auc = {2012: 0.70, 2013: 0.80, 2014: 0.90}[split_year]
            return {
                "status": "ok",
                "roc_auc": auc,
                "average_precision": auc - 0.1,
                "ece": 0.05,
            }

        report = run_rolling_origin_validation(
            scored,
            model_name="governance_linear",
            split_years=range(2012, 2015),
            horizon_years=5,
            model_evaluator=_fake_model_evaluator,
        )

        self.assertEqual(len(report.split_results), 3)
        self.assertTrue(np.isclose(report.mean_auc, np.mean([0.70, 0.80, 0.90])))
        self.assertTrue(
            np.isclose(
                report.auc_stability_metric,
                np.std([0.70, 0.80, 0.90]) / np.mean([0.70, 0.80, 0.90]),
            )
        )
        self.assertEqual(report.split_results[0].status, "ok")


if __name__ == "__main__":
    unittest.main()
