from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


from plasmid_priority.validation import (
    average_precision,
    bootstrap_intervals,
    expected_calibration_error,
    paired_bootstrap_delta,
    paired_bootstrap_deltas,
    roc_auc_score,
)


class ValidationMetricTests(unittest.TestCase):
    def test_roc_auc_score_handles_tied_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 0.75)

    def test_expected_calibration_error_is_zero_for_perfect_bins(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(y_true, y_score, n_bins=2), 0.0)

    def test_paired_bootstrap_delta_returns_positive_gain_when_first_model_is_better(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        better = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        worse = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6])
        delta = paired_bootstrap_delta(y_true, better, worse, roc_auc_score, n_bootstrap=50, seed=1)
        self.assertGreater(delta["delta"], 0.0)

    def test_bootstrap_intervals_returns_requested_metric_keys(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.35, 0.65, 0.8, 0.9])
        intervals = bootstrap_intervals(
            y_true,
            y_score,
            {
                "roc_auc": roc_auc_score,
                "average_precision": average_precision,
            },
            n_bootstrap=40,
            seed=3,
        )
        self.assertEqual(set(intervals), {"roc_auc", "average_precision"})
        self.assertIn("lower", intervals["roc_auc"])
        self.assertIn("upper", intervals["average_precision"])

    def test_paired_bootstrap_deltas_returns_requested_metric_keys(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        better = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        worse = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6])
        deltas = paired_bootstrap_deltas(
            y_true,
            better,
            worse,
            {
                "roc_auc": roc_auc_score,
                "average_precision": average_precision,
            },
            n_bootstrap=50,
            seed=5,
        )
        self.assertEqual(set(deltas), {"roc_auc", "average_precision"})
        self.assertGreater(deltas["roc_auc"]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
