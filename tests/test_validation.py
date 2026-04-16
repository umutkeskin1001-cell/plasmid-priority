from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from plasmid_priority.validation import (
    average_precision,
    bootstrap_intervals,
    bootstrap_spearman_ci,
    brier_decomposition,
    calibration_curve_data,
    calibration_slope_intercept,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    ndcg_at_k,
    novelty_adjusted_average_precision,
    paired_bootstrap_delta,
    paired_bootstrap_deltas,
    permutation_pvalue,
    roc_auc_score,
    weighted_classification_cost,
)


class ValidationMetricTests(unittest.TestCase):
    def test_average_precision_is_tie_invariant_and_matches_sklearn(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        expected = average_precision_score(y_true, y_score)

        self.assertAlmostEqual(average_precision(y_true, y_score), expected)
        self.assertAlmostEqual(average_precision(y_true[::-1], y_score[::-1]), expected)

    def test_roc_auc_score_handles_tied_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 0.75)

    def test_expected_calibration_error_is_zero_for_perfect_bins(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(y_true, y_score, n_bins=2), 0.0)
        self.assertAlmostEqual(max_calibration_error(y_true, y_score, n_bins=2), 0.0)

    def test_calibration_curve_data_and_brier_decomposition_track_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        curve = calibration_curve_data(y_true, y_score, n_bins=2)
        decomposition = brier_decomposition(y_true, y_score, n_bins=2)

        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0]["n_obs"], 2)
        self.assertAlmostEqual(float(decomposition["reliability"]), 0.0)
        self.assertGreater(float(decomposition["resolution"]), 0.0)
        self.assertAlmostEqual(float(decomposition["uncertainty"]), 0.25)

    def test_calibration_slope_intercept_are_finite_for_nontrivial_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.75, 0.85, 0.3, 0.65, 0.25, 0.9])

        slope, intercept = calibration_slope_intercept(y_true, y_score)

        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(intercept))
        self.assertGreater(slope, 0.0)

    def test_log_loss_and_ndcg_reward_better_rankings(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        better = np.array([0.9, 0.2, 0.8, 0.1])
        worse = np.array([0.6, 0.5, 0.4, 0.3])

        self.assertLess(log_loss(y_true, better), log_loss(y_true, worse))
        self.assertGreater(ndcg_at_k(y_true, better, k=2), ndcg_at_k(y_true, worse, k=2))

    def test_novelty_adjusted_average_precision_rewards_low_knownness_hits(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        knownness = np.array([0.1, 0.9, 0.5, 0.5])

        weighted = novelty_adjusted_average_precision(y_true, y_score, knownness, gamma=2.0)
        unweighted = average_precision(y_true, y_score)
        self.assertGreater(weighted, 0.0)
        self.assertNotAlmostEqual(weighted, unweighted)

    def test_weighted_classification_cost_penalizes_false_negatives_more(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0, 0])
        better = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
        worse = np.array([0.7, 0.2, 0.9, 0.8, 0.1, 0.05])
        self.assertLess(
            weighted_classification_cost(y_true, better),
            weighted_classification_cost(y_true, worse),
        )

    def test_permutation_pvalue_is_small_for_nearly_perfect_ranking(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        p_value, null = permutation_pvalue(y_true, y_score, n_permutations=100, rng_seed=7)
        self.assertLess(p_value, 0.05)
        self.assertEqual(len(null), 100)

    def test_bootstrap_spearman_ci_returns_monotone_interval(self) -> None:
        x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        y = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        summary = bootstrap_spearman_ci(x, y, n_bootstrap=100, seed=11)
        self.assertGreater(summary["statistic"], 0.9)
        self.assertLessEqual(summary["lower"], summary["upper"])

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
