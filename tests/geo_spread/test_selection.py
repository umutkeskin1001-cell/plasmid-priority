from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.geo_spread import (
    build_geo_spread_adaptive_result,
    build_geo_spread_blended_result,
    build_geo_spread_meta_result,
    select_geo_spread_primary_model,
)
from plasmid_priority.modeling.module_a import ModelResult


class GeoSpreadSelectionTests(unittest.TestCase):
    def _result(
        self, name: str, scores: list[float], labels: list[int], auc: float, ap: float
    ) -> ModelResult:
        return ModelResult(
            name=name,
            metrics={
                "roc_auc": auc,
                "average_precision": ap,
                "decision_utility_score": 0.2,
                "brier_score": 0.18,
                "expected_calibration_error": 0.04,
                "novelty_adjusted_average_precision": 0.1,
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": [f"bb{i}" for i in range(len(scores))],
                    "oof_prediction": scores,
                    "spread_label": labels,
                }
            ),
        )

    def test_build_geo_spread_blended_result_returns_model_result(self) -> None:
        labels = [0, 0, 1, 1]
        results = {
            "geo_context_hybrid_priority": self._result(
                "geo_context_hybrid_priority",
                [0.1, 0.2, 0.8, 0.9],
                labels,
                0.8,
                0.7,
            ),
            "geo_phylo_ecology_priority": self._result(
                "geo_phylo_ecology_priority",
                [0.15, 0.25, 0.75, 0.95],
                labels,
                0.79,
                0.69,
            ),
        }
        blend = build_geo_spread_blended_result(results, include_ci=False)
        self.assertEqual(blend.status, "ok")
        self.assertEqual(blend.name, "geo_reliability_blend")
        self.assertIn("roc_auc", blend.metrics)
        self.assertEqual(len(blend.predictions), 4)

    def test_build_geo_spread_adaptive_result_returns_model_result(self) -> None:
        labels = [0, 0, 1, 1]
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(4)],
                "spread_label": labels,
                "n_new_countries": [1, 1, 4, 4],
                "split_year": [2015] * 4,
                "backbone_assignment_mode": ["training_only"] * 4,
                "max_resolved_year_train": [2014] * 4,
                "min_resolved_year_test": [2017] * 4,
                "training_only_future_unseen_backbone_flag": [False] * 4,
                "log1p_member_count_train": [0.1, 0.2, 1.0, 1.2],
                "log1p_n_countries_train": [0.1, 0.2, 0.8, 1.0],
                "refseq_share_train": [0.2, 0.2, 0.8, 0.9],
            }
        )
        results = {
            "geo_context_hybrid_priority": self._result(
                "geo_context_hybrid_priority",
                [0.1, 0.2, 0.8, 0.9],
                labels,
                0.8,
                0.7,
            ),
            "geo_phylo_ecology_priority": self._result(
                "geo_phylo_ecology_priority",
                [0.15, 0.25, 0.75, 0.95],
                labels,
                0.79,
                0.69,
            ),
        }
        results["geo_reliability_blend"] = build_geo_spread_blended_result(
            results, include_ci=False
        )
        adaptive = build_geo_spread_adaptive_result(results, scored=scored, include_ci=False)
        self.assertEqual(adaptive.status, "ok")
        self.assertEqual(adaptive.name, "geo_adaptive_knownness_priority")
        self.assertIn("knownness_threshold", adaptive.metrics)
        self.assertIn("adaptive_support_model_weight_mean", adaptive.metrics)
        self.assertEqual(len(adaptive.predictions), 4)

    def test_build_geo_spread_meta_result_returns_model_result(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 0, 1]
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(8)],
                "spread_label": labels,
                "n_new_countries": [1, 1, 1, 4, 4, 5, 1, 6],
                "split_year": [2015] * 8,
                "backbone_assignment_mode": ["training_only"] * 8,
                "max_resolved_year_train": [2014] * 8,
                "min_resolved_year_test": [2017] * 8,
                "training_only_future_unseen_backbone_flag": [False] * 8,
                "log1p_member_count_train": [0.1, 0.2, 0.3, 1.0, 1.2, 1.3, 0.4, 1.4],
                "log1p_n_countries_train": [0.1, 0.2, 0.2, 0.8, 1.0, 1.1, 0.3, 1.2],
                "refseq_share_train": [0.2, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.95],
            }
        )
        results = {
            "geo_context_hybrid_priority": self._result(
                "geo_context_hybrid_priority",
                [0.11, 0.17, 0.31, 0.72, 0.84, 0.89, 0.36, 0.93],
                labels,
                0.82,
                0.73,
            ),
            "geo_support_light_priority": self._result(
                "geo_support_light_priority",
                [0.12, 0.18, 0.32, 0.71, 0.83, 0.88, 0.35, 0.91],
                labels,
                0.81,
                0.72,
            ),
            "geo_phylo_ecology_priority": self._result(
                "geo_phylo_ecology_priority",
                [0.15, 0.20, 0.30, 0.68, 0.80, 0.84, 0.40, 0.89],
                labels,
                0.79,
                0.70,
            ),
        }
        results["geo_reliability_blend"] = build_geo_spread_blended_result(
            results, include_ci=False
        )
        results["geo_adaptive_knownness_priority"] = build_geo_spread_adaptive_result(
            results,
            scored=scored,
            include_ci=False,
        )
        meta = build_geo_spread_meta_result(results, scored=scored, include_ci=False)
        self.assertEqual(meta.status, "ok")
        self.assertEqual(meta.name, "geo_meta_knownness_priority")
        self.assertIn("knownness_threshold", meta.metrics)
        self.assertIn("meta_component_count", meta.metrics)
        self.assertEqual(len(meta.predictions), 8)

    def test_selection_scorecard_picks_highest_scoring_model(self) -> None:
        labels = [0, 0, 1, 1]
        results = {
            "geo_context_hybrid_priority": self._result(
                "geo_context_hybrid_priority",
                [0.1, 0.2, 0.8, 0.9],
                labels,
                0.81,
                0.71,
            ),
            "geo_phylo_ecology_priority": self._result(
                "geo_phylo_ecology_priority",
                [0.15, 0.25, 0.75, 0.95],
                labels,
                0.79,
                0.69,
            ),
        }
        calibration_summary = pd.DataFrame(
            {
                "model_name": ["geo_context_hybrid_priority", "geo_phylo_ecology_priority"],
                "abstain_rate": [0.45, 0.55],
                "calibrated_expected_calibration_error": [0.03, 0.02],
                "calibrated_brier_score": [0.16, 0.17],
            }
        )
        recommended, scorecard = select_geo_spread_primary_model(results, calibration_summary)
        self.assertFalse(scorecard.empty)
        self.assertEqual(recommended, scorecard.iloc[0]["model_name"])
        self.assertIn("selection_score", scorecard.columns)
        self.assertIn("low_knownness_roc_auc", scorecard.columns)

    def test_selection_scorecard_rejects_abstain_only_primary(self) -> None:
        labels = [0, 0, 1, 1]
        results = {
            "geo_meta_knownness_priority": self._result(
                "geo_meta_knownness_priority",
                [0.1, 0.2, 0.8, 0.9],
                labels,
                0.90,
                0.80,
            ),
            "geo_reliability_blend": self._result(
                "geo_reliability_blend",
                [0.12, 0.22, 0.78, 0.88],
                labels,
                0.88,
                0.78,
            ),
        }
        calibration_summary = pd.DataFrame(
            {
                "model_name": ["geo_meta_knownness_priority", "geo_reliability_blend"],
                "abstain_rate": [1.0, 0.35],
                "calibrated_expected_calibration_error": [0.015, 0.020],
                "calibrated_brier_score": [0.15, 0.16],
            }
        )
        recommended, scorecard = select_geo_spread_primary_model(results, calibration_summary)
        self.assertEqual(recommended, "geo_reliability_blend")
        self.assertFalse(
            bool(
                scorecard.loc[
                    scorecard["model_name"] == "geo_meta_knownness_priority", "primary_eligible"
                ].iloc[0]
            )
        )


if __name__ == "__main__":
    unittest.main()
