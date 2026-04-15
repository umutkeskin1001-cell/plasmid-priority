from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.modeling.module_a import build_single_model_pareto_screen
from plasmid_priority.modeling.module_a_support import build_single_model_candidate_family
from plasmid_priority.modeling.single_model_pareto import (
    add_failure_severity,
    add_weighted_objective,
    build_pareto_shortlist,
    rank_single_model_candidates,
)


class SingleModelParetoTests(unittest.TestCase):
    def test_build_single_model_candidate_family_returns_parent_derived_variants(self) -> None:
        family = build_single_model_candidate_family()

        self.assertEqual(
            list(family.columns),
            ["model_name", "parent_model_name", "feature_set", "feature_count", "candidate_kind"],
        )
        names = set(family["model_name"].astype(str))
        self.assertTrue(
            {
                "phylo_support_fusion_priority",
                "discovery_12f_source",
                "support_synergy_priority",
                "knownness_robust_priority",
                "parsimonious_priority",
            }.issubset(names)
        )
        self.assertTrue(
            any(name.startswith("phylo_support_fusion_priority__pruned") for name in names)
        )
        self.assertTrue(any(name.startswith("discovery_12f_source__pruned") for name in names))
        self.assertTrue(any(name.startswith("support_synergy_priority__pruned") for name in names))
        self.assertTrue(any(name.startswith("knownness_robust_priority__pruned") for name in names))
        self.assertTrue(any(name.startswith("parsimonious_priority__pruned") for name in names))
        self.assertTrue((family["feature_count"] == family["feature_set"].map(len)).all())

    def test_build_single_model_pareto_screen_returns_expected_columns(self) -> None:
        family = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "parent_model_name": "parsimonious_priority",
                    "feature_set": (
                        "T_eff_norm",
                        "H_obs_specialization_norm",
                        "A_eff_norm",
                        "coherence_score",
                    ),
                    "feature_count": 4,
                    "candidate_kind": "parent",
                },
                {
                    "model_name": "parsimonious_priority__pruned",
                    "parent_model_name": "parsimonious_priority",
                    "feature_set": (
                        "T_eff_norm",
                        "H_obs_specialization_norm",
                        "A_eff_norm",
                    ),
                    "feature_count": 3,
                    "candidate_kind": "pruned",
                },
            ]
        )
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(40)],
                "spread_label": [1, 0] * 20,
                "dominant_source": ["refseq_leaning"] * 20 + ["insd_leaning"] * 20,
                "dominant_region_train": ["region_a"] * 10
                + ["region_b"] * 10
                + ["region_a"] * 10
                + ["region_b"] * 10,
                "log1p_member_count_train": [0.0, 0.2, 0.4, 0.6] * 10,
                "log1p_n_countries_train": [0.1, 0.3, 0.5, 0.7] * 10,
                "refseq_share_train": [0.2, 0.4, 0.6, 0.8] * 10,
                "T_eff_norm": np.linspace(0.05, 0.95, 40),
                "H_obs_specialization_norm": np.linspace(0.95, 0.05, 40),
                "A_eff_norm": np.linspace(0.15, 0.85, 40),
                "coherence_score": np.linspace(0.25, 0.75, 40),
            }
        )

        screen = build_single_model_pareto_screen(
            scored,
            family=family,
            n_splits=2,
            n_repeats=1,
            min_group_size=5,
            max_groups_per_column=2,
        )

        expected_columns = {
            "model_name",
            "parent_model_name",
            "feature_count",
            "roc_auc",
            "average_precision",
            "knownness_matched_gap",
            "source_holdout_gap",
            "blocked_holdout_weighted_roc_auc",
            "screen_fit_seconds",
            "predictive_power_score",
            "reliability_score",
            "compute_efficiency_score",
            "weighted_objective_score",
        }
        self.assertTrue(expected_columns.issubset(set(screen.columns)))
        self.assertEqual(
            set(screen["model_name"].astype(str)), set(family["model_name"].astype(str))
        )
        self.assertTrue(screen["screen_fit_seconds"].ge(0.0).all())
        self.assertTrue(screen["weighted_objective_score"].notna().all())

    def test_weighted_objective_prefers_better_reliability_when_power_is_close(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "model_name": "a",
                    "reliability_score": 0.82,
                    "predictive_power_score": 0.80,
                    "compute_efficiency_score": 0.40,
                },
                {
                    "model_name": "b",
                    "reliability_score": 0.70,
                    "predictive_power_score": 0.83,
                    "compute_efficiency_score": 0.40,
                },
            ]
        )

        ranked = rank_single_model_candidates(candidates)

        self.assertEqual(str(ranked.iloc[0]["model_name"]), "a")
        self.assertGreater(
            float(ranked.iloc[0]["weighted_objective_score"]),
            float(ranked.iloc[1]["weighted_objective_score"]),
        )

    def test_failure_severity_penalizes_multi_guardrail_failures_and_worse_gaps(self) -> None:
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "a",
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:knownness_matched",
                    "knownness_matched_gap": -0.010,
                    "source_holdout_gap": -0.001,
                    "blocked_holdout_raw_ece": 0.03,
                },
                {
                    "model_name": "b",
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:knownness_matched,source_holdout,calibration",
                    "knownness_matched_gap": -0.040,
                    "source_holdout_gap": -0.050,
                    "blocked_holdout_raw_ece": 0.12,
                },
            ]
        )

        enriched = add_failure_severity(scorecard)

        severity_a = float(enriched.loc[enriched["model_name"] == "a", "failure_severity"].iloc[0])
        severity_b = float(enriched.loc[enriched["model_name"] == "b", "failure_severity"].iloc[0])

        self.assertLess(severity_a, severity_b)

    def test_failure_severity_increases_when_knownness_matched_gap_worsens(self) -> None:
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "mild",
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:knownness_matched",
                    "knownness_matched_gap": -0.010,
                },
                {
                    "model_name": "severe",
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:knownness_matched",
                    "knownness_matched_gap": -0.040,
                },
            ]
        )

        enriched = add_failure_severity(scorecard)

        mild = float(enriched.loc[enriched["model_name"] == "mild", "failure_severity"].iloc[0])
        severe = float(enriched.loc[enriched["model_name"] == "severe", "failure_severity"].iloc[0])

        self.assertLess(mild, severe)

    def test_rank_single_model_candidates_uses_severity_then_objective_then_name(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "model_name": "beta",
                    "reliability_score": 0.80,
                    "predictive_power_score": 0.80,
                    "compute_efficiency_score": 0.50,
                    "failure_severity": 0.20,
                },
                {
                    "model_name": "alpha",
                    "reliability_score": 0.80,
                    "predictive_power_score": 0.80,
                    "compute_efficiency_score": 0.50,
                    "failure_severity": 0.20,
                },
                {
                    "model_name": "zeta",
                    "reliability_score": 0.90,
                    "predictive_power_score": 0.72,
                    "compute_efficiency_score": 0.40,
                    "failure_severity": 0.05,
                },
            ]
        )

        ranked = rank_single_model_candidates(candidates)

        self.assertEqual(list(ranked["model_name"]), ["zeta", "alpha", "beta"])

    def test_build_pareto_shortlist_is_deterministic_and_excludes_dominated_candidates(
        self,
    ) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "model_name": "m1",
                    "reliability_score": 0.90,
                    "predictive_power_score": 0.88,
                    "compute_efficiency_score": 0.60,
                    "failure_severity": 0.10,
                },
                {
                    "model_name": "m2",
                    "reliability_score": 0.85,
                    "predictive_power_score": 0.82,
                    "compute_efficiency_score": 0.55,
                    "failure_severity": 0.20,
                },
                {
                    "model_name": "m3",
                    "reliability_score": 0.90,
                    "predictive_power_score": 0.88,
                    "compute_efficiency_score": 0.60,
                    "failure_severity": 0.10,
                },
            ]
        )

        shortlist = build_pareto_shortlist(candidates)

        self.assertEqual(list(shortlist["model_name"]), ["m1", "m3"])
        self.assertNotIn("m2", set(shortlist["model_name"]))
        self.assertTrue(shortlist["weighted_objective_score"].is_monotonic_decreasing)

    def test_weighted_objective_matches_protocol_weights(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_name": "x",
                    "reliability_score": 0.5,
                    "predictive_power_score": 0.75,
                    "compute_efficiency_score": 0.25,
                }
            ]
        )

        scored = add_weighted_objective(frame)

        self.assertAlmostEqual(float(scored.iloc[0]["weighted_objective_score"]), 0.55)


if __name__ == "__main__":
    unittest.main()
