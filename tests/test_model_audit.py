from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.reporting import (
    build_amrfinder_coverage_table,
    build_benchmark_protocol_table,
    build_blocked_holdout_calibration_summary,
    build_blocked_holdout_calibration_table,
    build_blocked_holdout_summary,
    build_calibration_metric_table,
    build_candidate_dossier_table,
    build_candidate_portfolio_table,
    build_candidate_risk_table,
    build_candidate_universe_table,
    build_component_floor_diagnostics,
    build_consensus_candidate_ranking,
    build_decision_yield_table,
    build_frozen_scientific_acceptance_audit,
    build_future_sentinel_audit,
    build_gate_consistency_audit,
    build_group_holdout_performance,
    build_h_feature_diagnostics,
    build_knownness_audit_tables,
    build_logistic_implementation_audit,
    build_magic_number_sensitivity_table,
    build_model_comparison_table,
    build_model_family_summary,
    build_model_selection_scorecard,
    build_model_simplicity_summary,
    build_model_subgroup_performance,
    build_negative_control_audit,
    build_novelty_margin_summary,
    build_official_benchmark_panel,
    build_permutation_null_tables,
    build_primary_model_selection_summary,
    build_priority_bootstrap_stability_table,
    build_score_distribution_diagnostics,
    build_selection_adjusted_permutation_null,
    build_single_model_official_decision,
    build_single_model_pareto_finalists,
    build_sleeper_threat_table,
    build_source_balance_resampling_table,
    build_temporal_drift_summary,
    build_temporal_rank_stability_table,
    build_threshold_flip_table,
    build_threshold_utility_table,
    build_variant_rank_consistency_table,
    sanitize_adaptive_gated_predictions,
)
from plasmid_priority.validation import decision_utility_summary


class ModelAuditTests(unittest.TestCase):
    def test_build_single_model_pareto_finalists_keeps_pareto_shortlist(self) -> None:
        screen = pd.DataFrame(
            [
                {
                    "model_name": "dominant",
                    "parent_model_name": "dominant",
                    "feature_set": ("a", "b"),
                    "feature_count": 2,
                    "candidate_kind": "parent",
                    "reliability_score": 0.90,
                    "predictive_power_score": 0.88,
                    "compute_efficiency_score": 0.60,
                    "weighted_objective_score": 0.832,
                    "failure_severity": 0.10,
                    "roc_auc": 0.84,
                    "average_precision": 0.80,
                },
                {
                    "model_name": "dominated",
                    "parent_model_name": "dominated",
                    "feature_set": ("a", "b", "c"),
                    "feature_count": 3,
                    "candidate_kind": "pruned",
                    "reliability_score": 0.82,
                    "predictive_power_score": 0.80,
                    "compute_efficiency_score": 0.55,
                    "weighted_objective_score": 0.758,
                    "failure_severity": 0.30,
                    "roc_auc": 0.80,
                    "average_precision": 0.76,
                },
                {
                    "model_name": "tradeoff",
                    "parent_model_name": "tradeoff",
                    "feature_set": ("a",),
                    "feature_count": 1,
                    "candidate_kind": "pruned",
                    "reliability_score": 0.75,
                    "predictive_power_score": 0.81,
                    "compute_efficiency_score": 0.95,
                    "weighted_objective_score": 0.814,
                    "failure_severity": 0.05,
                    "roc_auc": 0.79,
                    "average_precision": 0.75,
                },
            ]
        )

        finalists = build_single_model_pareto_finalists(screen, max_finalists=3)

        self.assertEqual(list(finalists["model_name"]), ["dominant", "tradeoff"])
        self.assertNotIn("dominated", set(finalists["model_name"]))

    def test_build_single_model_official_decision_prefers_lower_failure_severity_before_auc(
        self,
    ) -> None:
        finalists = pd.DataFrame(
            [
                {
                    "model_name": "high_auc_fail_hard",
                    "weighted_objective_score": 0.84,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:matched_knownness,source_holdout",
                    "failure_severity": 0.70,
                    "roc_auc": 0.84,
                    "average_precision": 0.79,
                    "compute_efficiency_score": 0.30,
                    "screen_fit_seconds": 12.0,
                },
                {
                    "model_name": "slightly_lower_auc_fail_soft",
                    "weighted_objective_score": 0.83,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:matched_knownness",
                    "failure_severity": 0.18,
                    "roc_auc": 0.82,
                    "average_precision": 0.78,
                    "compute_efficiency_score": 0.45,
                    "screen_fit_seconds": 8.0,
                },
            ]
        )

        decision = build_single_model_official_decision(finalists)

        self.assertEqual(
            str(decision.iloc[0]["official_model_name"]), "slightly_lower_auc_fail_soft"
        )
        self.assertEqual(
            str(decision.iloc[0]["decision_reason"]),
            "lowest_failure_severity_with_competitive_auc",
        )

    def test_build_gate_consistency_audit_summarizes_half_gate_routes(self) -> None:
        adaptive_predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "model_name": ["adaptive_knownness_blend_priority"] * 8,
                "knownness_score": [0.42, 0.45, 0.48, 0.49, 0.51, 0.52, 0.55, 0.58],
                "knownness_half": ["lower_half"] * 4 + ["upper_half"] * 4,
                "knownness_quartile": [
                    "q1_lowest",
                    "q1_lowest",
                    "q2",
                    "q2",
                    "q3",
                    "q3",
                    "q4_highest",
                    "q4_highest",
                ],
                "lower_half_route_prediction": [0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48],
                "upper_half_route_prediction": [0.59, 0.57, 0.55, 0.53, 0.51, 0.49, 0.47, 0.45],
            }
        )
        audit = build_gate_consistency_audit(adaptive_predictions, near_fraction=0.5, min_n=2)
        self.assertEqual(set(audit["gate_name"]), {"half_boundary"})
        self.assertEqual(int(audit.iloc[0]["n_near_gate"]), 4)
        self.assertIn("mean_abs_route_delta_near_gate", audit.columns)
        self.assertGreaterEqual(float(audit.iloc[0]["route_spearman_near_gate"]), 0.9)

    def test_sanitize_adaptive_gated_predictions_drops_internal_specialist_columns(self) -> None:
        adaptive_predictions = pd.DataFrame(
            {
                "backbone_id": ["bb_1"],
                "adaptive_prediction": [0.62],
                "spread_label": [1],
                "knownness_score": [0.23],
                "knownness_half": ["lower_half"],
                "knownness_quartile": ["q1_lowest"],
                "model_name": ["adaptive_knownness_robust_priority"],
                "base_model_name": ["knownness_robust_priority"],
                "specialist_model_name": ["novelty_specialist_priority"],
                "gating_rule": ["lower_half_specialist_switch"],
                "prediction_source": ["novelty_specialist_priority"],
                "specialist_weight_lower_half": [1.0],
                "base_oof_prediction": [0.41],
                "novelty_specialist_prediction": [0.62],
                "upper_half_route_prediction": [0.41],
                "lower_half_route_prediction": [0.62],
                "novelty_specialist_full_fit_prediction": [0.88],
                "quartile_specialist_full_fit_prediction": [0.91],
                "quartile_specialist_prediction": [0.77],
                "specialist_weight_q1": [1.0],
                "specialist_weight_q2": [0.5],
            }
        )

        sanitized = sanitize_adaptive_gated_predictions(adaptive_predictions)

        forbidden = {
            "novelty_specialist_full_fit_prediction",
            "quartile_specialist_full_fit_prediction",
            "quartile_specialist_prediction",
            "specialist_weight_q1",
            "specialist_weight_q2",
        }
        self.assertTrue(forbidden.isdisjoint(sanitized.columns))
        self.assertIn("specialist_weight_lower_half", sanitized.columns)
        self.assertEqual(float(sanitized.iloc[0]["adaptive_prediction"]), 0.62)

    def test_build_model_family_summary_keeps_priority_models(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "source_only", "roc_auc": 0.45},
                {"model_name": "baseline_both", "roc_auc": 0.68},
                {"model_name": "full_priority", "roc_auc": 0.75},
                {"model_name": "bio_residual_synergy_priority", "roc_auc": 0.79},
                {"model_name": "hybrid_agreement_priority", "roc_auc": 0.82},
                {"model_name": "firth_parsimonious_priority", "roc_auc": 0.78},
                {"model_name": "T_plus_H_plus_A", "roc_auc": 0.76},
                {"model_name": "proxy_light_priority", "roc_auc": 0.77},
                {"model_name": "enhanced_priority", "roc_auc": 0.78},
                {"model_name": "monotonic_latent_priority", "roc_auc": 0.81},
                {"model_name": "regime_stability_priority", "roc_auc": 0.80},
            ]
        )
        summary = build_model_family_summary(model_metrics)
        self.assertIn("evidence_role", summary.columns)
        self.assertIn("delta_auc_vs_enhanced_priority", summary.columns)
        self.assertIn("model_track", summary.columns)
        self.assertIn("track_summary", summary.columns)
        self.assertIn("bio_residual_synergy_priority", set(summary["model_name"]))
        self.assertIn("hybrid_agreement_priority", set(summary["model_name"]))
        self.assertIn("firth_parsimonious_priority", set(summary["model_name"]))
        self.assertIn("enhanced_priority", set(summary["model_name"]))
        self.assertIn("monotonic_latent_priority", set(summary["model_name"]))
        self.assertIn("regime_stability_priority", set(summary["model_name"]))
        baseline_row = summary.loc[summary["model_name"] == "baseline_both"].iloc[0]
        synergy_row = summary.loc[summary["model_name"] == "bio_residual_synergy_priority"].iloc[0]
        governance_row = summary.loc[summary["model_name"] == "regime_stability_priority"].iloc[0]
        self.assertEqual(str(baseline_row["model_track"]), "baseline")
        self.assertEqual(str(synergy_row["model_track"]), "discovery")
        self.assertEqual(str(governance_row["model_track"]), "governance")

    def test_build_model_family_summary_flags_suspicious_auc(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "monotonic_latent_priority", "roc_auc": 0.91},
                {"model_name": "support_synergy_priority", "roc_auc": 0.82},
            ]
        )
        summary = build_model_family_summary(model_metrics)
        flagged = summary.loc[summary["model_name"] == "monotonic_latent_priority"].iloc[0]
        self.assertTrue(bool(flagged["leakage_review_required"]))
        self.assertEqual(
            str(flagged["leakage_review_reason"]), "roc_auc_ge_0p90_on_current_feature_universe"
        )

    def test_build_model_selection_scorecard_returns_composite_ranks(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "bio_residual_synergy_priority",
                    "roc_auc": 0.80,
                    "average_precision": 0.71,
                    "decision_utility_score": 0.31,
                    "optimal_decision_threshold": 0.42,
                },
                {
                    "model_name": "phylo_support_fusion_priority",
                    "roc_auc": 0.79,
                    "average_precision": 0.70,
                    "decision_utility_score": 0.29,
                    "optimal_decision_threshold": 0.38,
                },
                {
                    "model_name": "baseline_both",
                    "roc_auc": 0.72,
                    "average_precision": 0.65,
                    "decision_utility_score": 0.18,
                    "optimal_decision_threshold": 0.50,
                },
                {
                    "model_name": "failed_model",
                    "roc_auc": 0.99,
                    "average_precision": 0.99,
                    "status": "failed",
                    "error_message": "forced failure",
                },
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 4,
                "model_name": ["bio_residual_synergy_priority"] * 12
                + ["phylo_support_fusion_priority"] * 12
                + ["baseline_both"] * 12
                + ["failed_model"] * 12,
                "oof_prediction": [
                    0.95,
                    0.90,
                    0.88,
                    0.80,
                    0.70,
                    0.65,
                    0.45,
                    0.35,
                    0.30,
                    0.20,
                    0.18,
                    0.12,
                ]
                + [0.92, 0.88, 0.86, 0.82, 0.72, 0.67, 0.46, 0.37, 0.32, 0.21, 0.19, 0.14]
                + [0.80, 0.78, 0.75, 0.70, 0.68, 0.66, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
                + [0.55, 0.60, 0.58, 0.52, 0.48, 0.42, 0.38, 0.34, 0.28, 0.22, 0.18, 0.15],
                "spread_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 4,
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "log1p_member_count_train": [0.0] * 6 + [1.0] * 6,
                "log1p_n_countries_train": [0.0] * 6 + [1.0] * 6,
                "refseq_share_train": [0.0] * 6 + [1.0] * 6,
            }
        )
        matched = pd.DataFrame(
            [
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "bio_residual_synergy_priority",
                    "roc_auc": 0.74,
                },
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "phylo_support_fusion_priority",
                    "roc_auc": 0.73,
                },
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "baseline_both",
                    "roc_auc": 0.60,
                },
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "model_name": "bio_residual_synergy_priority",
                    "status": "ok",
                    "roc_auc": 0.76,
                    "n_test_backbones": 9,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "bio_residual_synergy_priority",
                    "status": "ok",
                    "roc_auc": 0.70,
                    "n_test_backbones": 3,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "phylo_support_fusion_priority",
                    "status": "ok",
                    "roc_auc": 0.75,
                    "n_test_backbones": 9,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "phylo_support_fusion_priority",
                    "status": "ok",
                    "roc_auc": 0.69,
                    "n_test_backbones": 3,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "baseline_both",
                    "status": "ok",
                    "roc_auc": 0.62,
                    "n_test_backbones": 9,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "baseline_both",
                    "status": "ok",
                    "roc_auc": 0.58,
                    "n_test_backbones": 3,
                },
            ]
        )
        scorecard = build_model_selection_scorecard(
            model_metrics,
            predictions,
            scored,
            knownness_matched_validation=matched,
            group_holdout=holdout,
        )
        self.assertIn("selection_composite_score", scorecard.columns)
        self.assertIn("selection_rank", scorecard.columns)
        self.assertIn("decision_utility_score", scorecard.columns)
        self.assertIn("optimal_decision_threshold", scorecard.columns)
        self.assertIn("model_track", scorecard.columns)
        self.assertIn("track_rank", scorecard.columns)
        self.assertIn("discovery_track_rank", scorecard.columns)
        self.assertIn("governance_track_rank", scorecard.columns)
        self.assertIn("baseline_track_rank", scorecard.columns)
        self.assertEqual(str(scorecard.iloc[0]["model_name"]), "bio_residual_synergy_priority")
        self.assertNotIn("failed_model", set(scorecard["model_name"]))
        discovery_row = scorecard.loc[
            scorecard["model_name"] == "bio_residual_synergy_priority"
        ].iloc[0]
        governance_row = scorecard.loc[
            scorecard["model_name"] == "phylo_support_fusion_priority"
        ].iloc[0]
        baseline_row = scorecard.loc[scorecard["model_name"] == "baseline_both"].iloc[0]
        self.assertEqual(str(discovery_row["model_track"]), "discovery")
        self.assertEqual(int(discovery_row["discovery_track_rank"]), 1)
        self.assertEqual(str(governance_row["model_track"]), "governance")
        self.assertEqual(int(governance_row["governance_track_rank"]), 1)
        self.assertEqual(str(baseline_row["model_track"]), "baseline")
        self.assertEqual(int(baseline_row["baseline_track_rank"]), 1)

    def test_decision_utility_summary_prefers_best_threshold(self) -> None:
        summary = decision_utility_summary(
            np.array([1, 1, 0, 0, 0, 1]),
            np.array([0.95, 0.85, 0.80, 0.35, 0.20, 0.10]),
            thresholds=(0.2, 0.5, 0.8),
            false_positive_cost=1.0,
            false_negative_cost=4.0,
        )
        self.assertIn("optimal_threshold", summary)
        self.assertIn("optimal_threshold_utility_per_sample", summary)
        self.assertGreaterEqual(float(summary["optimal_threshold"]), 0.2)
        self.assertLessEqual(float(summary["optimal_threshold"]), 0.8)
        self.assertGreater(float(summary["optimal_threshold_precision"]), 0.5)

    def test_build_model_selection_scorecard_penalizes_missing_metrics(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "complete_model", "roc_auc": 0.79, "average_precision": 0.70},
                {"model_name": "missing_holdout_model", "roc_auc": 0.79, "average_precision": 0.70},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["complete_model"] * 12 + ["missing_holdout_model"] * 12,
                "oof_prediction": [
                    0.90,
                    0.85,
                    0.80,
                    0.75,
                    0.70,
                    0.65,
                    0.40,
                    0.35,
                    0.30,
                    0.25,
                    0.20,
                    0.10,
                ]
                + [0.95, 0.92, 0.88, 0.80, 0.78, 0.75, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
                "spread_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 2,
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "log1p_member_count_train": [0.0] * 6 + [1.0] * 6,
                "log1p_n_countries_train": [0.0] * 6 + [1.0] * 6,
                "refseq_share_train": [0.0] * 6 + [1.0] * 6,
            }
        )
        matched = pd.DataFrame(
            [
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "complete_model",
                    "roc_auc": 0.74,
                },
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "missing_holdout_model",
                    "roc_auc": 0.74,
                },
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "model_name": "complete_model",
                    "status": "ok",
                    "roc_auc": 0.76,
                    "n_test_backbones": 9,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "complete_model",
                    "status": "ok",
                    "roc_auc": 0.70,
                    "n_test_backbones": 3,
                },
            ]
        )
        scorecard = build_model_selection_scorecard(
            model_metrics,
            predictions,
            scored,
            knownness_matched_validation=matched,
            group_holdout=holdout,
        )
        missing_row = scorecard.loc[scorecard["model_name"] == "missing_holdout_model"].iloc[0]
        complete_row = scorecard.loc[scorecard["model_name"] == "complete_model"].iloc[0]
        self.assertIn("selection_metrics_complete", scorecard.columns)
        self.assertEqual(int(missing_row["selection_missing_metric_count"]), 1)
        self.assertFalse(bool(missing_row["selection_metrics_complete"]))
        self.assertTrue(bool(complete_row["selection_metrics_complete"]))
        self.assertGreater(
            float(complete_row["selection_composite_score"]),
            float(missing_row["selection_composite_score"]),
        )
        self.assertTrue(pd.isna(missing_row["selection_rank"]))
        self.assertEqual(int(complete_row["selection_rank"]), 1)

    def test_build_threshold_utility_table_returns_optimal_thresholds(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(6)] * 2,
                "model_name": ["utility_a"] * 6 + ["utility_b"] * 6,
                "oof_prediction": [0.95, 0.9, 0.75, 0.3, 0.2, 0.1]
                + [0.9, 0.7, 0.6, 0.4, 0.35, 0.2],
                "spread_label": [1, 1, 1, 0, 0, 0] * 2,
            }
        )
        utility = build_threshold_utility_table(
            predictions,
            model_names=["utility_a", "utility_b"],
            thresholds=(0.25, 0.5, 0.75),
        )
        self.assertEqual(set(utility["model_name"]), {"utility_a", "utility_b"})
        self.assertIn("optimal_threshold", utility.columns)
        self.assertIn("optimal_threshold_utility_per_sample", utility.columns)
        self.assertGreaterEqual(float(utility.iloc[0]["optimal_threshold_utility_per_sample"]), 0.0)

    def test_build_model_selection_scorecard_marks_strict_knownness_acceptance(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "accepted_model", "roc_auc": 0.80, "average_precision": 0.72},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "model_name": ["accepted_model"] * 8,
                "oof_prediction": [0.90, 0.84, 0.81, 0.78, 0.32, 0.28, 0.22, 0.10],
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "log1p_member_count_train": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
                "log1p_n_countries_train": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
                "refseq_share_train": [0.2, 0.2, 0.3, 0.3, 0.8, 0.8, 0.9, 0.9],
            }
        )
        matched = pd.DataFrame(
            [
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "accepted_model",
                    "roc_auc": 0.81,
                },
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "model_name": "accepted_model",
                    "status": "ok",
                    "roc_auc": 0.805,
                    "n_test_backbones": 8,
                },
            ]
        )

        scorecard = build_model_selection_scorecard(
            model_metrics,
            predictions,
            scored,
            knownness_matched_validation=matched,
            group_holdout=holdout,
        )

        self.assertIn("strict_knownness_acceptance_flag", scorecard.columns)
        self.assertIn("knownness_matched_gap", scorecard.columns)
        self.assertIn("source_holdout_gap", scorecard.columns)
        self.assertIn("guardrail_loss", scorecard.columns)
        self.assertIn("governance_priority_score", scorecard.columns)
        self.assertIn("governance_rank", scorecard.columns)
        self.assertTrue(bool(scorecard.iloc[0]["strict_knownness_acceptance_flag"]))

    def test_build_model_selection_scorecard_marks_frozen_scientific_acceptance(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "accepted_model",
                    "roc_auc": 0.80,
                    "average_precision": 0.72,
                    "ece": 0.03,
                    "spatial_holdout_roc_auc": 0.78,
                    "selection_adjusted_empirical_p_roc_auc": 0.009,
                },
                {
                    "model_name": "calibration_fail_model",
                    "roc_auc": 0.80,
                    "average_precision": 0.72,
                    "ece": 0.08,
                    "spatial_holdout_roc_auc": 0.79,
                    "selection_adjusted_empirical_p_roc_auc": 0.009,
                },
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)] * 2,
                "model_name": ["accepted_model"] * 8 + ["calibration_fail_model"] * 8,
                "oof_prediction": [0.90, 0.84, 0.81, 0.78, 0.32, 0.28, 0.22, 0.10] * 2,
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0] * 2,
            }
        )
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "log1p_member_count_train": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
                "log1p_n_countries_train": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
                "refseq_share_train": [0.2, 0.2, 0.3, 0.3, 0.8, 0.8, 0.9, 0.9],
            }
        )
        matched = pd.DataFrame(
            [
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "accepted_model",
                    "roc_auc": 0.80,
                },
                {
                    "matched_stratum": "__weighted_overall__",
                    "model_name": "calibration_fail_model",
                    "roc_auc": 0.80,
                },
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "model_name": "accepted_model",
                    "status": "ok",
                    "roc_auc": 0.80,
                    "n_test_backbones": 8,
                },
                {
                    "group_column": "dominant_source",
                    "model_name": "calibration_fail_model",
                    "status": "ok",
                    "roc_auc": 0.80,
                    "n_test_backbones": 8,
                },
            ]
        )

        scorecard = build_model_selection_scorecard(
            model_metrics,
            predictions,
            scored,
            knownness_matched_validation=matched,
            group_holdout=holdout,
        )

        accepted_row = scorecard.loc[scorecard["model_name"] == "accepted_model"].iloc[0]
        calibration_fail_row = scorecard.loc[
            scorecard["model_name"] == "calibration_fail_model"
        ].iloc[0]
        self.assertIn("scientific_acceptance_flag", scorecard.columns)
        self.assertIn("scientific_acceptance_status", scorecard.columns)
        self.assertIn("scientific_acceptance_failed_criteria", scorecard.columns)
        self.assertTrue(bool(accepted_row["scientific_acceptance_flag"]))
        self.assertEqual(str(accepted_row["scientific_acceptance_status"]), "pass")
        self.assertFalse(bool(calibration_fail_row["scientific_acceptance_flag"]))
        self.assertEqual(str(calibration_fail_row["scientific_acceptance_status"]), "fail")
        self.assertIn(
            "calibration",
            str(calibration_fail_row["scientific_acceptance_failed_criteria"]),
        )

    def test_build_frozen_scientific_acceptance_audit_filters_official_surfaces(self) -> None:
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "phylo_support_fusion_priority",
                    "selection_rank": 2,
                    "model_track": "governance",
                    "roc_auc": 0.83,
                    "average_precision": 0.77,
                    "matched_knownness_weighted_roc_auc": 0.78,
                    "knownness_matched_gap": -0.05,
                    "source_holdout_weighted_roc_auc": 0.76,
                    "source_holdout_gap": -0.07,
                    "spatial_holdout_roc_auc": 0.81,
                    "spatial_holdout_gap": -0.02,
                    "ece": 0.08,
                    "selection_adjusted_empirical_p_roc_auc": 0.005,
                    "matched_knownness_gate_pass": False,
                    "source_holdout_gate_pass": False,
                    "spatial_holdout_gate_pass": True,
                    "calibration_gate_pass": False,
                    "selection_adjusted_gate_pass": True,
                    "leakage_review_gate_pass": True,
                    "scientific_acceptance_scored": True,
                    "scientific_acceptance_flag": False,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:matched_knownness,source_holdout,calibration",
                },
                {
                    "model_name": "bio_clean_priority",
                    "selection_rank": 1,
                    "model_track": "discovery",
                    "roc_auc": 0.73,
                    "average_precision": 0.64,
                    "matched_knownness_weighted_roc_auc": 0.68,
                    "knownness_matched_gap": -0.05,
                    "source_holdout_weighted_roc_auc": 0.72,
                    "source_holdout_gap": -0.01,
                    "spatial_holdout_roc_auc": 0.73,
                    "spatial_holdout_gap": -0.01,
                    "ece": 0.04,
                    "selection_adjusted_empirical_p_roc_auc": 0.005,
                    "matched_knownness_gate_pass": False,
                    "source_holdout_gate_pass": False,
                    "spatial_holdout_gate_pass": True,
                    "calibration_gate_pass": True,
                    "selection_adjusted_gate_pass": True,
                    "leakage_review_gate_pass": True,
                    "scientific_acceptance_scored": True,
                    "scientific_acceptance_flag": False,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:matched_knownness,source_holdout",
                },
                {
                    "model_name": "baseline_both",
                    "selection_rank": 3,
                    "model_track": "baseline",
                    "roc_auc": 0.72,
                    "average_precision": 0.65,
                    "matched_knownness_weighted_roc_auc": 0.59,
                    "knownness_matched_gap": -0.13,
                    "source_holdout_weighted_roc_auc": 0.74,
                    "source_holdout_gap": 0.01,
                    "spatial_holdout_roc_auc": 0.74,
                    "spatial_holdout_gap": 0.02,
                    "ece": 0.04,
                    "selection_adjusted_empirical_p_roc_auc": 0.005,
                    "matched_knownness_gate_pass": False,
                    "source_holdout_gate_pass": True,
                    "spatial_holdout_gate_pass": True,
                    "calibration_gate_pass": True,
                    "selection_adjusted_gate_pass": True,
                    "leakage_review_gate_pass": True,
                    "scientific_acceptance_scored": True,
                    "scientific_acceptance_flag": False,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:matched_knownness",
                },
            ]
        )

        audit = build_frozen_scientific_acceptance_audit(scorecard)

        self.assertEqual(
            audit["model_name"].tolist(),
            [
                "baseline_both",
                "bio_clean_priority",
                "phylo_support_fusion_priority",
            ],
        )
        self.assertIn("matched_knownness_gap_min", audit.columns)
        self.assertIn("selection_adjusted_p_max", audit.columns)
        self.assertEqual(str(audit.iloc[0]["model_track"]), "baseline")
        self.assertEqual(str(audit.iloc[1]["model_track"]), "discovery")
        self.assertEqual(str(audit.iloc[2]["model_track"]), "governance")
        self.assertFalse(bool(audit.iloc[1]["scientific_acceptance_flag"]))

    def test_build_future_sentinel_audit_flags_future_outcome_canary_as_excluded(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "model_name": ["bio_clean_priority"] * 8,
                "oof_prediction": [0.90, 0.84, 0.81, 0.78, 0.32, 0.28, 0.22, 0.10],
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        audit = build_future_sentinel_audit(
            scored,
            predictions=predictions,
            primary_model_name="bio_clean_priority",
            model_names=["bio_clean_priority", "baseline_both"],
        )

        row = audit.iloc[0]
        self.assertEqual(str(row["audit_status"]), "pass")
        self.assertTrue(bool(row["discovery_contract_forbidden"]))
        self.assertFalse(bool(row["official_discovery_models_use_sentinel"]))
        self.assertEqual(float(row["sentinel_only_roc_auc"]), 1.0)
        self.assertGreaterEqual(float(row["delta_roc_auc_vs_primary"]), 0.0)

    def test_build_benchmark_protocol_table_marks_primary_and_preferred_adaptive(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "support_synergy_priority",
                    "roc_auc": 0.818,
                    "average_precision": 0.740,
                },
                {
                    "model_name": "knownness_robust_priority",
                    "roc_auc": 0.806,
                    "average_precision": 0.719,
                },
                {"model_name": "baseline_both", "roc_auc": 0.729, "average_precision": 0.675},
                {"model_name": "source_only", "roc_auc": 0.448, "average_precision": 0.401},
            ]
        )
        selection_summary = pd.DataFrame(
            [
                {
                    "published_primary_model": "support_synergy_priority",
                    "conservative_model_name": "knownness_robust_priority",
                    "strongest_metric_model": "support_synergy_priority",
                    "governance_primary_model": "knownness_robust_priority",
                    "governance_primary_strict_knownness_acceptance_flag": True,
                    "published_primary_top_10_precision": 0.41,
                    "published_primary_blocked_holdout_raw_brier_score": 0.14,
                    "published_primary_blocked_holdout_best_calibration_method": "platt",
                    "published_primary_blocked_holdout_best_calibration_gain_vs_raw_brier": 0.02,
                    "governance_primary_top_10_precision": 0.52,
                    "governance_primary_blocked_holdout_best_calibration_method": "beta",
                    "governance_primary_blocked_holdout_best_calibration_gain_vs_raw_ece": 0.05,
                }
            ]
        )
        adaptive_metrics = pd.DataFrame(
            [
                {
                    "model_name": "adaptive_support_synergy_blend_priority",
                    "status": "ok",
                    "roc_auc": 0.823,
                    "average_precision": 0.743,
                    "specialist_weight_lower_half": 0.5,
                },
                {
                    "model_name": "adaptive_knownness_robust_priority",
                    "status": "ok",
                    "roc_auc": 0.826,
                    "average_precision": 0.732,
                    "specialist_weight_lower_half": 1.0,
                },
            ]
        )
        gate_consistency = pd.DataFrame(
            [
                {
                    "model_name": "adaptive_support_synergy_blend_priority",
                    "gate_consistency_tier": "stable",
                },
                {
                    "model_name": "adaptive_knownness_robust_priority",
                    "gate_consistency_tier": "unstable",
                },
            ]
        )
        protocol = build_benchmark_protocol_table(
            model_metrics,
            selection_summary,
            adaptive_gated_metrics=adaptive_metrics,
            gate_consistency_audit=gate_consistency,
            model_selection_scorecard=pd.DataFrame(
                [
                    {
                        "model_name": "support_synergy_priority",
                        "selection_rank": 1,
                        "strict_knownness_acceptance_flag": True,
                        "knownness_matched_gap": -0.002,
                        "source_holdout_gap": -0.001,
                        "leakage_review_required": False,
                        "guardrail_loss": 0.003,
                        "governance_priority_score": 0.815,
                        "governance_rank": 2,
                    },
                    {
                        "model_name": "knownness_robust_priority",
                        "selection_rank": 2,
                        "strict_knownness_acceptance_flag": True,
                        "knownness_matched_gap": -0.003,
                        "source_holdout_gap": -0.002,
                        "leakage_review_required": False,
                        "guardrail_loss": 0.005,
                        "governance_priority_score": 0.801,
                        "governance_rank": 1,
                    },
                ]
            ),
            governance_model_name="knownness_robust_priority",
        )
        self.assertEqual(
            str(
                protocol.loc[protocol["benchmark_role"] == "primary_benchmark", "model_name"].iloc[
                    0
                ]
            ),
            "support_synergy_priority",
        )
        self.assertEqual(
            str(
                protocol.loc[
                    protocol["benchmark_role"] == "preferred_adaptive_audit", "model_name"
                ].iloc[0]
            ),
            "adaptive_support_synergy_blend_priority",
        )
        self.assertEqual(
            str(
                protocol.loc[
                    protocol["benchmark_role"] == "strongest_adaptive_upper_bound", "model_name"
                ].iloc[0]
            ),
            "adaptive_knownness_robust_priority",
        )
        primary_row = protocol.loc[protocol["benchmark_role"] == "primary_benchmark"].iloc[0]
        governance_row = protocol.loc[protocol["benchmark_role"] == "governance_benchmark"].iloc[0]
        self.assertIn("strict_knownness_acceptance_flag", protocol.columns)
        self.assertIn("scientific_acceptance_status", protocol.columns)
        self.assertIn("benchmark_track", protocol.columns)
        self.assertIn("published_primary_top_10_precision", protocol.columns)
        self.assertIn(
            "published_primary_blocked_holdout_best_calibration_method", protocol.columns
        )
        self.assertIn("governance_primary_top_10_precision", protocol.columns)
        self.assertEqual(str(primary_row["benchmark_guardrail_status"]), "passes_strict_acceptance")
        self.assertEqual(str(governance_row["benchmark_track"]), "governance")
        self.assertEqual(
            str(governance_row["benchmark_guardrail_status"]), "passes_strict_acceptance"
        )
        self.assertEqual(float(primary_row["published_primary_top_10_precision"]), 0.41)
        self.assertEqual(
            str(primary_row["published_primary_blocked_holdout_best_calibration_method"]),
            "platt",
        )
        self.assertEqual(
            float(primary_row["published_primary_blocked_holdout_best_calibration_gain_vs_raw_brier"]),
            0.02,
        )
        self.assertEqual(float(governance_row["governance_primary_top_10_precision"]), 0.52)
        self.assertEqual(
            str(governance_row["governance_primary_blocked_holdout_best_calibration_method"]),
            "beta",
        )
        self.assertEqual(
            float(
                governance_row[
                    "governance_primary_blocked_holdout_best_calibration_gain_vs_raw_ece"
                ]
            ),
            0.05,
        )
        official = build_official_benchmark_panel(protocol)
        self.assertEqual(
            set(official["benchmark_role"]),
            {
                "primary_benchmark",
                "governance_benchmark",
                "conservative_benchmark",
                "counts_baseline",
                "source_control",
            },
        )
        self.assertNotIn("preferred_adaptive_audit", set(official["benchmark_role"]))
        self.assertNotIn("strongest_adaptive_upper_bound", set(official["benchmark_role"]))

    def test_build_primary_model_selection_summary_separates_discovery_and_governance_tracks(
        self,
    ) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "discovery_primary", "roc_auc": 0.83, "average_precision": 0.77},
                {
                    "model_name": "regime_stability_priority",
                    "roc_auc": 0.79,
                    "average_precision": 0.73,
                },
                {"model_name": "conservative_model", "roc_auc": 0.72, "average_precision": 0.64},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)] * 3,
                "model_name": ["discovery_primary"] * 8
                + ["regime_stability_priority"] * 8
                + ["conservative_model"] * 8,
                "oof_prediction": [0.90, 0.88, 0.84, 0.80, 0.31, 0.25, 0.20, 0.12] * 3,
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0] * 3,
            }
        )
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "discovery_primary",
                    "selection_rank": 1,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.061,
                    "source_holdout_gap": -0.085,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.146,
                    "governance_priority_score": 0.684,
                },
                {
                    "model_name": "regime_stability_priority",
                    "selection_rank": 3,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.057,
                    "source_holdout_gap": -0.010,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.067,
                    "governance_priority_score": 0.723,
                },
            ]
        )
        decision_yield = pd.DataFrame(
            [
                {
                    "model_name": "discovery_primary",
                    "top_k": 10,
                    "precision_at_k": 0.40,
                    "recall_at_k": 0.50,
                },
                {
                    "model_name": "discovery_primary",
                    "top_k": 25,
                    "precision_at_k": 0.32,
                    "recall_at_k": 0.80,
                },
                {
                    "model_name": "conservative_model",
                    "top_k": 10,
                    "precision_at_k": 0.50,
                    "recall_at_k": 0.60,
                },
                {
                    "model_name": "conservative_model",
                    "top_k": 25,
                    "precision_at_k": 0.36,
                    "recall_at_k": 0.72,
                },
                {
                    "model_name": "regime_stability_priority",
                    "top_k": 10,
                    "precision_at_k": 0.60,
                    "recall_at_k": 0.70,
                },
                {
                    "model_name": "regime_stability_priority",
                    "top_k": 25,
                    "precision_at_k": 0.48,
                    "recall_at_k": 0.84,
                },
            ]
        )
        blocked_holdout_calibration_summary = pd.DataFrame(
            [
                {
                    "model_name": "discovery_primary",
                    "calibration_method": "raw",
                    "brier_score": 0.14,
                    "ece": 0.06,
                },
                {
                    "model_name": "discovery_primary",
                    "calibration_method": "platt",
                    "brier_score": 0.12,
                    "ece": 0.05,
                    "calibration_gain_vs_raw_brier": 0.02,
                    "calibration_gain_vs_raw_ece": 0.01,
                },
                {
                    "model_name": "conservative_model",
                    "calibration_method": "raw",
                    "brier_score": 0.15,
                    "ece": 0.07,
                },
                {
                    "model_name": "conservative_model",
                    "calibration_method": "isotonic",
                    "brier_score": 0.13,
                    "ece": 0.04,
                    "calibration_gain_vs_raw_brier": 0.02,
                    "calibration_gain_vs_raw_ece": 0.03,
                },
                {
                    "model_name": "regime_stability_priority",
                    "calibration_method": "raw",
                    "brier_score": 0.16,
                    "ece": 0.08,
                },
                {
                    "model_name": "regime_stability_priority",
                    "calibration_method": "beta",
                    "brier_score": 0.11,
                    "ece": 0.03,
                    "calibration_gain_vs_raw_brier": 0.05,
                    "calibration_gain_vs_raw_ece": 0.05,
                },
            ]
        )
        summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name="discovery_primary",
            conservative_model_name="conservative_model",
            predictions=predictions,
            decision_yield=decision_yield,
            blocked_holdout_calibration_summary=blocked_holdout_calibration_summary,
            model_selection_scorecard=scorecard,
        )
        row = summary.iloc[0]
        self.assertEqual(str(row["published_primary_track"]), "discovery")
        self.assertEqual(str(row["governance_primary_track"]), "governance_watch_only")
        self.assertEqual(str(row["governance_primary_benchmark_status"]), "governance_watch_only")
        self.assertEqual(str(row["governance_primary_model"]), "regime_stability_priority")
        self.assertEqual(float(row["published_primary_top_10_precision"]), 0.40)
        self.assertEqual(float(row["published_primary_top_25_recall"]), 0.80)
        self.assertEqual(float(row["governance_primary_top_10_precision"]), 0.60)
        self.assertEqual(float(row["governance_primary_top_25_recall"]), 0.84)
        self.assertEqual(
            str(row["published_primary_blocked_holdout_best_calibration_method"]), "platt"
        )
        self.assertEqual(
            float(row["published_primary_blocked_holdout_best_calibration_gain_vs_raw_brier"]),
            0.02,
        )
        self.assertEqual(
            str(row["governance_primary_blocked_holdout_best_calibration_method"]), "beta"
        )
        self.assertEqual(
            float(row["governance_primary_blocked_holdout_best_calibration_gain_vs_raw_ece"]),
            0.05,
        )
        self.assertIn("governance track", str(row["selection_rationale"]))
        self.assertIn("watch-only", str(row["governance_selection_rationale"]))
        self.assertIn("guardrail-aware candidate", str(row["governance_selection_rationale"]))

    def test_build_primary_model_selection_summary_prefers_explicit_governance_model_name(
        self,
    ) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "discovery_primary", "roc_auc": 0.83, "average_precision": 0.77},
                {
                    "model_name": "phylo_support_fusion_priority",
                    "roc_auc": 0.82,
                    "average_precision": 0.76,
                },
                {
                    "model_name": "structured_signal_priority",
                    "roc_auc": 0.79,
                    "average_precision": 0.73,
                },
                {"model_name": "conservative_model", "roc_auc": 0.72, "average_precision": 0.64},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)] * 4,
                "model_name": ["discovery_primary"] * 8
                + ["phylo_support_fusion_priority"] * 8
                + ["structured_signal_priority"] * 8
                + ["conservative_model"] * 8,
                "oof_prediction": [
                    0.90,
                    0.88,
                    0.84,
                    0.80,
                    0.31,
                    0.25,
                    0.20,
                    0.12,
                ]
                * 4,
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0] * 4,
            }
        )
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "discovery_primary",
                    "selection_rank": 1,
                    "strict_knownness_acceptance_flag": True,
                    "knownness_matched_gap": -0.061,
                    "source_holdout_gap": -0.085,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.146,
                    "governance_priority_score": 0.684,
                },
                {
                    "model_name": "phylo_support_fusion_priority",
                    "selection_rank": 2,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.057,
                    "source_holdout_gap": -0.010,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.067,
                    "governance_priority_score": 0.723,
                },
                {
                    "model_name": "structured_signal_priority",
                    "selection_rank": 3,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.030,
                    "source_holdout_gap": -0.020,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.091,
                    "governance_priority_score": 0.812,
                },
            ]
        )
        summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name="discovery_primary",
            conservative_model_name="conservative_model",
            governance_model_name="phylo_support_fusion_priority",
            predictions=predictions,
            model_selection_scorecard=scorecard,
        )
        row = summary.iloc[0]
        self.assertEqual(str(row["governance_primary_model"]), "phylo_support_fusion_priority")
        self.assertEqual(str(row["governance_primary_track"]), "governance_watch_only")

    def test_build_primary_model_selection_summary_prefers_governance_track_candidate(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "bio_clean_priority", "roc_auc": 0.83, "average_precision": 0.77},
                {
                    "model_name": "phylo_support_fusion_priority",
                    "roc_auc": 0.80,
                    "average_precision": 0.74,
                },
                {"model_name": "baseline_both", "roc_auc": 0.72, "average_precision": 0.64},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)] * 3,
                "model_name": ["bio_clean_priority"] * 8
                + ["phylo_support_fusion_priority"] * 8
                + ["baseline_both"] * 8,
                "oof_prediction": [0.90, 0.88, 0.84, 0.80, 0.31, 0.25, 0.20, 0.12] * 3,
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0] * 3,
            }
        )
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "bio_clean_priority",
                    "selection_rank": 1,
                    "strict_knownness_acceptance_flag": True,
                    "knownness_matched_gap": -0.001,
                    "source_holdout_gap": -0.001,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.002,
                    "governance_priority_score": 0.828,
                },
                {
                    "model_name": "phylo_support_fusion_priority",
                    "selection_rank": 2,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.010,
                    "source_holdout_gap": -0.012,
                    "leakage_review_required": False,
                    "guardrail_loss": 0.022,
                    "governance_priority_score": 0.778,
                },
            ]
        )

        summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name="bio_clean_priority",
            conservative_model_name="baseline_both",
            predictions=predictions,
            model_selection_scorecard=scorecard,
        )

        row = summary.iloc[0]
        self.assertEqual(str(row["governance_primary_model"]), "phylo_support_fusion_priority")
        self.assertEqual(str(row["governance_primary_track"]), "governance_watch_only")
        self.assertEqual(int(row["governance_primary_selection_rank"]), 2)
        self.assertIn("phylo_support_fusion_priority", str(row["governance_selection_rationale"]))

    def test_build_model_subgroup_performance_returns_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(20)],
                "refseq_share_train": [1.0] * 10 + [0.0] * 10,
                "member_count_train": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2] * 2,
                "n_countries_train": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 2,
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(20)],
                "model_name": ["enhanced_priority"] * 20,
                "oof_prediction": [0.1, 0.9] * 10,
                "spread_label": [0, 1] * 10,
            }
        )
        audit = build_model_subgroup_performance(
            predictions, scored, model_names=["enhanced_priority"]
        )
        self.assertIn("overall", set(audit["subgroup_name"]))
        self.assertIn("dominant_source", set(audit["subgroup_name"]))
        self.assertTrue((audit["model_name"] == "enhanced_priority").all())

    def test_build_model_comparison_table_returns_delta_columns(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["enhanced_priority"] * 12 + ["baseline_both"] * 12,
                "oof_prediction": [0.1, 0.2, 0.2, 0.8, 0.9, 0.7, 0.1, 0.2, 0.3, 0.8, 0.85, 0.9]
                + [0.4] * 12,
                "spread_label": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1] * 2,
            }
        )
        comparison = build_model_comparison_table(
            predictions,
            primary_model_name="enhanced_priority",
            comparison_model_names=["baseline_both"],
        )
        self.assertIn("delta_roc_auc", comparison.columns)
        self.assertEqual(comparison.iloc[0]["comparison_model_name"], "baseline_both")

    def test_build_calibration_metric_table_returns_ece(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "model_name": ["enhanced_priority"] * 8,
                "oof_prediction": [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
                "spread_label": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        calibration = build_calibration_metric_table(
            predictions,
            model_names=["enhanced_priority"],
            calibration_methods=("raw", "platt", "isotonic", "beta"),
            n_splits=4,
            n_repeats=1,
        )
        self.assertIn("expected_calibration_error", calibration.columns)
        self.assertIn("calibration_metric_family", calibration.columns)
        self.assertIn("calibration_metric_semantics", calibration.columns)
        self.assertEqual(set(calibration["evaluation_split"]), {"oof"})
        self.assertEqual(set(calibration["calibration_method"]), {"raw", "platt", "isotonic", "beta"})
        self.assertEqual(
            set(calibration["calibration_metric_family"]),
            {"fixed_bin_probability_calibration"},
        )
        self.assertEqual(calibration.iloc[0]["calibration_method"], "raw")

    def test_build_blocked_holdout_calibration_table_returns_method_rows(self) -> None:
        n = 50
        spread = [0, 1] * (n // 2)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": spread,
                "log1p_member_count_train": [0.2, 1.0] * (n // 2),
                "log1p_n_countries_train": [0.1, 0.9] * (n // 2),
                "refseq_share_train": [0.8] * (n // 2) + [0.2] * (n // 2),
                "dominant_source": ["source_a"] * 25 + ["source_b"] * 25,
            }
        )
        calibration = build_blocked_holdout_calibration_table(
            scored,
            model_names=["baseline_both"],
            group_columns=["dominant_source"],
            calibration_methods=("raw", "platt", "isotonic", "beta"),
            n_splits=3,
            n_repeats=1,
            seed=11,
        )
        self.assertFalse(calibration.empty)
        self.assertEqual(set(calibration["evaluation_split"]), {"blocked_holdout"})
        self.assertEqual(set(calibration["calibration_method"]), {"raw", "platt", "isotonic", "beta"})
        summary = build_blocked_holdout_calibration_summary(calibration)
        self.assertFalse(summary.empty)
        self.assertEqual(set(summary["evaluation_split"]), {"blocked_holdout"})
        self.assertEqual(set(summary["calibration_method"]), {"raw", "platt", "isotonic", "beta"})

    def test_build_source_balance_resampling_table_returns_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(24)],
                "spread_label": [0, 1] * 12,
                "priority_index": [0.2, 0.8] * 12,
                "arithmetic_priority_index": [0.2, 0.8] * 12,
                "T_eff_norm": [0.2, 0.8] * 12,
                "H_eff_norm": [0.3, 0.7] * 12,
                "A_eff_norm": [0.2, 0.8] * 12,
                "coherence_score": [0.4, 0.6] * 12,
                "orit_support": [0.3, 0.9] * 12,
                "log1p_member_count_train": [0.1, 1.2] * 12,
                "log1p_n_countries_train": [0.1, 0.8] * 12,
                "refseq_share_train": [1.0] * 12 + [0.0] * 12,
            }
        )
        resampling = build_source_balance_resampling_table(
            scored,
            model_name="enhanced_priority",
            n_resamples=3,
            seed=5,
        )
        self.assertEqual(len(resampling), 3)
        self.assertIn("roc_auc", resampling.columns)

    def test_build_permutation_null_tables_returns_summary(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "model_name": ["enhanced_priority"] * 12,
                "oof_prediction": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85],
                "spread_label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            }
        )
        detail, summary = build_permutation_null_tables(
            predictions, model_names=["enhanced_priority"], n_permutations=20, seed=3
        )
        self.assertEqual(len(detail), 20)
        self.assertEqual(summary.iloc[0]["model_name"], "enhanced_priority")
        self.assertIn("empirical_p_roc_auc", summary.columns)

    def test_build_selection_adjusted_permutation_null_returns_primary_summary(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "spread_label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "log1p_member_count_train": [
                    0.0,
                    0.1,
                    0.2,
                    0.2,
                    0.3,
                    0.4,
                    1.1,
                    1.2,
                    1.3,
                    1.2,
                    1.1,
                    1.0,
                ],
                "log1p_n_countries_train": [
                    0.0,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.3,
                    0.8,
                    0.9,
                    0.9,
                    1.0,
                    1.0,
                    1.1,
                ],
                "T_eff_norm": [0.1, 0.2, 0.1, 0.2, 0.3, 0.3, 0.8, 0.7, 0.8, 0.9, 0.8, 0.9],
                "H_eff_norm": [0.2, 0.2, 0.3, 0.3, 0.2, 0.1, 0.7, 0.8, 0.7, 0.8, 0.9, 0.8],
                "A_eff_norm": [0.1, 0.2, 0.2, 0.1, 0.2, 0.3, 0.7, 0.8, 0.8, 0.7, 0.9, 0.8],
                "coherence_score": [0.3, 0.3, 0.4, 0.4, 0.3, 0.2, 0.7, 0.8, 0.8, 0.9, 0.8, 0.9],
                "orit_support": [0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.7, 0.8, 0.9, 0.8, 0.7, 0.9],
            }
        )

        detail, summary = build_selection_adjusted_permutation_null(
            scored,
            model_names=["enhanced_priority", "baseline_both"],
            primary_model_name="enhanced_priority",
            n_permutations=3,
            n_splits=3,
            n_repeats=1,
            seed=7,
        )

        self.assertEqual(len(detail), 3)
        self.assertEqual(
            set(summary["model_name"].astype(str)), {"enhanced_priority", "baseline_both"}
        )
        self.assertIn("selection_adjusted_empirical_p_roc_auc", summary.columns)
        self.assertIn("modal_selected_model_name", summary.columns)

    def test_build_priority_bootstrap_stability_table_returns_candidate_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "member_count_train": [1] * 12,
                "n_countries_train": [1] * 12,
                "priority_index": [
                    0.95,
                    0.91,
                    0.89,
                    0.84,
                    0.8,
                    0.77,
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    0.5,
                    0.45,
                ],
                "T_eff": [0.9, 0.85, 0.82, 0.78, 0.76, 0.71, 0.66, 0.6, 0.55, 0.5, 0.44, 0.4],
                "H_eff": [0.88, 0.84, 0.8, 0.74, 0.7, 0.68, 0.63, 0.58, 0.52, 0.47, 0.42, 0.38],
                "A_eff": [0.92, 0.87, 0.81, 0.79, 0.75, 0.69, 0.61, 0.57, 0.51, 0.48, 0.43, 0.39],
            }
        )
        stability = build_priority_bootstrap_stability_table(
            scored, candidate_n=5, top_k=3, n_bootstrap=10, seed=2
        )
        self.assertEqual(len(stability), 5)
        self.assertIn("bootstrap_top_k_frequency", stability.columns)
        self.assertIn("bootstrap_top_10_frequency", stability.columns)

    def test_build_variant_rank_consistency_table_returns_frequency(self) -> None:
        base = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "member_count_train": [1, 1, 1, 1],
                "priority_index": [0.9, 0.8, 0.7, 0.6],
            }
        )
        variant = pd.DataFrame(
            {
                "backbone_id": ["bb2", "bb1", "bb3", "bb4"],
                "member_count_train": [1, 1, 1, 1],
                "priority_index": [0.91, 0.85, 0.65, 0.55],
            }
        )
        consistency = build_variant_rank_consistency_table(
            base, {"variant_a": variant}, candidate_n=3, top_k=2
        )
        self.assertEqual(len(consistency), 3)
        self.assertIn("variant_top_k_frequency", consistency.columns)
        self.assertIn("variant_top_10_frequency", consistency.columns)

    def test_build_knownness_audit_tables_returns_summary_and_strata(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(30)],
                "priority_index": [0.1 + 0.02 * i for i in range(30)],
                "log1p_member_count_train": [1.0] * 30,
                "log1p_n_countries_train": [0.7] * 30,
                "refseq_share_train": [0.0] * 30,
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(30)] * 2,
                "model_name": ["proxy_light_priority"] * 30 + ["baseline_both"] * 30,
                "oof_prediction": ([0.2, 0.8, 0.3, 0.7, 0.4, 0.9] * 5)
                + ([0.4, 0.6, 0.45, 0.55, 0.5, 0.6] * 5),
                "spread_label": ([0, 1, 0, 1, 0, 1] * 5) * 2,
            }
        )
        summary, strata = build_knownness_audit_tables(
            predictions,
            scored,
            primary_model_name="proxy_light_priority",
            baseline_model_name="baseline_both",
            top_k=10,
        )
        self.assertEqual(len(summary), 1)
        self.assertIn("overall_delta_roc_auc", summary.columns)
        self.assertIn("matched_strata_weighted_delta_roc_auc", summary.columns)
        self.assertIn("top_k_lower_half_knownness_fraction", summary.columns)
        self.assertFalse(strata.empty)

    def test_build_knownness_audit_tables_keeps_q1_empty_when_quartiles_unsupported(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "priority_index": np.linspace(0.1, 0.9, 12),
                "log1p_member_count_train": [0.0] * 12,
                "log1p_n_countries_train": [0.0] * 12,
                "refseq_share_train": [0.0] * 12,
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["proxy_light_priority"] * 12 + ["baseline_both"] * 12,
                "oof_prediction": ([0.2, 0.8] * 6) + ([0.45, 0.55] * 6),
                "spread_label": ([0, 1] * 6) * 2,
            }
        )

        summary, _ = build_knownness_audit_tables(
            predictions,
            scored,
            primary_model_name="proxy_light_priority",
            baseline_model_name="baseline_both",
            top_k=6,
        )

        row = summary.iloc[0]
        self.assertFalse(bool(row["lowest_knownness_quartile_supported"]))
        self.assertEqual(int(row["lowest_knownness_quartile_n_backbones"]), 0)
        self.assertTrue(pd.isna(row["lowest_knownness_quartile_primary_roc_auc"]))

    def test_build_novelty_margin_summary_returns_watchlist_columns(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(24)],
                "priority_index": [0.1 + 0.02 * i for i in range(24)],
                "log1p_member_count_train": [0.2 + 0.05 * (i % 6) for i in range(24)],
                "log1p_n_countries_train": [0.1 + 0.05 * (i % 4) for i in range(24)],
                "refseq_share_train": [0.0, 1.0] * 12,
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(24)] * 2,
                "model_name": ["proxy_light_priority"] * 24 + ["baseline_both"] * 24,
                "oof_prediction": ([0.2, 0.8, 0.3, 0.7, 0.35, 0.85] * 4)
                + ([0.4, 0.6, 0.45, 0.55, 0.5, 0.6] * 4),
                "spread_label": ([0, 1, 0, 1, 0, 1] * 4) * 2,
            }
        )
        summary = build_novelty_margin_summary(
            predictions,
            scored,
            primary_model_name="proxy_light_priority",
            baseline_model_name="baseline_both",
            top_k=8,
        )
        self.assertEqual(len(summary), 1)
        self.assertIn("watchlist_positive_fraction", summary.columns)
        self.assertIn("lower_half_knownness_novelty_margin_roc_auc", summary.columns)

    def test_build_group_holdout_performance_returns_rows(self) -> None:
        n = 40
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": [0, 1] * (n // 2),
                "priority_index": [0.1, 0.9] * (n // 2),
                "arithmetic_priority_index": [0.1, 0.9] * (n // 2),
                "T_eff_norm": [0.1, 0.9] * (n // 2),
                "H_eff_norm": [0.2, 0.8] * (n // 2),
                "A_eff_norm": [0.15, 0.85] * (n // 2),
                "coherence_score": [0.3, 0.8] * (n // 2),
                "orit_support": [0.2, 0.95] * (n // 2),
                "log1p_member_count_train": [0.2, 1.3] * (n // 2),
                "log1p_n_countries_train": [0.1, 0.9] * (n // 2),
                "refseq_share_train": [1.0] * 20 + [0.0] * 20,
                "dominant_source": ["refseq_leaning"] * 20 + ["insd_leaning"] * 20,
                "dominant_genus_train": ["Escherichia"] * 20 + ["Klebsiella"] * 20,
            }
        )
        holdout = build_group_holdout_performance(
            scored,
            model_names=["enhanced_priority"],
            group_columns=["dominant_source", "dominant_genus_train"],
            min_group_size=10,
            max_groups_per_column=4,
        )
        self.assertIn("group_column", holdout.columns)
        self.assertTrue((holdout["model_name"] == "enhanced_priority").all())

    def test_build_blocked_holdout_summary_filters_failed_rows(self) -> None:
        group_holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "group_value": "refseq_leaning",
                    "model_name": "enhanced_priority",
                    "status": "ok",
                    "roc_auc": 0.84,
                    "n_test_backbones": 20,
                },
                {
                    "group_column": "dominant_source",
                    "group_value": "insd_leaning",
                    "model_name": "enhanced_priority",
                    "status": "ok",
                    "roc_auc": 0.76,
                    "n_test_backbones": 18,
                },
                {
                    "group_column": "dominant_source",
                    "group_value": "refseq_leaning",
                    "model_name": "enhanced_priority",
                    "status": "ok",
                    "roc_auc": 0.82,
                    "n_test_backbones": 19,
                },
                {
                    "group_column": "dominant_region_train",
                    "group_value": "Europe",
                    "model_name": "enhanced_priority",
                    "status": "ok",
                    "roc_auc": 0.81,
                    "n_test_backbones": 22,
                },
                {
                    "group_column": "dominant_region_train",
                    "group_value": "Asia",
                    "model_name": "enhanced_priority",
                    "status": "failed",
                    "roc_auc": 0.99,
                    "n_test_backbones": 5,
                },
                {
                    "group_column": "dominant_source",
                    "group_value": "refseq_leaning",
                    "model_name": "baseline_both",
                    "status": "ok",
                    "roc_auc": 0.78,
                    "n_test_backbones": 20,
                },
            ]
        )
        summary = build_blocked_holdout_summary(group_holdout)
        self.assertIn("blocked_holdout_roc_auc", summary.columns)
        self.assertIn("pooled_overlap_summary", summary.columns)
        self.assertTrue(summary["pooled_overlap_summary"].all())
        self.assertEqual(set(summary["model_name"]), {"enhanced_priority", "baseline_both"})
        self.assertEqual(
            set(summary.loc[summary["model_name"] == "enhanced_priority", "blocked_holdout_group_columns"]),
            {"dominant_source", "dominant_region_train"},
        )
        source = summary.loc[
            (summary["model_name"] == "enhanced_priority")
            & (summary["blocked_holdout_group_columns"] == "dominant_source")
        ].iloc[0]
        region = summary.loc[
            (summary["model_name"] == "enhanced_priority")
            & (summary["blocked_holdout_group_columns"] == "dominant_region_train")
        ].iloc[0]
        self.assertEqual(int(source["blocked_holdout_group_count"]), 2)
        self.assertEqual(int(region["blocked_holdout_group_count"]), 1)
        self.assertNotIn("dominant_region_train:Asia", str(region["worst_blocked_holdout_group"]))

    def test_blocked_holdout_count_coherence_with_pooled_overlap_flag(self) -> None:
        """Test that blocked-holdout summary marks pooled-overlap to prevent count misinterpretation."""
        group_holdout = pd.DataFrame(
            [
                {
                    "group_column": "dominant_source",
                    "group_value": "refseq_leaning",
                    "model_name": "test_model",
                    "status": "ok",
                    "roc_auc": 0.80,
                    "n_test_backbones": 50,
                },
                {
                    "group_column": "dominant_source",
                    "group_value": "insd_leaning",
                    "model_name": "test_model",
                    "status": "ok",
                    "roc_auc": 0.75,
                    "n_test_backbones": 40,
                },
                {
                    "group_column": "dominant_region_train",
                    "group_value": "Europe",
                    "model_name": "test_model",
                    "status": "ok",
                    "roc_auc": 0.78,
                    "n_test_backbones": 45,
                },
                {
                    "group_column": "dominant_region_train",
                    "group_value": "Asia",
                    "model_name": "test_model",
                    "status": "ok",
                    "roc_auc": 0.72,
                    "n_test_backbones": 35,
                },
            ]
        )
        summary = build_blocked_holdout_summary(group_holdout)
        
        # Verify pooled_overlap_summary flag is present and True
        self.assertIn("pooled_overlap_summary", summary.columns)
        self.assertTrue(summary["pooled_overlap_summary"].all())
        
        # Verify that separate axes (source and region) are reported
        self.assertEqual(len(summary), 2)  # One for source, one for region
        
        # The pooled n_backbones may exceed a single disjoint cohort
        # This is expected and allowed when pooled_overlap_summary is True
        source_n = summary.loc[summary["blocked_holdout_group_columns"] == "dominant_source", "blocked_holdout_n_backbones"].iloc[0]
        region_n = summary.loc[summary["blocked_holdout_group_columns"] == "dominant_region_train", "blocked_holdout_n_backbones"].iloc[0]
        
        # These counts are from different axes and may overlap
        # The pooled_overlap_summary flag signals this to prevent misinterpretation
        self.assertGreater(source_n, 0)
        self.assertGreater(region_n, 0)

    def test_build_negative_control_audit_returns_noise_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(20)],
                "spread_label": [0, 1] * 10,
                "log1p_member_count_train": [0.2, 1.3] * 10,
                "log1p_n_countries_train": [0.1, 0.9] * 10,
                "T_eff_norm": [0.1, 0.8] * 10,
                "H_eff_norm": [0.2, 0.7] * 10,
                "A_eff_norm": [0.15, 0.75] * 10,
                "coherence_score": [0.3, 0.8] * 10,
                "orit_support": [0.2, 0.9] * 10,
            }
        )
        audit = build_negative_control_audit(
            scored, primary_model_name="enhanced_priority", n_splits=4, n_repeats=2, seed=3
        )
        self.assertIn("primary_model", set(audit["audit_name"]))
        self.assertIn("negative_control_noise_a_only", set(audit["audit_name"]))
        self.assertIn("delta_roc_auc_vs_primary", audit.columns)

    def test_build_model_simplicity_summary_returns_overlap_columns(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "enhanced_priority",
                    "roc_auc": 0.78,
                    "average_precision": 0.81,
                    "brier_score": 0.19,
                },
                {
                    "model_name": "proxy_light_priority",
                    "roc_auc": 0.77,
                    "average_precision": 0.80,
                    "brier_score": 0.20,
                },
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["enhanced_priority"] * 12 + ["proxy_light_priority"] * 12,
                "oof_prediction": [0.95, 0.9, 0.88, 0.83, 0.8, 0.78, 0.4, 0.35, 0.3, 0.2, 0.1, 0.05]
                + [0.94, 0.89, 0.84, 0.8, 0.76, 0.74, 0.38, 0.33, 0.28, 0.18, 0.09, 0.04],
                "spread_label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 2,
            }
        )
        summary = build_model_simplicity_summary(
            model_metrics,
            predictions,
            primary_model_name="enhanced_priority",
            conservative_model_name="proxy_light_priority",
            top_ks=(3, 5),
        )
        self.assertEqual(summary.iloc[0]["top_3_overlap_count"], 3)
        self.assertIn("roc_auc_delta_primary_minus_conservative", summary.columns)

    def test_build_temporal_drift_summary_returns_year_rows(self) -> None:
        records = pd.DataFrame(
            {
                "resolved_year": [2014, 2014, 2015, 2016],
                "backbone_id": ["bb1", "bb2", "bb1", "bb3"],
                "country": ["TR", "US", "TR", "DE"],
                "genus": ["Escherichia", "Klebsiella", "Escherichia", "Bacillus"],
                "record_origin": ["refseq", "insd", "refseq", "insd"],
                "is_mobilizable": [True, False, True, True],
                "is_conjugative": [False, False, True, False],
            }
        )
        summary = build_temporal_drift_summary(records)
        self.assertEqual(set(summary["resolved_year"]), {2014, 2015, 2016})

    def test_build_h_feature_diagnostics_returns_core_columns(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "spread_label": [0, 1, 0, 1],
                "priority_index": [0.2, 0.8, 0.25, 0.85],
                "H_eff_norm": [0.1, 0.7, 0.2, 0.8],
                "log1p_member_count_train": [0.1, 1.0, 0.2, 1.1],
                "log1p_n_countries_train": [0.1, 0.8, 0.1, 0.9],
            }
        )
        coefficient_table = pd.DataFrame({"feature_name": ["H_eff_norm"], "coefficient": [0.25]})
        dropout_table = pd.DataFrame(
            {"feature_name": ["H_eff_norm"], "roc_auc_drop_vs_full": [0.03]}
        )
        diagnostics = build_h_feature_diagnostics(
            scored,
            coefficient_table=coefficient_table,
            dropout_table=dropout_table,
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("primary_model_h_coefficient", diagnostics.columns)
        self.assertIn("h_eff_norm_vs_spread_label_spearman", diagnostics.columns)

    def test_build_score_distribution_diagnostics_splits_low_cluster(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(6)],
                "priority_index": [0.1, 0.12, 0.18, 0.75, 0.82, 0.9],
                "member_count_train": [0, 0, 2, 2, 3, 4],
                "T_eff_norm": [0.4, 0.4, 0.3, 0.8, 0.85, 0.9],
                "H_eff_norm": [0.0, 0.05, 0.1, 0.75, 0.8, 0.82],
                "A_eff_norm": [0.3, 0.2, 0.4, 0.7, 0.82, 0.87],
            }
        )
        diagnostics = build_score_distribution_diagnostics(scored, low_score_threshold=0.25)
        self.assertIn("low_score_cluster", set(diagnostics["segment"]))
        self.assertIn("dominant_floor_H_fraction", diagnostics.columns)

    def test_build_component_floor_diagnostics_reports_zero_floor_values(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [1, 2, 3],
                "spread_label": [0, 1, 1],
                "T_eff": [0.0, 0.2, 0.6],
                "H_eff": [0.0, 0.3, 0.7],
                "A_eff": [0.0, 0.0, 0.8],
                "T_eff_norm": [0.0, 0.4, 0.9],
                "H_eff_norm": [0.0, 0.5, 0.95],
                "A_eff_norm": [0.0, 0.0, 0.92],
            }
        )
        diagnostics = build_component_floor_diagnostics(scored)
        self.assertEqual(set(diagnostics["component"]), {"T", "H", "A"})
        a_row = diagnostics.loc[diagnostics["component"] == "A"].iloc[0]
        self.assertEqual(float(a_row["normalized_value_when_raw_zero_median"]), 0.0)
        self.assertGreater(float(a_row["zero_fraction_training_reference"]), 0.0)

    def test_build_amrfinder_coverage_table_returns_fraction_columns(self) -> None:
        summary = pd.DataFrame(
            {
                "priority_group": ["high", "low", "overall"],
                "n_sequences": [6, 6, 12],
                "n_with_amrfinder_hits": [3, 0, 3],
                "n_with_any_amr_evidence": [4, 0, 4],
                "mean_gene_jaccard": [0.7, 1.0, 0.85],
                "mean_class_jaccard": [0.8, 1.0, 0.9],
            }
        )
        coverage = build_amrfinder_coverage_table(summary)
        self.assertEqual(set(coverage["priority_group"]), {"high", "low"})
        self.assertIn("amr_evidence_fraction", coverage.columns)

    def test_build_candidate_dossier_and_risk_tables_return_support_fields(self) -> None:
        base = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "freeze_rank": [1, 2],
                "priority_index": [0.9, 0.8],
                "coherence_score": [0.7, 0.6],
                "consensus_support_count": [3, 2],
                "member_count_train": [4, 1],
                "n_countries_train": [3, 1],
                "refseq_share_train": [0.6, 1.0],
                "insd_share_train": [0.4, 0.0],
            }
        )
        stability = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "bootstrap_top_k_frequency": [0.9, 0.72],
                "variant_top_k_frequency": [0.8, 0.7],
                "primary_model_full_fit_prediction_std": [0.18, 0.60],
                "assignment_confidence_score": [0.92, 0.55],
                "mash_graph_novelty_score": [0.18, 0.84],
                "mash_graph_bridge_fraction": [0.12, 0.78],
                "amr_agreement_score": [0.81, 0.42],
                "mean_amr_uncertainty_score": [0.12, 0.52],
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb1", "bb2"],
                "model_name": [
                    "enhanced_priority",
                    "enhanced_priority",
                    "proxy_light_priority",
                    "proxy_light_priority",
                ],
                "oof_prediction": [0.8, 0.6, 0.7, 0.3],
                "spread_label": [1, 0, 1, 0],
            }
        )
        who = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "who_mia_any_support": [True],
                "who_mia_any_hpecia": [True],
                "who_mia_mapped_fraction": [1.0],
            }
        )
        card = pd.DataFrame(
            {"backbone_id": ["bb1"], "card_any_support": [True], "card_match_fraction": [0.5]}
        )
        mobsuite = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "mobsuite_any_literature_support": [True],
                "mobsuite_any_cluster_support": [True],
            }
        )
        pathogen = pd.DataFrame(
            {
                "pathogen_dataset": ["combined"],
                "backbone_id": ["bb1"],
                "pd_any_support": [True],
                "pd_matching_fraction": [0.4],
            }
        )
        amrfinder = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "amrfinder_any_hit": [True],
                "gene_jaccard": [0.5],
                "class_jaccard": [0.5],
            }
        )
        dossier = build_candidate_dossier_table(
            base,
            candidate_stability=stability,
            predictions=predictions,
            primary_model_name="enhanced_priority",
            conservative_model_name="proxy_light_priority",
            who_detail=who,
            card_detail=card,
            mobsuite_detail=mobsuite,
            pathogen_support=pathogen,
            amrfinder_detail=amrfinder,
        )
        risk = build_candidate_risk_table(dossier)
        self.assertIn("candidate_confidence_tier", dossier.columns)
        self.assertIn("external_support_modalities_count", dossier.columns)
        self.assertIn("primary_driver_axis", dossier.columns)
        self.assertIn("mechanistic_rationale", dossier.columns)
        self.assertIn("monitoring_rationale", dossier.columns)
        self.assertIn("candidate_confidence_score", dossier.columns)
        self.assertIn("multiverse_stability_score", dossier.columns)
        self.assertIn("multiverse_stability_tier", dossier.columns)
        self.assertIn("candidate_explanation_summary", dossier.columns)
        self.assertIn("low_candidate_confidence_risk", dossier.columns)
        self.assertIn("model_prediction_uncertainty", dossier.columns)
        self.assertIn("uncertainty_review_tier", dossier.columns)
        self.assertIn("assignment_confidence_score", dossier.columns)
        self.assertIn("mash_graph_novelty_score", dossier.columns)
        self.assertIn("amr_agreement_score", dossier.columns)
        self.assertFalse(
            any(column.endswith("_x") or column.endswith("_y") for column in dossier.columns)
        )
        self.assertEqual(
            int(
                dossier.loc[
                    dossier["backbone_id"] == "bb1", "external_support_modalities_count"
                ].iloc[0]
            ),
            4,
        )
        self.assertGreater(
            float(dossier.loc[dossier["backbone_id"] == "bb2", "risk_uncertainty"].iloc[0]),
            float(dossier.loc[dossier["backbone_id"] == "bb1", "risk_uncertainty"].iloc[0]),
        )
        self.assertGreater(
            float(dossier.loc[dossier["backbone_id"] == "bb1", "candidate_confidence_score"].iloc[0]),
            0.80,
        )
        self.assertEqual(
            str(dossier.loc[dossier["backbone_id"] == "bb1", "uncertainty_review_tier"].iloc[0]),
            "review",
        )
        self.assertEqual(
            str(dossier.loc[dossier["backbone_id"] == "bb2", "uncertainty_review_tier"].iloc[0]),
            "abstain",
        )
        self.assertIn(
            "assignment_confidence_high",
            str(dossier.loc[dossier["backbone_id"] == "bb1", "monitoring_rationale"].iloc[0]),
        )
        self.assertEqual(
            str(dossier.loc[dossier["backbone_id"] == "bb1", "primary_driver_axis"].iloc[0]),
            "assignment_confidence",
        )
        self.assertEqual(
            str(dossier.loc[dossier["backbone_id"] == "bb2", "primary_driver_axis"].iloc[0]),
            "graph_novelty",
        )
        self.assertEqual(
            str(dossier.loc[dossier["backbone_id"] == "bb2", "secondary_driver_axis"].iloc[0]),
            "graph_bridge",
        )
        self.assertIn("false_positive_risk_tier", risk.columns)
        self.assertIn("uncertainty_review_tier", risk.columns)
        self.assertIn("low_assignment_confidence_risk", risk.columns)
        self.assertIn("graph_novelty_risk", risk.columns)
        self.assertIn("amr_uncertainty_risk", risk.columns)
        self.assertIn("low_candidate_confidence_risk", risk.columns)
        self.assertTrue(bool(risk.loc[risk["backbone_id"] == "bb2", "low_assignment_confidence_risk"].iloc[0]))
        self.assertTrue(bool(risk.loc[risk["backbone_id"] == "bb2", "graph_novelty_risk"].iloc[0]))
        self.assertTrue(bool(risk.loc[risk["backbone_id"] == "bb2", "amr_uncertainty_risk"].iloc[0]))
        self.assertTrue(bool(risk.loc[risk["backbone_id"] == "bb2", "low_candidate_confidence_risk"].iloc[0]))

    def test_build_consensus_candidate_ranking_prefers_cross_model_agreement(self) -> None:
        candidate_context = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "member_count_train": [3, 3, 3, 3],
                "primary_model_full_fit_prediction": [0.95, 0.94, 0.60, 0.20],
                "conservative_model_full_fit_prediction": [0.92, 0.20, 0.91, 0.10],
                "bio_priority_index": [0.88, 0.30, 0.86, 0.05],
                "operational_priority_index": [0.81, 0.50, 0.79, 0.10],
                "coherence_score": [0.9, 0.6, 0.8, 0.4],
                "spread_label": [1, 0, pd.NA, 1],
            }
        )
        consensus = build_consensus_candidate_ranking(
            candidate_context,
            primary_score_column="primary_model_full_fit_prediction",
            conservative_score_column="conservative_model_full_fit_prediction",
            top_k=4,
        )
        self.assertEqual(consensus.iloc[0]["backbone_id"], "bb1")
        self.assertNotIn("bb3", set(consensus["backbone_id"]))
        self.assertIn("consensus_candidate_score", consensus.columns)
        self.assertIn("consensus_support_count", consensus.columns)

    def test_build_candidate_portfolio_table_keeps_both_tracks(self) -> None:
        candidate_dossiers = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "candidate_confidence_tier": ["tier_a", "tier_b"],
                "candidate_confidence_score": [0.88, 0.63],
                "candidate_explanation_summary": [
                    "Confidence 0.88; primary mobility; review clear.",
                    "Confidence 0.63; primary graph_novelty; review review.",
                ],
                "priority_index": [0.9, 0.8],
                "primary_model_candidate_score": [0.92, 0.83],
                "baseline_both_candidate_score": [0.71, 0.62],
                "novelty_margin_vs_baseline": [0.21, 0.21],
                "candidate_prediction_source": ["oof", "full_fit"],
                "eligible_for_oof": [True, False],
                "knownness_score": [0.8, 0.7],
                "knownness_half": ["upper_half", "upper_half"],
                "assignment_confidence_score": [0.94, 0.57],
                "mash_graph_novelty_score": [0.21, 0.88],
                "mash_graph_bridge_fraction": [0.11, 0.79],
                "amr_agreement_score": [0.80, 0.44],
                "mean_amr_uncertainty_score": [0.12, 0.48],
                "refseq_share_train": [0.9, 0.6],
                "insd_share_train": [0.1, 0.4],
                "bootstrap_top_10_frequency": [0.9, 0.4],
                "who_mia_any_support": [True, pd.NA],
                "card_any_support": [False, pd.NA],
                "mobsuite_any_literature_support": [pd.NA, pd.NA],
                "pd_any_support": [True, pd.NA],
                "false_positive_risk_tier": ["low", "medium"],
                "uncertainty_review_tier": ["clear", "review"],
            }
        )
        novelty_watchlist = pd.DataFrame(
            {
                "backbone_id": ["bb3", "bb4"],
                "priority_index": [0.6, 0.58],
                "primary_model_oof_prediction": [0.84, 0.8],
                "baseline_both_oof_prediction": [0.35, 0.4],
                "primary_model_candidate_score": [0.84, 0.8],
                "baseline_both_candidate_score": [0.35, 0.4],
                "novelty_margin_vs_baseline": [0.49, 0.4],
                "candidate_prediction_source": ["oof", "oof"],
                "eligible_for_oof": [True, True],
                "knownness_score": [0.3, 0.32],
                "knownness_half": ["lower_half", "lower_half"],
                "spread_label": [1, 0],
            }
        )
        portfolio = build_candidate_portfolio_table(
            candidate_dossiers, novelty_watchlist, established_n=1, novel_n=1
        )
        self.assertEqual(
            set(portfolio["portfolio_track"]), {"established_high_risk", "novel_signal"}
        )
        self.assertIn("primary_model_candidate_score", portfolio.columns)
        self.assertIn("candidate_prediction_source", portfolio.columns)
        self.assertIn("recommended_monitoring_tier", portfolio.columns)
        self.assertIn("source_support_tier", portfolio.columns)
        self.assertIn("uncertainty_review_tier", portfolio.columns)
        self.assertIn("candidate_confidence_score", portfolio.columns)
        self.assertIn("multiverse_stability_score", portfolio.columns)
        self.assertIn("multiverse_stability_tier", portfolio.columns)
        self.assertIn("candidate_explanation_summary", portfolio.columns)
        self.assertIn("low_candidate_confidence_risk", portfolio.columns)
        self.assertEqual(
            int(
                portfolio.loc[
                    portfolio["backbone_id"] == "bb1", "external_support_modalities_count"
                ].iloc[0]
            ),
            2,
        )
        self.assertGreater(
            float(portfolio.loc[portfolio["backbone_id"] == "bb1", "candidate_confidence_score"].iloc[0]),
            0.80,
        )
        self.assertEqual(
            str(portfolio.loc[portfolio["backbone_id"] == "bb1", "uncertainty_review_tier"].iloc[0]),
            "clear",
        )
        self.assertIn("assignment_confidence_score", portfolio.columns)
        self.assertIn("mash_graph_novelty_score", portfolio.columns)
        self.assertIn("amr_agreement_score", portfolio.columns)

    def test_build_candidate_portfolio_table_preserves_source_coverage_diversity(self) -> None:
        candidate_dossiers = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "candidate_confidence_tier": ["tier_a", "tier_a", "tier_a"],
                "priority_index": [0.95, 0.93, 0.91],
                "primary_model_candidate_score": [0.95, 0.93, 0.91],
                "baseline_both_candidate_score": [0.72, 0.70, 0.69],
                "novelty_margin_vs_baseline": [0.23, 0.23, 0.22],
                "candidate_prediction_source": ["oof", "oof", "oof"],
                "eligible_for_oof": [True, True, True],
                "knownness_score": [0.82, 0.75, 0.74],
                "knownness_half": ["upper_half", "upper_half", "upper_half"],
                "refseq_share_train": [0.92, 0.48, 0.08],
                "insd_share_train": [0.08, 0.52, 0.92],
                "bootstrap_top_10_frequency": [0.9, 0.9, 0.9],
                "who_mia_any_support": [True, True, True],
                "card_any_support": [False, False, False],
                "mobsuite_any_literature_support": [pd.NA, pd.NA, pd.NA],
                "pd_any_support": [True, True, True],
                "false_positive_risk_tier": ["low", "low", "low"],
                "spread_label": [1, 1, 1],
            }
        )
        portfolio = build_candidate_portfolio_table(
            candidate_dossiers, pd.DataFrame(), established_n=2, novel_n=0
        )
        self.assertEqual(len(portfolio), 2)
        self.assertEqual(len(set(portfolio["source_support_tier"])), 2)
        self.assertIn("cross_source_supported", set(portfolio["source_support_tier"]))
        self.assertTrue(
            {"refseq_dominant", "insd_dominant"} & set(portfolio["source_support_tier"])
        )

    def test_build_candidate_portfolio_table_reserves_low_knownness_coverage(self) -> None:
        candidate_dossiers = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "candidate_confidence_tier": ["tier_a", "tier_a", "tier_a"],
                "priority_index": [0.99, 0.98, 0.97],
                "primary_model_candidate_score": [0.99, 0.98, 0.97],
                "baseline_both_candidate_score": [0.80, 0.79, 0.78],
                "novelty_margin_vs_baseline": [0.19, 0.19, 0.19],
                "candidate_prediction_source": ["oof", "oof", "oof"],
                "eligible_for_oof": [True, True, True],
                "knownness_score": [0.9, 0.88, 0.28],
                "knownness_half": ["upper_half", "upper_half", "lower_half"],
                "refseq_share_train": [0.9, 0.9, 0.9],
                "insd_share_train": [0.1, 0.1, 0.1],
                "bootstrap_top_10_frequency": [0.9, 0.9, 0.9],
                "who_mia_any_support": [True, True, True],
                "card_any_support": [False, False, False],
                "mobsuite_any_literature_support": [pd.NA, pd.NA, pd.NA],
                "pd_any_support": [True, True, True],
                "false_positive_risk_tier": ["low", "low", "low"],
                "spread_label": [1, 1, 1],
            }
        )
        portfolio = build_candidate_portfolio_table(
            candidate_dossiers, pd.DataFrame(), established_n=2, novel_n=0
        )
        self.assertEqual(len(portfolio), 2)
        self.assertGreaterEqual(
            int((portfolio["knownness_half"].astype(str) == "lower_half").sum()), 1
        )
        self.assertIn("bb3", set(portfolio["backbone_id"]))

    def test_build_logistic_implementation_audit_returns_similarity_metrics(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(20)],
                "spread_label": [0, 1] * 10,
                "T_raw_norm": [0.2, 0.8] * 10,
                "H_specialization_norm": [0.3, 0.7] * 10,
                "A_raw_norm": [0.15, 0.85] * 10,
                "orit_support": [0.2, 0.95] * 10,
                "H_support_norm_residual": [0.05, 0.18] * 10,
                "refseq_share_train": [1.0] * 10 + [0.0] * 10,
                "log1p_member_count_train": [0.1, 1.2] * 10,
                "log1p_n_countries_train": [0.1, 0.8] * 10,
            }
        )
        audit = build_logistic_implementation_audit(
            scored,
            model_name="visibility_adjusted_priority",
            columns=[
                "T_raw_norm",
                "H_specialization_norm",
                "A_raw_norm",
                "orit_support",
                "H_support_norm_residual",
            ],
            n_splits=2,
            n_repeats=1,
            seed=1,
        )
        self.assertEqual(len(audit), 1)
        self.assertIn("pearson_prediction_correlation", audit.columns)
        self.assertIn("max_absolute_prediction_difference", audit.columns)
        self.assertGreater(float(audit.iloc[0]["pearson_prediction_correlation"]), 0.95)

    def test_build_decision_yield_table_returns_topk_metrics(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(6)],
                "model_name": ["parsimonious_priority"] * 6,
                "oof_prediction": [0.95, 0.9, 0.8, 0.3, 0.2, 0.1],
                "spread_label": [1, 1, 0, 0, 1, 0],
            }
        )
        decision_yield = build_decision_yield_table(
            predictions, model_names=["parsimonious_priority"], top_ks=(2, 4)
        )
        self.assertEqual(set(decision_yield["top_k"]), {2, 4})
        top2 = decision_yield.loc[decision_yield["top_k"] == 2].iloc[0]
        self.assertEqual(int(top2["n_positive_selected"]), 2)
        self.assertAlmostEqual(float(top2["precision_at_k"]), 1.0)

    def test_build_threshold_flip_table_marks_eligibility_and_flips(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "member_count_train": [2, 5],
                "n_countries_train": [2, 5],
                "n_new_countries": [2, 4],
                "priority_index": [0.9, 0.7],
                "spread_label": [0, float("nan")],
            }
        )
        audit = build_threshold_flip_table(scored, thresholds=(1, 2, 3, 4), default_threshold=3)
        bb1 = audit.loc[audit["backbone_id"] == "bb1"].iloc[0]
        bb2 = audit.loc[audit["backbone_id"] == "bb2"].iloc[0]
        self.assertTrue(bool(bb1["eligible_for_threshold_audit"]))
        self.assertEqual(int(bb1["threshold_flip_count"]), 2)
        self.assertFalse(bool(bb2["eligible_for_threshold_audit"]))

    def test_build_candidate_universe_table_sets_origin_flags(self) -> None:
        scored = pd.DataFrame(
            {"backbone_id": ["bb1", "bb2"], "priority_index": [0.9, 0.7], "spread_label": [1, 0]}
        )
        consensus = pd.DataFrame({"backbone_id": ["bb1"], "consensus_rank": [1]})
        dossiers = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "candidate_confidence_tier": ["tier_a"],
                "recommended_monitoring_tier": ["core_surveillance"],
            }
        )
        portfolio = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "portfolio_track": ["established_high_risk"],
                "track_rank": [1],
            }
        )
        novelty = pd.DataFrame(
            {
                "backbone_id": ["bb2"],
                "knownness_half": ["lower_half"],
                "novelty_margin_vs_baseline": [0.2],
            }
        )
        freeze = pd.DataFrame({"backbone_id": ["bb2"], "freeze_rank": [2]})
        high_conf = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "candidate_confidence_tier": ["tier_a"],
                "false_positive_risk_tier": ["low"],
            }
        )
        risk = pd.DataFrame(
            {
                "backbone_id": ["bb2"],
                "false_positive_risk_tier": ["medium"],
                "risk_flag_count": [1],
                "risk_flags": ["stability_risk"],
            }
        )
        universe = build_candidate_universe_table(
            scored=scored,
            consensus_candidates=consensus,
            candidate_dossiers=dossiers,
            candidate_portfolio=portfolio,
            novelty_watchlist=novelty,
            prospective_freeze=freeze,
            high_confidence_candidates=high_conf,
            candidate_risk=risk,
        )
        bb1 = universe.loc[universe["backbone_id"] == "bb1"].iloc[0]
        bb2 = universe.loc[universe["backbone_id"] == "bb2"].iloc[0]
        self.assertTrue(bool(bb1["in_candidate_portfolio"]))
        self.assertEqual(str(bb1["candidate_universe_origin"]), "portfolio_established")
        self.assertTrue(bool(bb2["in_novelty_watchlist"]))

    def test_build_primary_model_selection_summary_reports_strongest_overlap(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "roc_auc": 0.76,
                    "average_precision": 0.67,
                    "brier_score": 0.18,
                },
                {
                    "model_name": "evidence_aware_priority",
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "brier_score": 0.17,
                },
                {
                    "model_name": "bio_clean_priority",
                    "roc_auc": 0.77,
                    "average_precision": 0.68,
                    "brier_score": 0.18,
                },
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 3,
                "model_name": ["parsimonious_priority"] * 12
                + ["evidence_aware_priority"] * 12
                + ["bio_clean_priority"] * 12,
                "oof_prediction": [
                    0.99,
                    0.98,
                    0.97,
                    0.96,
                    0.2,
                    0.19,
                    0.18,
                    0.17,
                    0.16,
                    0.15,
                    0.14,
                    0.13,
                ]
                + [0.11, 0.12, 0.13, 0.14, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]
                + [0.99, 0.95, 0.94, 0.93, 0.92, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04],
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 3,
            }
        )
        summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name="parsimonious_priority",
            conservative_model_name="bio_clean_priority",
            predictions=predictions,
        )
        row = summary.iloc[0]
        self.assertEqual(int(row["primary_vs_strongest_top_10_overlap_count"]), 8)
        self.assertAlmostEqual(float(row["primary_vs_strongest_top_10_overlap_fraction"]), 0.8)
        self.assertIn("headline benchmark", str(row["selection_rationale"]))

    def test_build_primary_model_selection_summary_uses_guardrail_failure_to_keep_primary(
        self,
    ) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "primary_model", "roc_auc": 0.82, "average_precision": 0.74},
                {"model_name": "strongest_model", "roc_auc": 0.83, "average_precision": 0.75},
                {"model_name": "conservative_model", "roc_auc": 0.79, "average_precision": 0.70},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)] * 3,
                "model_name": ["primary_model"] * 8
                + ["strongest_model"] * 8
                + ["conservative_model"] * 8,
                "oof_prediction": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1] * 3,
                "spread_label": [1, 1, 1, 1, 0, 0, 0, 0] * 3,
            }
        )
        scorecard = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "selection_rank": 2,
                    "strict_knownness_acceptance_flag": True,
                    "knownness_matched_gap": -0.002,
                    "source_holdout_gap": -0.004,
                    "leakage_review_required": False,
                },
                {
                    "model_name": "strongest_model",
                    "selection_rank": 1,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.030,
                    "source_holdout_gap": -0.020,
                    "leakage_review_required": False,
                },
            ]
        )

        summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name="primary_model",
            conservative_model_name="conservative_model",
            predictions=predictions,
            model_selection_scorecard=scorecard,
        )

        row = summary.iloc[0]
        self.assertTrue(bool(row["published_primary_strict_knownness_acceptance_flag"]))
        self.assertFalse(bool(row["strongest_metric_model_strict_knownness_acceptance_flag"]))
        self.assertIn(
            "passes strict matched-knownness and source-holdout guardrails",
            str(row["selection_rationale"]),
        )

    def test_build_temporal_rank_stability_table_reports_kendall_tau(self) -> None:
        predictions = pd.DataFrame(
            {
                "split_year": [2014] * 6 + [2015] * 6,
                "model_name": ["bio_clean_priority"] * 12,
                "backbone_id": [f"bb_{i}" for i in range(6)] * 2,
                "oof_prediction": [
                    0.90,
                    0.80,
                    0.70,
                    0.30,
                    0.20,
                    0.10,
                    0.88,
                    0.79,
                    0.69,
                    0.31,
                    0.22,
                    0.11,
                ],
            }
        )
        summary = build_temporal_rank_stability_table(predictions)
        self.assertEqual(len(summary), 1)
        self.assertEqual(str(summary.iloc[0]["status"]), "ok")
        self.assertGreater(float(summary.iloc[0]["kendall_tau"]), 0.8)

    def test_build_sleeper_threat_table_compares_ap_and_naap(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "discovery_model",
                    "average_precision": 0.55,
                    "novelty_adjusted_average_precision": 0.62,
                },
                {
                    "model_name": "bias_riding_model",
                    "average_precision": 0.60,
                    "novelty_adjusted_average_precision": 0.52,
                },
            ]
        )
        summary = build_sleeper_threat_table(model_metrics)
        discovery_row = summary.loc[summary["model_name"] == "discovery_model"].iloc[0]
        bias_row = summary.loc[summary["model_name"] == "bias_riding_model"].iloc[0]
        self.assertGreater(float(discovery_row["naap_minus_ap"]), 0.0)
        self.assertEqual(
            str(discovery_row["sleeper_threat_advantage"]),
            "favors_low_knownness_positives",
        )
        self.assertLess(float(bias_row["naap_minus_ap"]), 0.0)

    def test_build_magic_number_sensitivity_table_flags_large_auc_shifts(self) -> None:
        sensitivity = pd.DataFrame(
            [
                {
                    "variant": "default",
                    "parameter_name": "host_evenness_bias_power",
                    "parameter_value": 0.5,
                    "roc_auc": 0.80,
                },
                {
                    "variant": "bias_power_low",
                    "parameter_name": "host_evenness_bias_power",
                    "parameter_value": 0.3,
                    "roc_auc": 0.79,
                },
                {
                    "variant": "bias_power_high",
                    "parameter_name": "host_evenness_bias_power",
                    "parameter_value": 1.0,
                    "roc_auc": 0.70,
                },
            ]
        )
        summary = build_magic_number_sensitivity_table(sensitivity)
        high_row = summary.loc[summary["variant"] == "bias_power_high"].iloc[0]
        self.assertFalse(bool(high_row["passes_auc_tolerance"]))
        self.assertGreater(float(high_row["abs_relative_auc_delta"]), 0.05)


if __name__ == "__main__":
    unittest.main()
