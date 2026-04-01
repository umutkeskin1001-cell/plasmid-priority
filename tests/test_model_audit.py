from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd


from plasmid_priority.reporting import (
    build_amrfinder_coverage_table,
    build_benchmark_protocol_table,
    build_consensus_candidate_ranking,
    build_candidate_dossier_table,
    build_candidate_portfolio_table,
    build_candidate_risk_table,
    build_candidate_universe_table,
    build_calibration_metric_table,
    build_component_floor_diagnostics,
    build_decision_yield_table,
    build_group_holdout_performance,
    build_gate_consistency_audit,
    build_h_feature_diagnostics,
    build_knownness_audit_tables,
    build_logistic_implementation_audit,
    build_model_comparison_table,
    build_model_family_summary,
    build_model_selection_scorecard,
    build_model_simplicity_summary,
    build_primary_model_selection_summary,
    build_novelty_margin_summary,
    build_model_subgroup_performance,
    build_negative_control_audit,
    build_permutation_null_tables,
    build_priority_bootstrap_stability_table,
    build_score_distribution_diagnostics,
    build_source_balance_resampling_table,
    build_temporal_drift_summary,
    build_threshold_flip_table,
    build_variant_rank_consistency_table,
)


class ModelAuditTests(unittest.TestCase):
    def test_build_gate_consistency_audit_summarizes_half_gate_routes(self) -> None:
        adaptive_predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "model_name": ["adaptive_knownness_blend_priority"] * 8,
                "knownness_score": [0.42, 0.45, 0.48, 0.49, 0.51, 0.52, 0.55, 0.58],
                "knownness_half": ["lower_half"] * 4 + ["upper_half"] * 4,
                "knownness_quartile": ["q1_lowest", "q1_lowest", "q2", "q2", "q3", "q3", "q4_highest", "q4_highest"],
                "lower_half_route_prediction": [0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48],
                "upper_half_route_prediction": [0.59, 0.57, 0.55, 0.53, 0.51, 0.49, 0.47, 0.45],
            }
        )
        audit = build_gate_consistency_audit(adaptive_predictions, near_fraction=0.5, min_n=2)
        self.assertEqual(set(audit["gate_name"]), {"half_boundary"})
        self.assertEqual(int(audit.iloc[0]["n_near_gate"]), 4)
        self.assertIn("mean_abs_route_delta_near_gate", audit.columns)
        self.assertGreaterEqual(float(audit.iloc[0]["route_spearman_near_gate"]), 0.9)

    def test_build_model_family_summary_keeps_priority_models(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "source_only", "roc_auc": 0.45},
                {"model_name": "baseline_both", "roc_auc": 0.68},
                {"model_name": "full_priority", "roc_auc": 0.75},
                {"model_name": "T_plus_H_plus_A", "roc_auc": 0.76},
                {"model_name": "proxy_light_priority", "roc_auc": 0.77},
                {"model_name": "enhanced_priority", "roc_auc": 0.78},
            ]
        )
        summary = build_model_family_summary(model_metrics)
        self.assertIn("evidence_role", summary.columns)
        self.assertIn("delta_auc_vs_enhanced_priority", summary.columns)
        self.assertIn("enhanced_priority", set(summary["model_name"]))

    def test_build_model_selection_scorecard_returns_composite_ranks(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "knownness_robust_priority", "roc_auc": 0.80, "average_precision": 0.71},
                {"model_name": "baseline_both", "roc_auc": 0.72, "average_precision": 0.65},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["knownness_robust_priority"] * 12 + ["baseline_both"] * 12,
                "oof_prediction": [0.95, 0.90, 0.88, 0.80, 0.70, 0.65, 0.45, 0.35, 0.30, 0.20, 0.18, 0.12]
                + [0.80, 0.78, 0.75, 0.70, 0.68, 0.66, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25],
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
                {"matched_stratum": "__weighted_overall__", "model_name": "knownness_robust_priority", "roc_auc": 0.74},
                {"matched_stratum": "__weighted_overall__", "model_name": "baseline_both", "roc_auc": 0.60},
            ]
        )
        holdout = pd.DataFrame(
            [
                {"group_column": "dominant_source", "model_name": "knownness_robust_priority", "status": "ok", "roc_auc": 0.76, "n_test_backbones": 9},
                {"group_column": "dominant_source", "model_name": "knownness_robust_priority", "status": "ok", "roc_auc": 0.70, "n_test_backbones": 3},
                {"group_column": "dominant_source", "model_name": "baseline_both", "status": "ok", "roc_auc": 0.62, "n_test_backbones": 9},
                {"group_column": "dominant_source", "model_name": "baseline_both", "status": "ok", "roc_auc": 0.58, "n_test_backbones": 3},
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
        self.assertEqual(str(scorecard.iloc[0]["model_name"]), "knownness_robust_priority")

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
                "oof_prediction": [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.40, 0.35, 0.30, 0.25, 0.20, 0.10]
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
                {"matched_stratum": "__weighted_overall__", "model_name": "complete_model", "roc_auc": 0.74},
                {"matched_stratum": "__weighted_overall__", "model_name": "missing_holdout_model", "roc_auc": 0.74},
            ]
        )
        holdout = pd.DataFrame(
            [
                {"group_column": "dominant_source", "model_name": "complete_model", "status": "ok", "roc_auc": 0.76, "n_test_backbones": 9},
                {"group_column": "dominant_source", "model_name": "complete_model", "status": "ok", "roc_auc": 0.70, "n_test_backbones": 3},
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
        self.assertEqual(int(missing_row["selection_missing_metric_count"]), 1)
        self.assertGreater(float(complete_row["selection_composite_score"]), float(missing_row["selection_composite_score"]))

    def test_build_benchmark_protocol_table_marks_primary_and_preferred_adaptive(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "support_synergy_priority", "roc_auc": 0.818, "average_precision": 0.740},
                {"model_name": "knownness_robust_priority", "roc_auc": 0.806, "average_precision": 0.719},
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
                {"model_name": "adaptive_support_synergy_blend_priority", "gate_consistency_tier": "stable"},
                {"model_name": "adaptive_knownness_robust_priority", "gate_consistency_tier": "unstable"},
            ]
        )
        protocol = build_benchmark_protocol_table(
            model_metrics,
            selection_summary,
            adaptive_gated_metrics=adaptive_metrics,
            gate_consistency_audit=gate_consistency,
        )
        self.assertEqual(
            str(protocol.loc[protocol["benchmark_role"] == "primary_benchmark", "model_name"].iloc[0]),
            "support_synergy_priority",
        )
        self.assertEqual(
            str(protocol.loc[protocol["benchmark_role"] == "preferred_adaptive_audit", "model_name"].iloc[0]),
            "adaptive_support_synergy_blend_priority",
        )
        self.assertEqual(
            str(protocol.loc[protocol["benchmark_role"] == "strongest_adaptive_upper_bound", "model_name"].iloc[0]),
            "adaptive_knownness_robust_priority",
        )

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
        audit = build_model_subgroup_performance(predictions, scored, model_names=["enhanced_priority"])
        self.assertIn("overall", set(audit["subgroup_name"]))
        self.assertIn("dominant_source", set(audit["subgroup_name"]))
        self.assertTrue((audit["model_name"] == "enhanced_priority").all())

    def test_build_model_comparison_table_returns_delta_columns(self) -> None:
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 2,
                "model_name": ["enhanced_priority"] * 12 + ["baseline_both"] * 12,
                "oof_prediction": [0.1, 0.2, 0.2, 0.8, 0.9, 0.7, 0.1, 0.2, 0.3, 0.8, 0.85, 0.9] + [0.4] * 12,
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
        calibration = build_calibration_metric_table(predictions, model_names=["enhanced_priority"])
        self.assertIn("expected_calibration_error", calibration.columns)

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
        detail, summary = build_permutation_null_tables(predictions, model_names=["enhanced_priority"], n_permutations=20, seed=3)
        self.assertEqual(len(detail), 20)
        self.assertEqual(summary.iloc[0]["model_name"], "enhanced_priority")
        self.assertIn("empirical_p_roc_auc", summary.columns)

    def test_build_priority_bootstrap_stability_table_returns_candidate_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "member_count_train": [1] * 12,
                "priority_index": [0.95, 0.91, 0.89, 0.84, 0.8, 0.77, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
                "T_eff": [0.9, 0.85, 0.82, 0.78, 0.76, 0.71, 0.66, 0.6, 0.55, 0.5, 0.44, 0.4],
                "H_eff": [0.88, 0.84, 0.8, 0.74, 0.7, 0.68, 0.63, 0.58, 0.52, 0.47, 0.42, 0.38],
                "A_eff": [0.92, 0.87, 0.81, 0.79, 0.75, 0.69, 0.61, 0.57, 0.51, 0.48, 0.43, 0.39],
            }
        )
        stability = build_priority_bootstrap_stability_table(scored, candidate_n=5, top_k=3, n_bootstrap=10, seed=2)
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
        consistency = build_variant_rank_consistency_table(base, {"variant_a": variant}, candidate_n=3, top_k=2)
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
                "oof_prediction": ([0.2, 0.8, 0.3, 0.7, 0.4, 0.9] * 5) + ([0.4, 0.6, 0.45, 0.55, 0.5, 0.6] * 5),
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
                "oof_prediction": ([0.2, 0.8, 0.3, 0.7, 0.35, 0.85] * 4) + ([0.4, 0.6, 0.45, 0.55, 0.5, 0.6] * 4),
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
        audit = build_negative_control_audit(scored, primary_model_name="enhanced_priority", n_splits=4, n_repeats=2, seed=3)
        self.assertIn("primary_model", set(audit["audit_name"]))
        self.assertIn("negative_control_noise_a_only", set(audit["audit_name"]))
        self.assertIn("delta_roc_auc_vs_primary", audit.columns)

    def test_build_model_simplicity_summary_returns_overlap_columns(self) -> None:
        model_metrics = pd.DataFrame(
            [
                {"model_name": "enhanced_priority", "roc_auc": 0.78, "average_precision": 0.81, "brier_score": 0.19},
                {"model_name": "proxy_light_priority", "roc_auc": 0.77, "average_precision": 0.80, "brier_score": 0.20},
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
        dropout_table = pd.DataFrame({"feature_name": ["H_eff_norm"], "roc_auc_drop_vs_full": [0.03]})
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
                "coherence_score": [0.7, 0.4],
                "member_count_train": [4, 1],
                "n_countries_train": [3, 1],
                "refseq_share_train": [0.6, 1.0],
                "insd_share_train": [0.4, 0.0],
            }
        )
        stability = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "bootstrap_top_k_frequency": [0.9, 0.3],
                "variant_top_k_frequency": [0.8, 0.2],
            }
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb1", "bb2"],
                "model_name": ["enhanced_priority", "enhanced_priority", "proxy_light_priority", "proxy_light_priority"],
                "oof_prediction": [0.8, 0.6, 0.7, 0.3],
                "spread_label": [1, 0, 1, 0],
            }
        )
        who = pd.DataFrame({"backbone_id": ["bb1"], "who_mia_any_support": [True], "who_mia_any_hpecia": [True], "who_mia_mapped_fraction": [1.0]})
        card = pd.DataFrame({"backbone_id": ["bb1"], "card_any_support": [True], "card_match_fraction": [0.5]})
        mobsuite = pd.DataFrame({"backbone_id": ["bb1"], "mobsuite_any_literature_support": [True], "mobsuite_any_cluster_support": [True]})
        pathogen = pd.DataFrame({"pathogen_dataset": ["combined"], "backbone_id": ["bb1"], "pd_any_support": [True], "pd_matching_fraction": [0.4]})
        amrfinder = pd.DataFrame({"backbone_id": ["bb1"], "amrfinder_any_hit": [True], "gene_jaccard": [0.5], "class_jaccard": [0.5]})
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
        self.assertFalse(any(column.endswith("_x") or column.endswith("_y") for column in dossier.columns))
        self.assertEqual(int(dossier.loc[dossier["backbone_id"] == "bb1", "external_support_modalities_count"].iloc[0]), 4)
        self.assertIn("false_positive_risk_tier", risk.columns)

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
                "priority_index": [0.9, 0.8],
                "primary_model_candidate_score": [0.92, 0.83],
                "baseline_both_candidate_score": [0.71, 0.62],
                "novelty_margin_vs_baseline": [0.21, 0.21],
                "candidate_prediction_source": ["oof", "full_fit"],
                "eligible_for_oof": [True, False],
                "knownness_score": [0.8, 0.7],
                "knownness_half": ["upper_half", "upper_half"],
                "refseq_share_train": [0.9, 0.6],
                "insd_share_train": [0.1, 0.4],
                "bootstrap_top_10_frequency": [0.9, 0.4],
                "who_mia_any_support": [True, pd.NA],
                "card_any_support": [False, pd.NA],
                "mobsuite_any_literature_support": [pd.NA, pd.NA],
                "pd_any_support": [True, pd.NA],
                "false_positive_risk_tier": ["low", "medium"],
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
        portfolio = build_candidate_portfolio_table(candidate_dossiers, novelty_watchlist, established_n=1, novel_n=1)
        self.assertEqual(set(portfolio["portfolio_track"]), {"established_high_risk", "novel_signal"})
        self.assertIn("primary_model_candidate_score", portfolio.columns)
        self.assertIn("candidate_prediction_source", portfolio.columns)
        self.assertIn("recommended_monitoring_tier", portfolio.columns)
        self.assertIn("source_support_tier", portfolio.columns)
        self.assertEqual(int(portfolio.loc[portfolio["backbone_id"] == "bb1", "external_support_modalities_count"].iloc[0]), 2)

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
            columns=["T_raw_norm", "H_specialization_norm", "A_raw_norm", "orit_support", "H_support_norm_residual"],
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
        decision_yield = build_decision_yield_table(predictions, model_names=["parsimonious_priority"], top_ks=(2, 4))
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
        scored = pd.DataFrame({"backbone_id": ["bb1", "bb2"], "priority_index": [0.9, 0.7], "spread_label": [1, 0]})
        consensus = pd.DataFrame({"backbone_id": ["bb1"], "consensus_rank": [1]})
        dossiers = pd.DataFrame({"backbone_id": ["bb1"], "candidate_confidence_tier": ["tier_a"], "recommended_monitoring_tier": ["core_surveillance"]})
        portfolio = pd.DataFrame({"backbone_id": ["bb1"], "portfolio_track": ["established_high_risk"], "track_rank": [1]})
        novelty = pd.DataFrame({"backbone_id": ["bb2"], "knownness_half": ["lower_half"], "novelty_margin_vs_baseline": [0.2]})
        freeze = pd.DataFrame({"backbone_id": ["bb2"], "freeze_rank": [2]})
        high_conf = pd.DataFrame({"backbone_id": ["bb1"], "candidate_confidence_tier": ["tier_a"], "false_positive_risk_tier": ["low"]})
        risk = pd.DataFrame({"backbone_id": ["bb2"], "false_positive_risk_tier": ["medium"], "risk_flag_count": [1], "risk_flags": ["stability_risk"]})
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
                {"model_name": "parsimonious_priority", "roc_auc": 0.76, "average_precision": 0.67, "brier_score": 0.18},
                {"model_name": "evidence_aware_priority", "roc_auc": 0.81, "average_precision": 0.73, "brier_score": 0.17},
                {"model_name": "bio_clean_priority", "roc_auc": 0.77, "average_precision": 0.68, "brier_score": 0.18},
            ]
        )
        predictions = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)] * 3,
                "model_name": ["parsimonious_priority"] * 12 + ["evidence_aware_priority"] * 12 + ["bio_clean_priority"] * 12,
                "oof_prediction": [0.99, 0.98, 0.97, 0.96, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]
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


if __name__ == "__main__":
    unittest.main()
