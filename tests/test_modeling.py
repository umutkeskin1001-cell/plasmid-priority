from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd


from plasmid_priority.modeling import (
    CORE_MODEL_NAMES,
    CONSERVATIVE_MODEL_NAME,
    NOVELTY_SPECIALIST_FEATURES,
    PRIMARY_MODEL_NAME,
    RESEARCH_MODEL_NAMES,
    ABLATION_MODEL_NAMES,
    assert_feature_columns_present,
    annotate_knownness_metadata,
    build_feature_dropout_audit,
    build_logistic_convergence_audit,
    build_standardized_coefficient_table,
    build_coefficient_stability_table,
    evaluate_feature_columns,
    evaluate_model_name,
    fit_feature_columns_predictions,
    fit_full_model_predictions,
    fit_predict_model_holdout,
    get_conservative_model_name,
    get_module_a_model_names,
    get_primary_model_name,
    run_module_a,
)
from plasmid_priority.modeling import module_a as module_a_impl


class ModelingTests(unittest.TestCase):
    def test_run_module_a_returns_clean_default_results(self) -> None:
        n = 40
        rng = np.random.default_rng(42)
        priority = np.linspace(0.1, 0.9, n)
        spread = (priority > 0.5).astype(int)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": spread,
                "priority_index": priority,
                "arithmetic_priority_index": priority,
                "T_eff_norm": priority,
                "H_eff_norm": priority[::-1],
                "A_eff_norm": priority,
                "T_raw_norm": np.clip(priority * 0.9, 0.0, 1.0),
                "H_breadth_norm": np.clip(priority[::-1] * 0.7, 0.0, 1.0),
                "H_phylogenetic_norm": np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "H_phylogenetic_specialization_norm": 1.0 - np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "host_taxon_evenness_norm": np.clip(priority[::-1] * 0.55, 0.0, 1.0),
                "A_raw_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "A_recurrence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "support_shrinkage_norm": np.clip(priority * 0.95, 0.0, 1.0),
                "amr_support_norm": np.clip(priority * 0.75, 0.0, 1.0),
                "H_support_norm": np.clip(priority[::-1] * 0.65, 0.0, 1.0),
                "H_support_norm_residual": np.linspace(-0.3, 0.3, n),
                "amr_support_norm_residual": np.linspace(0.25, -0.25, n),
                "coherence_score": np.clip(priority + 0.05, 0.0, 1.0),
                "orit_support": np.clip(priority, 0.0, 1.0),
                "H_external_host_range_support": np.clip(priority[::-1], 0.0, 1.0),
                "pmlst_presence_fraction_train": np.clip(priority * 0.4, 0.0, 1.0),
                "log1p_member_count_train": np.log1p(rng.integers(1, 10, size=n)),
                "log1p_n_countries_train": np.log1p(rng.integers(1, 4, size=n)),
                "refseq_share_train": rng.uniform(0, 1, size=n),
            }
        )
        results = run_module_a(scored, n_splits=4, n_repeats=2, seed=42)
        self.assertTrue(set(CORE_MODEL_NAMES).issubset(results))
        self.assertIn("random_score_control", results)
        self.assertIn("label_permutation", results)
        self.assertIn("full_priority", results)
        self.assertIn("bio_clean_priority", results)
        self.assertIn("natural_auc_priority", results)
        self.assertIn("phylogeny_aware_priority", results)
        self.assertIn("structured_signal_priority", results)
        self.assertIn("ecology_clinical_priority", results)
        self.assertIn("knownness_robust_priority", results)
        self.assertIn("support_calibrated_priority", results)
        self.assertIn("support_synergy_priority", results)
        self.assertIn("host_transfer_synergy_priority", results)
        self.assertIn("baseline_both", results)
        self.assertEqual(len(results["full_priority"].predictions), n)
        self.assertEqual(len(results["bio_clean_priority"].predictions), n)
        self.assertEqual(len(results["natural_auc_priority"].predictions), n)
        self.assertEqual(len(results["phylogeny_aware_priority"].predictions), n)
        self.assertEqual(len(results["structured_signal_priority"].predictions), n)
        self.assertEqual(len(results["ecology_clinical_priority"].predictions), n)
        self.assertEqual(len(results["knownness_robust_priority"].predictions), n)
        self.assertEqual(len(results["support_calibrated_priority"].predictions), n)
        self.assertEqual(len(results["support_synergy_priority"].predictions), n)
        self.assertEqual(len(results["host_transfer_synergy_priority"].predictions), n)
        # T6: Verify parsimonious_priority model (bio_clean without H_support_norm_residual)
        self.assertIn("parsimonious_priority", results)
        self.assertEqual(len(results["parsimonious_priority"].predictions), n)
        # parsimonious should not use H_support_norm_residual
        from plasmid_priority.modeling import MODULE_A_FEATURE_SETS
        self.assertNotIn("H_support_norm_residual", MODULE_A_FEATURE_SETS["parsimonious_priority"])
        self.assertIn("H_support_norm_residual", MODULE_A_FEATURE_SETS["visibility_adjusted_priority"])

    def test_run_module_a_can_include_research_and_ablation_models(self) -> None:
        n = 24
        rng = np.random.default_rng(7)
        priority = np.linspace(0.1, 0.9, n)
        spread = (priority > 0.5).astype(int)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": spread,
                "priority_index": priority,
                "arithmetic_priority_index": priority,
                "T_eff_norm": priority,
                "H_eff_norm": priority[::-1],
                "A_eff_norm": priority,
                "T_raw_norm": np.clip(priority * 0.9, 0.0, 1.0),
                "H_breadth_norm": np.clip(priority[::-1] * 0.7, 0.0, 1.0),
                "H_phylogenetic_norm": np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "H_phylogenetic_specialization_norm": 1.0 - np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "host_taxon_evenness_norm": np.clip(priority[::-1] * 0.55, 0.0, 1.0),
                "A_raw_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "A_recurrence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "support_shrinkage_norm": np.clip(priority * 0.95, 0.0, 1.0),
                "amr_support_norm": np.clip(priority * 0.75, 0.0, 1.0),
                "H_support_norm": np.clip(priority[::-1] * 0.65, 0.0, 1.0),
                "H_support_norm_residual": np.linspace(-0.3, 0.3, n),
                "amr_support_norm_residual": np.linspace(0.25, -0.25, n),
                "coherence_score": np.clip(priority + 0.05, 0.0, 1.0),
                "orit_support": np.clip(priority, 0.0, 1.0),
                "H_external_host_range_support": np.clip(priority[::-1], 0.0, 1.0),
                "pmlst_presence_fraction_train": np.clip(priority * 0.4, 0.0, 1.0),
                "log1p_member_count_train": np.log1p(rng.integers(1, 10, size=n)),
                "log1p_n_countries_train": np.log1p(rng.integers(1, 4, size=n)),
                "refseq_share_train": rng.uniform(0, 1, size=n),
            }
        )
        model_names = get_module_a_model_names(include_research=True, include_ablations=True)
        results = run_module_a(scored, model_names=model_names, n_splits=4, n_repeats=2, seed=42)
        self.assertTrue(set(RESEARCH_MODEL_NAMES).issubset(results))
        self.assertTrue(set(ABLATION_MODEL_NAMES).issubset(results))

    def test_evaluate_model_name_returns_predictions(self) -> None:
        n = 30
        rng = np.random.default_rng(11)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": [0, 1] * (n // 2),
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        result = evaluate_model_name(scored, model_name="enhanced_priority", n_splits=3, n_repeats=2, seed=4)
        self.assertEqual(result.name, "enhanced_priority")
        self.assertEqual(len(result.predictions), n)

    def test_evaluate_model_name_accepts_class_balanced_override(self) -> None:
        n = 30
        rng = np.random.default_rng(13)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": ([0] * 20) + ([1] * 10),
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        result = evaluate_model_name(
            scored,
            model_name="parsimonious_priority",
            n_splits=3,
            n_repeats=2,
            seed=7,
            fit_config={"sample_weight_mode": "class_balanced"},
        )
        self.assertEqual(len(result.predictions), n)
        self.assertIn("recall_at_top_25", result.metrics)

    def test_evaluate_model_name_can_skip_confidence_intervals(self) -> None:
        n = 30
        rng = np.random.default_rng(21)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": [0, 1] * (n // 2),
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        result = evaluate_model_name(
            scored,
            model_name="parsimonious_priority",
            n_splits=3,
            n_repeats=2,
            seed=9,
            include_ci=False,
        )
        self.assertIn("roc_auc", result.metrics)
        self.assertNotIn("roc_auc_ci_lower", result.metrics)

    def test_fit_predict_model_holdout_returns_holdout_scores(self) -> None:
        n = 30
        rng = np.random.default_rng(17)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": [0, 1] * (n // 2),
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        train = scored.iloc[:20].copy()
        test = scored.iloc[20:].copy()
        preds = fit_predict_model_holdout(train, test, model_name="enhanced_priority")
        self.assertEqual(len(preds), len(test))
        self.assertIn("prediction", preds.columns)
        self.assertTrue(preds["prediction"].between(0.0, 1.0).all())

    def test_fit_full_model_predictions_scores_all_backbones(self) -> None:
        n = 24
        rng = np.random.default_rng(23)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": ([0, 1] * 10) + [np.nan, np.nan, np.nan, np.nan],
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        preds = fit_full_model_predictions(scored, model_name="evidence_aware_priority")
        self.assertEqual(len(preds), n)
        self.assertIn("prediction", preds.columns)
        self.assertTrue(preds["prediction"].between(0.0, 1.0).all())

    def test_annotate_knownness_metadata_adds_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "log1p_member_count_train": [0.1, 0.3, 0.6, 1.0],
                "log1p_n_countries_train": [0.0, 0.2, 0.5, 0.8],
                "refseq_share_train": [0.0, 0.2, 0.8, 1.0],
            }
        )
        annotated = annotate_knownness_metadata(frame)
        self.assertIn("knownness_score", annotated.columns)
        self.assertIn("knownness_half", annotated.columns)
        self.assertIn("knownness_quartile", annotated.columns)
        self.assertTrue(annotated["knownness_score"].between(0.0, 1.0).all())

    def test_annotate_knownness_metadata_keeps_tied_knownness_scores_in_same_quartile(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(8)],
                "log1p_member_count_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.6, 1.6],
                "log1p_n_countries_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.6, 1.6],
                "refseq_share_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0],
            }
        )
        annotated = annotate_knownness_metadata(frame)
        grouped = annotated.groupby("knownness_score", sort=False)["knownness_quartile"].nunique()
        self.assertTrue((grouped <= 1).all())

    def test_annotate_knownness_metadata_disables_fake_q1_when_quartiles_collapse(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(8)],
                "log1p_member_count_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.6, 1.6],
                "log1p_n_countries_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.6, 1.6],
                "refseq_share_train": [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0],
            }
        )
        annotated = annotate_knownness_metadata(frame)
        self.assertFalse(bool(annotated["knownness_quartile_supported"].any()))
        self.assertTrue(annotated["knownness_quartile"].isna().all())

    def test_evaluate_feature_columns_accepts_fit_config(self) -> None:
        n = 30
        rng = np.random.default_rng(29)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": ([0] * 18) + ([1] * 12),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "replicon_architecture_norm": rng.uniform(0.1, 0.9, size=n),
                "mash_neighbor_distance_train_norm": rng.uniform(0.1, 0.9, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        result = evaluate_feature_columns(
            scored,
            columns=NOVELTY_SPECIALIST_FEATURES,
            label="novelty_specialist_priority",
            n_splits=3,
            n_repeats=2,
            seed=9,
            fit_config={"sample_weight_mode": "class_balanced", "l2": 3.0},
        )
        self.assertEqual(len(result.predictions), n)
        self.assertIn("roc_auc", result.metrics)

    def test_fit_feature_columns_predictions_scores_requested_rows(self) -> None:
        n = 24
        rng = np.random.default_rng(31)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": ([0, 1] * 10) + [np.nan, np.nan, np.nan, np.nan],
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "replicon_architecture_norm": rng.uniform(0.1, 0.9, size=n),
                "mash_neighbor_distance_train_norm": rng.uniform(0.1, 0.9, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        train = scored.loc[scored["spread_label"].notna()].copy()
        score = scored.iloc[:8].copy()
        preds = fit_feature_columns_predictions(
            train,
            score,
            columns=NOVELTY_SPECIALIST_FEATURES,
            fit_config={"l2": 3.0},
        )
        self.assertEqual(len(preds), len(score))
        self.assertTrue(preds["prediction"].between(0.0, 1.0).all())

    def test_primary_model_name_prefers_primary_model(self) -> None:
        model_name = get_primary_model_name(["baseline_both", "full_priority", PRIMARY_MODEL_NAME])
        self.assertEqual(model_name, PRIMARY_MODEL_NAME)

    def test_knownness_sample_weight_is_order_invariant(self) -> None:
        eligible = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "log1p_member_count_train": [1.0, 1.0, 2.0, 2.0],
                "log1p_n_countries_train": [0.0, 0.0, 1.0, 1.0],
                "refseq_share_train": [0.0, 0.0, 1.0, 1.0],
                "spread_label": [0, 1, 0, 1],
            }
        )
        weight_a = pd.Series(
            module_a_impl._compute_sample_weight(eligible, mode="knownness_balanced"),
            index=eligible["backbone_id"],
        ).sort_index()
        shuffled = eligible.sample(frac=1.0, random_state=42).reset_index(drop=True)
        weight_b = pd.Series(
            module_a_impl._compute_sample_weight(shuffled, mode="knownness_balanced"),
            index=shuffled["backbone_id"],
        ).sort_index()
        pd.testing.assert_series_equal(weight_a, weight_b)

    def test_knownness_sample_weight_handles_duplicate_bootstrap_index(self) -> None:
        eligible = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "log1p_member_count_train": [1.0, 1.2, 2.0, 2.2],
                "log1p_n_countries_train": [0.0, 0.1, 1.0, 1.1],
                "refseq_share_train": [0.0, 0.2, 0.8, 1.0],
                "spread_label": [0, 1, 0, 1],
            }
        )
        bootstrap_train = eligible.iloc[[0, 1, 1, 3]].copy()
        bootstrap_train.index = [0, 1, 1, 3]
        weights = module_a_impl._compute_sample_weight(bootstrap_train, mode="knownness_balanced")
        self.assertEqual(len(weights), len(bootstrap_train))
        self.assertTrue(np.isfinite(weights).all())

    def test_annotate_knownness_metadata_scopes_groups_to_eligible_rows(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4", "bb5", "bb6"],
                "spread_label": [0, 1, 0, 1, np.nan, np.nan],
                "log1p_member_count_train": [0.1, 0.2, 0.8, 0.9, 0.0, 0.0],
                "log1p_n_countries_train": [0.1, 0.2, 0.8, 0.9, 0.0, 0.0],
                "refseq_share_train": [0.1, 0.2, 0.8, 0.9, 0.0, 0.0],
            }
        )
        annotated = annotate_knownness_metadata(scored)
        eligible = annotated.loc[annotated["spread_label"].notna()]
        out_of_scope = annotated.loc[annotated["spread_label"].isna()]
        self.assertSetEqual(set(eligible["knownness_half"].astype(str)), {"lower_half", "upper_half"})
        self.assertTrue(out_of_scope["knownness_half"].astype(str).eq("out_of_scope").all())

    def test_assert_feature_columns_present_raises_for_missing_engineered_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "spread_label": [0, 1],
                "T_eff_norm": [0.2, 0.8],
            }
        )
        with self.assertRaisesRegex(ValueError, "clinical_context_sparse_penalty_norm"):
            assert_feature_columns_present(
                frame,
                ["T_eff_norm", "clinical_context_sparse_penalty_norm"],
                label="test frame",
            )

    def test_conservative_model_name_prefers_bio_clean_priority(self) -> None:
        model_name = get_conservative_model_name(["baseline_both", "T_plus_H_plus_A", CONSERVATIVE_MODEL_NAME])
        self.assertEqual(model_name, CONSERVATIVE_MODEL_NAME)

    def test_audit_helpers_return_expected_columns(self) -> None:
        n = 24
        rng = np.random.default_rng(7)
        spread = np.array([0, 1] * (n // 2))
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": spread,
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.4, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        columns = [
            "T_raw_norm",
            "H_breadth_norm",
            "A_raw_norm",
            "H_support_norm_residual",
            "coherence_score",
            "orit_support",
        ]
        coefficients = build_standardized_coefficient_table(scored, model_name=PRIMARY_MODEL_NAME, columns=columns)
        dropout = build_feature_dropout_audit(scored, model_name=PRIMARY_MODEL_NAME, columns=columns, n_splits=4, n_repeats=2, seed=42)
        stability = build_coefficient_stability_table(
            scored,
            model_name=PRIMARY_MODEL_NAME,
            columns=columns,
            n_splits=4,
            n_repeats=2,
            seed=42,
        )
        self.assertEqual(set(coefficients["feature_name"]), set(columns))
        self.assertIn("__full_model__", set(dropout["feature_name"]))
        self.assertEqual(len(dropout), len(columns) + 1)
        self.assertEqual(set(stability["feature_name"]), set(columns))

    def test_convergence_audit_returns_fold_rows(self) -> None:
        n = 24
        rng = np.random.default_rng(71)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": np.array([0, 1] * (n // 2)),
                "priority_index": rng.uniform(0.1, 0.9, size=n),
                "arithmetic_priority_index": rng.uniform(0.1, 0.9, size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "T_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "H_breadth_norm": rng.uniform(0.1, 0.9, size=n),
                "A_raw_norm": rng.uniform(0.1, 0.9, size=n),
                "support_shrinkage_norm": rng.uniform(0.1, 0.9, size=n),
                "amr_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm": rng.uniform(0.1, 0.9, size=n),
                "H_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "amr_support_norm_residual": rng.uniform(-0.3, 0.3, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0.1, 0.9, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        audit = build_logistic_convergence_audit(
            scored,
            model_names=["parsimonious_priority"],
            n_splits=3,
            n_repeats=2,
            seed=5,
        )
        self.assertFalse(audit.empty)
        self.assertIn("converged", audit.columns)
        self.assertIn("used_pinv", audit.columns)


if __name__ == "__main__":
    unittest.main()
