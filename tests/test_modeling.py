from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression

from plasmid_priority.modeling import (
    ABLATION_MODEL_NAMES,
    CONSERVATIVE_MODEL_NAME,
    CORE_MODEL_NAMES,
    FEATURE_PROVENANCE_REGISTRY,
    GOVERNANCE_MODEL_NAME,
    MODULE_A_FEATURE_SETS,
    MODULE_A_MODEL_TRACKS,
    NOVELTY_SPECIALIST_FEATURES,
    PRIMARY_MODEL_NAME,
    RESEARCH_MODEL_NAMES,
    annotate_knownness_metadata,
    assert_feature_columns_present,
    build_cmim_feature_selection_table,
    build_coefficient_stability_table,
    build_feature_dropout_audit,
    build_logistic_convergence_audit,
    build_standardized_coefficient_table,
    evaluate_feature_columns,
    evaluate_model_name,
    fit_feature_columns_predictions,
    fit_full_model_predictions,
    fit_predict_model_holdout,
    get_conservative_model_name,
    get_feature_track,
    get_model_track,
    get_module_a_model_names,
    get_official_model_names,
    get_primary_model_name,
    run_module_a,
    select_cmim_features,
)
from plasmid_priority.modeling import module_a as module_a
from plasmid_priority.modeling import module_a_support as module_a_support_impl
from plasmid_priority.modeling import tree_models as tree_models_impl
from plasmid_priority.modeling.nested_cv import nested_cross_validate
from plasmid_priority.modeling.nested_cv_bayesian import NestedCVEvaluator


class ModelingTests(unittest.TestCase):
    def _hybrid_ready_scored(self, *, n: int = 36, include_unlabeled_tail: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(101)
        total = n + include_unlabeled_tail
        spread = np.array(([0, 1] * (n // 2))[:n], dtype=float)
        if include_unlabeled_tail:
            spread = np.concatenate([spread, np.full(include_unlabeled_tail, np.nan)])
        transfer = rng.uniform(0.1, 0.9, size=total)
        host_obs = rng.uniform(0.1, 0.9, size=total)
        amr = rng.uniform(0.1, 0.9, size=total)
        coherence = rng.uniform(0.1, 0.9, size=total)
        return pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(total)],
                "spread_label": spread,
                "T_eff_norm": transfer,
                "H_obs_norm": host_obs,
                "H_phylogenetic_norm": host_obs,
                "H_breadth_norm": host_obs,
                "A_eff_norm": amr,
                "coherence_score": coherence,
                "orit_support": rng.uniform(0.1, 0.9, size=total),
                "T_H_obs_synergy_norm": transfer * host_obs,
                "A_H_obs_synergy_norm": amr * host_obs,
                "T_coherence_synergy_norm": transfer * coherence,
                "T_A_synergy_norm": transfer * amr,
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=total),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=total),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=total),
            }
        )

    def test_prepare_feature_matrices_fits_knn_imputer_on_train_only(self) -> None:
        train = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "f1": [0.0, 10.0, np.nan],
                "f2": [0.0, 10.0, 0.0],
            }
        )
        score = pd.DataFrame(
            {
                "backbone_id": ["bb4"],
                "f1": [1000.0],
                "f2": [0.0],
            }
        )

        train_matrix, score_matrix = module_a._prepare_feature_matrices(
            train, score, ["f1", "f2"]
        )

        expected_train = KNNImputer(
            n_neighbors=5,
            weights="distance",
            keep_empty_features=True,
        ).fit_transform(train[["f1", "f2"]].to_numpy(dtype=float))
        leaked_train = KNNImputer(
            n_neighbors=5,
            weights="distance",
            keep_empty_features=True,
        ).fit_transform(
            pd.concat([train[["f1", "f2"]], score[["f1", "f2"]]], ignore_index=True).to_numpy(
                dtype=float
            )
        )[: len(train)]

        self.assertTrue(np.allclose(train_matrix, expected_train))
        self.assertFalse(np.allclose(train_matrix, leaked_train))
        self.assertEqual(score_matrix.shape, (1, 2))

    def test_knownness_residualizer_auto_alpha_selects_from_grid(self) -> None:
        rng = np.random.default_rng(19)
        n = 24
        train = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": np.array([0, 1] * (n // 2)),
                "log1p_member_count_train": np.linspace(0.1, 2.0, n),
                "log1p_n_countries_train": np.linspace(0.0, 1.2, n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
                "f1": np.linspace(0.0, 1.0, n),
                "f2": np.linspace(1.0, 0.0, n),
            }
        )
        alpha = module_a._select_knownness_residualizer_alpha(
            train,
            ["f1", "f2"],
            fit_kwargs={
                "preprocess_mode": "knownness_residualized",
                "preprocess_alpha": "auto",
                "preprocess_alpha_grid": [0.1, 1.0, 10.0],
                "l2": 1.0,
                "max_iter": 100,
                "sample_weight_mode": None,
            },
        )
        self.assertIn(alpha, {0.1, 1.0, 10.0})

    def test_grouped_knownness_alpha_assigns_feature_family_specific_penalties(self) -> None:
        alphas = module_a._resolve_knownness_grouped_alpha(
            ["T_eff_norm", "H_obs_specialization_norm", "A_eff_norm", "coherence_score"],
            base_alpha=1.5,
            fit_kwargs={
                "preprocess_alpha_grouped": True,
                "preprocess_alpha_T": 0.1,
                "preprocess_alpha_H": 2.0,
                "preprocess_alpha_A": 1.0,
            },
        )
        self.assertTrue(np.allclose(alphas, np.array([0.1, 2.0, 1.0, 1.5])))

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
                "H_obs_specialization_norm": 1.0 - np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "host_taxon_evenness_norm": np.clip(priority[::-1] * 0.55, 0.0, 1.0),
                "host_phylogenetic_dispersion_norm": np.clip(priority[::-1] * 0.5, 0.0, 1.0),
                "A_raw_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "A_recurrence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "support_shrinkage_norm": np.clip(priority * 0.95, 0.0, 1.0),
                "amr_support_norm": np.clip(priority * 0.75, 0.0, 1.0),
                "H_support_norm": np.clip(priority[::-1] * 0.65, 0.0, 1.0),
                "H_support_norm_residual": np.linspace(-0.3, 0.3, n),
                "amr_support_norm_residual": np.linspace(0.25, -0.25, n),
                "coherence_score": np.clip(priority + 0.05, 0.0, 1.0),
                "orit_support": np.clip(priority, 0.0, 1.0),
                "backbone_purity_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "assignment_confidence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "mash_neighbor_distance_train_norm": np.clip(1.0 - priority * 0.5, 0.0, 1.0),
                "replicon_architecture_norm": np.clip(priority * 0.6, 0.0, 1.0),
                "clinical_context_fraction_norm": np.clip(priority * 0.4, 0.0, 1.0),
                "ecology_context_diversity_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "amr_gene_burden_norm": np.clip(priority * 0.65, 0.0, 1.0),
                "evolutionary_jump_score_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_agreement_score": np.clip(priority * 0.7, 0.0, 1.0),
                "mean_amr_uncertainty_score": np.clip(1.0 - priority * 0.3, 0.0, 1.0),
                "mash_graph_novelty_score": np.clip(priority * 0.45, 0.0, 1.0),
                "mash_graph_bridge_fraction": np.clip(priority * 0.3, 0.0, 1.0),
                "plasmidfinder_complexity_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_class_richness_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "pmlst_coherence_norm": np.clip(priority * 0.6, 0.0, 1.0),
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
        # Verify all core models produce predictions
        for model_name in CORE_MODEL_NAMES:
            self.assertIn(model_name, results)
            self.assertEqual(len(results[model_name].predictions), n)
        # parsimonious should not use H_support_norm_residual
        self.assertNotIn("H_support_norm_residual", MODULE_A_FEATURE_SETS["parsimonious_priority"])
        self.assertIn(
            "H_support_norm_residual", MODULE_A_FEATURE_SETS["visibility_adjusted_priority"]
        )

    def test_get_active_model_names_prefers_successful_results(self) -> None:
        payload = {
            "primary_model_name": "bio_clean_priority",
            "conservative_model_name": "parsimonious_priority",
            "bio_clean_priority": {"roc_auc": 0.81, "status": "ok"},
            "parsimonious_priority": {
                "roc_auc": 0.76,
                "status": "failed",
                "error_message": "forced failure",
            },
            "baseline_both": {"roc_auc": 0.70, "status": "ok"},
        }
        self.assertEqual(
            module_a_support_impl.get_active_model_names(payload),
            ["bio_clean_priority", "baseline_both"],
        )

        failed_only = pd.DataFrame(
            {
                "model_name": ["failed_a", "failed_b"],
                "status": ["failed", "failed"],
            }
        )
        self.assertEqual(
            module_a_support_impl.get_active_model_names(failed_only),
            ["failed_a", "failed_b"],
        )

    def test_threat_architecture_priority_uses_requested_threat_feature_set(self) -> None:
        self.assertEqual(
            MODULE_A_FEATURE_SETS["threat_architecture_priority"],
            [
                "A_raw_norm",
                "A_eff_norm",
                "amr_class_richness_norm",
                "amr_gene_burden_norm",
                "amr_clinical_threat_norm",
                "amr_mdr_proxy_norm",
                "amr_xdr_proxy_norm",
                "amr_clinical_escalation_norm",
                "last_resort_convergence_norm",
                "amr_mechanism_diversity_norm",
                "H_specialization_norm",
                "replicon_architecture_norm",
                "mean_n_replicon_types_norm",
                "metadata_support_depth_norm",
                "T_A_synergy_norm",
                "clinical_A_synergy_norm",
                "amr_load_density_norm",
                "H_evenness_T_synergy_norm",
                "silent_carrier_risk_norm",
                "evolutionary_jump_score_norm",
            ],
        )

    def test_support_synergy_priority_includes_new_engineered_features(self) -> None:
        for feature_name in (
            "pmlst_presence_norm",
            "plasmidfinder_support_norm",
            "amr_support_norm_residual",
            "context_support_guard_norm",
            "amr_load_density_norm",
            "amr_clinical_escalation_norm",
            "amr_mechanism_diversity_norm",
            "last_resort_convergence_norm",
            "H_evenness_T_synergy_norm",
            "silent_carrier_risk_norm",
            "evolutionary_jump_score_norm",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["support_synergy_priority"])

    def test_monotonic_latent_priority_uses_saturating_latent_axes(self) -> None:
        for feature_name in (
            "amr_burden_saturation_norm",
            "replicon_multiplicity_saturation_norm",
            "host_range_saturation_norm",
            "eco_clinical_context_saturation_norm",
            "monotonic_latent_priority_index",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["monotonic_latent_priority"])

    def test_structured_signal_priority_includes_augmented_phylogeny_features(self) -> None:
        for feature_name in (
            "H_phylogenetic_augmented_specialization_norm",
            "host_phylogenetic_dispersion_norm",
            "mean_n_replicon_types_norm",
            "plasmidfinder_complexity_norm",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["structured_signal_priority"])

    def test_discovery_and_governance_feature_sets_are_split_by_provenance(self) -> None:
        for feature_name in (
            "H_obs_specialization_norm",
            "T_raw_norm",
            "A_raw_norm",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["bio_clean_priority"])
        for feature_name in (
            "H_obs_specialization_norm",
            "T_eff_norm",
            "A_eff_norm",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["parsimonious_priority"])
        for feature_name in (
            "H_obs_specialization_norm",
            "H_support_norm",
            "H_support_norm_residual",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["phylo_support_fusion_priority"])

    def test_feature_sets_do_not_include_exact_complement_pairs(self) -> None:
        forbidden_pairs = (
            ("H_obs_norm", "H_obs_specialization_norm"),
            ("H_breadth_norm", "H_specialization_norm"),
            ("H_phylogenetic_norm", "H_phylogenetic_specialization_norm"),
        )
        for model_name, columns in MODULE_A_FEATURE_SETS.items():
            feature_names = set(columns)
            for left, right in forbidden_pairs:
                self.assertFalse(left in feature_names and right in feature_names, msg=model_name)

    def test_cmim_feature_selection_prioritizes_nonredundant_signal(self) -> None:
        n = 200
        rng = np.random.default_rng(123)
        y = rng.integers(0, 2, size=n)
        primary_noise = rng.random(n) < 0.10
        secondary_noise = rng.random(n) < 0.20
        signal_primary = np.logical_xor(y == 1, primary_noise).astype(float)
        signal_secondary = np.logical_xor(y == 1, secondary_noise).astype(float)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": y,
                "signal_primary": signal_primary,
                "signal_secondary": signal_secondary,
                "zz_duplicate_signal": signal_primary.copy(),
                "noise": rng.random(n),
            }
        )
        ranking = build_cmim_feature_selection_table(
            scored,
            columns=["signal_primary", "signal_secondary", "zz_duplicate_signal", "noise"],
            top_n=3,
            n_bins=4,
        )
        self.assertEqual(ranking.loc[0, "feature_name"], "signal_primary")
        self.assertEqual(ranking.loc[1, "feature_name"], "signal_secondary")
        self.assertEqual(
            select_cmim_features(
                scored,
                columns=["signal_primary", "signal_secondary", "zz_duplicate_signal", "noise"],
                top_n=2,
                n_bins=4,
            ),
            ["signal_primary", "signal_secondary"],
        )

    def test_phylo_support_fusion_priority_combines_support_and_augmented_phylogeny(self) -> None:
        for feature_name in (
            "H_support_norm",
            "amr_support_norm_residual",
            "context_support_guard_norm",
            "H_phylogenetic_augmented_specialization_norm",
            "host_phylogenetic_dispersion_norm",
            "mean_n_replicon_types_norm",
            "plasmidfinder_support_norm",
            "plasmidfinder_complexity_norm",
        ):
            self.assertIn(feature_name, MODULE_A_FEATURE_SETS["phylo_support_fusion_priority"])

    def test_feature_provenance_registry_separates_discovery_and_governance_inputs(self) -> None:
        self.assertEqual(get_feature_track("T_eff_norm"), "discovery")
        self.assertEqual(get_feature_track("H_support_norm"), "governance")
        self.assertEqual(get_feature_track("log1p_member_count_train"), "baseline")
        self.assertEqual(FEATURE_PROVENANCE_REGISTRY["T_H_obs_synergy_norm"].track, "discovery")

    def test_model_track_registry_marks_official_surfaces_correctly(self) -> None:
        self.assertEqual(get_model_track("bio_clean_priority"), "discovery")
        self.assertEqual(get_model_track("phylo_support_fusion_priority"), "governance")
        self.assertEqual(get_model_track("baseline_both"), "baseline")
        self.assertEqual(MODULE_A_MODEL_TRACKS["bio_residual_synergy_priority"], "discovery")

    def test_bio_residual_synergy_priority_uses_only_discovery_safe_interactions(self) -> None:
        self.assertEqual(
            MODULE_A_FEATURE_SETS["bio_residual_synergy_priority"],
            [
                "T_eff_norm",
                "H_obs_norm",
                "A_eff_norm",
                "coherence_score",
                "orit_support",
                "T_H_obs_synergy_norm",
                "A_H_obs_synergy_norm",
                "T_coherence_synergy_norm",
                "T_A_synergy_norm",
            ],
        )

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
                "H_obs_specialization_norm": 1.0 - np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "host_taxon_evenness_norm": np.clip(priority[::-1] * 0.55, 0.0, 1.0),
                "host_phylogenetic_dispersion_norm": np.clip(priority[::-1] * 0.5, 0.0, 1.0),
                "A_raw_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "A_recurrence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "support_shrinkage_norm": np.clip(priority * 0.95, 0.0, 1.0),
                "amr_support_norm": np.clip(priority * 0.75, 0.0, 1.0),
                "H_support_norm": np.clip(priority[::-1] * 0.65, 0.0, 1.0),
                "H_support_norm_residual": np.linspace(-0.3, 0.3, n),
                "amr_support_norm_residual": np.linspace(0.25, -0.25, n),
                "coherence_score": np.clip(priority + 0.05, 0.0, 1.0),
                "orit_support": np.clip(priority, 0.0, 1.0),
                "backbone_purity_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "assignment_confidence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "mash_neighbor_distance_train_norm": np.clip(1.0 - priority * 0.5, 0.0, 1.0),
                "replicon_architecture_norm": np.clip(priority * 0.6, 0.0, 1.0),
                "clinical_context_fraction_norm": np.clip(priority * 0.4, 0.0, 1.0),
                "ecology_context_diversity_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "amr_gene_burden_norm": np.clip(priority * 0.65, 0.0, 1.0),
                "evolutionary_jump_score_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_agreement_score": np.clip(priority * 0.7, 0.0, 1.0),
                "mean_amr_uncertainty_score": np.clip(1.0 - priority * 0.3, 0.0, 1.0),
                "mash_graph_novelty_score": np.clip(priority * 0.45, 0.0, 1.0),
                "mash_graph_bridge_fraction": np.clip(priority * 0.3, 0.0, 1.0),
                "plasmidfinder_complexity_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_class_richness_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "pmlst_coherence_norm": np.clip(priority * 0.6, 0.0, 1.0),
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
        result = evaluate_model_name(
            scored, model_name="enhanced_priority", n_splits=3, n_repeats=2, seed=4
        )
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

    def test_evaluate_model_name_reports_selected_split_strategy(self) -> None:
        scored = self._hybrid_ready_scored(n=36)
        scored["resolved_year"] = np.repeat(np.arange(2010, 2019), 4)[: len(scored)]
        discovery = evaluate_model_name(
            scored,
            model_name="parsimonious_priority",
            n_splits=3,
            n_repeats=2,
            seed=19,
        )
        governance = evaluate_model_name(
            scored,
            model_name="governance_linear",
            n_splits=3,
            n_repeats=2,
            seed=19,
        )

        self.assertEqual(discovery.metrics.get("split_strategy"), "stratified_repeated")
        self.assertEqual(governance.metrics.get("split_strategy"), "temporal_group")

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
        self.assertIn("prediction_std", preds.columns)
        self.assertIn("prediction_ci_lower", preds.columns)
        self.assertIn("prediction_ci_upper", preds.columns)
        self.assertTrue(preds["prediction_std"].ge(0.0).all())
        self.assertTrue(preds["prediction_ci_lower"].between(0.0, 1.0).all())
        self.assertTrue(preds["prediction_ci_upper"].between(0.0, 1.0).all())

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
        self.assertIn("prediction_posterior_mean", preds.columns)
        self.assertIn("prediction_std", preds.columns)
        self.assertTrue(preds["prediction_posterior_mean"].between(0.0, 1.0).all())
        self.assertTrue(preds["prediction_std"].ge(0.0).all())

    def test_hybrid_agreement_priority_emits_agreement_surface(self) -> None:
        scored = self._hybrid_ready_scored(n=36)
        result = evaluate_model_name(
            scored,
            model_name="hybrid_agreement_priority",
            n_splits=3,
            n_repeats=2,
            seed=17,
        )
        self.assertEqual(len(result.predictions), len(scored))
        self.assertIn("agreement_score", result.predictions.columns)
        self.assertIn("agreement_review_flag", result.predictions.columns)
        self.assertIn("logistic_base_prediction", result.predictions.columns)
        self.assertIn("nonlinear_base_prediction", result.predictions.columns)
        self.assertIn("nonlinear_backend_requested", result.predictions.columns)
        self.assertIn("nonlinear_backend_resolved", result.predictions.columns)
        self.assertIn("nonlinear_backend_resolution_status", result.predictions.columns)
        self.assertIn("mean_agreement_score", result.metrics)
        self.assertIn("review_fraction", result.metrics)
        self.assertTrue(result.predictions["agreement_score"].between(0.0, 1.0).all())

    def test_hybrid_agreement_priority_falls_back_when_ebm_fit_fails(self) -> None:
        scored = self._hybrid_ready_scored(n=36)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with (
                mock.patch.object(
                    module_a,
                    "_resolve_nonlinear_backend_name",
                    return_value="ebm",
                ),
                mock.patch.object(
                    module_a,
                    "_fit_ebm_classifier",
                    side_effect=PermissionError("blocked by environment"),
                ),
            ):
                result = evaluate_model_name(
                    scored,
                    model_name="hybrid_agreement_priority",
                    n_splits=3,
                    n_repeats=2,
                    seed=17,
                )
        self.assertEqual(len(result.predictions), len(scored))
        self.assertIn("oof_prediction", result.predictions.columns)
        self.assertIn("agreement_score", result.predictions.columns)
        self.assertIn("agreement_review_flag", result.predictions.columns)
        self.assertIn("logistic_base_prediction", result.predictions.columns)
        self.assertIn("nonlinear_base_prediction", result.predictions.columns)
        self.assertIn("nonlinear_backend_requested", result.predictions.columns)
        self.assertIn("nonlinear_backend_resolved", result.predictions.columns)
        self.assertIn("nonlinear_backend_resolution_status", result.predictions.columns)
        self.assertTrue(result.predictions["oof_prediction"].between(0.0, 1.0).all())
        self.assertTrue(result.predictions["nonlinear_base_prediction"].between(0.0, 1.0).all())
        self.assertEqual(str(result.predictions["nonlinear_backend_resolved"].iloc[0]), "hist_gbm")
        self.assertIn(
            "fallback", str(result.predictions["nonlinear_backend_resolution_status"].iloc[0])
        )
        self.assertIn("mean_agreement_score", result.metrics)
        self.assertIn("review_fraction", result.metrics)

    def test_hybrid_backend_prefers_ebm_when_interpret_is_available(self) -> None:
        backend = module_a._resolve_nonlinear_backend_name({"nonlinear_backend": "ebm"})
        if module_a.ExplainableBoostingClassifier is None:
            self.assertEqual(backend, "hist_gbm")
        else:
            self.assertEqual(backend, "ebm")

    def test_hybrid_full_fit_predictions_include_base_models(self) -> None:
        scored = self._hybrid_ready_scored(n=32, include_unlabeled_tail=4)
        preds = fit_full_model_predictions(scored, model_name="hybrid_agreement_priority")
        self.assertEqual(len(preds), len(scored))
        self.assertIn("prediction", preds.columns)
        self.assertIn("logistic_base_prediction", preds.columns)
        self.assertIn("nonlinear_base_prediction", preds.columns)
        self.assertIn("agreement_score", preds.columns)
        self.assertIn("agreement_review_flag", preds.columns)
        self.assertIn("nonlinear_backend_requested", preds.columns)
        self.assertIn("nonlinear_backend_resolved", preds.columns)
        self.assertIn("nonlinear_backend_resolution_status", preds.columns)
        self.assertTrue(preds["prediction"].between(0.0, 1.0).all())

    def test_full_fit_predictions_can_skip_posterior_uncertainty(self) -> None:
        n = 24
        rng = np.random.default_rng(22)
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": ([0, 1] * 10) + [np.nan, np.nan, np.nan, np.nan],
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
        preds = fit_full_model_predictions(
            scored,
            model_name="evidence_aware_priority",
            include_posterior_uncertainty=False,
        )
        self.assertEqual(list(preds.columns), ["backbone_id", "prediction"])
        self.assertEqual(len(preds), n)
        self.assertTrue(preds["prediction"].between(0.0, 1.0).all())

    def test_run_module_a_continues_when_one_model_fails_in_worker_fallback(self) -> None:
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
                "H_obs_specialization_norm": 1.0 - np.clip(priority[::-1] * 0.6, 0.0, 1.0),
                "host_taxon_evenness_norm": np.clip(priority[::-1] * 0.55, 0.0, 1.0),
                "host_phylogenetic_dispersion_norm": np.clip(priority[::-1] * 0.5, 0.0, 1.0),
                "A_raw_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "A_recurrence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "support_shrinkage_norm": np.clip(priority * 0.95, 0.0, 1.0),
                "amr_support_norm": np.clip(priority * 0.75, 0.0, 1.0),
                "H_support_norm": np.clip(priority[::-1] * 0.65, 0.0, 1.0),
                "H_support_norm_residual": np.linspace(-0.3, 0.3, n),
                "amr_support_norm_residual": np.linspace(0.25, -0.25, n),
                "coherence_score": np.clip(priority + 0.05, 0.0, 1.0),
                "orit_support": np.clip(priority, 0.0, 1.0),
                "backbone_purity_norm": np.clip(priority * 0.8, 0.0, 1.0),
                "assignment_confidence_norm": np.clip(priority * 0.7, 0.0, 1.0),
                "mash_neighbor_distance_train_norm": np.clip(1.0 - priority * 0.5, 0.0, 1.0),
                "replicon_architecture_norm": np.clip(priority * 0.6, 0.0, 1.0),
                "clinical_context_fraction_norm": np.clip(priority * 0.4, 0.0, 1.0),
                "ecology_context_diversity_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "amr_gene_burden_norm": np.clip(priority * 0.65, 0.0, 1.0),
                "evolutionary_jump_score_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_agreement_score": np.clip(priority * 0.7, 0.0, 1.0),
                "mean_amr_uncertainty_score": np.clip(1.0 - priority * 0.3, 0.0, 1.0),
                "mash_graph_novelty_score": np.clip(priority * 0.45, 0.0, 1.0),
                "mash_graph_bridge_fraction": np.clip(priority * 0.3, 0.0, 1.0),
                "plasmidfinder_complexity_norm": np.clip(priority * 0.5, 0.0, 1.0),
                "amr_class_richness_norm": np.clip(priority * 0.55, 0.0, 1.0),
                "pmlst_coherence_norm": np.clip(priority * 0.6, 0.0, 1.0),
                "H_external_host_range_support": np.clip(priority[::-1], 0.0, 1.0),
                "pmlst_presence_fraction_train": np.clip(priority * 0.4, 0.0, 1.0),
                "log1p_member_count_train": np.log1p(rng.integers(1, 10, size=n)),
                "log1p_n_countries_train": np.log1p(rng.integers(1, 4, size=n)),
                "refseq_share_train": rng.uniform(0, 1, size=n),
            }
        )
        original_evaluate_model_name = module_a.evaluate_model_name

        def _evaluate_model_name_side_effect(*args: object, **kwargs: object) -> object:
            model_name = str(kwargs.get("model_name", ""))
            if model_name == "T_only":
                raise RuntimeError("forced model failure")
            return original_evaluate_model_name(*args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with (
                mock.patch.object(
                    module_a,
                    "ProcessPoolExecutor",
                    side_effect=OSError("process pool unavailable"),
                ),
                mock.patch.object(
                    module_a,
                    "evaluate_model_name",
                    side_effect=_evaluate_model_name_side_effect,
                ),
            ):
                results = run_module_a(scored, n_splits=4, n_repeats=2, seed=42, n_jobs=2)

        self.assertIn("T_only", results)
        self.assertEqual(results["T_only"].status, "failed")
        self.assertEqual(results["T_only"].metrics, {})
        self.assertTrue(results["T_only"].predictions.empty)
        self.assertIn("RuntimeError", results["T_only"].error_message or "")
        self.assertEqual(results["baseline_both"].status, "ok")
        self.assertEqual(len(results["baseline_both"].predictions), n)

    def test_firth_parsimonious_priority_scores_rows(self) -> None:
        scored = self._hybrid_ready_scored(n=28)
        scored["H_obs_specialization_norm"] = 1.0 - scored["H_obs_norm"]
        result = evaluate_model_name(
            scored,
            model_name="firth_parsimonious_priority",
            n_splits=3,
            n_repeats=2,
            seed=19,
        )
        self.assertEqual(len(result.predictions), len(scored))
        self.assertIn("roc_auc", result.metrics)
        self.assertTrue(result.predictions["oof_prediction"].between(0.0, 1.0).all())

    def test_firthlogist_sidecar_backend_returns_coefficients_when_available(self) -> None:
        X = np.array(
            [
                [2.0, 0.5],
                [1.5, 0.3],
                [1.0, 0.2],
                [-1.0, -0.2],
                [-1.5, -0.3],
                [-2.0, -0.5],
            ],
            dtype=float,
        )
        y = np.array([1, 1, 1, 0, 0, 0], dtype=int)
        beta, diagnostics = module_a_support_impl._fit_logistic_regression_with_diagnostics(
            X,
            y,
            max_iter=100,
            fit_backend="firthlogist",
        )
        self.assertEqual(beta.shape[0], X.shape[1] + 1)
        self.assertIn(
            str(diagnostics["solver"]),
            {"firthlogist_sidecar", "firth_warm_start_fallback", "firth_newton"},
        )

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

    def test_sovereign_fit_kwargs_use_native_class_balanced_training(self) -> None:
        fit_kwargs = module_a_support_impl._model_fit_kwargs("sovereign_precision_priority")

        self.assertEqual(fit_kwargs["sample_weight_mode"], "class_balanced")
        self.assertEqual(fit_kwargs["l2"], 1.5)
        self.assertNotIn("preprocess_mode", fit_kwargs)

    def test_sovereign_v2_feature_surface_prunes_dead_signal(self) -> None:
        features = MODULE_A_FEATURE_SETS["sovereign_v2_priority"]

        self.assertNotIn("assignment_confidence_norm", features)
        self.assertNotIn("context_support_guard_norm", features)
        self.assertIn("T_eff_norm", features)
        self.assertIn("T_raw_norm", features)
        self.assertIn("A_eff_norm", features)
        self.assertIn("metadata_support_depth_norm", features)
        self.assertIn("evolutionary_jump_score_norm", features)

    def test_single_model_candidate_family_includes_sovereign_parents(self) -> None:
        family = module_a_support_impl.build_single_model_candidate_family()
        parents = set(
            family.loc[family["candidate_kind"].astype(str) == "parent", "model_name"].astype(str)
        )

        self.assertIn("sovereign_precision_priority", parents)
        self.assertIn("sovereign_v2_priority", parents)

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

    def test_official_model_names_resolve_discovery_governance_and_baseline(self) -> None:
        model_names = get_official_model_names(
            [
                "baseline_both",
                PRIMARY_MODEL_NAME,
                "full_priority",
                GOVERNANCE_MODEL_NAME,
            ]
        )
        self.assertEqual(
            model_names,
            (PRIMARY_MODEL_NAME, GOVERNANCE_MODEL_NAME, "baseline_both"),
        )

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
            module_a._compute_sample_weight(eligible, mode="knownness_balanced"),
            index=eligible["backbone_id"],
        ).sort_index()
        shuffled = eligible.sample(frac=1.0, random_state=42).reset_index(drop=True)
        weight_b = pd.Series(
            module_a._compute_sample_weight(shuffled, mode="knownness_balanced"),
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
        weights = module_a._compute_sample_weight(bootstrap_train, mode="knownness_balanced")
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
        self.assertSetEqual(
            set(eligible["knownness_half"].astype(str)), {"lower_half", "upper_half"}
        )
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
        model_name = get_conservative_model_name(
            ["baseline_both", "T_plus_H_plus_A", CONSERVATIVE_MODEL_NAME]
        )
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
        coefficients = build_standardized_coefficient_table(
            scored, model_name=PRIMARY_MODEL_NAME, columns=columns
        )
        dropout = build_feature_dropout_audit(
            scored, model_name=PRIMARY_MODEL_NAME, columns=columns, n_splits=4, n_repeats=2, seed=42
        )
        stability = build_coefficient_stability_table(
            scored,
            model_name=PRIMARY_MODEL_NAME,
            columns=columns,
            n_splits=4,
            n_repeats=2,
            seed=42,
        )
        self.assertEqual(set(coefficients["feature_name"]), set(columns))
        self.assertIn("standard_error", coefficients.columns)
        self.assertIn("coefficient_ci_lower", coefficients.columns)
        self.assertIn("coefficient_ci_upper", coefficients.columns)
        self.assertTrue(coefficients["standard_error"].ge(0.0).all())
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

    def test_convergence_audit_includes_pairwise_model_rows(self) -> None:
        import plasmid_priority.modeling.module_a as module_a_runtime

        scored = self._hybrid_ready_scored(n=40)
        audit = module_a_runtime.build_logistic_convergence_audit(
            scored,
            model_names=["pairwise_rank_priority"],
            n_splits=4,
            n_repeats=2,
            seed=17,
        )
        self.assertFalse(audit.empty)
        self.assertEqual(set(audit["model_name"]), {"pairwise_rank_priority"})
        self.assertTrue(audit["repeat_index"].notna().all())
        self.assertTrue(audit["fold_index"].notna().all())
        self.assertIn("converged", audit.columns)
        self.assertIn("used_pinv", audit.columns)

    def test_pairwise_rank_priority_emits_predictions(self) -> None:
        scored = self._hybrid_ready_scored(n=40)
        result = evaluate_model_name(
            scored,
            model_name="pairwise_rank_priority",
            n_splits=4,
            n_repeats=2,
            seed=23,
        )

        self.assertEqual(result.name, "pairwise_rank_priority")
        self.assertEqual(len(result.predictions), len(scored))
        self.assertIn("oof_prediction", result.predictions.columns)
        self.assertTrue(result.predictions["oof_prediction"].between(0.0, 1.0).all())
        self.assertIn("roc_auc", result.metrics)

    def test_pu_negative_downweight_reduces_low_knownness_negative_weights(self) -> None:
        eligible = pd.DataFrame(
            {
                "backbone_id": ["bb_low", "bb_high", "bb_pos"],
                "spread_label": [0, 0, 1],
                "log1p_member_count_train": [0.0, 2.5, 2.5],
                "log1p_n_countries_train": [0.0, 1.5, 1.5],
                "refseq_share_train": [0.0, 0.9, 0.9],
            }
        )

        weights = module_a._compute_sample_weight(
            eligible,
            mode="pu_negative_downweight",
            fit_kwargs={"pu_negative_min_weight": 0.2, "pu_negative_power": 1.0},
        )

        assert weights is not None
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[0]), float(weights[2]))

    def test_inverse_knownness_weighting_gives_higher_weight_to_low_knownness_samples(self) -> None:
        eligible = pd.DataFrame(
            {
                "backbone_id": ["bb_low", "bb_high", "bb_pos_low", "bb_pos_high"],
                "spread_label": [0, 0, 1, 1],
                "log1p_member_count_train": [0.1, 3.0, 0.2, 2.8],
                "log1p_n_countries_train": [0.0, 1.5, 0.1, 1.4],
                "refseq_share_train": [0.05, 0.95, 0.1, 0.9],
            }
        )
        weights = module_a._compute_sample_weight(eligible, mode="inverse_knownness_weighting")
        assert weights is not None
        self.assertEqual(len(weights), 4)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        # Low-knownness negative should have higher weight than high-knownness negative
        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_inverse_knownness_weighting_combines_with_class_balanced(self) -> None:
        eligible = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "spread_label": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                "log1p_member_count_train": [
                    0.1,
                    0.2,
                    0.3,
                    2.5,
                    2.8,
                    3.0,
                    0.15,
                    0.25,
                    0.1,
                    0.2,
                    2.5,
                    2.8,
                ],
                "log1p_n_countries_train": [
                    0.0,
                    0.1,
                    0.2,
                    1.0,
                    1.2,
                    1.5,
                    0.05,
                    0.15,
                    0.0,
                    0.1,
                    1.0,
                    1.2,
                ],
                "refseq_share_train": [
                    0.05,
                    0.1,
                    0.15,
                    0.8,
                    0.9,
                    0.95,
                    0.03,
                    0.08,
                    0.05,
                    0.1,
                    0.8,
                    0.9,
                ],
            }
        )
        weights_combined = module_a._compute_sample_weight(
            eligible, mode="class_balanced+inverse_knownness_weighting"
        )
        weights_class_only = module_a._compute_sample_weight(eligible, mode="class_balanced")
        weights_ipw_only = module_a._compute_sample_weight(
            eligible, mode="inverse_knownness_weighting"
        )
        assert weights_combined is not None
        assert weights_class_only is not None
        assert weights_ipw_only is not None
        self.assertEqual(len(weights_combined), 12)
        self.assertTrue(np.isfinite(weights_combined).all())
        self.assertTrue((weights_combined > 0).all())
        # Combined weights should differ from either token alone
        self.assertFalse(np.allclose(weights_combined, weights_class_only))
        self.assertFalse(np.allclose(weights_combined, weights_ipw_only))

    def test_graph_evidence_priority_can_score_structural_columns(self) -> None:
        rng = np.random.default_rng(313)
        n = 30
        label = np.array([0, 1] * (n // 2))
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": label,
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "backbone_purity_norm": rng.uniform(0.1, 0.9, size=n),
                "assignment_confidence_norm": rng.uniform(0.1, 0.9, size=n),
                "mash_neighbor_distance_train_norm": rng.uniform(0.1, 0.9, size=n),
                "replicon_architecture_norm": rng.uniform(0.1, 0.9, size=n),
                "clinical_context_fraction_norm": rng.uniform(0.1, 0.9, size=n),
                "A_recurrence_norm": rng.uniform(0.1, 0.9, size=n),
                "pmlst_coherence_norm": rng.uniform(0.1, 0.9, size=n),
                "ecology_context_diversity_norm": rng.uniform(0.1, 0.9, size=n),
                "mash_graph_novelty_score": np.where(label == 1, 0.8, 0.2),
                "mash_graph_bridge_fraction": np.where(label == 1, 0.7, 0.1),
                "amr_agreement_score": np.where(label == 1, 0.9, 0.4),
                "mean_amr_uncertainty_score": np.where(label == 1, 0.1, 0.5),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )

        result = evaluate_model_name(
            scored,
            model_name="graph_evidence_priority",
            n_splits=3,
            n_repeats=2,
            seed=29,
        )

        self.assertEqual(len(result.predictions), n)
        self.assertIn("oof_prediction", result.predictions.columns)
        self.assertTrue(result.predictions["oof_prediction"].between(0.0, 1.0).all())

    def test_tree_extract_xy_uses_configured_features_allow_list(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "spread_label": [0.0, 1.0, 0.0, 1.0],
                "T_raw_norm": [0.2, 0.8, 0.3, 0.7],
                "H_breadth_norm": [0.1, 0.9, 0.2, 0.8],
                "A_raw_norm": [0.4, 0.6, 0.5, 0.55],
                "future_signal_proxy": [0.0, 1.0, 0.0, 1.0],
                "n_new_countries_future": [1.0, 2.0, 3.0, 4.0],
            }
        )
        X, y = tree_models_impl._extract_xy(scored)

        self.assertListEqual(X.columns.tolist(), ["T_raw_norm", "H_breadth_norm", "A_raw_norm"])
        self.assertNotIn("future_signal_proxy", X.columns)
        self.assertNotIn("n_new_countries_future", X.columns)
        self.assertListEqual(y.astype(int).tolist(), [0, 1, 0, 1])

    def test_lightgbm_classifier_fit_and_predict(self) -> None:
        """_fit_lightgbm_classifier should produce a model with predict_proba."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.uniform(0, 1, size=(n, 4))
        y = (X[:, 0] > 0.5).astype(int)
        model = module_a._fit_lightgbm_classifier(
            X, y, fit_kwargs={"n_estimators": 20, "max_depth": 3, "verbose": -1}
        )
        self.assertTrue(hasattr(model, "predict_proba"))
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (n, 2))
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_oof_lightgbm_predictions_returns_well_formed_output(self) -> None:
        """_oof_lightgbm_predictions_from_eligible should return OOF preds without standardization."""
        rng = np.random.default_rng(77)
        n = 50
        eligible = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": rng.choice([0, 1], size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0, 1, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        columns = [
            "T_eff_norm",
            "H_specialization_norm",
            "A_eff_norm",
            "coherence_score",
            "orit_support",
        ]
        preds, y = module_a._oof_lightgbm_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=3,
            n_repeats=1,
            seed=42,
            fit_kwargs={"n_estimators": 20, "max_depth": 3},
        )
        self.assertEqual(len(preds), n)
        self.assertEqual(len(y), n)
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue((preds >= 0).all() and (preds <= 1).all())

    def test_oof_predictions_routes_lightgbm_model_type(self) -> None:
        """_oof_predictions_from_eligible should route model_type='lightgbm' correctly."""
        rng = np.random.default_rng(88)
        n = 40
        eligible = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": rng.choice([0, 1], size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
                "orit_support": rng.uniform(0, 1, size=n),
                "log1p_member_count_train": rng.uniform(0.1, 2.0, size=n),
                "log1p_n_countries_train": rng.uniform(0.1, 1.5, size=n),
                "refseq_share_train": rng.uniform(0.0, 1.0, size=n),
            }
        )
        columns = [
            "T_eff_norm",
            "H_specialization_norm",
            "A_eff_norm",
            "coherence_score",
            "orit_support",
        ]
        preds, y = module_a._oof_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=3,
            n_repeats=1,
            seed=42,
            fit_kwargs={"model_type": "lightgbm", "n_estimators": 20, "max_depth": 3},
        )
        self.assertEqual(len(preds), n)
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue((preds >= 0).all() and (preds <= 1).all())

    def test_compute_shap_tree_values_with_lightgbm(self) -> None:
        """compute_shap_tree_values should return exact SHAP values for LightGBM."""
        rng = np.random.default_rng(99)
        n = 80
        X_df = pd.DataFrame(
            {
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
            }
        )
        y = (X_df["T_eff_norm"] > 0.5).astype(int).to_numpy()
        model = module_a._fit_lightgbm_classifier(
            X_df.to_numpy(), y, fit_kwargs={"n_estimators": 20, "max_depth": 3}
        )
        from plasmid_priority.modeling.shap_explainer import compute_shap_tree_values

        result = compute_shap_tree_values(model, X_df)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["feature_names"]), 4)
        self.assertEqual(result["shap_values"].shape, (n, 4))
        self.assertIsInstance(result["base_value"], float)

    def test_build_global_feature_importance_from_tree_shap(self) -> None:
        """build_global_feature_importance should rank features by mean |SHAP|."""
        rng = np.random.default_rng(99)
        n = 80
        X_df = pd.DataFrame(
            {
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
            }
        )
        y = (X_df["T_eff_norm"] > 0.5).astype(int).to_numpy()
        model = module_a._fit_lightgbm_classifier(
            X_df.to_numpy(), y, fit_kwargs={"n_estimators": 20, "max_depth": 3}
        )
        from plasmid_priority.modeling.shap_explainer import (
            build_global_feature_importance,
            compute_shap_tree_values,
        )

        shap_result = compute_shap_tree_values(model, X_df)
        importance = build_global_feature_importance(shap_result)
        self.assertEqual(len(importance), 4)
        self.assertIn("feature", importance.columns)
        self.assertIn("mean_abs_shap", importance.columns)
        self.assertIn("direction", importance.columns)
        # Should be sorted descending
        self.assertTrue((importance["mean_abs_shap"].diff().dropna() <= 0).all())

    def test_fit_global_lightgbm_model(self) -> None:
        """fit_global_lightgbm_model should fit a single model for interpretability."""
        rng = np.random.default_rng(42)
        n = 60
        eligible = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(n)],
                "spread_label": rng.choice([0, 1], size=n),
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
            }
        )
        columns = ["T_eff_norm", "H_specialization_norm", "A_eff_norm"]
        from plasmid_priority.modeling.shap_explainer import fit_global_lightgbm_model

        result = fit_global_lightgbm_model(
            eligible, columns, fit_kwargs={"n_estimators": 20, "max_depth": 3}
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(hasattr(result["model"], "predict_proba"))
        self.assertEqual(result["feature_names"], columns)
        self.assertEqual(result["X"].shape, (n, 3))

    def test_build_shap_dependence_data(self) -> None:
        """build_shap_dependence_data should return dependence plot data for top features."""
        rng = np.random.default_rng(99)
        n = 80
        X_df = pd.DataFrame(
            {
                "T_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "H_specialization_norm": rng.uniform(0.1, 0.9, size=n),
                "A_eff_norm": rng.uniform(0.1, 0.9, size=n),
                "coherence_score": rng.uniform(0.1, 0.9, size=n),
            }
        )
        y = (X_df["T_eff_norm"] > 0.5).astype(int).to_numpy()
        model = module_a._fit_lightgbm_classifier(
            X_df.to_numpy(), y, fit_kwargs={"n_estimators": 20, "max_depth": 3}
        )
        from plasmid_priority.modeling.shap_explainer import (
            build_shap_dependence_data,
            compute_shap_tree_values,
        )

        shap_result = compute_shap_tree_values(model, X_df)
        dep_data = build_shap_dependence_data(shap_result, X_df, top_features=3)
        self.assertEqual(len(dep_data), 3)
        for entry in dep_data:
            self.assertIn("feature", entry)
            self.assertIn("feature_values", entry)
            self.assertIn("shap_values", entry)
            self.assertIn("interaction_feature", entry)
            self.assertEqual(len(entry["feature_values"]), n)
            self.assertEqual(len(entry["shap_values"]), n)

    def test_nested_cross_validate_uses_inner_cv_to_select_fit_config(self) -> None:
        scored = self._hybrid_ready_scored(n=36)
        inner_fit_configs = [{"l2": 0.1}, {"l2": 10.0}]

        def _fake_inner_eval(*_args: object, **kwargs: object) -> SimpleNamespace:
            fit_config = kwargs.get("fit_config")
            candidate = fit_config if isinstance(fit_config, dict) else {}
            l2 = float(candidate.get("l2", 1.0))
            auc = 0.91 if np.isclose(l2, 0.1) else 0.61
            return SimpleNamespace(metrics={"roc_auc": auc})

        def _fake_holdout(
            _train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            *,
            model_name: str,
            fit_config: dict[str, object] | None = None,
            **_kwargs: object,
        ) -> pd.DataFrame:
            assert model_name == "bio_clean_priority"
            candidate = fit_config if fit_config is not None else {}
            l2 = float(candidate.get("l2", 1.0))
            high = 0.9 if np.isclose(l2, 0.1) else 0.6
            low = 1.0 - high
            y = test_df["spread_label"].to_numpy(dtype=int)
            preds = np.where(y == 1, high, low)
            return pd.DataFrame(
                {
                    "backbone_id": test_df["backbone_id"].astype(str).tolist(),
                    "spread_label": y.tolist(),
                    "prediction": preds.tolist(),
                },
            )

        with (
            mock.patch(
                "plasmid_priority.modeling.module_a.evaluate_model_name",
                side_effect=_fake_inner_eval,
            ) as inner_mock,
            mock.patch(
                "plasmid_priority.modeling.module_a.fit_predict_model_holdout",
                side_effect=_fake_holdout,
            ) as holdout_mock,
        ):
            result = nested_cross_validate(
                scored,
                model_name="bio_clean_priority",
                n_outer_splits=3,
                n_inner_splits=2,
                n_repeats=1,
                seed=7,
                inner_fit_configs=inner_fit_configs,
            )

        self.assertEqual(inner_mock.call_count, 3 * len(inner_fit_configs))
        self.assertEqual(holdout_mock.call_count, 3)
        for called in holdout_mock.call_args_list:
            fit_config = called.kwargs.get("fit_config")
            self.assertIsInstance(fit_config, dict)
            self.assertTrue(np.isclose(float(fit_config.get("l2", 1.0)), 0.1))
        self.assertEqual(result["selection_mode"], "inner_cv_tuning")
        self.assertTrue(bool(result["inner_cv_participated"]))
        self.assertEqual(len(result["fold_selection_summary"]), 3)
        self.assertTrue(
            all(
                fold.get("selection_mode") == "inner_cv_tuning"
                for fold in result["fold_selection_summary"]
            ),
        )

    def test_nested_cross_validate_reports_explicit_no_tuning_mode(self) -> None:
        scored = self._hybrid_ready_scored(n=36)

        def _fake_holdout(
            _train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            *,
            model_name: str,
            fit_config: dict[str, object] | None = None,
            **_kwargs: object,
        ) -> pd.DataFrame:
            assert model_name == "bio_clean_priority"
            y = test_df["spread_label"].to_numpy(dtype=int)
            preds = np.where(y == 1, 0.85, 0.15)
            return pd.DataFrame(
                {
                    "backbone_id": test_df["backbone_id"].astype(str).tolist(),
                    "spread_label": y.tolist(),
                    "prediction": preds.tolist(),
                },
            )

        with (
            mock.patch(
                "plasmid_priority.modeling.module_a.evaluate_model_name",
            ) as inner_mock,
            mock.patch(
                "plasmid_priority.modeling.module_a.fit_predict_model_holdout",
                side_effect=_fake_holdout,
            ),
        ):
            result = nested_cross_validate(
                scored,
                model_name="bio_clean_priority",
                n_outer_splits=3,
                n_inner_splits=2,
                n_repeats=1,
                seed=7,
                inner_fit_configs=[{}],
            )

        inner_mock.assert_not_called()
        self.assertEqual(result["selection_mode"], "explicit_no_tuning")
        self.assertFalse(bool(result["inner_cv_participated"]))
        self.assertEqual(len(result["fold_selection_summary"]), 3)
        self.assertTrue(
            all(
                fold.get("selection_mode") == "explicit_no_tuning"
                for fold in result["fold_selection_summary"]
            ),
        )

    def test_nested_cv_evaluator_tracks_inner_parameter_selection(self) -> None:
        rng = np.random.default_rng(313)
        X = rng.normal(size=(120, 4))
        linear_signal = (1.8 * X[:, 0]) - (0.7 * X[:, 1]) + rng.normal(scale=0.4, size=120)
        y = (linear_signal > 0.0).astype(int)
        evaluator = NestedCVEvaluator(outer_cv=3, inner_cv=3, random_state=13)

        result = evaluator.evaluate(
            X,
            y,
            model_factory=lambda: LogisticRegression(max_iter=200, solver="liblinear"),
        )

        self.assertEqual(result["selection_mode"], "inner_cv_tuning")
        self.assertTrue(bool(result["inner_selection_performed"]))
        trace = result["selection_trace"]
        self.assertEqual(len(trace), 3)
        self.assertTrue(
            all(entry.get("selection_mode") == "inner_cv_tuning" for entry in trace),
        )
        self.assertTrue(
            all(int(entry.get("n_candidates_evaluated", 0)) >= 2 for entry in trace),
        )
        self.assertTrue(all("C" in dict(entry.get("selected_params", {})) for entry in trace))

    def test_nested_cv_evaluator_reports_explicit_no_tuning(self) -> None:
        rng = np.random.default_rng(909)
        X = rng.normal(size=(90, 3))
        y = np.array([0, 1] * 45, dtype=int)
        evaluator = NestedCVEvaluator(outer_cv=3, inner_cv=3, random_state=19)

        result = evaluator.evaluate(
            X,
            y,
            model_factory=lambda: DummyClassifier(strategy="prior"),
        )

        self.assertEqual(result["selection_mode"], "explicit_no_tuning")
        self.assertFalse(bool(result["inner_selection_performed"]))
        trace = result["selection_trace"]
        self.assertEqual(len(trace), 3)
        self.assertTrue(
            all(entry.get("selection_mode") == "explicit_no_tuning" for entry in trace),
        )

    def test_shap_functions_return_skipped_when_unavailable(self) -> None:
        """SHAP functions should return skipped status when shap is not available."""
        from plasmid_priority.modeling import shap_explainer as se

        # Monkey-patch to simulate missing shap
        original = se.SHAP_AVAILABLE
        try:
            se.SHAP_AVAILABLE = False
            result = se.compute_shap_tree_values(None, pd.DataFrame())
            self.assertEqual(result["status"], "skipped")
            result = se.compute_shap_interactions(None, pd.DataFrame())
            self.assertEqual(result["status"], "skipped")
            result = se.build_shap_dependence_data({"status": "skipped"}, pd.DataFrame())
            self.assertEqual(result, [])
        finally:
            se.SHAP_AVAILABLE = original


if __name__ == "__main__":
    unittest.main()
