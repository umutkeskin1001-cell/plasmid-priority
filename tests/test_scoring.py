from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.scoring import (
    DEFAULT_NORMALIZATION_METHOD,
    build_scored_backbone_table,
    recompute_priority_from_reference,
)


class ScoringTests(unittest.TestCase):
    def test_default_normalization_method_is_rank_percentile(self) -> None:
        self.assertEqual(DEFAULT_NORMALIZATION_METHOD, "rank_percentile")

    def test_build_scored_backbone_table_supports_robust_sigmoid_normalization(self) -> None:
        backbone_table = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [2, 3, 4],
                "member_count_total": [2, 3, 4],
                "n_countries_train": [1, 2, 3],
                "n_countries_test": [0, 1, 2],
                "n_new_countries": [0, 1, 1],
                "spread_label": [0, 1, 1],
                "refseq_share_train": [0.0, 1.0, 1.0],
                "insd_share_train": [1.0, 0.0, 0.0],
                "coherence_score": [0.4, 0.6, 0.8],
                "mean_n_replicon_types_train": [1.0, 1.5, 2.0],
                "multi_replicon_fraction_train": [0.0, 0.5, 1.0],
                "primary_replicon_diversity_train": [0.0, 0.2, 0.4],
                "plasmidfinder_support_score": [0.0, 0.4, 0.8],
                "plasmidfinder_complexity_score": [0.0, 0.3, 0.7],
            }
        )
        feature_t = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [2, 3, 4],
                "relaxase_support": [0.0, 0.5, 1.0],
                "mpf_support": [0.0, 0.5, 1.0],
                "orit_support": [0.0, 0.5, 1.0],
                "mobilizable_support": [0.0, 0.5, 1.0],
                "support_shrinkage": [0.4, 0.5, 0.6],
                "T_raw": [0.1, 0.4, 0.8],
                "T_eff": [0.04, 0.2, 0.48],
            }
        )
        feature_h = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "H_genus_norm": [0.1, 0.2, 0.3],
                "phylo_breadth_score": [0.2, 0.4, 0.6],
                "phylo_pairwise_dispersion_score": [0.1, 0.3, 0.8],
                "phylo_breadth_augmented_score": [0.2, 0.4, 0.8],
                "host_taxon_evenness_score": [0.0, 0.4, 0.8],
                "host_support_factor": [0.4, 0.5, 0.6],
                "H_external_host_range_support": [0.0, 0.5, 1.0],
                "H_external_host_range_score": [0.0, 0.4, 0.9],
                "H_obs": [0.18, 0.36, 0.58],
                "H_raw": [0.14, 0.28, 0.42],
                "H_phylogenetic_raw": [0.14, 0.28, 0.49],
                "H_support": [0.2, 0.5, 0.8],
                "H_augmented_raw": [0.14, 0.28, 0.42],
                "H_phylogenetic_augmented_raw": [0.14, 0.28, 0.49],
                "H_eff": [0.056, 0.14, 0.252],
                "H_phylogenetic_eff": [0.056, 0.14, 0.294],
                "H_augmented_eff": [0.056, 0.14, 0.252],
                "H_phylogenetic_augmented_eff": [0.056, 0.14, 0.294],
            }
        )
        feature_a = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "mean_amr_class_count": [0.0, 1.0, 2.0],
                "mean_amr_gene_count": [0.0, 2.0, 4.0],
                "mdr_proxy_fraction": [0.0, 0.5, 1.0],
                "xdr_proxy_fraction": [0.0, 0.0, 0.5],
                "mean_last_resort_convergence_score": [0.0, 0.5, 1.0],
                "mean_amr_mechanism_diversity_proxy": [0.0, 0.5, 1.0],
                "mean_amr_clinical_threat_score": [9.0, 1.0, 0.0],
                "A_consistency": [0.1, 0.5, 0.9],
                "A_recurrence": [0.0, 0.4, 0.8],
                "amr_support_factor": [0.3, 0.5, 0.8],
                "canonical_member_count_train": [2, 3, 4],
            }
        )
        scored = build_scored_backbone_table(
            backbone_table,
            feature_t,
            feature_h,
            feature_a,
            normalization_method="robust_sigmoid",
        )
        for column in [
            "T_eff_norm",
            "H_eff_norm",
            "H_obs_norm",
            "A_eff_norm",
            "T_raw_norm",
            "A_raw_norm",
            "amr_clinical_threat_norm",
            "amr_load_density_norm",
            "amr_mdr_proxy_norm",
            "amr_xdr_proxy_norm",
            "amr_clinical_escalation_norm",
            "last_resort_convergence_norm",
            "amr_mechanism_diversity_norm",
            "amr_burden_latent_norm",
            "amr_burden_saturation_norm",
            "support_shrinkage_norm",
            "amr_support_norm",
            "priority_index",
            "bio_priority_index",
            "evidence_support_index",
            "mean_n_replicon_types_norm",
            "replicon_architecture_norm",
            "replicon_multiplicity_latent_norm",
            "replicon_multiplicity_saturation_norm",
            "host_range_latent_norm",
            "host_range_saturation_norm",
            "eco_clinical_context_latent_norm",
            "eco_clinical_context_saturation_norm",
            "monotonic_latent_priority_index",
            "host_phylogenetic_dispersion_norm",
            "host_taxon_evenness_norm",
            "H_phylogenetic_norm",
            "H_obs_specialization_norm",
            "H_phylogenetic_specialization_norm",
            "A_recurrence_norm",
            "clinical_context_sparse_penalty_norm",
            "external_t_synergy_norm",
            "T_H_obs_synergy_norm",
            "A_H_obs_synergy_norm",
            "T_coherence_synergy_norm",
            "clinical_weapon_synergy_norm",
            "endemic_resistance_norm",
            "H_evenness_T_synergy_norm",
            "pmlst_presence_norm",
            "plasmidfinder_support_norm",
            "plasmidfinder_complexity_norm",
            "metadata_support_depth_norm",
            "metadata_missingness_burden",
            "context_support_guard_norm",
            "silent_carrier_risk_norm",
            "evolutionary_jump_score_norm",
        ]:
            self.assertTrue(scored[column].between(0.0, 1.0).all())

        # Core A-content should follow the documented burden + richness design,
        # not the descriptive WHO-derived clinical-threat column itself.
        bb2 = scored.loc[scored["backbone_id"] == "bb2"].iloc[0]
        self.assertAlmostEqual(
            float(bb2["A_content_raw"]),
            0.5 * float(bb2["amr_class_richness_norm"]) + 0.5 * float(bb2["amr_gene_burden_norm"]),
        )
        self.assertTrue(scored["amr_clinical_threat_norm"].between(0.0, 1.0).all())
        expected_evenness_t = float(np.clip(0.4 * float(bb2["T_eff_norm"]), 0.0, 1.0))
        self.assertAlmostEqual(float(bb2["H_evenness_T_synergy_norm"]), expected_evenness_t)
        expected_t_h_obs = float(
            np.clip(float(bb2["T_eff_norm"]) * float(bb2["H_obs_norm"]), 0.0, 1.0)
        )
        self.assertAlmostEqual(float(bb2["T_H_obs_synergy_norm"]), expected_t_h_obs)
        expected_a_h_obs = float(
            np.clip(float(bb2["A_eff_norm"]) * float(bb2["H_obs_norm"]), 0.0, 1.0)
        )
        self.assertAlmostEqual(float(bb2["A_H_obs_synergy_norm"]), expected_a_h_obs)
        expected_clinical_weapon = float(
            np.clip(
                float(bb2["T_eff_norm"])
                * float(bb2["A_eff_norm"])
                * float(bb2["H_specialization_norm"]),
                0.0,
                1.0,
            )
        )
        self.assertAlmostEqual(
            float(bb2["clinical_weapon_synergy_norm"]), expected_clinical_weapon
        )
        expected_endemic_resistance = float(
            np.clip(
                float(bb2["A_recurrence_norm"]) * float(bb2["H_specialization_norm"]),
                0.0,
                1.0,
            )
        )
        self.assertAlmostEqual(float(bb2["endemic_resistance_norm"]), expected_endemic_resistance)
        expected_metadata_support = float(
            np.clip(
                (0.26 * float(bb2["H_support_norm"]))
                + (0.20 * float(bb2["amr_support_norm"]))
                + (0.18 * float(bb2["H_external_host_range_support"]))
                + (0.16 * float(bb2["pmlst_presence_norm"]))
                + (0.10 * float(bb2["host_support_observed_flag"]))
                + (0.10 * float(bb2["context_support_observed_flag"])),
                0.0,
                1.0,
            )
        )
        self.assertAlmostEqual(float(bb2["metadata_support_depth_norm"]), expected_metadata_support)
        expected_silent_carrier = float(
            np.clip(0.5 * (1.0 - float(bb2["metadata_support_depth_norm"])), 0.0, 1.0)
        )
        self.assertAlmostEqual(float(bb2["silent_carrier_risk_norm"]), expected_silent_carrier)
        self.assertIn("H_obs_norm", scored.columns)
        self.assertIn("H_support_norm", scored.columns)
        self.assertIn("H_obs_specialization_norm", scored.columns)
        self.assertIn("T_H_obs_synergy_norm", scored.columns)

    def test_monotonic_latent_axes_saturate_with_diminishing_returns(self) -> None:
        backbone_table = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "member_count_train": [1, 2, 3, 4],
                "member_count_total": [1, 2, 3, 4],
                "n_countries_train": [1, 2, 3, 4],
                "n_countries_test": [0, 1, 2, 3],
                "n_new_countries": [0, 1, 2, 3],
                "spread_label": [0, 1, 1, 1],
                "refseq_share_train": [0.0, 0.3, 0.6, 0.9],
                "insd_share_train": [1.0, 0.7, 0.4, 0.1],
                "coherence_score": [0.2, 0.4, 0.7, 0.9],
                "mean_n_replicon_types_train": [1.0, 1.5, 2.5, 4.0],
                "multi_replicon_fraction_train": [0.0, 0.2, 0.6, 1.0],
                "primary_replicon_diversity_train": [0.0, 0.1, 0.3, 0.8],
                "plasmidfinder_support_score": [0.0, 0.3, 0.6, 0.9],
                "plasmidfinder_complexity_score": [0.0, 0.2, 0.5, 0.8],
                "clinical_context_fraction_train": [0.0, 0.2, 0.5, 0.9],
                "ecology_context_diversity_train": [0.0, 0.1, 0.4, 0.85],
            }
        )
        feature_t = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "member_count_train": [1, 2, 3, 4],
                "relaxase_support": [0.0, 0.2, 0.6, 1.0],
                "mpf_support": [0.0, 0.2, 0.6, 1.0],
                "orit_support": [0.0, 0.2, 0.6, 1.0],
                "mobilizable_support": [0.0, 0.2, 0.6, 1.0],
                "support_shrinkage": [0.1, 0.3, 0.6, 0.9],
                "T_raw": [0.0, 0.2, 0.6, 1.0],
                "T_eff": [0.0, 0.18, 0.48, 0.88],
            }
        )
        feature_h = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "H_genus_norm": [0.1, 0.2, 0.3, 0.4],
                "phylo_breadth_score": [0.1, 0.3, 0.6, 0.9],
                "phylo_pairwise_dispersion_score": [0.1, 0.3, 0.6, 0.9],
                "phylo_breadth_augmented_score": [0.1, 0.3, 0.6, 0.9],
                "host_taxon_evenness_score": [0.0, 0.2, 0.5, 0.9],
                "host_support_factor": [0.1, 0.3, 0.6, 1.0],
                "H_external_host_range_support": [0.0, 0.2, 0.6, 1.0],
                "H_external_host_range_score": [0.0, 0.2, 0.6, 1.0],
                "H_raw": [0.1, 0.2, 0.4, 0.6],
                "H_phylogenetic_raw": [0.1, 0.2, 0.4, 0.6],
                "H_augmented_raw": [0.1, 0.2, 0.4, 0.6],
                "H_phylogenetic_augmented_raw": [0.1, 0.2, 0.4, 0.6],
                "H_eff": [0.04, 0.08, 0.16, 0.24],
                "H_phylogenetic_eff": [0.04, 0.08, 0.16, 0.24],
                "H_augmented_eff": [0.04, 0.08, 0.16, 0.24],
                "H_phylogenetic_augmented_eff": [0.04, 0.08, 0.16, 0.24],
                "clinical_context_fraction_train": [0.0, 0.2, 0.5, 0.9],
                "ecology_context_diversity_train": [0.0, 0.1, 0.4, 0.85],
            }
        )
        feature_a = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3", "bb4"],
                "mean_amr_class_count": [1.0, 1.0, 1.0, 1.0],
                "mean_amr_gene_count": [0.0, 1.0, 4.0, 9.0],
                "mean_amr_clinical_threat_score": [0.0, 0.2, 0.6, 1.0],
                "A_consistency": [0.1, 0.3, 0.6, 0.9],
                "A_recurrence": [0.0, 0.2, 0.5, 0.8],
                "amr_support_factor": [0.1, 0.3, 0.6, 0.9],
                "canonical_member_count_train": [1, 2, 3, 4],
            }
        )
        scored = build_scored_backbone_table(
            backbone_table,
            feature_t,
            feature_h,
            feature_a,
            normalization_method="robust_sigmoid",
        )
        ordered = scored.sort_values("amr_burden_latent_norm").reset_index(drop=True)
        self.assertTrue(ordered["amr_burden_saturation_norm"].between(0.0, 1.0).all())
        self.assertTrue(ordered["replicon_multiplicity_saturation_norm"].between(0.0, 1.0).all())
        self.assertTrue(ordered["host_range_saturation_norm"].between(0.0, 1.0).all())
        self.assertTrue(ordered["eco_clinical_context_saturation_norm"].between(0.0, 1.0).all())
        self.assertTrue(ordered["monotonic_latent_priority_index"].between(0.0, 1.0).all())
        self.assertTrue(ordered["amr_burden_saturation_norm"].is_monotonic_increasing)

    def test_build_scored_backbone_table_defaults_to_rank_percentile(self) -> None:
        backbone_table = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [2, 3, 4],
                "member_count_total": [2, 3, 4],
                "n_countries_train": [1, 2, 3],
                "n_countries_test": [0, 1, 2],
                "n_new_countries": [0, 1, 1],
                "spread_label": [0, 1, 1],
                "refseq_share_train": [0.0, 1.0, 1.0],
                "insd_share_train": [1.0, 0.0, 0.0],
                "coherence_score": [0.4, 0.6, 0.8],
                "mean_n_replicon_types_train": [1.0, 1.5, 2.0],
                "multi_replicon_fraction_train": [0.0, 0.5, 1.0],
                "primary_replicon_diversity_train": [0.0, 0.2, 0.4],
                "plasmidfinder_support_score": [0.0, 0.4, 0.8],
                "plasmidfinder_complexity_score": [0.0, 0.3, 0.7],
            }
        )
        feature_t = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [2, 3, 4],
                "relaxase_support": [0.0, 0.5, 1.0],
                "mpf_support": [0.0, 0.5, 1.0],
                "orit_support": [0.0, 0.5, 1.0],
                "mobilizable_support": [0.0, 0.5, 1.0],
                "support_shrinkage": [0.4, 0.5, 0.6],
                "T_raw": [0.1, 0.4, 0.8],
                "T_eff": [0.04, 0.2, 0.48],
            }
        )
        feature_h = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "H_genus_norm": [0.1, 0.2, 0.3],
                "phylo_breadth_score": [0.2, 0.4, 0.6],
                "phylo_pairwise_dispersion_score": [0.1, 0.3, 0.8],
                "phylo_breadth_augmented_score": [0.2, 0.4, 0.8],
                "host_taxon_evenness_score": [0.0, 0.4, 0.8],
                "host_support_factor": [0.4, 0.5, 0.6],
                "H_external_host_range_support": [0.0, 0.5, 1.0],
                "H_external_host_range_score": [0.0, 0.4, 0.9],
                "H_raw": [0.14, 0.28, 0.42],
                "H_phylogenetic_raw": [0.14, 0.28, 0.49],
                "H_augmented_raw": [0.14, 0.28, 0.42],
                "H_phylogenetic_augmented_raw": [0.14, 0.28, 0.49],
                "H_eff": [0.056, 0.14, 0.252],
                "H_phylogenetic_eff": [0.056, 0.14, 0.294],
                "H_augmented_eff": [0.056, 0.14, 0.252],
                "H_phylogenetic_augmented_eff": [0.056, 0.14, 0.294],
            }
        )
        feature_a = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "mean_amr_class_count": [0.0, 1.0, 2.0],
                "mean_amr_gene_count": [0.0, 2.0, 4.0],
                "mean_amr_clinical_threat_score": [9.0, 1.0, 0.0],
                "A_consistency": [0.1, 0.5, 0.9],
                "A_recurrence": [0.0, 0.4, 0.8],
                "amr_support_factor": [0.3, 0.5, 0.8],
                "canonical_member_count_train": [2, 3, 4],
            }
        )

        default_scored = build_scored_backbone_table(
            backbone_table, feature_t, feature_h, feature_a
        )
        explicit_scored = build_scored_backbone_table(
            backbone_table,
            feature_t,
            feature_h,
            feature_a,
            normalization_method="rank_percentile",
        )

        pd.testing.assert_frame_equal(default_scored, explicit_scored)

    def test_recompute_priority_from_reference_preserves_ordered_columns(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [1, 2, 3],
                "n_countries_train": [1, 2, 3],
                "T_eff": [0.1, 0.4, 0.8],
                "H_eff": [0.2, 0.3, 0.9],
                "A_eff": [0.05, 0.5, 0.7],
            }
        )
        rescored = recompute_priority_from_reference(scored, scored.iloc[[0, 2]].copy())
        self.assertEqual(list(rescored["backbone_id"]), ["bb1", "bb2", "bb3"])
        self.assertTrue(rescored["priority_index"].between(0.0, 1.0).all())

    def test_rank_percentile_normalization_preserves_zero_components(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [1, 2, 3],
                "n_countries_train": [1, 2, 3],
                "T_eff": [0.0, 0.2, 0.8],
                "H_eff": [0.0, 0.3, 0.9],
                "A_eff": [0.0, 0.5, 0.7],
            }
        )
        rescored = recompute_priority_from_reference(scored, scored.iloc[[1, 2]].copy())
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "T_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "H_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "A_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "priority_index"].iloc[0]), 0.0
        )

    def test_robust_sigmoid_normalization_preserves_zero_components(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "member_count_train": [1, 2, 3],
                "n_countries_train": [1, 2, 3],
                "T_eff": [0.0, 0.2, 0.8],
                "H_eff": [0.0, 0.3, 0.9],
                "A_eff": [0.0, 0.5, 0.7],
            }
        )
        rescored = recompute_priority_from_reference(
            scored,
            scored.iloc[[1, 2]].copy(),
            normalization_method="robust_sigmoid",
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "T_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "H_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "A_eff_norm"].iloc[0]), 0.0
        )
        self.assertEqual(
            float(rescored.loc[rescored["backbone_id"] == "bb1", "priority_index"].iloc[0]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
