from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd


from plasmid_priority.scoring import build_scored_backbone_table, recompute_priority_from_reference


class ScoringTests(unittest.TestCase):
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
            "A_eff_norm",
            "T_raw_norm",
            "A_raw_norm",
            "support_shrinkage_norm",
            "amr_support_norm",
            "priority_index",
            "bio_priority_index",
            "evidence_support_index",
            "mean_n_replicon_types_norm",
            "replicon_architecture_norm",
            "host_phylogenetic_dispersion_norm",
            "host_taxon_evenness_norm",
            "H_phylogenetic_norm",
            "H_phylogenetic_specialization_norm",
            "A_recurrence_norm",
            "clinical_context_sparse_penalty_norm",
            "external_t_synergy_norm",
            "pmlst_presence_norm",
            "metadata_support_depth_norm",
            "metadata_missingness_burden",
            "context_support_guard_norm",
        ]:
            self.assertTrue(scored[column].between(0.0, 1.0).all())

        # Core A-content should follow the documented burden + richness design,
        # not the descriptive WHO-derived clinical-threat column.
        bb2 = scored.loc[scored["backbone_id"] == "bb2"].iloc[0]
        self.assertAlmostEqual(
            float(bb2["A_content_raw"]),
            0.5 * float(bb2["amr_class_richness_norm"]) + 0.5 * float(bb2["amr_gene_burden_norm"]),
        )

    def test_recompute_priority_from_reference_preserves_ordered_columns(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
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
                "T_eff": [0.0, 0.2, 0.8],
                "H_eff": [0.0, 0.3, 0.9],
                "A_eff": [0.0, 0.5, 0.7],
            }
        )
        rescored = recompute_priority_from_reference(scored, scored.iloc[[1, 2]].copy())
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "T_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "H_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "A_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "priority_index"].iloc[0]), 0.0)

    def test_robust_sigmoid_normalization_preserves_zero_components(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
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
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "T_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "H_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "A_eff_norm"].iloc[0]), 0.0)
        self.assertEqual(float(rescored.loc[rescored["backbone_id"] == "bb1", "priority_index"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
