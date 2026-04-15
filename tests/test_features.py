from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.features import compute_feature_h
from plasmid_priority.features.core import (
    _mean_pairwise_host_taxonomy_distance,
    _pairwise_host_taxonomy_distance,
    _subsample_signatures_for_pairwise_distance,
)


class FeatureTests(unittest.TestCase):
    def test_pairwise_host_taxonomy_distance_caps_large_signature_sets_deterministically(
        self,
    ) -> None:
        signatures = [
            ("1224", "1236", "91347", "543", str(index // 2), str(index)) for index in range(60)
        ]
        sampled = _subsample_signatures_for_pairwise_distance(signatures)
        self.assertEqual(len(sampled), 50)
        self.assertEqual(sampled[0], signatures[0])
        self.assertEqual(sampled[-1], signatures[-1])

        full_distances = [
            _pairwise_host_taxonomy_distance(left, right)
            for i, left in enumerate(signatures)
            for right in signatures[i + 1 :]
        ]
        sampled_distance = _mean_pairwise_host_taxonomy_distance(signatures)
        self.assertTrue(np.isfinite(sampled_distance))
        self.assertLess(abs(sampled_distance - float(np.mean(full_distances))), 0.1)

    def test_compute_feature_h_rewards_broader_supported_host_diversity(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb_low"] * 4 + ["bb_high"] * 4,
                "resolved_year": [2014] * 8,
                "genus": [
                    "Escherichia",
                    "Escherichia",
                    "Escherichia",
                    "Escherichia",
                    "Escherichia",
                    "Klebsiella",
                    "Salmonella",
                    "Enterobacter",
                ],
                "TAXONOMY_family": [
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                    "Enterobacteriaceae",
                ],
                "TAXONOMY_order": ["Enterobacterales"] * 8,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 8,
                "TAXONOMY_phylum": ["Proteobacteria"] * 8,
            }
        )
        feature_h = compute_feature_h(records).set_index("backbone_id")
        self.assertGreater(
            float(feature_h.loc["bb_high", "H_eff"]), float(feature_h.loc["bb_low", "H_eff"])
        )
        self.assertGreater(
            float(feature_h.loc["bb_high", "H_genus_norm"]),
            float(feature_h.loc["bb_low", "H_genus_norm"]),
        )
        self.assertGreater(
            float(feature_h.loc["bb_high", "host_taxon_evenness_score"]),
            float(feature_h.loc["bb_low", "host_taxon_evenness_score"]),
        )

    def test_compute_feature_h_does_not_give_single_observation_maximum_raw_diversity(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb_single", "bb_broad", "bb_broad", "bb_broad"],
                "resolved_year": [2014, 2014, 2014, 2014],
                "genus": ["Escherichia", "Escherichia", "Klebsiella", "Salmonella"],
                "TAXONOMY_family": ["Enterobacteriaceae"] * 4,
                "TAXONOMY_order": ["Enterobacterales"] * 4,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 4,
                "TAXONOMY_phylum": ["Proteobacteria"] * 4,
            }
        )
        feature_h = compute_feature_h(records).set_index("backbone_id")
        self.assertLess(float(feature_h.loc["bb_single", "H_raw"]), 1.0)
        self.assertGreater(
            float(feature_h.loc["bb_broad", "H_raw"]), float(feature_h.loc["bb_single", "H_raw"])
        )

    def test_compute_feature_h_emits_external_host_range_columns_when_present(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb2", "bb2"],
                "resolved_year": [2014, 2014, 2014, 2014],
                "genus": ["Escherichia", "Klebsiella", "Escherichia", "Escherichia"],
                "TAXONOMY_family": ["Enterobacteriaceae"] * 4,
                "TAXONOMY_order": ["Enterobacterales"] * 4,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 4,
                "TAXONOMY_phylum": ["Proteobacteria"] * 4,
                "predicted_host_range_overall_rank": ["phylum", "phylum", "genus", "genus"],
                "reported_host_range_lit_rank": ["order", "order", "", ""],
            }
        )
        feature_h = compute_feature_h(records).set_index("backbone_id")
        self.assertIn("H_external_host_range_score", feature_h.columns)
        self.assertIn("H_obs", feature_h.columns)
        self.assertIn("H_support", feature_h.columns)
        self.assertGreater(
            float(feature_h.loc["bb1", "H_external_host_range_score"]),
            float(feature_h.loc["bb2", "H_external_host_range_score"]),
        )
        self.assertGreater(float(feature_h.loc["bb1", "H_obs"]), 0.0)
        self.assertGreater(float(feature_h.loc["bb1", "H_support"]), 0.0)
        self.assertGreater(float(feature_h.loc["bb1", "H_external_host_range_support"]), 0.0)
        self.assertGreater(float(feature_h.loc["bb1", "H_augmented_raw"]), 0.0)

    def test_compute_feature_h_uses_taxonomy_ids_to_refine_phylogenetic_breadth(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": [
                    "bb_same_species",
                    "bb_same_species",
                    "bb_species_split",
                    "bb_species_split",
                ],
                "resolved_year": [2014, 2014, 2014, 2014],
                "taxonomy_uid": [562, 562, 562, 61645],
                "genus": ["Escherichia"] * 4,
                "species": [
                    "Escherichia_coli",
                    "Escherichia_coli",
                    "Escherichia_coli",
                    "Escherichia_albertii",
                ],
                "TAXONOMY_phylum": ["Pseudomonadota"] * 4,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 4,
                "TAXONOMY_order": ["Enterobacterales"] * 4,
                "TAXONOMY_family": ["Enterobacteriaceae"] * 4,
                "TAXONOMY_phylum_id": [1224] * 4,
                "TAXONOMY_class_id": [1236] * 4,
                "TAXONOMY_order_id": [91347] * 4,
                "TAXONOMY_family_id": [543] * 4,
                "TAXONOMY_genus_id": [561] * 4,
                "TAXONOMY_species_id": [562, 562, 562, 61645],
            }
        )
        feature_h = compute_feature_h(records).set_index("backbone_id")
        self.assertEqual(
            float(feature_h.loc["bb_same_species", "H_raw"]),
            float(feature_h.loc["bb_species_split", "H_raw"]),
        )
        self.assertGreater(
            float(feature_h.loc["bb_species_split", "phylo_pairwise_dispersion_score"]),
            float(feature_h.loc["bb_same_species", "phylo_pairwise_dispersion_score"]),
        )
        self.assertGreater(
            float(feature_h.loc["bb_species_split", "H_phylogenetic_raw"]),
            float(feature_h.loc["bb_same_species", "H_phylogenetic_raw"]),
        )

    def test_compute_feature_h_accepts_evenness_bias_power_override(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb_biased"] * 6 + ["bb_balanced"] * 6,
                "resolved_year": [2014] * 12,
                "genus": [
                    "Escherichia",
                    "Escherichia",
                    "Escherichia",
                    "Escherichia",
                    "Klebsiella",
                    "Klebsiella",
                    "Escherichia",
                    "Klebsiella",
                    "Salmonella",
                    "Enterobacter",
                    "Citrobacter",
                    "Serratia",
                ],
                "TAXONOMY_family": ["Enterobacteriaceae"] * 12,
                "TAXONOMY_order": ["Enterobacterales"] * 12,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 12,
                "TAXONOMY_phylum": ["Proteobacteria"] * 12,
            }
        )
        low_power = compute_feature_h(records, host_evenness_bias_power=0.3).set_index(
            "backbone_id"
        )
        high_power = compute_feature_h(records, host_evenness_bias_power=1.0).set_index(
            "backbone_id"
        )
        self.assertGreater(
            float(low_power.loc["bb_biased", "H_raw"]), float(high_power.loc["bb_biased", "H_raw"])
        )
        self.assertAlmostEqual(
            float(low_power.loc["bb_balanced", "host_taxon_evenness_score"]),
            float(high_power.loc["bb_balanced", "host_taxon_evenness_score"]),
        )


class FeatureTTests(unittest.TestCase):
    def test_compute_feature_t_basic_mobility_scoring(self) -> None:
        from plasmid_priority.features import compute_feature_t

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb_conj"] * 3 + ["bb_none"] * 3,
                "canonical_id": [f"c{i}" for i in range(6)],
                "has_relaxase": [True, True, True, False, False, False],
                "has_mpf": [True, True, True, False, False, False],
                "has_orit": [True, True, False, False, False, False],
                "is_mobilizable": [True, True, True, False, False, False],
                "amr_gene_count": [0] * 6,
                "amr_class_count": [0] * 6,
                "amr_hit_count": [0] * 6,
                "amr_drug_classes": [""] * 6,
            }
        )
        feature_t = compute_feature_t(training_canonical).set_index("backbone_id")
        self.assertGreater(float(feature_t.loc["bb_conj", "T_raw"]), 0.0)
        self.assertEqual(float(feature_t.loc["bb_none", "T_raw"]), 0.0)
        self.assertGreater(
            float(feature_t.loc["bb_conj", "T_eff"]), float(feature_t.loc["bb_none", "T_eff"])
        )

    def test_compute_feature_t_support_shrinkage_increases_with_members(self) -> None:
        from plasmid_priority.features import compute_feature_t

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb_small"] * 1 + ["bb_large"] * 10,
                "canonical_id": [f"c{i}" for i in range(11)],
                "has_relaxase": [True] * 11,
                "has_mpf": [True] * 11,
                "has_orit": [True] * 11,
                "is_mobilizable": [True] * 11,
                "amr_gene_count": [0] * 11,
                "amr_class_count": [0] * 11,
                "amr_hit_count": [0] * 11,
                "amr_drug_classes": [""] * 11,
            }
        )
        feature_t = compute_feature_t(training_canonical).set_index("backbone_id")
        self.assertGreater(
            float(feature_t.loc["bb_large", "support_shrinkage"]),
            float(feature_t.loc["bb_small", "support_shrinkage"]),
        )

    def test_compute_feature_t_single_member_backbone(self) -> None:
        from plasmid_priority.features import compute_feature_t

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb_solo"],
                "canonical_id": ["c0"],
                "has_relaxase": [True],
                "has_mpf": [False],
                "has_orit": [True],
                "is_mobilizable": [True],
                "amr_gene_count": [0],
                "amr_class_count": [0],
                "amr_hit_count": [0],
                "amr_drug_classes": [""],
            }
        )
        feature_t = compute_feature_t(training_canonical).set_index("backbone_id")
        self.assertEqual(int(feature_t.loc["bb_solo", "member_count_train"]), 1)
        self.assertGreater(float(feature_t.loc["bb_solo", "T_raw"]), 0.0)
        self.assertLess(
            float(feature_t.loc["bb_solo", "T_eff"]), float(feature_t.loc["bb_solo", "T_raw"])
        )


class FeatureATests(unittest.TestCase):
    def test_compute_feature_a_basic_amr_scoring(self) -> None:
        from plasmid_priority.features import compute_feature_a

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb_amr"] * 3 + ["bb_clean"] * 3,
                "canonical_id": [f"c{i}" for i in range(6)],
                "has_relaxase": [False] * 6,
                "has_mpf": [False] * 6,
                "has_orit": [False] * 6,
                "is_mobilizable": [False] * 6,
                "amr_gene_count": [3, 2, 4, 0, 0, 0],
                "amr_class_count": [2, 1, 3, 0, 0, 0],
                "amr_hit_count": [5, 3, 6, 0, 0, 0],
                "amr_gene_symbols": [
                    "blaKPC-2,aac(6')-Ib,tetA",
                    "blaKPC-2",
                    "blaNDM-1,qnrS1,tetA,sul1",
                    "",
                    "",
                    "",
                ],
                "amr_drug_classes": [
                    "CARBAPENEM,AMINOGLYCOSIDE",
                    "BETA-LACTAM",
                    "CARBAPENEM,POLYMYXIN,TETRACYCLINE,FLUOROQUINOLONE",
                    "",
                    "",
                    "",
                ],
            }
        )
        feature_a = compute_feature_a(training_canonical).set_index("backbone_id")
        self.assertGreater(float(feature_a.loc["bb_amr", "mean_amr_class_count"]), 0.0)
        self.assertEqual(float(feature_a.loc["bb_clean", "mean_amr_class_count"]), 0.0)
        self.assertGreater(float(feature_a.loc["bb_amr", "A_consistency"]), 0.0)
        self.assertGreater(float(feature_a.loc["bb_amr", "A_recurrence"]), 0.0)
        self.assertGreater(float(feature_a.loc["bb_amr", "mdr_proxy_fraction"]), 0.0)
        self.assertGreater(
            float(feature_a.loc["bb_amr", "mean_last_resort_convergence_score"]), 0.0
        )
        self.assertGreater(
            float(feature_a.loc["bb_amr", "mean_amr_mechanism_diversity_proxy"]), 0.0
        )
        self.assertEqual(float(feature_a.loc["bb_clean", "A_recurrence"]), 0.0)

    def test_compute_feature_a_all_empty_amr(self) -> None:
        from plasmid_priority.features import compute_feature_a

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1"],
                "canonical_id": ["c0", "c1", "c2"],
                "has_relaxase": [False] * 3,
                "has_mpf": [False] * 3,
                "has_orit": [False] * 3,
                "is_mobilizable": [False] * 3,
                "amr_gene_count": [0, 0, 0],
                "amr_class_count": [0, 0, 0],
                "amr_hit_count": [0, 0, 0],
                "amr_gene_symbols": ["", "", ""],
                "amr_drug_classes": ["", "", ""],
            }
        )
        feature_a = compute_feature_a(training_canonical).set_index("backbone_id")
        self.assertEqual(float(feature_a.loc["bb1", "mean_amr_class_count"]), 0.0)
        self.assertEqual(float(feature_a.loc["bb1", "A_consistency"]), 0.0)
        self.assertEqual(float(feature_a.loc["bb1", "mdr_proxy_fraction"]), 0.0)
        self.assertEqual(float(feature_a.loc["bb1", "amr_support_factor"]), 0.0)

    def test_compute_feature_a_single_member_with_amr(self) -> None:
        from plasmid_priority.features import compute_feature_a

        training_canonical = pd.DataFrame(
            {
                "backbone_id": ["bb_solo"],
                "canonical_id": ["c0"],
                "has_relaxase": [False],
                "has_mpf": [False],
                "has_orit": [False],
                "is_mobilizable": [False],
                "amr_gene_count": [5],
                "amr_class_count": [3],
                "amr_hit_count": [7],
                "amr_gene_symbols": ["blaKPC-2,aac(6')-Ib,tetA"],
                "amr_drug_classes": ["CARBAPENEM,AMINOGLYCOSIDE,TETRACYCLINE"],
            }
        )
        feature_a = compute_feature_a(training_canonical).set_index("backbone_id")
        self.assertEqual(int(feature_a.loc["bb_solo", "canonical_member_count_train"]), 1)
        self.assertGreater(float(feature_a.loc["bb_solo", "mean_amr_class_count"]), 0.0)
        self.assertGreaterEqual(float(feature_a.loc["bb_solo", "mdr_proxy_fraction"]), 1.0)


if __name__ == "__main__":
    unittest.main()
