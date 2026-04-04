from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.features import (
    build_training_canonical_table,
    compute_feature_a,
    compute_feature_h,
    compute_feature_t,
)


class FeaturePipelineIntegrationTests(unittest.TestCase):
    def test_build_training_canonical_table_excludes_post_split_rows(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["seq1", "seq2", "seq3", "seq4"],
                "canonical_id": ["c1", "c2", "c3", "c4"],
                "backbone_id": ["bb1", "bb1", "bb1", "bb2"],
                "resolved_year": [2014, 2015, 2018, 2014],
                "has_relaxase": [True, True, False, False],
                "has_mpf": [True, False, False, False],
                "has_orit": [True, False, False, False],
                "is_mobilizable": [True, False, False, False],
            }
        )
        amr_consensus = pd.DataFrame(
            {
                "sequence_accession": ["seq1", "seq2", "seq3", "seq4"],
                "amr_gene_count": [2, 1, 9, 0],
                "amr_class_count": [1, 1, 4, 0],
                "amr_hit_count": [2, 1, 9, 0],
                "amr_drug_classes": [
                    "BETA-LACTAM",
                    "BETA-LACTAM",
                    "AMINOGLYCOSIDE,TETRACYCLINE",
                    "",
                ],
            }
        )

        training_canonical = build_training_canonical_table(records, amr_consensus, split_year=2015)
        self.assertEqual(set(training_canonical["canonical_id"]), {"c1", "c2", "c4"})
        self.assertNotIn("c3", set(training_canonical["canonical_id"]))

        feature_t = compute_feature_t(training_canonical).set_index("backbone_id")
        feature_a = compute_feature_a(training_canonical).set_index("backbone_id")
        self.assertEqual(int(feature_t.loc["bb1", "member_count_train"]), 2)
        self.assertLess(float(feature_a.loc["bb1", "mean_amr_gene_count"]), 9.0)

    def test_build_training_canonical_table_defaults_missing_amr_columns(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["seq1", "seq2", "seq3", "seq4"],
                "canonical_id": ["c1", "c2", "c3", "c4"],
                "backbone_id": ["bb1", "bb1", "bb1", "bb2"],
                "resolved_year": [2014, 2015, 2018, 2014],
                "has_relaxase": [True, True, False, False],
                "has_mpf": [True, False, False, False],
                "has_orit": [True, False, False, False],
                "is_mobilizable": [True, False, False, False],
            }
        )
        amr_consensus = pd.DataFrame(
            {
                "sequence_accession": ["seq1", "seq2", "seq3", "seq4"],
            }
        )

        training_canonical = build_training_canonical_table(records, amr_consensus, split_year=2015)
        self.assertEqual(set(training_canonical["canonical_id"]), {"c1", "c2", "c4"})
        self.assertTrue((training_canonical["amr_gene_count"] == 0).all())
        self.assertTrue((training_canonical["amr_class_count"] == 0).all())
        self.assertTrue((training_canonical["amr_hit_count"] == 0).all())
        self.assertTrue((training_canonical["amr_gene_symbols"] == "").all())
        self.assertTrue((training_canonical["amr_drug_classes"] == "").all())

    def test_compute_feature_h_uses_only_training_period_rows(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb2", "bb2"],
                "resolved_year": [2014, 2015, 2019, 2014, 2018],
                "genus": ["Escherichia", "Escherichia", "Salmonella", "Klebsiella", "Enterobacter"],
                "TAXONOMY_family": ["Enterobacteriaceae"] * 5,
                "TAXONOMY_order": ["Enterobacterales"] * 5,
                "TAXONOMY_class": ["Gammaproteobacteria"] * 5,
                "TAXONOMY_phylum": ["Pseudomonadota"] * 5,
            }
        )

        train_only = records.loc[records["resolved_year"] <= 2015].copy()
        feature_h_train_only = compute_feature_h(train_only, split_year=2015).set_index(
            "backbone_id"
        )
        feature_h_full = compute_feature_h(records, split_year=2015).set_index("backbone_id")
        feature_h_with_leak = compute_feature_h(records, split_year=2020).set_index("backbone_id")

        self.assertEqual(
            float(feature_h_train_only.loc["bb1", "H_raw"]),
            float(feature_h_full.loc["bb1", "H_raw"]),
        )
        self.assertGreater(
            float(feature_h_with_leak.loc["bb1", "H_raw"]),
            float(feature_h_full.loc["bb1", "H_raw"]),
        )
        self.assertEqual(
            int(feature_h_full.loc["bb1", "host_observation_count"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
