from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd


from plasmid_priority.backbone import assign_backbone_ids_training_only, fallback_backbone_key
from plasmid_priority.config import DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD
from plasmid_priority.features import build_backbone_table


class BackboneTests(unittest.TestCase):
    def test_assign_backbone_ids_training_only_marks_unseen_future_cluster(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["train_a", "test_seen", "test_unseen"],
                "resolved_year": [2014, 2017, 2017],
                "primary_cluster_id": ["AA001", "AA001", "BB999"],
                "predicted_mobility": ["mobilizable"] * 3,
                "mpf_type": ["MPFT"] * 3,
                "primary_replicon": ["IncFIB"] * 3,
            }
        )
        assigned = assign_backbone_ids_training_only(records, split_year=2015)
        lookup = assigned.set_index("sequence_accession")["backbone_id"].to_dict()
        seen_lookup = assigned.set_index("sequence_accession")["backbone_seen_in_training"].to_dict()
        self.assertEqual(lookup["train_a"], "AA001")
        self.assertEqual(lookup["test_seen"], "AA001")
        self.assertTrue(lookup["test_unseen"].startswith("UNSEEN::"))
        self.assertTrue(bool(seen_lookup["test_seen"]))
        self.assertFalse(bool(seen_lookup["test_unseen"]))

    def test_fallback_backbone_key_uses_expected_parts(self) -> None:
        key = fallback_backbone_key(
            pd.Series(
                {
                    "predicted_mobility": "mobilizable",
                    "mpf_type": "MPFT",
                    "primary_replicon": "IncN",
                }
            )
        )
        self.assertEqual(key, "OP::mobilizable::MPFT::IncN")

    def test_build_backbone_table_respects_test_year_end(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1"],
                "canonical_id": ["c1", "c2", "c3"],
                "resolved_year": [2014, 2016, 2020],
                "country": ["TR", "DE", "US"],
                "record_origin": ["insd", "insd", "refseq"],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})
        table = build_backbone_table(records, coherence, split_year=2015, test_year_end=2017)
        row = table.iloc[0]
        self.assertEqual(int(row["n_countries_test"]), 1)
        self.assertEqual(int(row["test_year_end"]), 2017)

    def test_build_backbone_table_default_threshold_follows_project_config(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb1"],
                "canonical_id": ["c1", "c2", "c3", "c4"],
                "resolved_year": [2014, 2015, 2016, 2017],
                "country": ["TR", "TR", "DE", "US"],
                "record_origin": ["insd", "insd", "refseq", "refseq"],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})
        table = build_backbone_table(records, coherence, split_year=2015, test_year_end=2023)
        row = table.iloc[0]
        expected_label = int(2 >= DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD)
        self.assertEqual(int(row["spread_label"]), expected_label)

    def test_build_backbone_table_emits_purity_and_assignment_columns(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1"],
                "canonical_id": ["c1", "c2", "c3"],
                "resolved_year": [2014, 2015, 2017],
                "country": ["TR", "TR", "DE"],
                "record_origin": ["insd", "insd", "refseq"],
                "genus": ["Escherichia", "Escherichia", "Escherichia"],
                "TAXONOMY_family": ["Enterobacteriaceae"] * 3,
                "predicted_mobility": ["mobilizable"] * 3,
                "primary_replicon": ["IncFIB"] * 3,
                "backbone_assignment_rule": ["primary_cluster_id"] * 3,
                "mash_neighbor_distance": [0.1, 0.2, 0.9],
                "PMLST_scheme": ["schemeA", "schemeA", ""],
                "PMLST_sequence_type": ["ST1", "ST1", ""],
                "PMLST_alleles": ["a1,b1", "a1,b1", ""],
                "associated_pmid(s)": ["12345,67890", "12345", ""],
                "BIOSAMPLE_package": ["clinical isolate", "clinical isolate", "environmental sample"],
                "BIOSAMPLE_title": ["patient blood isolate", "patient urine isolate", "wastewater isolate"],
                "BIOSAMPLE_pathogenicity": ["pathogenic", "", ""],
                "ECOSYSTEM_tags": ["clinical", "clinical", "wastewater"],
                "DISEASE_tags": ["sepsis", "uti", ""],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})
        table = build_backbone_table(records, coherence, split_year=2015, test_year_end=2023)
        row = table.iloc[0]
        self.assertIn("backbone_purity_score", table.columns)
        self.assertIn("assignment_confidence_score", table.columns)
        self.assertIn("mean_n_replicon_types_train", table.columns)
        self.assertIn("multi_replicon_fraction_train", table.columns)
        self.assertIn("primary_replicon_diversity_train", table.columns)
        self.assertIn("pmlst_coherence_score", table.columns)
        self.assertIn("clinical_context_fraction_train", table.columns)
        self.assertIn("ecology_context_diversity_train", table.columns)
        self.assertAlmostEqual(float(row["assignment_primary_fraction"]), 1.0)
        self.assertGreaterEqual(float(row["backbone_purity_score"]), 0.9)
        self.assertGreater(float(row["pmlst_coherence_score"]), 0.0)
        self.assertGreater(float(row["clinical_context_fraction_train"]), 0.0)
        self.assertGreater(float(row["mean_pmid_count_train"]), 0.0)


if __name__ == "__main__":
    unittest.main()
