from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.backbone import (
    assign_backbone_ids,
    assign_backbone_ids_training_only,
    fallback_backbone_key,
)
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
        seen_lookup = assigned.set_index("sequence_accession")[
            "backbone_seen_in_training"
        ].to_dict()
        self.assertEqual(lookup["train_a"], "AA001")
        self.assertEqual(lookup["test_seen"], "AA001")
        self.assertTrue(lookup["test_unseen"].startswith("UNSEEN::"))
        self.assertTrue(bool(seen_lookup["test_seen"]))
        self.assertFalse(bool(seen_lookup["test_unseen"]))
        self.assertEqual(
            set(assigned["backbone_assignment_mode"].astype(str)),
            {"training_only"},
        )

    def test_assign_backbone_ids_all_records_marks_assignment_mode(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["a", "b"],
                "primary_cluster_id": ["AA001", ""],
                "predicted_mobility": ["mobilizable", "mobilizable"],
                "mpf_type": ["MPFT", "MPFT"],
                "primary_replicon": ["IncFIB", "IncFIB"],
            }
        )
        assigned = assign_backbone_ids(records)
        self.assertEqual(set(assigned["backbone_assignment_mode"].astype(str)), {"all_records"})

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
        table = build_backbone_table(
            records,
            coherence,
            split_year=2015,
            test_year_end=2017,
            backbone_assignment_mode="training_only",
        )
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
        table = build_backbone_table(
            records,
            coherence,
            split_year=2015,
            test_year_end=2023,
            backbone_assignment_mode="training_only",
        )
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
                "plasmidfinder_hit_count": [2, 1, 0],
                "plasmidfinder_type_count": [2, 1, 0],
                "plasmidfinder_dominant_type": ["IncFIB(K)", "IncFIB(K)", ""],
                "plasmidfinder_max_identity": [99.0, 97.0, 0.0],
                "plasmidfinder_mean_coverage": [100.0, 95.0, 0.0],
                "associated_pmid(s)": ["12345,67890", "12345", ""],
                "BIOSAMPLE_package": [
                    "clinical isolate",
                    "clinical isolate",
                    "environmental sample",
                ],
                "BIOSAMPLE_title": [
                    "patient blood isolate",
                    "patient urine isolate",
                    "wastewater isolate",
                ],
                "BIOSAMPLE_pathogenicity": ["pathogenic", "", ""],
                "ECOSYSTEM_tags": ["clinical", "clinical", "wastewater"],
                "DISEASE_tags": ["sepsis", "uti", ""],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})
        table = build_backbone_table(
            records,
            coherence,
            split_year=2015,
            test_year_end=2023,
            backbone_assignment_mode="training_only",
        )
        row = table.iloc[0]
        self.assertIn("backbone_purity_score", table.columns)
        self.assertIn("backbone_assignment_mode", table.columns)
        self.assertIn("max_resolved_year_train", table.columns)
        self.assertIn("min_resolved_year_test", table.columns)
        self.assertIn("training_only_future_unseen_backbone_flag", table.columns)
        self.assertIn("assignment_confidence_score", table.columns)
        self.assertIn("mean_n_replicon_types_train", table.columns)
        self.assertIn("multi_replicon_fraction_train", table.columns)
        self.assertIn("primary_replicon_diversity_train", table.columns)
        self.assertIn("plasmidfinder_support_score", table.columns)
        self.assertIn("plasmidfinder_complexity_score", table.columns)
        self.assertIn("pmlst_coherence_score", table.columns)
        self.assertIn("clinical_context_fraction_train", table.columns)
        self.assertIn("ecology_context_diversity_train", table.columns)
        self.assertAlmostEqual(float(row["assignment_primary_fraction"]), 1.0)
        self.assertGreaterEqual(float(row["backbone_purity_score"]), 0.9)
        self.assertGreater(float(row["plasmidfinder_support_score"]), 0.0)
        self.assertGreater(float(row["plasmidfinder_complexity_score"]), 0.0)
        self.assertGreater(float(row["pmlst_coherence_score"]), 0.0)
        self.assertGreater(float(row["clinical_context_fraction_train"]), 0.0)
        self.assertGreater(float(row["mean_pmid_count_train"]), 0.0)

    def test_build_backbone_table_propagates_training_only_metadata(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "UNSEEN::bb2"],
                "canonical_id": ["c1", "c2", "c3"],
                "resolved_year": [2014, 2017, 2018],
                "country": ["TR", "DE", "US"],
                "record_origin": ["insd", "refseq", "refseq"],
                "backbone_assignment_mode": ["training_only", "training_only", "training_only"],
                "backbone_seen_in_training": [True, True, False],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1", "UNSEEN::bb2"], "coherence_score": [0.8, 0.1]})
        table = build_backbone_table(
            records,
            coherence,
            split_year=2015,
            test_year_end=2023,
            backbone_assignment_mode="training_only",
        )
        bb1 = table.set_index("backbone_id").loc["bb1"]
        bb2 = table.set_index("backbone_id").loc["UNSEEN::bb2"]
        self.assertEqual(str(bb1["backbone_assignment_mode"]), "training_only")
        self.assertEqual(int(bb1["max_resolved_year_train"]), 2014)
        self.assertEqual(int(bb1["min_resolved_year_test"]), 2017)
        self.assertFalse(bool(bb1["training_only_future_unseen_backbone_flag"]))
        self.assertTrue(bool(bb2["training_only_future_unseen_backbone_flag"]))
        self.assertTrue(pd.isna(bb2["spread_label"]))

    def test_build_backbone_table_emits_event_timing_and_macro_region_outcomes(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb1"],
                "canonical_id": ["c1", "c2", "c3", "c4"],
                "resolved_year": [2014, 2016, 2017, 2019],
                "country": ["USA", "Germany", "Japan", "Brazil"],
                "record_origin": ["insd", "refseq", "refseq", "refseq"],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})

        table = build_backbone_table(
            records,
            coherence,
            split_year=2015,
            test_year_end=2023,
            backbone_assignment_mode="training_only",
        )
        row = table.iloc[0]

        self.assertIn("time_to_first_new_country_years", table.columns)
        self.assertIn("time_to_third_new_country_years", table.columns)
        self.assertIn("event_within_3y_label", table.columns)
        self.assertIn("three_countries_within_5y_label", table.columns)
        self.assertIn("spread_severity_bin", table.columns)
        self.assertIn("macro_region_jump_label", table.columns)
        self.assertEqual(int(row["n_new_countries_recomputed"]), 3)
        self.assertEqual(float(row["time_to_first_new_country_years"]), 1.0)
        self.assertEqual(float(row["time_to_third_new_country_years"]), 4.0)
        self.assertEqual(int(row["event_within_3y_label"]), 1)
        self.assertEqual(int(row["three_countries_within_3y_label"]), 0)
        self.assertEqual(int(row["three_countries_within_5y_label"]), 1)
        self.assertEqual(int(row["spread_severity_bin"]), 2)
        self.assertEqual(int(row["n_new_macro_regions"]), 3)
        self.assertEqual(int(row["macro_region_jump_label"]), 1)

    def test_build_backbone_table_rejects_invalid_assignment_mode(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "canonical_id": ["c1"],
                "resolved_year": [2014],
                "country": ["TR"],
                "record_origin": ["insd"],
            }
        )
        coherence = pd.DataFrame({"backbone_id": ["bb1"], "coherence_score": [0.8]})
        with self.assertRaisesRegex(ValueError, "backbone_assignment_mode"):
            build_backbone_table(
                records,
                coherence,
                split_year=2015,
                test_year_end=2023,
                backbone_assignment_mode="something_else",
            )


if __name__ == "__main__":
    unittest.main()
