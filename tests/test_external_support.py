from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
import unittest

import pandas as pd


from plasmid_priority.reporting import (
    build_priority_backbone_support_frame,
    build_who_mia_support,
    build_who_mia_reference_catalog,
    build_card_support,
    build_mobsuite_support,
    build_pathogen_strata_group_summary,
)


class ExternalSupportTests(unittest.TestCase):
    def test_build_priority_backbone_support_frame_can_follow_primary_model_scores(self) -> None:
        scored = pd.DataFrame(
            [
                {"backbone_id": "bb_high", "member_count_train": 4, "spread_label": 1, "priority_index": 0.1, "primary_model_oof_prediction": 0.95},
                {"backbone_id": "bb_low", "member_count_train": 4, "spread_label": 0, "priority_index": 0.9, "primary_model_oof_prediction": 0.05},
                {"backbone_id": "bb_unevaluable", "member_count_train": 4, "spread_label": pd.NA, "priority_index": 0.99, "primary_model_oof_prediction": pd.NA},
            ]
        )
        backbones = pd.DataFrame(
            [
                {"backbone_id": "bb_high", "sequence_accession": "acc1", "species": "Escherichia_coli", "genus": "Escherichia", "primary_replicon": "IncFIB", "replicon_types": "IncFIB"},
                {"backbone_id": "bb_low", "sequence_accession": "acc2", "species": "Klebsiella_pneumoniae", "genus": "Klebsiella", "primary_replicon": "IncN", "replicon_types": "IncN"},
                {"backbone_id": "bb_unevaluable", "sequence_accession": "acc3", "species": "Escherichia_coli", "genus": "Escherichia", "primary_replicon": "IncA", "replicon_types": "IncA"},
            ]
        )
        amr = pd.DataFrame(
            [
                {"sequence_accession": "acc1", "amr_gene_symbols": "blaTEM-1", "amr_drug_classes": "BETA-LACTAM", "amr_gene_count": 1, "amr_class_count": 1},
                {"sequence_accession": "acc2", "amr_gene_symbols": "", "amr_drug_classes": "", "amr_gene_count": 0, "amr_class_count": 0},
                {"sequence_accession": "acc3", "amr_gene_symbols": "qnrS1", "amr_drug_classes": "FLUOROQUINOLONE", "amr_gene_count": 1, "amr_class_count": 1},
            ]
        )

        frame = build_priority_backbone_support_frame(
            scored,
            backbones,
            amr,
            n_per_group=1,
            score_column="primary_model_oof_prediction",
            eligible_only=True,
        )
        self.assertEqual(set(frame["backbone_id"]), {"bb_high", "bb_low"})
        self.assertEqual(float(frame.loc[frame["backbone_id"] == "bb_high", "selection_score"].iloc[0]), 0.95)
        self.assertEqual(str(frame.loc[frame["backbone_id"] == "bb_high", "selection_score_column"].iloc[0]), "primary_model_oof_prediction")

    def test_build_card_support_matches_gene_symbols(self) -> None:
        priority_backbones = pd.DataFrame(
            [
                {
                    "backbone_id": "bb_high",
                    "priority_group": "high",
                    "priority_index": 0.9,
                    "dominant_species": "Escherichia_coli",
                    "dominant_genus": "Escherichia",
                    "primary_replicon": "IncFIB",
                    "amr_gene_symbols": "blaTEM-1,sul2",
                    "amr_gene_count": 2,
                },
                {
                    "backbone_id": "bb_low",
                    "priority_group": "low",
                    "priority_index": 0.1,
                    "dominant_species": "Klebsiella_pneumoniae",
                    "dominant_genus": "Klebsiella",
                    "primary_replicon": "IncN",
                    "amr_gene_symbols": "unknownGene",
                    "amr_gene_count": 1,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "card-data.tar.bz2"
            content = "\n".join(
                [
                    "CARD Short Name\tARO Name\tAMR Gene Family\tDrug Class\tResistance Mechanism",
                    "blaTEM-1\tblaTEM beta-lactamase\tTEM family\tbeta-lactam antibiotic\tantibiotic inactivation",
                    "sul2\tsul2\tSulfonamide-resistant dihydropteroate synthase\tsulfonamide antibiotic\tantibiotic target replacement",
                ]
            ).encode("utf-8")
            with tarfile.open(archive_path, "w:bz2") as archive:
                info = tarfile.TarInfo("card/aro_index.tsv")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

            detail, summary, family_comparison, mechanism_comparison = build_card_support(
                priority_backbones,
                archive_path,
            )

        high_row = detail.loc[detail["backbone_id"] == "bb_high"].iloc[0]
        self.assertEqual(int(high_row["card_matched_gene_count"]), 2)
        self.assertAlmostEqual(float(high_row["card_match_fraction"]), 1.0)
        self.assertEqual(int(summary.loc[summary["priority_group"] == "high", "n_with_any_card_support"].iloc[0]), 1)
        self.assertIn("TEM family", set(family_comparison["card_amr_gene_family"]))
        self.assertIn("antibiotic inactivation", set(mechanism_comparison["card_resistance_mechanism"]))

    def test_build_mobsuite_support_joins_primary_replicon(self) -> None:
        priority_backbones = pd.DataFrame(
            [
                {
                    "backbone_id": "bb_high",
                    "priority_group": "high",
                    "priority_index": 0.9,
                    "dominant_species": "Escherichia_coli",
                    "primary_replicon": "IncFIB",
                    "replicon_types": "IncFIB",
                },
                {
                    "backbone_id": "bb_low",
                    "priority_group": "low",
                    "priority_index": 0.1,
                    "dominant_species": "Klebsiella_pneumoniae",
                    "primary_replicon": "IncN",
                    "replicon_types": "IncN",
                },
            ]
        )

        host_range_content = "\n".join(
            [
                "sample_id\trep_type(s)\thost_species\treported_host_range_taxid\tpmid\tyear\tnotes",
                "KC001\tIncFIB\tEscherichia coli\t562\t1\t2018\tplasmid conjugative",
                "KC002\tIncFIB\tKlebsiella pneumoniae\t573\t2\t2019\tplasmid conjugative",
                "KC003\tIncN\tEscherichia coli\t562\t3\t2017\t",
            ]
        ).encode("utf-8")
        cluster_content = "\n".join(
            [
                "sample_id\ttaxid\trep_type(s)\trelaxase_type(s)\tmpf_type\torit_type(s)\tpredicted_mobility",
                "CL001\t562\tIncFIB\tMOBF\tMPFT\toriT_1\tconjugative",
                "CL002\t573\tIncFIB\tMOBF\tMPFT\toriT_1\tconjugative",
                "CL003\t562\tIncN\tMOBF\t-\t-\tmobilizable",
            ]
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "mobsuite.tar"
            with tarfile.open(archive_path, "w") as archive:
                apple_double = tarfile.TarInfo("data/._host_range_literature_plasmidDB.txt")
                apple_double_bytes = b"ignored"
                apple_double.size = len(apple_double_bytes)
                archive.addfile(apple_double, io.BytesIO(apple_double_bytes))

                host_info = tarfile.TarInfo("data/host_range_literature_plasmidDB.txt")
                host_info.size = len(host_range_content)
                archive.addfile(host_info, io.BytesIO(host_range_content))

                cluster_info = tarfile.TarInfo("data/clusters.txt")
                cluster_info.size = len(cluster_content)
                archive.addfile(cluster_info, io.BytesIO(cluster_content))

            detail, summary = build_mobsuite_support(priority_backbones, archive_path)

        high_row = detail.loc[detail["backbone_id"] == "bb_high"].iloc[0]
        self.assertEqual(int(high_row["mobsuite_literature_record_count"]), 2)
        self.assertEqual(int(high_row["mobsuite_cluster_taxid_count"]), 2)
        self.assertTrue(bool(high_row["mobsuite_any_literature_support"]))
        self.assertAlmostEqual(float(high_row["mobsuite_cluster_mobilizable_fraction"]), 0.0)
        self.assertEqual(int(summary.loc[summary["priority_group"] == "high", "n_with_literature_support"].iloc[0]), 1)

    def test_build_pathogen_strata_group_summary_adds_dataset_label(self) -> None:
        combined = pd.DataFrame(
            [
                {
                    "priority_group": "high",
                    "n_backbones": 10,
                    "n_with_any_support": 5,
                    "mean_matching_records": 3.0,
                    "median_matching_records": 1.0,
                    "mean_matching_fraction": 0.4,
                    "median_matching_fraction": 0.1,
                    "mean_matching_genes": 2.0,
                }
            ]
        )
        clinical = pd.DataFrame(
            [
                {
                    "priority_group": "low",
                    "n_backbones": 10,
                    "n_with_any_support": 2,
                    "mean_matching_records": 1.0,
                    "median_matching_records": 0.0,
                    "mean_matching_fraction": 0.2,
                    "median_matching_fraction": 0.0,
                    "mean_matching_genes": 1.0,
                }
            ]
        )

        summary = build_pathogen_strata_group_summary({"combined": combined, "clinical": clinical})
        self.assertEqual(set(summary["pathogen_dataset"]), {"combined", "clinical"})
        self.assertEqual(len(summary), 2)

    def test_build_who_mia_support_maps_unambiguous_classes(self) -> None:
        priority_backbones = pd.DataFrame(
            [
                {
                    "backbone_id": "bb_high",
                    "priority_group": "high",
                    "priority_index": 0.9,
                    "dominant_species": "Escherichia_coli",
                    "dominant_genus": "Escherichia",
                    "amr_drug_classes": "FLUOROQUINOLONE ANTIBIOTIC,AMINOGLYCOSIDE,TETRACYCLINE",
                },
                {
                    "backbone_id": "bb_low",
                    "priority_group": "low",
                    "priority_index": 0.1,
                    "dominant_species": "Enterococcus_faecium",
                    "dominant_genus": "Enterococcus",
                    "amr_drug_classes": "UNKNOWN,PHENICOL ANTIBIOTIC",
                },
            ]
        )

        detail, summary, comparison = build_who_mia_support(priority_backbones)
        high_row = detail.loc[detail["backbone_id"] == "bb_high"].iloc[0]
        self.assertTrue(bool(high_row["who_mia_any_hpecia"]))
        self.assertTrue(bool(high_row["who_mia_any_cia"]))
        self.assertTrue(bool(high_row["who_mia_any_hia"]))
        self.assertAlmostEqual(float(high_row["who_mia_mapped_fraction"]), 1.0)
        self.assertEqual(int(summary.loc[summary["priority_group"] == "low", "n_with_hia_support"].iloc[0]), 1)
        self.assertIn("HPCIA", set(comparison["who_mia_category"]))

    def test_build_who_mia_support_uses_total_group_size_for_prevalence(self) -> None:
        priority_backbones = pd.DataFrame(
            [
                {
                    "backbone_id": "bb_high_supported",
                    "priority_group": "high",
                    "priority_index": 0.9,
                    "dominant_species": "Escherichia_coli",
                    "dominant_genus": "Escherichia",
                    "amr_drug_classes": "FLUOROQUINOLONE ANTIBIOTIC",
                },
                {
                    "backbone_id": "bb_high_unmapped",
                    "priority_group": "high",
                    "priority_index": 0.8,
                    "dominant_species": "Escherichia_coli",
                    "dominant_genus": "Escherichia",
                    "amr_drug_classes": "UNKNOWN",
                },
                {
                    "backbone_id": "bb_low_supported",
                    "priority_group": "low",
                    "priority_index": 0.1,
                    "dominant_species": "Klebsiella_pneumoniae",
                    "dominant_genus": "Klebsiella",
                    "amr_drug_classes": "FLUOROQUINOLONE ANTIBIOTIC",
                },
            ]
        )

        _, _, comparison = build_who_mia_support(priority_backbones)
        hpecia = comparison.loc[comparison["who_mia_category"] == "HPCIA"].iloc[0]
        self.assertAlmostEqual(float(hpecia["high"]), 0.5)
        self.assertAlmostEqual(float(hpecia["low"]), 1.0)

    def test_build_who_mia_reference_catalog_marks_present_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            who_text = Path(tmp_dir) / "who.txt"
            who_text.write_text(
                "Quinolones\nGlycopeptides and\nlipoglycopeptides\nPleuromutilins\n",
                encoding="utf-8",
            )
            catalog = build_who_mia_reference_catalog(who_text)

        quinolones = catalog.loc[catalog["who_mia_class"] == "Quinolones"].iloc[0]
        glycopeptides = catalog.loc[catalog["who_mia_class"] == "Glycopeptides and lipoglycopeptides"].iloc[0]
        self.assertTrue(bool(quinolones["reference_class_present_in_text"]))
        self.assertTrue(bool(glycopeptides["reference_class_present_in_text"]))


if __name__ == "__main__":
    unittest.main()
