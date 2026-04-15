from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from plasmid_priority.harmonize.metadata import (
    BRONZE_INVENTORY_COLUMNS,
    build_plsdb_canonical_metadata,
    write_bronze_inventory,
)


class HarmonizeMetadataTests(unittest.TestCase):
    def test_build_plsdb_canonical_metadata_normalizes_text_nulls(self) -> None:
        plsdb_frame = pd.DataFrame(
            [
                {
                    "NUCCORE_ACC": "  ABC123  ",
                    "NUCCORE_UID": 10,
                    "ASSEMBLY_UID": 20,
                    "BIOSAMPLE_UID": 30,
                    "NUCCORE_Source": " None ",
                    "NUCCORE_CreateDate": "2024-01-15",
                    "NUCCORE_Description": "  Example description  ",
                    "NUCCORE_Length": 1234,
                    "TAXONOMY_UID": 99,
                    "NUCCORE_Topology": " Circular ",
                    "STATUS": " null ",
                }
            ]
        )
        taxonomy_frame = pd.DataFrame(
            [
                {
                    "TAXONOMY_UID": 99,
                    "TAXONOMY_superkingdom": "Bacteria",
                    "TAXONOMY_phylum": "  Nan ",
                    "TAXONOMY_class": "Alpha",
                    "TAXONOMY_order": " None ",
                    "TAXONOMY_family": "  FamilyA  ",
                    "TAXONOMY_genus": " GenusA ",
                    "TAXONOMY_species": " null ",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            plsdb_path = tmp_dir_path / "plsdb.tsv"
            taxonomy_path = tmp_dir_path / "taxonomy.tsv"
            plsdb_frame.to_csv(plsdb_path, sep="\t", index=False)
            taxonomy_frame.to_csv(taxonomy_path, sep="\t", index=False)

            canonical = build_plsdb_canonical_metadata(plsdb_path, taxonomy_path)

        self.assertEqual(canonical.loc[0, "sequence_accession"], "ABC123")
        self.assertEqual(canonical.loc[0, "record_origin"], "")
        self.assertEqual(canonical.loc[0, "fasta_description"], "Example description")
        self.assertEqual(canonical.loc[0, "TAXONOMY_phylum"], "")
        self.assertEqual(canonical.loc[0, "TAXONOMY_class"], "Alpha")
        self.assertEqual(canonical.loc[0, "TAXONOMY_order"], "")
        self.assertEqual(canonical.loc[0, "TAXONOMY_family"], "FamilyA")
        self.assertEqual(canonical.loc[0, "genus"], "GenusA")
        self.assertEqual(canonical.loc[0, "species"], "")
        self.assertEqual(canonical.loc[0, "topology"], "circular")
        self.assertEqual(canonical.loc[0, "status"], "")

    def test_build_plsdb_canonical_metadata_duckdb_matches_pandas_path(self) -> None:
        plsdb_frame = pd.DataFrame(
            [
                {
                    "NUCCORE_ACC": "ABC123",
                    "NUCCORE_UID": 10,
                    "ASSEMBLY_UID": 20,
                    "BIOSAMPLE_UID": 30,
                    "NUCCORE_Source": "plasmid",
                    "NUCCORE_CreateDate": "2024-01-15",
                    "NUCCORE_Description": "Example description",
                    "NUCCORE_Length": 1234,
                    "TAXONOMY_UID": 99,
                    "NUCCORE_Topology": "Circular",
                    "STATUS": "ok",
                },
                {
                    "NUCCORE_ACC": "DEF456",
                    "NUCCORE_UID": 11,
                    "ASSEMBLY_UID": 21,
                    "BIOSAMPLE_UID": 31,
                    "NUCCORE_Source": "RefSeq",
                    "NUCCORE_CreateDate": "2023-07-04",
                    "NUCCORE_Description": "Second description",
                    "NUCCORE_Length": 4321,
                    "TAXONOMY_UID": 100,
                    "NUCCORE_Topology": "Linear",
                    "STATUS": "null",
                },
            ]
        )
        taxonomy_frame = pd.DataFrame(
            [
                {
                    "TAXONOMY_UID": 99,
                    "TAXONOMY_superkingdom": "Bacteria",
                    "TAXONOMY_phylum": "Proteobacteria",
                    "TAXONOMY_class": "Gammaproteobacteria",
                    "TAXONOMY_order": "Enterobacterales",
                    "TAXONOMY_family": "Enterobacteriaceae",
                    "TAXONOMY_genus": "Escherichia",
                    "TAXONOMY_species": "coli",
                },
                {
                    "TAXONOMY_UID": 100,
                    "TAXONOMY_superkingdom": "Bacteria",
                    "TAXONOMY_phylum": "Firmicutes",
                    "TAXONOMY_class": "Bacilli",
                    "TAXONOMY_order": "Lactobacillales",
                    "TAXONOMY_family": "Streptococcaceae",
                    "TAXONOMY_genus": "Streptococcus",
                    "TAXONOMY_species": "pneumoniae",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            plsdb_path = tmp_dir_path / "plsdb.tsv"
            taxonomy_path = tmp_dir_path / "taxonomy.tsv"
            plsdb_frame.to_csv(plsdb_path, sep="\t", index=False)
            taxonomy_frame.to_csv(taxonomy_path, sep="\t", index=False)

            pandas_frame = build_plsdb_canonical_metadata(
                plsdb_path,
                taxonomy_path,
                use_duckdb=False,
            )
            duckdb_frame = build_plsdb_canonical_metadata(
                plsdb_path,
                taxonomy_path,
                use_duckdb=True,
            )

        pd.testing.assert_frame_equal(duckdb_frame, pandas_frame, check_like=True)

    def test_write_bronze_inventory_streams_rows_without_to_dict(self) -> None:
        plsdb_frame = pd.DataFrame(
            [
                {
                    "source_dataset": "plsdb",
                    "sequence_accession": "ABC123",
                    "nuccore_uid": 1,
                    "assembly_uid": 2,
                    "biosample_uid": 3,
                    "record_origin": "plasmid",
                    "resolved_year": 2020,
                    "fasta_description": "desc one",
                    "sequence_length": 111,
                    "taxonomy_uid": 4,
                    "TAXONOMY_phylum": "P",
                    "TAXONOMY_class": "C",
                    "TAXONOMY_order": "O",
                    "TAXONOMY_family": "F",
                    "genus": "GenusA",
                    "species": "SpeciesA",
                    "topology": "circular",
                    "status": "ok",
                    "metadata_status": "canonical_plsdb",
                },
                {
                    "source_dataset": "plsdb",
                    "sequence_accession": "DEF456",
                    "nuccore_uid": 5,
                    "assembly_uid": 6,
                    "biosample_uid": 7,
                    "record_origin": "plasmid",
                    "resolved_year": 2021,
                    "fasta_description": "desc two",
                    "sequence_length": 222,
                    "taxonomy_uid": 8,
                    "TAXONOMY_phylum": "P2",
                    "TAXONOMY_class": "C2",
                    "TAXONOMY_order": "O2",
                    "TAXONOMY_family": "F2",
                    "genus": "GenusB",
                    "species": "SpeciesB",
                    "topology": "linear",
                    "status": "ok",
                    "metadata_status": "canonical_plsdb",
                },
            ]
        )[list(reversed(BRONZE_INVENTORY_COLUMNS))]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "bronze.tsv"
            with mock.patch.object(
                pd.DataFrame,
                "to_dict",
                side_effect=AssertionError("write_bronze_inventory should not materialize rows"),
            ):
                row_count = write_bronze_inventory(plsdb_frame, iter(()), output_path)

            self.assertEqual(row_count, 2)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0].split("\t"), BRONZE_INVENTORY_COLUMNS)
            self.assertIn("ABC123", lines[1])
            self.assertIn("DEF456", lines[2])


if __name__ == "__main__":
    unittest.main()
