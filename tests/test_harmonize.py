from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plasmid_priority.harmonize import build_harmonized_plasmid_table, normalize_country


class HarmonizeTests(unittest.TestCase):
    def test_build_harmonized_plasmid_table_merges_plasmidfinder_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inventory_path = root / "inventory.tsv"
            typing_path = root / "typing.csv"
            biosample_path = root / "biosample.csv"
            plasmidfinder_path = root / "plasmidfinder.csv"

            pd.DataFrame(
                [
                    {
                        "sequence_accession": "acc1",
                        "biosample_uid": 1,
                        "record_origin": "refseq",
                        "resolved_year": 2020,
                    }
                ]
            ).to_csv(inventory_path, sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "NUCCORE_ACC": "acc1",
                        "gc": 0.5,
                        "size": 5000,
                        "num_contigs": 1,
                        "rep_type(s)": "rep_cluster_1",
                        "relaxase_type(s)": "",
                        "mpf_type": "",
                        "orit_type(s)": "",
                        "predicted_mobility": "mobilizable",
                        "mash_neighbor_distance": 0.1,
                        "predicted_host_range_overall_rank": "genus",
                        "predicted_host_range_overall_name": "Escherichia",
                        "reported_host_range_lit_rank": "",
                        "reported_host_range_lit_name": "",
                        "associated_pmid(s)": "",
                        "primary_cluster_id": "AA001",
                        "secondary_cluster_id": "",
                        "observed_host_range_ncbi_rank": "genus",
                        "observed_host_range_ncbi_name": "Escherichia",
                        "PMLST_scheme": "",
                        "PMLST_sequence_type": "",
                        "PMLST_alleles": "",
                    }
                ]
            ).to_csv(typing_path, index=False)
            pd.DataFrame(
                [
                    {
                        "BIOSAMPLE_UID": 1,
                        "LOCATION_name": "Istanbul, Turkey",
                        "LOCATION_query": "",
                        "BIOSAMPLE_title": "sample",
                        "BIOSAMPLE_package": "Microbe; version 1.0",
                        "BIOSAMPLE_pathogenicity": "",
                        "ECOSYSTEM_tags": "",
                        "DISEASE_tags": "",
                    }
                ]
            ).to_csv(biosample_path, index=False)
            pd.DataFrame(
                [
                    {
                        "NUCCORE_ACC": "acc1",
                        "typing": "IncFIB(K)",
                        "identity": 98.0,
                        "coverage": 100.0,
                    },
                    {
                        "NUCCORE_ACC": "acc1",
                        "typing": "IncFIB(K)",
                        "identity": 99.0,
                        "coverage": 100.0,
                    },
                    {
                        "NUCCORE_ACC": "acc1",
                        "typing": "ColRNAI",
                        "identity": 95.0,
                        "coverage": 90.0,
                    },
                ]
            ).to_csv(plasmidfinder_path, index=False)

            harmonized = build_harmonized_plasmid_table(
                inventory_path,
                typing_path,
                biosample_path,
                plasmidfinder_path,
            )

        self.assertEqual(len(harmonized), 1)
        row = harmonized.iloc[0]
        self.assertEqual(row["country"], "Turkey")
        self.assertEqual(row["plasmidfinder_hit_count"], 3.0)
        self.assertEqual(row["plasmidfinder_type_count"], 2.0)
        self.assertEqual(row["plasmidfinder_dominant_type"], "IncFIB(K)")
        self.assertAlmostEqual(float(row["plasmidfinder_dominant_type_share"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(row["plasmidfinder_max_identity"]), 99.0)
        self.assertAlmostEqual(float(row["plasmidfinder_mean_coverage"]), 96.6666666667, places=6)

    def test_normalize_country_extracts_real_country_from_address_suffix(self) -> None:
        self.assertEqual(
            normalize_country("Erciyes University, Cevre Sokak, Kayseri, Turkey"),
            "Turkey",
        )
        self.assertEqual(
            normalize_country("Bethesda Metro Center Bus Station, Maryland, USA"),
            "USA",
        )

    def test_normalize_country_returns_empty_for_non_country_free_text(self) -> None:
        self.assertEqual(normalize_country("University of Texas MD Anderson Cancer Center"), "")
        self.assertEqual(normalize_country("40.15N;26.40E"), "")

    def test_normalize_country_handles_embedded_country_aliases(self) -> None:
        self.assertEqual(normalize_country("Turkey,Hatay"), "Turkey")
        self.assertEqual(normalize_country("Wanshan Islands, China"), "China")
        self.assertEqual(
            normalize_country("Alcitepe, Eceabat, Canakkale, Marmara Region, Turkey"), "Turkey"
        )

    def test_normalize_country_uk_constituent_nations(self) -> None:
        """England, Scotland, Wales, Northern Ireland should all resolve to UK."""
        self.assertEqual(normalize_country("London, England"), "UK")
        self.assertEqual(normalize_country("Edinburgh, Scotland"), "UK")
        self.assertEqual(normalize_country("Cardiff, Wales"), "UK")
        self.assertEqual(normalize_country("Belfast, Northern Ireland"), "UK")
        self.assertEqual(normalize_country("United Kingdom"), "UK")

    def test_normalize_country_guinea_variants(self) -> None:
        """Guinea, Equatorial Guinea, Guinea-Bissau, Papua New Guinea must not cross-match."""
        self.assertEqual(normalize_country("Guinea"), "Guinea")
        self.assertEqual(normalize_country("Equatorial Guinea"), "Equatorial Guinea")
        self.assertEqual(normalize_country("Papua New Guinea"), "Papua New Guinea")

    def test_normalize_country_empty_and_missing(self) -> None:
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(float("nan")), "")
        self.assertEqual(normalize_country("   "), "")

    def test_normalize_country_usa_variants(self) -> None:
        self.assertEqual(normalize_country("United States of America"), "USA")
        self.assertEqual(normalize_country("USA"), "USA")
        self.assertEqual(normalize_country("Los Angeles, CA, United States"), "USA")

    def test_normalize_country_korea_variants(self) -> None:
        self.assertEqual(normalize_country("South Korea"), "South Korea")
        self.assertEqual(normalize_country("Republic of Korea"), "South Korea")

    def test_normalize_country_case_insensitive(self) -> None:
        self.assertEqual(normalize_country("GERMANY"), "Germany")
        self.assertEqual(normalize_country("germany"), "Germany")
        self.assertEqual(normalize_country("GeRmAnY"), "Germany")

    def test_normalize_country_china_variants(self) -> None:
        self.assertEqual(normalize_country("People's Republic of China"), "China")
        self.assertEqual(normalize_country("PR China"), "China")


if __name__ == "__main__":
    unittest.main()
