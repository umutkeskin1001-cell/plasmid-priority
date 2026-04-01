from __future__ import annotations

from pathlib import Path
import unittest


from plasmid_priority.harmonize import normalize_country


class HarmonizeTests(unittest.TestCase):
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
        self.assertEqual(normalize_country("Alcitepe, Eceabat, Canakkale, Marmara Region, Turkey"), "Turkey")

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

