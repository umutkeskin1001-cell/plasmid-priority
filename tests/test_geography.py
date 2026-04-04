from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.utils.geography import (
    build_country_quality_summary,
    country_to_macro_region,
    dominant_macro_region_table,
)


class GeographyTests(unittest.TestCase):
    def test_country_to_macro_region_maps_known_countries(self) -> None:
        self.assertEqual(country_to_macro_region("Germany"), "Europe")
        self.assertEqual(country_to_macro_region("Turkey"), "Middle East and West Asia")
        self.assertEqual(country_to_macro_region("Brazil"), "Latin America and Caribbean")

    def test_dominant_macro_region_table_uses_training_rows_only(self) -> None:
        records = pd.DataFrame(
            [
                {"backbone_id": "AA1", "resolved_year": 2014, "country": "Germany"},
                {"backbone_id": "AA1", "resolved_year": 2015, "country": "France"},
                {"backbone_id": "AA1", "resolved_year": 2018, "country": "Brazil"},
                {"backbone_id": "AA2", "resolved_year": 2014, "country": "Brazil"},
            ]
        )
        result = dominant_macro_region_table(records, split_year=2015)
        aa1 = result.loc[result["backbone_id"] == "AA1", "dominant_region_train"].iloc[0]
        aa2 = result.loc[result["backbone_id"] == "AA2", "dominant_region_train"].iloc[0]
        self.assertEqual(aa1, "Europe")
        self.assertEqual(aa2, "Latin America and Caribbean")

    def test_country_quality_summary_reports_completeness(self) -> None:
        records = pd.DataFrame(
            [
                {"backbone_id": "AA1", "resolved_year": 2014, "country": "Germany"},
                {"backbone_id": "AA1", "resolved_year": 2018, "country": ""},
                {"backbone_id": "AA2", "resolved_year": 2013, "country": "Brazil"},
            ]
        )
        summary = build_country_quality_summary(records, split_year=2015)
        overall = summary.loc[summary["period"] == "all_rows"].iloc[0]
        self.assertEqual(int(overall["n_rows"]), 3)
        self.assertAlmostEqual(float(overall["country_non_null_fraction"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
