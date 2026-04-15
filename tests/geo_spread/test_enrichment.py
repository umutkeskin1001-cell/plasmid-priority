from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.geo_spread import enrich_geo_spread_scored_table


class GeoSpreadEnrichmentTests(unittest.TestCase):
    def test_enrich_geo_spread_scored_table_adds_context_features(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "spread_label": [1.0, 0.0],
                "n_new_countries": [4, 1],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2014],
                "min_resolved_year_test": [2017, 2017],
                "training_only_future_unseen_backbone_flag": [False, False],
            }
        )
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb2", "bb2"],
                "country": ["USA", "Canada", "USA", "Germany", "Germany"],
                "resolved_year": [2014, 2015, 2013, 2015, 2014],
            }
        )
        enriched = enrich_geo_spread_scored_table(scored, split_year=2015, records=records)
        for column in (
            "geo_country_record_count_train",
            "geo_country_entropy_train",
            "geo_macro_region_entropy_train",
            "geo_dominant_region_share_train",
        ):
            self.assertIn(column, enriched.columns)
        self.assertGreater(
            float(
                enriched.loc[enriched["backbone_id"] == "bb1", "geo_country_entropy_train"].iloc[0]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                enriched.loc[
                    enriched["backbone_id"] == "bb2", "geo_dominant_region_share_train"
                ].iloc[0]
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
