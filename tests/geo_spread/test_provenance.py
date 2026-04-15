from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from plasmid_priority.geo_spread import build_geo_spread_run_provenance


class GeoSpreadProvenanceTests(unittest.TestCase):
    def _scored(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "spread_label": [1, 0],
                "n_new_countries": [4, 1],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2014],
                "min_resolved_year_test": [2017, 2017],
                "training_only_future_unseen_backbone_flag": [False, False],
            }
        )

    def test_build_geo_spread_run_provenance_is_stable(self) -> None:
        provenance_1 = build_geo_spread_run_provenance(
            self._scored(),
            model_names=("geo_counts_baseline", "geo_parsimonious_priority"),
            config={
                "geo_spread": {
                    "fit_config": {
                        "geo_counts_baseline": {"calibration": "none"},
                        "geo_parsimonious_priority": {"calibration": "isotonic"},
                    }
                }
            },
            source_paths=[Path(__file__)],
        )
        provenance_2 = build_geo_spread_run_provenance(
            self._scored(),
            model_names=("geo_counts_baseline", "geo_parsimonious_priority"),
            config={
                "geo_spread": {
                    "fit_config": {
                        "geo_counts_baseline": {"calibration": "none"},
                        "geo_parsimonious_priority": {"calibration": "isotonic"},
                    }
                }
            },
            source_paths=[Path(__file__)],
        )
        self.assertEqual(provenance_1["run_signature"], provenance_2["run_signature"])
        self.assertEqual(provenance_1["benchmark_name"], "geo_spread_v1")
        self.assertEqual(provenance_1["n_positive"], 1)
        self.assertIn("feature_surface_hash", provenance_1)

    def test_build_geo_spread_run_provenance_changes_when_config_changes(self) -> None:
        base = build_geo_spread_run_provenance(
            self._scored(),
            model_names=("geo_counts_baseline",),
            config={"geo_spread": {"fit_config": {"geo_counts_baseline": {"calibration": "none"}}}},
            source_paths=[Path(__file__)],
        )
        changed = build_geo_spread_run_provenance(
            self._scored(),
            model_names=("geo_counts_baseline",),
            config={
                "geo_spread": {"fit_config": {"geo_counts_baseline": {"calibration": "isotonic"}}}
            },
            source_paths=[Path(__file__)],
        )
        self.assertNotEqual(base["config_hash"], changed["config_hash"])
        self.assertNotEqual(base["run_signature"], changed["run_signature"])


if __name__ == "__main__":
    unittest.main()
