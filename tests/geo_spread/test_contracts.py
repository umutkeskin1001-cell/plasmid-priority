from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.geo_spread import (
    GeoSpreadBenchmarkSpec,
    GeoSpreadConfig,
    build_geo_spread_input_contract,
    load_geo_spread_config,
    validate_geo_spread_input_contract,
)


class GeoSpreadConfigTests(unittest.TestCase):
    def test_load_geo_spread_config_parses_branch_defaults(self) -> None:
        config = load_geo_spread_config()
        self.assertIsInstance(config, GeoSpreadConfig)
        self.assertIsInstance(config.benchmark, GeoSpreadBenchmarkSpec)
        self.assertEqual(config.benchmark.split_year, 2015)
        self.assertEqual(config.primary_model_name, "geo_reliability_blend")
        self.assertIn("geo_counts_baseline", config.feature_sets)
        self.assertIn("geo_support_light_priority", config.fit_config)
        self.assertEqual(config.fit_config["geo_support_light_priority"].calibration, "isotonic")

    def test_build_geo_spread_input_contract_uses_config_defaults(self) -> None:
        contract = build_geo_spread_input_contract()
        self.assertEqual(contract.benchmark.min_new_countries_for_spread, 3)
        self.assertEqual(contract.required_columns[0], "backbone_id")


class GeoSpreadContractTests(unittest.TestCase):
    def _valid_backbone_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2", "bb3"],
                "spread_label": [1.0, 0.0, pd.NA],
                "n_new_countries": [4, 1, 0],
                "split_year": [2015, 2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only", "training_only"],
                "max_resolved_year_train": [2014, 2015, 2015],
                "min_resolved_year_test": [2017, 2018, 2019],
                "training_only_future_unseen_backbone_flag": [False, False, False],
            }
        )

    def test_validate_geo_spread_input_contract_accepts_valid_table(self) -> None:
        validate_geo_spread_input_contract(
            self._valid_backbone_table(),
            contract=build_geo_spread_input_contract(),
            label="unit-test geo spread table",
        )

    def test_validate_geo_spread_input_contract_rejects_bad_assignment_mode(self) -> None:
        table = self._valid_backbone_table().copy()
        table["backbone_assignment_mode"] = "all_records"
        with self.assertRaisesRegex(ValueError, "non-discovery backbone assignment modes"):
            validate_geo_spread_input_contract(
                table,
                contract=build_geo_spread_input_contract(),
                label="unit-test geo spread table",
            )

    def test_validate_geo_spread_input_contract_rejects_label_threshold_mismatch(self) -> None:
        table = self._valid_backbone_table().copy()
        table.loc[0, "n_new_countries"] = 1
        with self.assertRaisesRegex(ValueError, "does not match `n_new_countries` thresholding"):
            validate_geo_spread_input_contract(
                table,
                contract=build_geo_spread_input_contract(),
                label="unit-test geo spread table",
            )


if __name__ == "__main__":
    unittest.main()
