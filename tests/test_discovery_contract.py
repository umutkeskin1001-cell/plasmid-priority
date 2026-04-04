from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.modeling import (
    build_discovery_input_contract,
    validate_discovery_input_contract,
)


class DiscoveryContractTests(unittest.TestCase):
    def test_discovery_contract_accepts_training_only_scored_input(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2015],
                "min_resolved_year_test": [2017, 2018],
                "training_only_future_unseen_backbone_flag": [False, False],
                "spread_label": [1.0, 0.0],
            }
        )
        validate_discovery_input_contract(
            scored,
            model_names=["bio_clean_priority"],
            contract=build_discovery_input_contract(2015),
            label="unit-test discovery table",
        )

    def test_discovery_contract_rejects_all_records_assignment(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "split_year": [2015],
                "backbone_assignment_mode": ["all_records"],
                "max_resolved_year_train": [2015],
                "min_resolved_year_test": [2017],
                "training_only_future_unseen_backbone_flag": [False],
                "spread_label": [1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-discovery backbone assignment modes"):
            validate_discovery_input_contract(
                scored,
                model_names=["bio_clean_priority"],
                contract=build_discovery_input_contract(2015),
                label="unit-test discovery table",
            )

    def test_discovery_contract_ignores_governance_only_model_sets(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "split_year": [2015],
                "backbone_assignment_mode": ["all_records"],
                "max_resolved_year_train": [2015],
                "min_resolved_year_test": [2017],
                "training_only_future_unseen_backbone_flag": [False],
                "spread_label": [1.0],
            }
        )
        validate_discovery_input_contract(
            scored,
            model_names=["phylo_support_fusion_priority"],
            contract=build_discovery_input_contract(2015),
            label="unit-test governance table",
        )


if __name__ == "__main__":
    unittest.main()
