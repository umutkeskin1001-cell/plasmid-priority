from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.bio_transfer.contracts import (
    build_bio_transfer_input_contract,
    validate_bio_transfer_input_contract,
)
from plasmid_priority.clinical_hazard.contracts import (
    build_clinical_hazard_input_contract,
    validate_clinical_hazard_input_contract,
)
from plasmid_priority.shared.labels import (
    build_bio_transfer_labels,
    build_clinical_hazard_labels,
    build_geo_spread_labels,
)


class SharedLabelTests(unittest.TestCase):
    def test_geo_spread_label_factory_counts_new_countries(self) -> None:
        scored = pd.DataFrame({"backbone_id": ["bb1", "bb2"]})
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb1", "bb2", "bb2"],
                "resolved_year": [2014, 2016, 2017, 2018, 2014, 2018],
                "country": ["A", "B", "C", "D", "A", "A"],
                "host_genus": ["g1"] * 6,
                "host_family": ["f1"] * 6,
            }
        )
        labels = build_geo_spread_labels(scored, records, 2015, 5)
        row = labels.loc[labels["backbone_id"] == "bb1"].iloc[0]
        self.assertEqual(int(row["n_new_countries_future"]), 3)
        self.assertEqual(int(row["spread_label"]), 1)

    def test_bio_transfer_label_factory_uses_host_expansion(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb1", "bb2", "bb2"],
                "resolved_year": [2014, 2016, 2017, 2018, 2014, 2018],
                "country": ["A", "B", "C", "D", "A", "A"],
                "host_genus": ["g1", "g2", "g3", "g4", "g1", "g1"],
                "host_family": ["f1", "f1", "f2", "f3", "f1", "f1"],
            }
        )
        labels = build_bio_transfer_labels(records, 2015, 5)
        row = labels.loc[labels["backbone_id"] == "bb1"].iloc[0]
        self.assertGreaterEqual(int(row["n_new_host_genera_future"]), 2)
        self.assertEqual(int(row["bio_transfer_label"]), 1)

    def test_clinical_hazard_label_factory_requires_two_escalation_conditions(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb1", "bb1", "bb2", "bb2"],
                "resolved_year": [2014, 2016, 2017, 2014, 2018],
                "country": ["A", "B", "B", "A", "A"],
                "clinical_context": [
                    "environmental",
                    "clinical",
                    "clinical",
                    "environmental",
                    "environmental",
                ],
                "amr_class": ["A", "CARBAPENEM", "CARBAPENEM", "A", "A"],
                "drug_class_count": [1, 4, 4, 1, 1],
                "pd_clinical_support": [0, 1, 1, 0, 0],
            }
        )
        pd_metadata = pd.DataFrame(
            {"backbone_id": ["bb1", "bb2"], "context_label": ["clinical", "environmental"]}
        )
        labels = build_clinical_hazard_labels(records, pd_metadata, 2015, 5)
        row = labels.loc[labels["backbone_id"] == "bb1"].iloc[0]
        self.assertEqual(int(row["clinical_hazard_label"]), 1)

    def test_clinical_hazard_label_factory_uses_gain_not_absolute_future_level(self) -> None:
        records = pd.DataFrame.from_records(
            [
                *[
                    {
                        "backbone_id": "bb1",
                        "resolved_year": 2014,
                        "country": "A",
                        "clinical_context": "clinical" if idx < 9 else "environmental",
                        "amr_class": "A",
                        "drug_class_count": 1,
                        "pd_clinical_support": 0,
                    }
                    for idx in range(10)
                ],
                *[
                    {
                        "backbone_id": "bb1",
                        "resolved_year": 2016,
                        "country": "A",
                        "clinical_context": "clinical",
                        "amr_class": "A",
                        "drug_class_count": 1,
                        "pd_clinical_support": 0,
                    }
                    for _ in range(10)
                ],
                *[
                    {
                        "backbone_id": "bb2",
                        "resolved_year": 2014,
                        "country": "B",
                        "clinical_context": "environmental",
                        "amr_class": "A",
                        "drug_class_count": 1,
                        "pd_clinical_support": 0,
                    }
                    for _ in range(10)
                ],
                *[
                    {
                        "backbone_id": "bb2",
                        "resolved_year": 2016,
                        "country": "B",
                        "clinical_context": "clinical",
                        "amr_class": "CARBAPENEM",
                        "drug_class_count": 4,
                        "pd_clinical_support": 1,
                    }
                    for _ in range(10)
                ],
            ]
        )
        labels = build_clinical_hazard_labels(records, None, 2015, 5)
        row = labels.loc[labels["backbone_id"] == "bb1"].iloc[0]
        positive = labels.loc[labels["backbone_id"] == "bb2"].iloc[0]

        self.assertEqual(int(row["clinical_hazard_label"]), 0)
        self.assertEqual(int(positive["clinical_hazard_label"]), 1)


class SharedContractTests(unittest.TestCase):
    def test_bio_transfer_contract_accepts_valid_table(self) -> None:
        table = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "bio_transfer_label": [1.0, 0.0],
                "future_new_host_genera_count": [3, 0],
                "future_new_host_families_count": [1, 0],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2014],
                "min_resolved_year_test": [2016, 2016],
                "training_only_future_unseen_backbone_flag": [False, False],
            }
        )
        validate_bio_transfer_input_contract(
            table,
            contract=build_bio_transfer_input_contract(),
            label="bio transfer test table",
        )

    def test_clinical_hazard_contract_accepts_valid_table(self) -> None:
        table = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "clinical_hazard_label": [1.0, 0.0],
                "clinical_fraction_future": [0.3, 0.0],
                "last_resort_fraction_future": [0.2, 0.0],
                "mdr_proxy_fraction_future": [0.2, 0.0],
                "pd_clinical_support_future": [0.2, 0.0],
                "split_year": [2015, 2015],
                "backbone_assignment_mode": ["training_only", "training_only"],
                "max_resolved_year_train": [2014, 2014],
                "min_resolved_year_test": [2016, 2016],
                "training_only_future_unseen_backbone_flag": [False, False],
            }
        )
        validate_clinical_hazard_input_contract(
            table,
            contract=build_clinical_hazard_input_contract(),
            label="clinical hazard test table",
        )


if __name__ == "__main__":
    unittest.main()
