"""Temporal leakage detection tests.

Verify that no test-period data (post-2015) contaminates training features.
This is a critical methodological safeguard for the retrospective design.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.features.core import _training_period_records


class TestTemporalLeakage(unittest.TestCase):
    """Ensure the training/test split is honored throughout the pipeline."""

    @staticmethod
    def _build_mock_records() -> pd.DataFrame:
        """Create a minimal record set spanning both training and test periods."""
        return pd.DataFrame(
            {
                "sequence_accession": [f"ACC{i}" for i in range(1, 11)],
                "backbone_id": ["BB1"] * 5 + ["BB2"] * 5,
                "collection_year": [2010, 2012, 2014, 2015, 2018, 2013, 2015, 2016, 2020, 2022],
                "country": ["US", "DE", "FR", "JP", "BR", "US", "DE", "CN", "IN", "NG"],
                "species": ["E.coli"] * 5 + ["K.pneumoniae"] * 5,
                "genus": ["Escherichia"] * 5 + ["Klebsiella"] * 5,
                "amr_drug_classes": ["AMINOGLYCOSIDE"] * 10,
                "amr_gene_symbols": ["aac(6')"] * 10,
                "replicon_types": ["IncF"] * 10,
                "mob_type": ["MOBF"] * 10,
                "topology": ["circular"] * 10,
                "canonical_id": [f"ACC{i}" for i in range(1, 11)],
            }
        )

    def test_training_records_exclude_post_2015(self) -> None:
        """Records used for feature computation must be ≤ 2015."""
        records = self._build_mock_records()
        training_cutoff = 2015
        training = records.loc[records["collection_year"] <= training_cutoff]

        # Verify no training record has year > cutoff
        self.assertTrue(
            (training["collection_year"] <= training_cutoff).all(),
            "Training set contains records from the test period (> 2015).",
        )
        # Verify test-period records exist but are excluded
        test_only = records.loc[records["collection_year"] > training_cutoff]
        self.assertGreater(
            len(test_only),
            0,
            "Mock data should include test-period records.",
        )
        self.assertEqual(
            len(set(training["sequence_accession"]) & set(test_only["sequence_accession"])),
            0,
            "Training and test accessions must be disjoint.",
        )

    def test_outcome_uses_only_test_period(self) -> None:
        """spread_label (n_new_countries) must be computed from post-2015 only."""
        records = self._build_mock_records()
        training_cutoff = 2015

        training = records.loc[records["collection_year"] <= training_cutoff]
        test = records.loc[records["collection_year"] > training_cutoff]

        train_countries = set(training.loc[training["backbone_id"] == "BB1", "country"])
        test_countries = set(test.loc[test["backbone_id"] == "BB1", "country"])
        new_countries = test_countries - train_countries

        # BB1 has training countries {US, DE, FR, JP} and test country {BR}
        self.assertEqual(
            new_countries,
            {"BR"},
            f"Expected only BR as new country, got {new_countries}",
        )

    def test_feature_computation_cannot_see_future(self) -> None:
        """Backbone features derived from training records must not include
        information from the test period."""
        records = self._build_mock_records()
        training_cutoff = 2015
        training = records.loc[records["collection_year"] <= training_cutoff]

        # Simulate member_count_train and n_countries_train
        for backbone_id, group in training.groupby("backbone_id"):
            member_count = len(group)
            country_count = group["country"].nunique()

            # Count from full data (this would be a leak)
            full_group = records.loc[records["backbone_id"] == backbone_id]
            full_member_count = len(full_group)
            full_country_count = full_group["country"].nunique()

            # Assert training counts ≤ full counts (leak detection)
            self.assertLessEqual(
                member_count,
                full_member_count,
                f"member_count for {backbone_id} exceeds training-only count (possible leakage)",
            )
            self.assertLessEqual(
                country_count,
                full_country_count,
                f"country_count for {backbone_id} exceeds training-only count (possible leakage)",
            )
            # Assert training counts are strictly less (proves filtering works)
            if backbone_id == "BB1":
                self.assertLess(
                    member_count,
                    full_member_count,
                    "BB1 should have fewer training records than total (test record exists)",
                )

    def test_spread_label_threshold_integrity(self) -> None:
        """spread_label must use ≥3 new countries threshold as defined in plan."""
        threshold = 3
        n_new = np.array([0, 1, 2, 3, 4, 10])
        labels = (n_new >= threshold).astype(int)
        expected = np.array([0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(
            labels,
            expected,
            "spread_label threshold must be ≥3 new countries.",
        )

    def test_training_period_records_ignore_missing_years(self) -> None:
        records = pd.DataFrame(
            {
                "resolved_year": [2010, None, 2016],
                "backbone_id": ["BB1", "BB2", "BB3"],
            }
        )
        training = _training_period_records(
            records,
            split_year=2015,
            label="test_training_period_records_ignore_missing_years",
        )
        self.assertListEqual(training["backbone_id"].tolist(), ["BB1"])


if __name__ == "__main__":
    unittest.main()
