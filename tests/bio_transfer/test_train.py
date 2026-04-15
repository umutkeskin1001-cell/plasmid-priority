from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.bio_transfer.train import fit_bio_transfer_model


class BioTransferTrainSmokeTests(unittest.TestCase):
    def test_fit_bio_transfer_model_smoke(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["b1", "b2", "b3", "b4"],
                "split_year": [2015, 2015, 2015, 2015],
                "backbone_assignment_mode": ["training_only"] * 4,
                "max_resolved_year_train": [2014, 2014, 2014, 2014],
                "min_resolved_year_test": [2016, 2016, 2016, 2016],
                "training_only_future_unseen_backbone_flag": [False, False, False, False],
                "future_new_host_genera_count": [0, 2, 0, 2],
                "future_new_host_families_count": [0, 1, 0, 1],
            }
        )
        records = pd.DataFrame(
            {
                "backbone_id": ["b1"] * 4 + ["b2"] * 4 + ["b3"] * 4 + ["b4"] * 4,
                "resolved_year": [2012, 2014, 2016, 2018] * 4,
                "country": [
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "de",
                    "fr",
                    "us",
                    "us",
                    "gb",
                    "gb",
                ],
                "host_genus": [
                    "genus_a",
                    "genus_a",
                    "genus_a",
                    "genus_a",
                    "genus_a",
                    "genus_a",
                    "genus_b",
                    "genus_c",
                    "genus_d",
                    "genus_d",
                    "genus_d",
                    "genus_d",
                    "genus_e",
                    "genus_e",
                    "genus_f",
                    "genus_g",
                ],
                "host_family": [
                    "fam_a",
                    "fam_a",
                    "fam_a",
                    "fam_a",
                    "fam_a",
                    "fam_a",
                    "fam_b",
                    "fam_c",
                    "fam_d",
                    "fam_d",
                    "fam_d",
                    "fam_d",
                    "fam_e",
                    "fam_e",
                    "fam_f",
                    "fam_g",
                ],
            }
        )

        result = fit_bio_transfer_model(
            scored, model_name="bio_transfer_baseline", records=records, n_splits=2, n_repeats=1
        )
        self.assertEqual(getattr(result, "status", None), "ok")


if __name__ == "__main__":
    unittest.main()
