from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.clinical_hazard.train import fit_clinical_hazard_model


class ClinicalHazardTrainSmokeTests(unittest.TestCase):
    def test_fit_clinical_hazard_model_smoke(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": ["b1", "b2", "b3", "b4"],
                "split_year": [2015, 2015, 2015, 2015],
                "backbone_assignment_mode": ["training_only"] * 4,
                "max_resolved_year_train": [2014, 2014, 2014, 2014],
                "min_resolved_year_test": [2016, 2016, 2016, 2016],
                "training_only_future_unseen_backbone_flag": [False, False, False, False],
                "clinical_fraction_future": [0.0, 0.0, 0.3, 0.4],
                "last_resort_fraction_future": [0.0, 0.0, 0.2, 0.2],
                "mdr_proxy_fraction_future": [0.0, 0.0, 0.2, 0.2],
                "pd_clinical_support_future": [0.0, 0.0, 0.2, 0.2],
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
                "clinical_context": [
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "environmental",
                    "hospital",
                    "environmental",
                    "clinical",
                    "hospital",
                    "hospital",
                ],
                "last_resort_flag": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                "amr_class_count": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
                "amr_gene_count": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5],
                "pd_clinical_support": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            }
        )

        result = fit_clinical_hazard_model(
            scored, model_name="clinical_hazard_baseline", records=records, n_splits=2, n_repeats=1
        )
        self.assertEqual(getattr(result, "status", None), "ok")


if __name__ == "__main__":
    unittest.main()
