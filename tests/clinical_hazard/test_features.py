from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.clinical_hazard.features import build_clinical_hazard_features


class ClinicalHazardFeatureTests(unittest.TestCase):
    def test_feature_builder_fills_escalation_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "A_eff_norm": [0.7],
                "clinical_context_fraction_norm": [0.5],
                "H_external_host_range_norm": [0.2],
                "mash_neighbor_distance_train_norm": [0.4],
            }
        )
        built = build_clinical_hazard_features(frame)
        self.assertIn("A_clinical_context_synergy_norm", built.columns)
        self.assertIn("amr_mdr_proxy_norm", built.columns)
        self.assertNotIn("clinical_fraction_future_future", built.columns)
        self.assertAlmostEqual(float(built.loc[0, "A_clinical_context_synergy_norm"]), 0.35, places=6)

    def test_feature_builder_does_not_backfill_from_future_only_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "A_eff_norm": [0.7],
                "clinical_fraction_future": [0.9],
                "mdr_proxy_fraction_future": [0.8],
                "xdr_proxy_fraction_future": [0.6],
                "pd_clinical_support_future": [1.0],
                "last_resort_fraction_future": [0.5],
            }
        )
        built = build_clinical_hazard_features(frame)
        self.assertEqual(float(built.loc[0, "clinical_context_fraction_norm"]), 0.0)
        self.assertEqual(float(built.loc[0, "mdr_proxy_fraction_norm"]), 0.0)
        self.assertEqual(float(built.loc[0, "xdr_proxy_fraction_norm"]), 0.0)
        self.assertEqual(float(built.loc[0, "pd_clinical_support_norm"]), 0.0)
        self.assertEqual(float(built.loc[0, "last_resort_convergence_norm"]), 0.0)


if __name__ == "__main__":
    unittest.main()
