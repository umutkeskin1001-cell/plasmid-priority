from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.bio_transfer.features import build_bio_transfer_features


class BioTransferFeatureTests(unittest.TestCase):
    def test_feature_builder_fills_mobility_and_synergy_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "orit_support": [0.8],
                "T_eff_norm": [0.6],
                "H_external_host_range_norm": [0.4],
            }
        )
        built = build_bio_transfer_features(frame)
        self.assertIn("mobility_support_norm", built.columns)
        self.assertIn("host_breadth_mobility_synergy_norm", built.columns)
        self.assertNotIn("bio_transfer_label_future", built.columns)
        self.assertAlmostEqual(
            float(built.loc[0, "host_breadth_mobility_synergy_norm"]), 0.32, places=6
        )


if __name__ == "__main__":
    unittest.main()
