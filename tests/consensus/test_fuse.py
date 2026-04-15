from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.consensus.fuse import build_operational_consensus_frame, merge_branch_predictions


class ConsensusFuseTests(unittest.TestCase):
    def test_operational_consensus_is_monotonic_and_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "spread_label": [1, 0],
                "p_geo": [0.9, 0.1],
                "p_bio_transfer": [0.5, 0.5],
                "p_clinical_hazard": [0.5, 0.5],
                "confidence_geo": [0.9, 0.9],
                "confidence_bio_transfer": [0.9, 0.9],
                "confidence_clinical_hazard": [0.9, 0.9],
                "ood_geo": [False, False],
                "ood_bio_transfer": [False, False],
                "ood_clinical_hazard": [False, False],
            }
        )
        fused = build_operational_consensus_frame(frame)
        self.assertTrue(((fused["consensus_score"] >= 0.0) & (fused["consensus_score"] <= 1.0)).all())
        self.assertGreater(float(fused.loc[0, "consensus_score"]), float(fused.loc[1, "consensus_score"]))

    def test_merge_branch_predictions_aligns_on_backbone_id(self) -> None:
        geo = pd.DataFrame({"backbone_id": ["bb1"], "prediction_calibrated": [0.9], "confidence_score": [0.8], "ood_flag": [False], "spread_label": [1]})
        bio = pd.DataFrame({"backbone_id": ["bb1"], "prediction_calibrated": [0.6], "confidence_score": [0.7], "ood_flag": [False], "spread_label": [1]})
        clinical = pd.DataFrame({"backbone_id": ["bb1"], "prediction_calibrated": [0.4], "confidence_score": [0.6], "ood_flag": [False], "spread_label": [1]})
        merged = merge_branch_predictions(geo, bio, clinical)
        self.assertIn("p_geo", merged.columns)
        self.assertIn("p_bio_transfer", merged.columns)
        self.assertIn("p_clinical_hazard", merged.columns)
        self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()
