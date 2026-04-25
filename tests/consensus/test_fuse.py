from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.consensus.fuse import (
    build_operational_consensus_frame,
    build_research_consensus_frame,
    merge_branch_predictions,
)


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
        self.assertTrue(
            ((fused["consensus_score"] >= 0.0) & (fused["consensus_score"] <= 1.0)).all()
        )
        self.assertGreater(
            float(fused.loc[0, "consensus_score"]), float(fused.loc[1, "consensus_score"])
        )
        self.assertIn("consensus_attenuation", fused.columns)
        self.assertIn("consensus_uncertainty", fused.columns)
        self.assertIn("consensus_score_lower", fused.columns)
        self.assertIn("consensus_score_upper", fused.columns)
        self.assertTrue(
            (
                (fused["consensus_attenuation"] >= 0.0) & (fused["consensus_attenuation"] <= 1.0)
            ).all()
        )
        self.assertTrue((fused["consensus_score_lower"] <= fused["consensus_score"]).all())
        self.assertTrue((fused["consensus_score"] <= fused["consensus_score_upper"]).all())

    def test_disagreement_increases_uncertainty(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "spread_label": [1, 0],
                "p_geo": [0.95, 0.10],
                "p_bio_transfer": [0.94, 0.15],
                "p_clinical_hazard": [0.93, 0.20],
                "confidence_geo": [0.95, 0.95],
                "confidence_bio_transfer": [0.95, 0.95],
                "confidence_clinical_hazard": [0.95, 0.95],
                "ood_geo": [False, False],
                "ood_bio_transfer": [False, False],
                "ood_clinical_hazard": [False, False],
            }
        )
        fused = build_operational_consensus_frame(frame)
        self.assertGreater(float(fused.loc[1, "consensus_uncertainty"]), 0.0)
        self.assertLessEqual(
            float(fused.loc[1, "consensus_score_lower"]), float(fused.loc[1, "consensus_score"])
        )

    def test_merge_branch_predictions_aligns_on_backbone_id(self) -> None:
        geo = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "prediction_calibrated": [0.9],
                "confidence_score": [0.8],
                "ood_flag": [False],
                "spread_label": [1],
            }
        )
        bio = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "prediction_calibrated": [0.6],
                "confidence_score": [0.7],
                "ood_flag": [False],
                "spread_label": [1],
            }
        )
        clinical = pd.DataFrame(
            {
                "backbone_id": ["bb1"],
                "prediction_calibrated": [0.4],
                "confidence_score": [0.6],
                "ood_flag": [False],
                "spread_label": [1],
            }
        )
        merged = merge_branch_predictions(geo, bio, clinical)
        self.assertIn("p_geo", merged.columns)
        self.assertIn("p_bio_transfer", merged.columns)
        self.assertIn("p_clinical_hazard", merged.columns)
        self.assertEqual(len(merged), 1)

    def test_research_consensus_reports_disjoint_fold_score_when_partition_is_possible(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(12)],
                "spread_label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                "p_geo": [0.95, 0.90, 0.86, 0.80, 0.74, 0.70, 0.66, 0.52, 0.43, 0.35, 0.25, 0.15],
                "p_bio_transfer": [
                    0.82,
                    0.79,
                    0.75,
                    0.70,
                    0.65,
                    0.61,
                    0.56,
                    0.49,
                    0.41,
                    0.34,
                    0.24,
                    0.14,
                ],
                "p_clinical_hazard": [
                    0.78,
                    0.74,
                    0.70,
                    0.66,
                    0.60,
                    0.56,
                    0.52,
                    0.45,
                    0.38,
                    0.30,
                    0.22,
                    0.12,
                ],
                "confidence_geo": [0.92] * 12,
                "confidence_bio_transfer": [0.89] * 12,
                "confidence_clinical_hazard": [0.87] * 12,
                "ood_geo": [False] * 12,
                "ood_bio_transfer": [False] * 12,
                "ood_clinical_hazard": [False] * 12,
            }
        )
        fused = build_research_consensus_frame(frame)
        self.assertIn("research_score", fused.columns)
        self.assertIn("research_weight_geo", fused.columns)
        self.assertIn("research_weight_bio_transfer", fused.columns)
        self.assertIn("research_weight_clinical_hazard", fused.columns)
        research_score = float(fused.loc[0, "research_score"])
        self.assertTrue(np.isfinite(research_score))
        self.assertGreaterEqual(research_score, 0.0)
        self.assertLessEqual(research_score, 1.0)
        self.assertAlmostEqual(
            float(
                fused.loc[0, "research_weight_geo"]
                + fused.loc[0, "research_weight_bio_transfer"]
                + fused.loc[0, "research_weight_clinical_hazard"]
            ),
            1.0,
            places=6,
        )

    def test_research_consensus_leaves_score_nan_when_disjoint_partition_is_not_possible(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(6)],
                "spread_label": [1, 1, 1, 0, 0, 0],
                "p_geo": [0.90, 0.86, 0.78, 0.42, 0.34, 0.25],
                "p_bio_transfer": [0.80, 0.75, 0.70, 0.46, 0.38, 0.31],
                "p_clinical_hazard": [0.76, 0.71, 0.64, 0.44, 0.36, 0.28],
                "confidence_geo": [0.9] * 6,
                "confidence_bio_transfer": [0.9] * 6,
                "confidence_clinical_hazard": [0.9] * 6,
                "ood_geo": [False] * 6,
                "ood_bio_transfer": [False] * 6,
                "ood_clinical_hazard": [False] * 6,
            }
        )
        fused = build_research_consensus_frame(frame)
        self.assertTrue(pd.isna(fused.loc[0, "research_score"]))

    def test_research_consensus_without_labels_matches_operational_path(self) -> None:
        frame = pd.DataFrame(
            {
                "backbone_id": [f"bb{i}" for i in range(4)],
                "p_geo": [0.9, 0.7, 0.4, 0.2],
                "p_bio_transfer": [0.8, 0.6, 0.5, 0.3],
                "p_clinical_hazard": [0.85, 0.65, 0.45, 0.25],
                "confidence_geo": [0.9, 0.9, 0.9, 0.9],
                "confidence_bio_transfer": [0.9, 0.9, 0.9, 0.9],
                "confidence_clinical_hazard": [0.9, 0.9, 0.9, 0.9],
                "ood_geo": [False, False, False, False],
                "ood_bio_transfer": [False, False, False, False],
                "ood_clinical_hazard": [False, False, False, False],
            }
        )
        research = build_research_consensus_frame(frame)
        operational = build_operational_consensus_frame(frame)
        pd.testing.assert_series_equal(
            research["consensus_score"].reset_index(drop=True),
            operational["consensus_score"].reset_index(drop=True),
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
