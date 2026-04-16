from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from plasmid_priority.reporting.figures import (
    _pretty_model_label,
    plot_candidate_stability,
    plot_score_distribution,
)


class FigureSmokeTests(unittest.TestCase):
    def test_pretty_model_label_covers_published_primary_model(self) -> None:
        self.assertEqual(
            _pretty_model_label("parsimonious_priority"), "parsimonious priority model"
        )

    def test_plot_score_distribution_writes_png(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "priority_index": [0.05, 0.08, 0.2, 0.3, 0.55, 0.72, 0.81, 0.93],
                "member_count_train": [0, 0, 1, 2, 3, 4, 5, 6],
                "spread_label": [None, None, 0, 1, 0, 1, 1, 0],
                "bio_priority_index": [0.1, 0.15, 0.2, 0.35, 0.4, 0.6, 0.7, 0.8],
                "evidence_support_index": [0.05, 0.1, 0.18, 0.28, 0.45, 0.62, 0.74, 0.86],
                "T_eff_norm": [0.01, 0.05, 0.1, 0.2, 0.6, 0.8, 0.7, 0.9],
                "H_eff_norm": [0.2, 0.1, 0.2, 0.4, 0.7, 0.8, 0.65, 0.88],
                "A_eff_norm": [0.05, 0.04, 0.08, 0.3, 0.55, 0.75, 0.72, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "score_distribution.png"
            plot_score_distribution(scored, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_candidate_stability_writes_png(self) -> None:
        candidates = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(12)],
                "base_rank": list(range(1, 13)),
                "priority_index": [1 - 0.03 * i for i in range(12)],
                "bootstrap_top_k_frequency": [1.0] * 12,
                "bootstrap_top_10_frequency": [
                    0.95,
                    0.94,
                    0.92,
                    0.9,
                    0.88,
                    0.86,
                    0.83,
                    0.8,
                    0.77,
                    0.7,
                    0.25,
                    0.05,
                ],
                "bootstrap_top_25_frequency": [1.0] * 12,
                "variant_top_k_frequency": [0.75] * 12,
                "variant_top_10_frequency": [0.75] * 10 + [0.0, 0.0],
                "variant_top_25_frequency": [0.75] * 12,
                "bootstrap_mean_rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
                "bootstrap_rank_std": [1.2] * 12,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "candidate_stability.png"
            plot_candidate_stability(candidates, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_score_distribution_emits_no_layout_warning(self) -> None:
        scored = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(8)],
                "priority_index": [0.05, 0.08, 0.2, 0.3, 0.55, 0.72, 0.81, 0.93],
                "member_count_train": [0, 0, 1, 2, 3, 4, 5, 6],
                "spread_label": [None, None, 0, 1, 0, 1, 1, 0],
                "bio_priority_index": [0.1, 0.15, 0.2, 0.35, 0.4, 0.6, 0.7, 0.8],
                "evidence_support_index": [0.05, 0.1, 0.18, 0.28, 0.45, 0.62, 0.74, 0.86],
                "T_eff_norm": [0.01, 0.05, 0.1, 0.2, 0.6, 0.8, 0.7, 0.9],
                "H_eff_norm": [0.2, 0.1, 0.2, 0.4, 0.7, 0.8, 0.65, 0.88],
                "A_eff_norm": [0.05, 0.04, 0.08, 0.3, 0.55, 0.75, 0.72, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "score_distribution_warning_free.png"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plot_score_distribution(scored, output)
            self.assertTrue(output.exists())
            self.assertFalse(
                any("Tight layout not applied" in str(item.message) for item in caught)
            )


if __name__ == "__main__":
    unittest.main()
