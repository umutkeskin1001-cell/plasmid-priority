from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.reporting.enrichment import (
    build_candidate_signature_context,
    build_module_f_enrichment_table,
    build_module_f_top_hits,
)


class EnrichmentTests(unittest.TestCase):
    def test_build_module_f_tables_return_significant_rows(self) -> None:
        identity = pd.DataFrame(
            {
                "backbone_id": [f"bb_{i}" for i in range(24)],
                "spread_label": [1] * 12 + [0] * 12,
                "dominant_genus": ["Klebsiella"] * 10 + ["Escherichia"] * 14,
                "primary_replicon": ["IncHI2A"] * 8 + ["IncFIB"] * 16,
                "dominant_mpf_type": ["MPF_F"] * 12 + ["MPF_T"] * 12,
                "dominant_amr_gene_family": ["BLACTXM"] * 9 + ["SUL"] * 15,
                "amr_class_tokens": ["BETA-LACTAM,AMINOGLYCOSIDE"] * 10 + ["TETRACYCLINE"] * 14,
            }
        )
        enrichment = build_module_f_enrichment_table(identity, min_backbones=5)
        top_hits = build_module_f_top_hits(
            enrichment, q_threshold=1.0, max_per_group=2, max_total=10
        )
        self.assertFalse(enrichment.empty)
        self.assertIn("q_value", enrichment.columns)
        self.assertFalse(top_hits.empty)
        self.assertIn("log2_odds_ratio", top_hits.columns)

    def test_build_candidate_signature_context_matches_top_hits(self) -> None:
        candidates = pd.DataFrame({"backbone_id": ["bb1", "bb2"]})
        identity = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "dominant_genus": ["Klebsiella", "Escherichia"],
                "primary_replicon": ["IncHI2A", "IncFIB"],
                "dominant_mpf_type": ["MPF_F", "MPF_T"],
                "dominant_amr_gene_family": ["BLACTXM", "SUL"],
                "amr_class_tokens": ["BETA-LACTAM", "TETRACYCLINE"],
            }
        )
        enrichment = pd.DataFrame(
            {
                "feature_group": ["dominant_genus", "primary_replicon", "amr_class"],
                "feature_value": ["Klebsiella", "IncHI2A", "BETA-LACTAM"],
                "odds_ratio": [4.0, 5.0, 6.0],
                "q_value": [0.01, 0.02, 0.03],
                "enriched_in_positive": [True, True, True],
            }
        )
        context = build_candidate_signature_context(
            candidates, identity, enrichment, q_threshold=0.05
        )
        self.assertEqual(len(context), 2)
        self.assertIn("module_f_enriched_signatures", context.columns)
        bb1 = context.loc[context["backbone_id"] == "bb1"].iloc[0]
        self.assertGreater(int(bb1["module_f_enriched_signature_count"]), 0)


if __name__ == "__main__":
    unittest.main()
