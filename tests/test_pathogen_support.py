from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.reporting.pathogen_support import (
    build_pathogen_group_comparison,
    extract_pd_gene_symbols,
    normalize_species_token,
)


class PathogenSupportTests(unittest.TestCase):
    def test_normalize_species_token(self) -> None:
        self.assertEqual(normalize_species_token("Escherichia coli"), "Escherichia_coli")
        self.assertEqual(normalize_species_token(""), "")

    def test_extract_pd_gene_symbols(self) -> None:
        genes = extract_pd_gene_symbols("blaTEM=COMPLETE,qnrS1=HMM")
        self.assertEqual(genes, {"blaTEM", "qnrS1"})

    def test_build_pathogen_group_comparison_returns_effect_sizes(self) -> None:
        detail = pd.DataFrame(
            {
                "pathogen_dataset": ["combined"] * 6,
                "priority_group": ["high", "high", "high", "low", "low", "low"],
                "pd_matching_fraction": [0.2, 0.3, 0.1, 0.05, 0.08, 0.02],
                "pd_any_support": [True, True, False, True, False, False],
            }
        )
        comparison = build_pathogen_group_comparison(detail, n_permutations=50, seed=7)
        self.assertEqual(comparison.iloc[0]["pathogen_dataset"], "combined")
        self.assertIn("delta_mean_matching_fraction_high_minus_low", comparison.columns)
        self.assertIn("permutation_p_support_fraction", comparison.columns)


if __name__ == "__main__":
    unittest.main()
