from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plasmid_priority.config import build_context


class ScientificProtocolTests(unittest.TestCase):
    def test_repo_context_exposes_the_canonical_protocol(self) -> None:
        context = build_context()
        protocol = context.protocol

        self.assertEqual(protocol.split_year, 2015)
        self.assertEqual(protocol.min_new_countries_for_spread, 3)
        self.assertEqual(protocol.primary_model_name, "discovery_12f_source")
        self.assertEqual(protocol.governance_model_name, "phylo_support_fusion_priority")
        self.assertEqual(
            protocol.official_model_names,
            ("discovery_12f_source", "phylo_support_fusion_priority", "baseline_both"),
        )
        self.assertEqual(protocol.acceptance_thresholds["selection_adjusted_p_max"], 0.01)
        self.assertEqual(protocol.outcome_definition["split_year"], 2015)
        self.assertEqual(protocol.outcome_definition["min_new_countries_for_spread"], 3)

    def test_protocol_rejects_primary_model_when_it_is_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "config.yaml").write_text(
                """
pipeline:
  split_year: 2015
  min_new_countries_for_spread: 3
models:
  primary_model_name: discovery_12f_source
  primary_model_fallback: parsimonious_priority
  conservative_model_name: parsimonious_priority
  governance_model_name: phylo_support_fusion_priority
  governance_model_fallback: support_synergy_priority
  core_model_names:
    - baseline_both
    - parsimonious_priority
    - phylo_support_fusion_priority
  research_model_names:
    - discovery_12f_source
  ablation_model_names: []
""".strip(),
                encoding="utf-8",
            )
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [],
                    }
                ),
                encoding="utf-8",
            )

            context = build_context(root)
            with self.assertRaises(ValueError):
                _ = context.protocol


if __name__ == "__main__":
    unittest.main()
