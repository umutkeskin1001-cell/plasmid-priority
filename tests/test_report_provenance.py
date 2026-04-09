from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plasmid_priority.protocol import ScientificProtocol
from plasmid_priority.reporting.provenance import (
    build_artifact_provenance,
    load_provenance_json,
    provenance_matches_current,
    validate_provenance_record,
    write_provenance_json,
)


class ReportProvenanceTests(unittest.TestCase):
    def test_artifact_provenance_round_trip_and_freshness_gate(self) -> None:
        protocol = ScientificProtocol(
            split_year=2015,
            min_new_countries_for_spread=3,
            primary_model_name="bio_clean_priority",
            primary_model_fallback="parsimonious_priority",
            conservative_model_name="parsimonious_priority",
            governance_model_name="phylo_support_fusion_priority",
            governance_model_fallback="support_synergy_priority",
            core_model_names=(
                "bio_clean_priority",
                "parsimonious_priority",
                "phylo_support_fusion_priority",
                "baseline_both",
            ),
            research_model_names=(),
            ablation_model_names=(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.tsv"
            source_path = root / "script.py"
            input_path.write_text("a\n", encoding="utf-8")
            source_path.write_text("print('x')\n", encoding="utf-8")

            provenance = build_artifact_provenance(
                protocol=protocol,
                artifact_family="report_surface",
                input_paths=[input_path],
                source_paths=[source_path],
                generated_at="2026-04-08T00:00:00+00:00",
            )
            validate_provenance_record(provenance, artifact_name="report_provenance")
            path = root / "report_provenance.json"
            write_provenance_json(path, provenance)
            loaded = load_provenance_json(path)

            self.assertEqual(loaded["artifact_family"], "report_surface")
            self.assertTrue(
                provenance_matches_current(
                    loaded,
                    protocol=protocol,
                    input_paths=[input_path],
                    source_paths=[source_path],
                )
            )

            input_path.write_text("changed\n", encoding="utf-8")
            self.assertFalse(
                provenance_matches_current(
                    loaded,
                    protocol=protocol,
                    input_paths=[input_path],
                    source_paths=[source_path],
                )
            )

    def test_provenance_validation_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_provenance_record({}, artifact_name="broken_provenance")


if __name__ == "__main__":
    unittest.main()
