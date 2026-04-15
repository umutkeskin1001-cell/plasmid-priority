from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plasmid_priority.snapshots import profile_has_content, sync_profile_outputs


class SnapshotTests(unittest.TestCase):
    def test_report_pack_sync_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            source_root = Path(src_tmp)
            (source_root / "analysis").mkdir()
            (source_root / "scores").mkdir()
            (source_root / "silver").mkdir()
            (source_root / "analysis/module_a_metrics.json").write_text("{}", encoding="utf-8")
            (source_root / "scores/backbone_scored.tsv").write_text("x\n1\n", encoding="utf-8")
            (source_root / "silver/plasmid_backbones.tsv").write_text("x\n1\n", encoding="utf-8")
            (source_root / "silver/plasmid_amr_consensus.tsv").write_text(
                "x\n1\n", encoding="utf-8"
            )

            copied = sync_profile_outputs(
                source_root,
                Path(dst_tmp),
                "report-pack",
            )

            self.assertEqual(len(copied), 4)
            self.assertTrue((Path(dst_tmp) / "analysis/module_a_metrics.json").exists())
            self.assertTrue((Path(dst_tmp) / "scores/backbone_scored.tsv").exists())
            self.assertTrue(profile_has_content(Path(dst_tmp), "report-pack"))

    def test_clean_sync_removes_stale_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            source_root = Path(src_tmp)
            (source_root / "analysis").mkdir()
            (source_root / "scores").mkdir()
            (source_root / "silver").mkdir()
            (source_root / "analysis/module_a_metrics.json").write_text("{}", encoding="utf-8")
            (source_root / "scores/backbone_scored.tsv").write_text("x\n1\n", encoding="utf-8")
            (source_root / "silver/plasmid_backbones.tsv").write_text("x\n1\n", encoding="utf-8")
            (source_root / "silver/plasmid_amr_consensus.tsv").write_text(
                "x\n1\n", encoding="utf-8"
            )

            destination_root = Path(dst_tmp)
            stale_path = destination_root / "analysis/stale.tsv"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("stale\n", encoding="utf-8")

            sync_profile_outputs(
                source_root,
                destination_root,
                "report-pack",
                clean_first=True,
            )

            self.assertFalse(stale_path.exists())
            self.assertTrue((destination_root / "analysis/module_a_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
