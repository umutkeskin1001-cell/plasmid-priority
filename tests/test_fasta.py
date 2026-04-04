from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plasmid_priority.io.fasta import concatenate_fastas, iter_fasta_summaries


class FastaTests(unittest.TestCase):
    def test_iter_fasta_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fasta_path = Path(tmp_dir) / "small.fasta"
            fasta_path.write_text(">A alpha\nACTG\n>B beta\nAAA\nTT\n", encoding="utf-8")
            records = list(iter_fasta_summaries(fasta_path))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].accession, "A")
            self.assertEqual(records[0].sequence_length, 4)
            self.assertEqual(records[1].sequence_length, 5)

    def test_concatenate_fastas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left = root / "left.fasta"
            right = root / "right.fasta"
            output = root / "combined.fasta"

            left.write_text(">A alpha\nAAAA\n", encoding="utf-8")
            right.write_text(">B beta\nTTTT\n", encoding="utf-8")

            stats = concatenate_fastas([left, right], output, overwrite=False, dry_run=False)
            self.assertEqual(stats["record_count"], 2)
            self.assertTrue(output.exists())
            self.assertIn(">A alpha", output.read_text(encoding="utf-8"))
            self.assertIn(">B beta", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
