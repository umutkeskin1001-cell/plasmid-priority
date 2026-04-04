from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plasmid_priority.io.tabular import peek_table_columns, read_ncbi_assembly_summary_columns


class TabularTests(unittest.TestCase):
    def test_peek_table_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "table.tsv"
            path.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
            self.assertEqual(peek_table_columns(path, delimiter="\t"), ["a", "b", "c"])

    def test_read_ncbi_assembly_summary_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "assembly.txt"
            path.write_text(
                "## comment\n#assembly_accession\tbioproject\tseq_rel_date\n"
                "GCF_1\tPRJ\t2024-01-01\n",
                encoding="utf-8",
            )
            columns = read_ncbi_assembly_summary_columns(path)
            self.assertEqual(columns, ["assembly_accession", "bioproject", "seq_rel_date"])


if __name__ == "__main__":
    unittest.main()
