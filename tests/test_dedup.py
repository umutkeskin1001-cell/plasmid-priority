from __future__ import annotations

import unittest

import pandas as pd

from plasmid_priority.dedup import annotate_canonical_ids


class DedupTests(unittest.TestCase):
    def test_annotate_canonical_ids(self) -> None:
        records = pd.DataFrame(
            {
                "sequence_accession": ["B", "A", "C", "D"],
                "other": [1, 2, 3, 4],
            }
        )
        identical = pd.DataFrame(
            {
                "NUCCORE_ACC": ["B", "C"],
                "NUCCORE_Identical": ["A", "B"],
            }
        )
        annotated = annotate_canonical_ids(records, identical)
        lookup = dict(zip(annotated["sequence_accession"], annotated["canonical_id"]))
        self.assertEqual(lookup["A"], "A")
        self.assertEqual(lookup["B"], "A")
        self.assertEqual(lookup["C"], "A")
        self.assertEqual(lookup["D"], "D")


if __name__ == "__main__":
    unittest.main()
