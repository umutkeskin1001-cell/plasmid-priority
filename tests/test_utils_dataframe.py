from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - fallback in non-venv environments
    duckdb = None

from plasmid_priority.utils.dataframe import clear_read_tsv_cache, read_parquet, read_tsv


class DataFrameUtilsTests(unittest.TestCase):
    def test_read_tsv_reuses_cached_parse_for_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.tsv"
            path.write_text("a\tb\n1\t2\n", encoding="utf-8")
            clear_read_tsv_cache()
            with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as read_csv_mock:
                first = read_tsv(path)
                second = read_tsv(path)
            self.assertEqual(read_csv_mock.call_count, 1)
            self.assertTrue(first.equals(second))

    def test_read_parquet_prefers_duckdb_when_available(self) -> None:
        if duckdb is None:
            self.skipTest("duckdb is not available in this environment")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.parquet"
            frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            frame.to_parquet(path, index=False)
            with mock.patch.object(duckdb, "connect", wraps=duckdb.connect) as connect_mock:
                loaded = read_parquet(path)
            self.assertEqual(connect_mock.call_count, 1)
            pd.testing.assert_frame_equal(loaded, frame)


if __name__ == "__main__":
    unittest.main()
