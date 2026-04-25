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

from plasmid_priority.utils.dataframe import (
    clean_text_series,
    clear_read_tsv_cache,
    read_parquet,
    read_tsv,
)


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

    def test_clean_text_series_basic(self) -> None:
        """Test basic text cleaning."""
        s = pd.Series(["  Hello  ", "  World  ", "  Test  "])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "Hello")
        self.assertEqual(result.iloc[1], "World")
        self.assertEqual(result.iloc[2], "Test")

    def test_clean_text_series_with_nan(self) -> None:
        """Test text cleaning with NaN values."""
        s = pd.Series(["  Hello  ", None, "  World  "])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "Hello")
        # clean_text_series NaN'ları boş string'e çeviriyor
        self.assertEqual(result.iloc[1], "")
        self.assertEqual(result.iloc[2], "World")

    def test_clean_text_series_special_chars(self) -> None:
        """Test text cleaning with special characters."""
        s = pd.Series(["Hello-World", "Test_Case", "UPPER lower"])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "Hello-World")
        self.assertEqual(result.iloc[1], "Test_Case")
        self.assertEqual(result.iloc[2], "UPPER lower")

    def test_clean_text_series_numbers(self) -> None:
        """Test text cleaning with numbers."""
        s = pd.Series(["Test123", "456Test", "Test 789"])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "Test123")
        self.assertEqual(result.iloc[1], "456Test")
        self.assertEqual(result.iloc[2], "Test 789")

    def test_clean_text_series_empty(self) -> None:
        """Test text cleaning with empty strings."""
        s = pd.Series(["", "  ", "Hello"])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "")
        self.assertEqual(result.iloc[1], "")
        self.assertEqual(result.iloc[2], "Hello")

    def test_clean_text_series_unicode(self) -> None:
        """Test text cleaning with unicode characters."""
        s = pd.Series(["Héllo Wörld", "Tëst Çase"])
        result = clean_text_series(s)
        self.assertEqual(result.iloc[0], "Héllo Wörld")
        self.assertEqual(result.iloc[1], "Tëst Çase")


if __name__ == "__main__":
    unittest.main()
