"""Tests for report/figure incremental cache helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from plasmid_priority.reporting.cache import ReportCache, frame_fingerprint


def test_frame_fingerprint_changes_on_value_change() -> None:
    frame_a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    frame_b = pd.DataFrame({"x": [1, 9], "y": ["a", "b"]})
    assert frame_fingerprint(frame_a) != frame_fingerprint(frame_b)


def test_report_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cache = ReportCache(root / "cache")
        output = root / "report.md"
        output.write_text("ok", encoding="utf-8")

        key = cache.build_report_key(
            report_name="24_build_reports",
            input_hashes=[{"path": "a", "size": 1, "mtime_ns": 1}],
            config_hash="cfg",
            protocol_hash="protocol",
            code_hash="code",
            mode="report-full",
        )
        assert not cache.is_report_current(
            report_name="24_build_reports",
            report_key=key,
            outputs=[output],
        )
        cache.put_report(report_name="24_build_reports", report_key=key, outputs=[output])
        assert cache.is_report_current(
            report_name="24_build_reports",
            report_key=key,
            outputs=[output],
        )


def test_figure_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cache = ReportCache(root / "cache")
        output = root / "plot.png"
        output.write_bytes(b"png-bytes")

        key = cache.build_figure_key(
            figure_name="roc_curve",
            data_fingerprint="dfp",
            function_hash="fn",
            mode="report-diff",
        )
        assert not cache.is_figure_current(
            figure_name="roc_curve",
            figure_key=key,
            output_path=output,
        )
        cache.put_figure(figure_name="roc_curve", figure_key=key, output_path=output)
        assert cache.is_figure_current(
            figure_name="roc_curve",
            figure_key=key,
            output_path=output,
        )
