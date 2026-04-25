from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_reports_shim",
    PROJECT_ROOT / "scripts/24_build_reports.py",
)
assert SPEC is not None and SPEC.loader is not None
build_reports_shim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_reports_shim
SPEC.loader.exec_module(build_reports_shim)


def test_iter_analysis_backed_exports_routes_summary_outputs_to_analysis_sources(
    tmp_path: Path,
) -> None:
    (tmp_path / "data/analysis").mkdir(parents=True)
    (tmp_path / "data/tmp/logs").mkdir(parents=True)
    source = tmp_path / "data/analysis/model_comparison_summary.tsv"
    source.write_text("x\n1\n", encoding="utf-8")
    summary = {
        "output_files_written": [
            "reports/core_tables/model_comparison_summary.tsv",
            "reports/jury_brief.md",
        ]
    }
    (tmp_path / "data/tmp/logs/24_build_reports_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    exports = build_reports_shim._iter_analysis_backed_exports(tmp_path)

    assert exports == [
        (source, tmp_path / "reports/core_tables/model_comparison_summary.tsv"),
    ]


def test_sync_analysis_backed_exports_materializes_report_copies(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "data/analysis"
    logs_dir = project_root / "data/tmp/logs"
    source_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    source = source_dir / "rolling_temporal_validation.tsv"
    source.write_text("score\n0.8\n", encoding="utf-8")
    summary = {
        "output_files_written": [
            "reports/diagnostic_tables/rolling_temporal_validation.tsv",
        ]
    }
    (logs_dir / "24_build_reports_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    build_reports_shim._iter_analysis_backed_exports(project_root)
    with_source = build_reports_shim._iter_analysis_backed_exports(project_root)
    assert with_source == [
        (
            source,
            project_root / "reports/diagnostic_tables/rolling_temporal_validation.tsv",
        )
    ]

    original_file = build_reports_shim.__file__
    build_reports_shim.__file__ = str(project_root / "scripts/24_build_reports.py")
    try:
        build_reports_shim._sync_analysis_backed_exports()
    finally:
        build_reports_shim.__file__ = original_file

    destination = project_root / "reports/diagnostic_tables/rolling_temporal_validation.tsv"
    assert destination.read_text(encoding="utf-8") == "score\n0.8\n"
    assert destination.stat().st_mtime >= source.stat().st_mtime
