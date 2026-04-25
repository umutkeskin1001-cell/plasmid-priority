from __future__ import annotations

import json
from pathlib import Path

from plasmid_priority.performance.trend_analysis import analyze_workflow_trends


def test_analyze_workflow_trends_handles_missing_history(tmp_path: Path) -> None:
    report = analyze_workflow_trends(tmp_path / "missing.jsonl")
    assert report.n_runs == 0
    assert report.steps == []
    assert report.overall_regression is False
    assert report.cache_summary.total_steps == 0


def test_analyze_workflow_trends_detects_step_regression(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    rows = [
        {
            "mode": "release-full",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "steps": [
                {"step_name": "train", "duration_seconds": 10.0, "cache_status": "miss"},
                {"step_name": "report", "duration_seconds": 4.0, "cache_status": "hit"},
            ],
        },
        {
            "mode": "release-full",
            "generated_at": "2026-01-02T00:00:00+00:00",
            "steps": [
                {"step_name": "train", "duration_seconds": 11.0, "cache_status": "miss"},
                {"step_name": "report", "duration_seconds": 4.2, "cache_status": "hit"},
            ],
        },
        {
            "mode": "release-full",
            "generated_at": "2026-01-03T00:00:00+00:00",
            "steps": [
                {"step_name": "train", "duration_seconds": 40.0, "cache_status": "miss"},
                {"step_name": "report", "duration_seconds": 4.1, "cache_status": "hit"},
            ],
        },
    ]
    history_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = analyze_workflow_trends(history_path, mode="release-full", regression_threshold=2.0)
    by_name = {step.step_name: step for step in report.steps}
    assert report.n_runs == 3
    assert report.overall_regression is True
    assert by_name["train"].regression_detected is True
    assert by_name["train"].regression_zscore is not None
    assert by_name["report"].regression_detected is False
    assert report.cache_summary.cache_hits == 3
    assert report.cache_summary.cache_misses == 3
