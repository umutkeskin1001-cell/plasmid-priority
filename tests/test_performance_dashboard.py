from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_performance_dashboard_script",
    PROJECT_ROOT / "scripts/41_build_performance_dashboard.py",
)
assert SPEC is not None and SPEC.loader is not None
build_performance_dashboard_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_performance_dashboard_script)


def test_build_performance_dashboard_outputs_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        history_path = root / "workflow_profile_history.jsonl"
        rows = [
            {
                "generated_at": "2026-04-21T10:00:00+00:00",
                "budget_mode": "smoke-local",
                "total_duration_seconds": 100.0,
                "git_commit": "a" * 40,
            },
            {
                "generated_at": "2026-04-21T11:00:00+00:00",
                "budget_mode": "smoke-local",
                "total_duration_seconds": 80.0,
                "git_commit": "b" * 40,
            },
        ]
        history_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        output_dir = root / "dash"
        result = build_performance_dashboard_script.main(
            ["--history", str(history_path), "--output-dir", str(output_dir)],
        )
        assert result == 0
        json_path = output_dir / "workflow_performance_dashboard.json"
        md_path = output_dir / "workflow_performance_dashboard.md"
        html_path = output_dir / "workflow_performance_dashboard.html"
        tex_path = output_dir / "workflow_performance_dashboard.tex"
        assert json_path.exists()
        assert md_path.exists()
        assert html_path.exists()
        assert tex_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        summaries = payload.get("summaries", [])
        smoke = [item for item in summaries if item.get("mode") == "smoke-local"][0]
        assert float(smoke["latest_total_duration_seconds"]) == 80.0


def test_build_performance_dashboard_writes_workflow_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        history_path = root / "workflow_profile_history.jsonl"
        history_path.write_text("", encoding="utf-8")
        output_dir = root / "dash"
        data_root = root / "runtime-data"
        with mock.patch.dict(
            os.environ,
            {"PLASMID_PRIORITY_DATA_ROOT": str(data_root)},
            clear=False,
        ):
            result = build_performance_dashboard_script.main(
                ["--history", str(history_path), "--output-dir", str(output_dir)],
            )
        assert result == 0
        summary_path = data_root / "tmp" / "logs" / "41_build_performance_dashboard_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["status"] == "ok"
