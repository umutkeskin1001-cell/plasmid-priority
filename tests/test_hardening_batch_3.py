from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, script_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "scripts" / script_name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_hardening_summary_writes_json_when_markdown_enabled(
    tmp_path: Path,
    capsys: Any,
) -> None:
    module = _load_script("run_hardening_summary_batch3", "run_hardening_summary.py")
    backbone_path = tmp_path / "backbone.tsv"
    scored_path = tmp_path / "scored.tsv"
    output_json = tmp_path / "summary.json"
    backbone_df = pd.DataFrame({"backbone_id": ["bb1"]})
    scored_df = pd.DataFrame({"backbone_id": ["bb1"], "spread_label": [1]})
    backbone_df.to_csv(backbone_path, sep="\t", index=False)
    scored_df.to_csv(scored_path, sep="\t", index=False)

    summary = {
        "overall_status": "ok",
        "tables_audited": ["backbone_table", "scored_backbone_table"],
    }
    with mock.patch.object(module, "build_hardening_audit_summary", return_value=summary):
        rc = module.main(
            [
                "--backbone",
                str(backbone_path),
                "--scored",
                str(scored_path),
                "--markdown",
                "--output-json",
                str(output_json),
            ]
        )

    captured = capsys.readouterr()
    assert rc == 0
    assert output_json.exists()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["overall_status"] == "ok"
    assert "Hardening Audit Summary" in captured.out


def test_hardening_snapshot_marks_skipped_audits_as_incomplete_when_data_exists() -> None:
    module = _load_script("generate_hardening_snapshot_batch3", "generate_hardening_snapshot.py")
    with (
        mock.patch.object(
            module,
            "run_data_audits_if_available",
            return_value={
                "epv_audit": {"status": "skipped"},
                "lead_time_bias_audit": {"status": "ok"},
                "missingness_audit": {"status": "ok"},
                "schema_validation": {"status": "ok"},
            },
        ),
        mock.patch.object(
            module,
            "check_data_dependent_audits",
            return_value={
                "data_files": {
                    "backbone_table": True,
                    "scored_backbone": True,
                    "harmonized_plasmids": True,
                    "deduplicated_plasmids": True,
                },
                "any_data_available": True,
                "status": "available",
            },
        ),
    ):
        snapshot = module.build_hardening_snapshot()

    assert snapshot["overall_status"] == "incomplete"
