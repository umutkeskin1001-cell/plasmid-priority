from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_run_branch_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_branch_script", PROJECT_ROOT / "scripts" / "run_branch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_branch_cli_forwards_arguments(monkeypatch: Any) -> None:
    module = _load_run_branch_module()
    captured: dict[str, object] = {}

    def fake_run_branch(branch_name: str, *, branch_args: list[str]) -> int:
        captured["branch_name"] = branch_name
        captured["branch_args"] = list(branch_args)
        return 7

    monkeypatch.setattr(module, "run_branch", fake_run_branch)

    rc = module.main(["--branch", "geo_spread", "--", "--jobs", "4"])

    assert rc == 7
    assert captured == {"branch_name": "geo_spread", "branch_args": ["--jobs", "4"]}
