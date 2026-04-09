from __future__ import annotations

import os
import tempfile
from pathlib import Path

from plasmid_priority import runtime


def test_resolve_mode_workflow_uses_default_when_unspecified() -> None:
    assert runtime.resolve_mode_workflow("fast-local") == "reports-only"
    assert runtime.resolve_mode_workflow("full-local") == "pipeline"


def test_resolve_mode_workflow_rejects_invalid_explicit_workflow() -> None:
    try:
        runtime.resolve_mode_workflow("fast-local", "pipeline")
    except ValueError as exc:
        assert "Allowed" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected workflow validation to fail")


def test_resolve_mode_data_root_relative_path_uses_repo_root() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        relative = "tmp/external-data"
        expected = (runtime._RUNTIME_PROJECT_ROOT / relative).resolve()
        original_cwd = Path.cwd()
        try:
            # Ensure resolution does not depend on caller CWD.
            os.chdir(tmp_dir)
            resolved = runtime.resolve_mode_data_root("full-local", relative)
        finally:
            os.chdir(original_cwd)
        assert resolved == expected
