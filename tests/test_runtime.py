from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from plasmid_priority import runtime
from plasmid_priority.settings import get_settings


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


def test_resolve_mode_data_root_uses_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_DATA_ROOT", "tmp/configured-data-root")
    get_settings.cache_clear()
    try:
        resolved = runtime.resolve_mode_data_root("fast-local")
    finally:
        get_settings.cache_clear()
    expected = (runtime._RUNTIME_PROJECT_ROOT / "tmp/configured-data-root").resolve()
    assert resolved == expected


def test_resolve_mode_data_root_full_local_raises_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLASMID_PRIORITY_DATA_ROOT", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="PLASMID_PRIORITY_DATA_ROOT"):
            runtime.resolve_mode_data_root("full-local")
    finally:
        get_settings.cache_clear()
