from __future__ import annotations

import os
import tempfile
from pathlib import Path

from plasmid_priority import runtime


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
