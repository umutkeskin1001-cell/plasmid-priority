from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plasmid_priority.config import DATA_ROOT_ENV_VAR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_mode_script",
    PROJECT_ROOT / "scripts/run_mode.py",
)
assert SPEC is not None and SPEC.loader is not None
run_mode_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_mode_script
SPEC.loader.exec_module(run_mode_script)


class RunModeTests(unittest.TestCase):
    def test_full_local_prompts_for_data_root_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.object(run_mode_script, "prompt_for_data_root", return_value=str(data_root)),
                mock.patch.object(run_mode_script.subprocess, "run", return_value=completed) as run_mock,
            ):
                result = run_mode_script.main(["full-local", "--dry-run"])
            self.assertEqual(result, 0)
            self.assertEqual(
                run_mock.call_args.kwargs["env"][DATA_ROOT_ENV_VAR],
                str(data_root.resolve()),
            )

    def test_full_local_rejects_missing_data_root_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing"
            with self.assertRaises(FileNotFoundError):
                run_mode_script.main(["full-local", "--data-root", str(missing), "--dry-run"])

    def test_fast_local_uses_cache_root_by_default(self) -> None:
        completed = mock.Mock(returncode=0)
        cache_root = Path("/tmp/test-home/.cache/plasmid-priority/fast-local/data")
        with (
            mock.patch.object(run_mode_script.subprocess, "run", return_value=completed) as run_mock,
            mock.patch.object(run_mode_script, "resolve_mode_data_root", return_value=cache_root),
            mock.patch.object(run_mode_script, "profile_has_content", return_value=True),
        ):
            result = run_mode_script.main(["fast-local", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertEqual(
            run_mock.call_args.kwargs["env"][DATA_ROOT_ENV_VAR],
            str(cache_root.resolve()),
        )

    def test_invalid_workflow_for_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_mode_script.main(["fast-local", "--workflow", "pipeline", "--dry-run"])

    def test_fast_local_rejects_release_workflow(self) -> None:
        with self.assertRaises(ValueError):
            run_mode_script.main(["fast-local", "--workflow", "release", "--dry-run"])

    def test_fast_local_syncs_report_pack_before_workflow(self) -> None:
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir)
            with (
                mock.patch.object(
                    run_mode_script,
                    "resolve_mode_data_root",
                    return_value=Path("/tmp/cache-data"),
                ),
                mock.patch.object(run_mode_script, "sync_profile_outputs") as sync_mock,
                mock.patch.object(run_mode_script.subprocess, "run", return_value=completed),
            ):
                result = run_mode_script.main(
                    ["fast-local", "--source-data-root", str(source_root), "--dry-run"]
                )
        self.assertEqual(result, 0)
        self.assertEqual(sync_mock.call_args.args[2], "report-pack")

    def test_fast_local_requires_seeded_cache_without_source_data_root(self) -> None:
        with (
            mock.patch.object(run_mode_script, "resolve_mode_data_root", return_value=Path("/tmp/cache-data")),
            mock.patch.object(run_mode_script, "profile_has_content", return_value=False),
        ):
            with self.assertRaises(FileNotFoundError):
                run_mode_script.main(["fast-local", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
