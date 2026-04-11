from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from plasmid_priority.qc import ValidationReport
from plasmid_priority.qc.input_checks import AssetCheckResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smoke_runner_script",
    PROJECT_ROOT / "scripts/26_run_tests_or_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
smoke_runner_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke_runner_script
SPEC.loader.exec_module(smoke_runner_script)


class _FakeRun:
    def __init__(self) -> None:
        self.metrics: dict[str, int] = {}
        self.warnings: list[str] = []

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_metric(self, key: str, value: int) -> None:
        self.metrics[key] = value

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def set_rows_in(self, key: str, value: int) -> None:
        self.metrics[key] = value


@dataclass
class _SuccessfulTestResult:
    tests_run: int = 1
    test_failures: int = 0
    test_errors: int = 0
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    def was_successful(self) -> bool:
        return True


class _FakeContext:
    def asset_path(self, key: str) -> Path:
        return Path("/tmp") / key


class SmokeRunnerTests(unittest.TestCase):
    def test_smoke_does_not_run_unit_tests_by_default(self) -> None:
        fake_run = _FakeRun()
        report = ValidationReport(results=[], contract_notes=[])

        with (
            mock.patch.object(smoke_runner_script, "build_context", return_value=_FakeContext()),
            mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
            mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
            mock.patch.object(
                smoke_runner_script,
                "_run_cli_smoke",
                return_value=mock.Mock(returncode=0),
            ),
            mock.patch.object(
                smoke_runner_script,
                "build_plsdb_canonical_metadata",
                return_value=[],
            ),
            mock.patch.object(
                smoke_runner_script,
                "iter_fasta_summaries",
                return_value=iter([("h1", "", 1), ("h2", "", 1)]),
            ),
            mock.patch.object(smoke_runner_script, "run_unit_tests") as run_tests_mock,
        ):
            result = smoke_runner_script.main([])

        self.assertEqual(result, 0)
        self.assertEqual(fake_run.metrics["tests_run"], 0)
        run_tests_mock.assert_not_called()

    def test_smoke_can_run_unit_tests_when_requested(self) -> None:
        fake_run = _FakeRun()
        report = ValidationReport(results=[], contract_notes=[])

        with (
            mock.patch.object(smoke_runner_script, "build_context", return_value=_FakeContext()),
            mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
            mock.patch.object(
                smoke_runner_script, "run_unit_tests", return_value=_SuccessfulTestResult()
            ) as run_tests_mock,
            mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
            mock.patch.object(
                smoke_runner_script,
                "_run_cli_smoke",
                return_value=mock.Mock(returncode=0),
            ),
            mock.patch.object(
                smoke_runner_script,
                "build_plsdb_canonical_metadata",
                return_value=[],
            ),
            mock.patch.object(
                smoke_runner_script,
                "iter_fasta_summaries",
                return_value=iter([("h1", "", 1), ("h2", "", 1)]),
            ),
        ):
            result = smoke_runner_script.main(["--with-tests"])

        self.assertEqual(result, 0)
        self.assertEqual(fake_run.metrics["tests_run"], 1)
        run_tests_mock.assert_called_once()

    def test_smoke_skips_real_data_when_required_inputs_are_absent(self) -> None:
        fake_run = _FakeRun()
        missing_path = Path(tempfile.mkdtemp()) / "missing_plsdb_sequences.fasta"
        report = ValidationReport(
            results=[
                AssetCheckResult(
                    key="plsdb_sequences_fasta",
                    path=str(missing_path),
                    status="error",
                    required=True,
                    stage="raw",
                    details=["missing"],
                )
            ],
            contract_notes=[],
        )

        with (
            mock.patch.object(smoke_runner_script, "build_context", return_value=_FakeContext()),
            mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
            mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
            mock.patch.object(smoke_runner_script, "_run_cli_smoke") as cli_smoke_mock,
        ):
            with self.assertRaises(RuntimeError):
                smoke_runner_script.main([])

        self.assertEqual(fake_run.metrics["smoke_skipped_missing_required_inputs"], 1)
        self.assertTrue(fake_run.warnings)
        cli_smoke_mock.assert_not_called()

    def test_smoke_skip_is_allowed_when_tests_have_run(self) -> None:
        fake_run = _FakeRun()
        missing_path = Path(tempfile.mkdtemp()) / "missing_plsdb_sequences.fasta"
        report = ValidationReport(
            results=[
                AssetCheckResult(
                    key="plsdb_sequences_fasta",
                    path=str(missing_path),
                    status="error",
                    required=True,
                    stage="raw",
                    details=["missing"],
                )
            ],
            contract_notes=[],
        )

        with (
            mock.patch.object(smoke_runner_script, "build_context", return_value=_FakeContext()),
            mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
            mock.patch.object(
                smoke_runner_script, "run_unit_tests", return_value=_SuccessfulTestResult()
            ),
            mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
            mock.patch.object(smoke_runner_script, "_run_cli_smoke") as cli_smoke_mock,
        ):
            result = smoke_runner_script.main(["--with-tests"])

        self.assertEqual(result, 0)
        self.assertEqual(fake_run.metrics["tests_run"], 1)
        self.assertEqual(fake_run.metrics["smoke_skipped_missing_required_inputs"], 1)
        self.assertTrue(fake_run.warnings)
        cli_smoke_mock.assert_not_called()

    def test_smoke_fails_when_present_inputs_are_invalid(self) -> None:
        fake_run = _FakeRun()
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_path = Path(tmp_dir) / "present.tsv"
            existing_path.write_text("bad", encoding="utf-8")
            report = ValidationReport(
                results=[
                    AssetCheckResult(
                        key="pathogen_detection_metadata",
                        path=str(existing_path),
                        status="error",
                        required=True,
                        stage="external",
                        details=["invalid columns"],
                    )
                ],
                contract_notes=[],
            )

            with (
                mock.patch.object(smoke_runner_script, "build_context", return_value=_FakeContext()),
                mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
                mock.patch.object(
                    smoke_runner_script, "run_unit_tests", return_value=_SuccessfulTestResult()
                ),
                mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
                mock.patch.object(smoke_runner_script, "_run_cli_smoke") as cli_smoke_mock,
            ):
                with self.assertRaises(RuntimeError):
                    smoke_runner_script.main([])

        cli_smoke_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
