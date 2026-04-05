from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
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


class _SuccessfulTestResult:
    testsRun = 1
    failures: list[object] = []
    errors: list[object] = []

    def wasSuccessful(self) -> bool:
        return True


class SmokeRunnerTests(unittest.TestCase):
    def test_smoke_skips_real_data_when_required_inputs_are_absent(self) -> None:
        fake_run = _FakeRun()
        report = ValidationReport(
            results=[
                AssetCheckResult(
                    key="plsdb_sequences_fasta",
                    path=str(PROJECT_ROOT / "data/raw/plsdb_sequences.fasta"),
                    status="error",
                    required=True,
                    stage="raw",
                    details=["missing"],
                )
            ],
            contract_notes=[],
        )

        with (
            mock.patch.object(smoke_runner_script, "build_context", return_value=object()),
            mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
            mock.patch.object(
                smoke_runner_script, "run_unit_tests", return_value=_SuccessfulTestResult()
            ),
            mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
            mock.patch.object(smoke_runner_script, "_run_cli_smoke") as cli_smoke_mock,
        ):
            result = smoke_runner_script.main()

        self.assertEqual(result, 0)
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
                mock.patch.object(smoke_runner_script, "build_context", return_value=object()),
                mock.patch.object(smoke_runner_script, "ManagedScriptRun", return_value=fake_run),
                mock.patch.object(
                    smoke_runner_script, "run_unit_tests", return_value=_SuccessfulTestResult()
                ),
                mock.patch.object(smoke_runner_script, "run_input_checks", return_value=report),
                mock.patch.object(smoke_runner_script, "_run_cli_smoke") as cli_smoke_mock,
            ):
                with self.assertRaises(RuntimeError):
                    smoke_runner_script.main()

        cli_smoke_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
