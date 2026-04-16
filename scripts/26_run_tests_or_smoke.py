#!/usr/bin/env python3
"""Run a lightweight smoke check on real project data, optionally with unit tests."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.harmonize import build_plsdb_canonical_metadata
from plasmid_priority.io.fasta import iter_fasta_summaries
from plasmid_priority.qc import run_input_checks
from plasmid_priority.reporting import ManagedScriptRun


@dataclass(frozen=True)
class TestRunSummary:
    tests_run: int
    test_failures: int
    test_errors: int
    returncode: int
    stdout: str
    stderr: str

    def was_successful(self) -> bool:
        return self.returncode == 0


def _extract_pytest_counts(junit_xml: Path) -> tuple[int, int, int]:
    if not junit_xml.exists():
        return 0, 0, 0
    try:
        root = ET.parse(junit_xml).getroot()
    except ET.ParseError:
        return 0, 0, 0

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests_run = 0
    test_failures = 0
    test_errors = 0
    for suite in suites:
        tests_run += int(suite.attrib.get("tests", "0") or 0)
        test_failures += int(suite.attrib.get("failures", "0") or 0)
        test_errors += int(suite.attrib.get("errors", "0") or 0)
    return tests_run, test_failures, test_errors


def run_unit_tests() -> TestRunSummary:
    with tempfile.NamedTemporaryFile(
        suffix=".xml", prefix="pytest-smoke-", delete=False
    ) as temp_file:
        junit_path = Path(temp_file.name)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(PROJECT_ROOT / "tests"),
            f"--junitxml={junit_path}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(os.environ.get("PLASMID_PRIORITY_TEST_TIMEOUT_SECONDS", "86400"))),
    )
    tests_run, test_failures, test_errors = _extract_pytest_counts(junit_path)
    junit_path.unlink(missing_ok=True)
    return TestRunSummary(
        tests_run=tests_run,
        test_failures=test_failures,
        test_errors=test_errors,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_cli_smoke(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(os.environ.get("PLASMID_PRIORITY_SCRIPT_TIMEOUT_SECONDS", "3600"))),
    )


def _missing_required_input_errors(validation_report) -> list[object]:
    missing_errors = []
    for result in validation_report.errors:
        if Path(result.path).exists():
            return []
        missing_errors.append(result)
    return missing_errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data smoke checks. Use --with-tests to prepend unit tests."
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Run the full unittest suite before the smoke checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args([] if argv is None else argv)
    context = build_context(PROJECT_ROOT)
    tests_run_count = 0

    with ManagedScriptRun(context, "26_run_tests_or_smoke") as run:
        if args.with_tests:
            test_result = run_unit_tests()
            run.set_metric("tests_run", test_result.tests_run)
            run.set_metric("test_failures", test_result.test_failures)
            run.set_metric("test_errors", test_result.test_errors)
            tests_run_count = test_result.tests_run
            if test_result.tests_run <= 0:
                raise RuntimeError(
                    "Pytest did not execute any tests in --with-tests mode; failing closed."
                )
            if not test_result.was_successful():
                run.warn(
                    test_result.stderr.strip() or test_result.stdout.strip() or "pytest failed"
                )
                raise RuntimeError("Unit tests failed.")
        else:
            run.set_metric("tests_run", 0)
            run.set_metric("test_failures", 0)
            run.set_metric("test_errors", 0)
            tests_run_count = 0

        validation_report = run_input_checks(context)
        run.set_metric("input_check_errors", len(validation_report.errors))
        if not validation_report.ok:
            missing_required_inputs = _missing_required_input_errors(validation_report)
            if missing_required_inputs:
                missing_keys = ", ".join(result.key for result in missing_required_inputs)
                run.warn(
                    "Skipping real-data smoke because required inputs are not present in this "
                    f"checkout: {missing_keys}"
                )
                run.set_metric(
                    "smoke_skipped_missing_required_inputs",
                    len(missing_required_inputs),
                )
                if tests_run_count <= 0:
                    raise RuntimeError(
                        "No meaningful validation executed: required inputs missing and tests were not run."
                    )
                return 0
            raise RuntimeError("Input validation failed during smoke run.")

        for script_name in (
            "01_check_inputs.py",
            "27_run_advanced_audits.py",
            "24_build_reports.py",
            "25_export_tubitak_summary.py",
            "28_build_release_bundle.py",
        ):
            completed = _run_cli_smoke(script_name)
            run.set_metric(f"{script_name}_returncode", int(completed.returncode))
            if completed.returncode != 0:
                run.warn(
                    completed.stderr.strip() or completed.stdout.strip() or f"{script_name} failed"
                )
                raise RuntimeError(f"CLI smoke failed for {script_name}.")

        plsdb_metadata = context.asset_path("plsdb_metadata_tsv")
        taxonomy_csv = context.asset_path("plsdb_meta_tables_dir") / "taxonomy.csv"
        plsdb_frame = build_plsdb_canonical_metadata(plsdb_metadata, taxonomy_csv)
        run.set_rows_in("plsdb_canonical_smoke_rows", int(len(plsdb_frame)))

        plsdb_headers = list(
            itertools.islice(iter_fasta_summaries(context.asset_path("plsdb_sequences_fasta")), 3)
        )
        refseq_headers = list(
            itertools.islice(iter_fasta_summaries(context.asset_path("refseq_plasmids_fasta")), 3)
        )
        run.set_metric("plsdb_fasta_probe_count", len(plsdb_headers))
        run.set_metric("refseq_fasta_probe_count", len(refseq_headers))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
