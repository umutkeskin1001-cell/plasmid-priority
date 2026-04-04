#!/usr/bin/env python3
"""Run unit tests and a lightweight smoke check on real project data."""

from __future__ import annotations

import itertools
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.harmonize import build_plsdb_canonical_metadata
from plasmid_priority.io.fasta import iter_fasta_summaries
from plasmid_priority.qc import run_input_checks
from plasmid_priority.reporting import ManagedScriptRun


def run_unit_tests() -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    return unittest.TextTestRunner(verbosity=2).run(suite)


def _run_cli_smoke(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    context = build_context(PROJECT_ROOT)

    with ManagedScriptRun(context, "26_run_tests_or_smoke") as run:
        test_result = run_unit_tests()
        run.set_metric("tests_run", test_result.testsRun)
        run.set_metric("test_failures", len(test_result.failures))
        run.set_metric("test_errors", len(test_result.errors))
        if not test_result.wasSuccessful():
            raise RuntimeError("Unit tests failed.")

        validation_report = run_input_checks(context)
        run.set_metric("input_check_errors", len(validation_report.errors))
        if not validation_report.ok:
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
    raise SystemExit(main())
