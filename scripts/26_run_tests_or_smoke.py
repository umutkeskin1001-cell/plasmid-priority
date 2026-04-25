#!/usr/bin/env python3
"""Smoke-test and test runner entry point for CI and manual validation.

This script is the primary CI smoke-test entry point. It verifies that:
- Core packages are importable
- Primary model backend (LightGBM) is available
- Key configuration is loadable
- Optionally runs the full test suite (--with-tests flag)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from plasmid_priority.config import build_context
from plasmid_priority.harmonize import build_plsdb_canonical_metadata  # noqa: F401
from plasmid_priority.io import iter_fasta_summaries  # noqa: F401
from plasmid_priority.qc import run_input_checks
from plasmid_priority.reporting import ManagedScriptRun

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_primary_model_backend() -> list[str]:
    """Verify LightGBM (primary model backend) is importable."""
    issues: list[str] = []
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        issues.append(
            "LightGBM is NOT importable. The primary model (discovery_boosted) will silently "
            "fall back to hist_gbm. Install tree-models extra: "
            "pip install -e '.[analysis,dev,tree-models]'",
        )
    return issues


def _check_core_imports() -> list[str]:
    """Verify all core plasmid_priority modules are importable."""
    issues: list[str] = []
    modules = [
        "plasmid_priority",
        "plasmid_priority.modeling",
        "plasmid_priority.validation",
        "plasmid_priority.scoring",
        "plasmid_priority.backbone",
        "plasmid_priority.features",
        "plasmid_priority.reporting",
        "plasmid_priority.exceptions",
        "plasmid_priority.protocol",
    ]
    for module in modules:
        try:
            __import__(module)
        except ImportError as exc:
            issues.append(f"Cannot import {module}: {exc}")
    return issues


def _check_primary_model_config() -> list[str]:
    """Verify primary model name is present in feature sets."""
    issues: list[str] = []
    try:
        from plasmid_priority.modeling import MODULE_A_FEATURE_SETS  # noqa: PLC0415
        from plasmid_priority.modeling.module_a import get_primary_model_name  # noqa: PLC0415

        primary_name = get_primary_model_name(MODULE_A_FEATURE_SETS.keys())  # type: ignore
        if primary_name not in MODULE_A_FEATURE_SETS:
            issues.append(
                f"Primary model '{primary_name}' is not defined in MODULE_A_FEATURE_SETS. "
                "Check config.yaml: models.primary_model_name",
            )
        else:
            n_features = len(MODULE_A_FEATURE_SETS[primary_name])
            print(f"  ✓ Primary model '{primary_name}' found with {n_features} features.")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Failed to load primary model config: {exc}")
    return issues


def _check_lightgbm_has_flag() -> list[str]:
    """Verify _HAS_LIGHTGBM is True — critical for matching CI to production."""
    issues: list[str] = []
    try:
        import plasmid_priority.modeling.module_a as mod_a  # noqa: PLC0415

        _HAS_LIGHTGBM = getattr(mod_a, "_HAS_LIGHTGBM", False)

        if _HAS_LIGHTGBM:
            print("  ✓ _HAS_LIGHTGBM = True (primary model uses LightGBM backend)")
        else:
            issues.append(
                "_HAS_LIGHTGBM is False — LightGBM is not installed. "
                "Primary model (discovery_boosted) will silently use hist_gbm fallback. "
                "This means CI is testing a DIFFERENT model than production. "
                "Fix: pip install -e '.[analysis,dev,tree-models]'",
            )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Cannot check _HAS_LIGHTGBM: {exc}")
    return issues


def run_smoke_checks() -> int:
    """Run all smoke checks. Returns 0 if OK, 1 if any issues found."""
    print("\n=== Plasmid Priority Smoke Checks ===\n")

    all_issues: list[str] = []

    print("[1/4] Core imports...")
    issues = _check_core_imports()
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  ✗ {i}")
    else:
        print("  ✓ All core modules importable.")

    print("[2/4] Primary model backend (LightGBM)...")
    issues = _check_primary_model_backend()
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  ✗ {i}")
    else:
        print("  ✓ LightGBM importable.")

    print("[3/4] LightGBM flag check...")
    issues = _check_lightgbm_has_flag()
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  ✗ {i}")

    print("[4/4] Primary model config...")
    issues = _check_primary_model_config()
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  ✗ {i}")

    if all_issues:
        print(f"\n=== SMOKE FAILED — {len(all_issues)} issue(s) found ===")
        for issue in all_issues:
            print(f"  • {issue}")
        return 1

    print("\n=== All smoke checks passed ===")
    return 0


def _run_cli_smoke(context: object | None = None) -> subprocess.CompletedProcess[None]:
    """Run the CLI smoke checks as a subprocess-compatible result."""
    del context
    return subprocess.CompletedProcess(args=["smoke"], returncode=run_smoke_checks())


def run_unit_tests() -> subprocess.CompletedProcess[None]:
    """Run pytest test suite."""
    print("\n=== Running test suite ===\n")
    env = dict(os.environ)
    # Matplotlib reads MPLCONFIGDIR from environment; there is no rcParam key for it.
    # Always force a temporary, writable config directory so inherited shell env cannot
    # point to a stale or permission-denied location.
    with tempfile.TemporaryDirectory(prefix="mplconfig-") as mplconfigdir:
        env["MPLCONFIGDIR"] = mplconfigdir
        # Use a headless backend by default for CI/sandbox test environments.
        env.setdefault("MPLBACKEND", "Agg")
        return subprocess.run(  # type: ignore
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )


def run_tests() -> int:
    """Backward-compatible wrapper around run_unit_tests()."""
    return run_unit_tests().returncode


def _has_missing_required_raw_inputs(report) -> bool:  # type: ignore
    for result in getattr(report, "errors", []):
        path = Path(str(getattr(result, "path", "")))
        if (
            getattr(result, "required", False)
            and getattr(result, "stage", "") == "raw"
            and not path.exists()
        ):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plasmid Priority smoke test and test runner.")
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Also run the full pytest suite after smoke checks.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only smoke checks, skip test suite (default behavior).",
    )
    parser.add_argument(
        "--strict-inputs",
        action="store_true",
        help="Fail even with --with-tests when required raw inputs are missing.",
    )
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    with ManagedScriptRun(context, "26_run_tests_or_smoke") as run:
        report = run_input_checks(context)

        missing_required_raw_inputs = _has_missing_required_raw_inputs(report)
        if missing_required_raw_inputs:
            run.set_metric("smoke_skipped_missing_required_inputs", 1)
            run.warn("Missing required raw input(s); skipping CLI smoke checks.")

        if not getattr(report, "ok", True):
            invalid_errors = [
                result
                for result in getattr(report, "errors", [])
                if not (
                    getattr(result, "required", False)
                    and getattr(result, "stage", "") == "raw"
                    and not Path(str(getattr(result, "path", ""))).exists()
                )
            ]
            if invalid_errors:
                raise RuntimeError("Input validation failed. See validation report for details.")

        if args.with_tests:
            test_result = run_unit_tests()
            tests_run = getattr(test_result, "tests_run", None)
            if tests_run is None:
                tests_run = 1 if getattr(test_result, "returncode", 1) == 0 else 0
            run.set_metric("tests_run", int(tests_run))
            if getattr(test_result, "returncode", 0) != 0:
                return int(getattr(test_result, "returncode", 1))
        else:
            run.set_metric("tests_run", 0)

        if missing_required_raw_inputs:
            strict_inputs = args.strict_inputs or os.environ.get(
                "PLASMID_PRIORITY_STRICT_INPUTS",
                "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if strict_inputs or not args.with_tests:
                raise RuntimeError("Required raw inputs are missing.")
            return 0

        smoke_result = _run_cli_smoke(context)
        if getattr(smoke_result, "returncode", 0) != 0:
            return int(getattr(smoke_result, "returncode", 1))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
