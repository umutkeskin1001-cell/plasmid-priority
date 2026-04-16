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
import subprocess
import sys
from pathlib import Path

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
            "pip install -e '.[analysis,dev,tree-models]'"
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

        primary_name = get_primary_model_name(MODULE_A_FEATURE_SETS.keys())
        if primary_name not in MODULE_A_FEATURE_SETS:
            issues.append(
                f"Primary model '{primary_name}' is not defined in MODULE_A_FEATURE_SETS. "
                "Check config.yaml: models.primary_model_name"
            )
        else:
            n_features = len(MODULE_A_FEATURE_SETS[primary_name])
            print(
                f"  ✓ Primary model '{primary_name}' found with {n_features} features."
            )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Failed to load primary model config: {exc}")
    return issues


def _check_lightgbm_has_flag() -> list[str]:
    """Verify _HAS_LIGHTGBM is True — critical for matching CI to production."""
    issues: list[str] = []
    try:
        from plasmid_priority.modeling.module_a import _HAS_LIGHTGBM  # noqa: PLC0415

        if _HAS_LIGHTGBM:
            print("  ✓ _HAS_LIGHTGBM = True (primary model uses LightGBM backend)")
        else:
            issues.append(
                "_HAS_LIGHTGBM is False — LightGBM is not installed. "
                "Primary model (discovery_boosted) will silently use hist_gbm fallback. "
                "This means CI is testing a DIFFERENT model than production. "
                "Fix: pip install -e '.[analysis,dev,tree-models]'"
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


def run_tests() -> int:
    """Run pytest test suite."""
    print("\n=== Running test suite ===\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plasmid Priority smoke test and test runner."
    )
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
    args = parser.parse_args()

    rc = run_smoke_checks()
    if rc != 0:
        return rc

    if args.with_tests:
        rc = run_tests()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
