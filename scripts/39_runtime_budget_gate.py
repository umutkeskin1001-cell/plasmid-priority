#!/usr/bin/env python3
"""Validate runtime budget config and optionally enforce against latest profile report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_MODES = (
    "smoke-local",
    "dev-refresh",
    "model-refresh",
    "report-refresh",
    "release-full",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", type=str, default="smoke-local")
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    budgets_path = root / "config" / "performance_budgets.yaml"
    payload = _read_yaml(budgets_path)
    modes = payload.get("modes", {}) if isinstance(payload.get("modes"), dict) else {}
    errors: list[str] = []
    for mode in REQUIRED_MODES:
        mode_cfg = modes.get(mode)
        if not isinstance(mode_cfg, dict):
            errors.append(f"missing budget mode: {mode}")
            continue
        budget_seconds = mode_cfg.get("budget_seconds")
        if not isinstance(budget_seconds, (int, float)) or float(budget_seconds) <= 0:
            errors.append(f"invalid budget_seconds for mode {mode}")
    enforcement = payload.get("enforcement", {})
    tolerance = 0.1
    if isinstance(enforcement, dict):
        tol = enforcement.get("default_exceedance_tolerance", 0.1)
        if isinstance(tol, (int, float)):
            tolerance = float(tol)
    if tolerance < 0:
        errors.append("default_exceedance_tolerance must be >= 0")

    logs_report = root / "data" / "tmp" / "logs" / "workflow_profile_report.json"
    if logs_report.exists() and isinstance(modes.get(args.mode), dict):
        report = _read_json(logs_report)
        duration = report.get("total_duration_seconds")
        budget_seconds = modes[args.mode].get("budget_seconds")
        if isinstance(duration, (int, float)) and isinstance(budget_seconds, (int, float)):
            threshold = float(budget_seconds) * (1.0 + tolerance)
            if float(duration) > threshold:
                errors.append(
                    f"latest profile runtime exceeds budget: {duration:.2f}s > {threshold:.2f}s "
                    f"(mode={args.mode})",
                )

    if errors:
        print("runtime-budget gate failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("runtime-budget gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
