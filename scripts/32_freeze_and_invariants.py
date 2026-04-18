#!/usr/bin/env python3
"""Create baseline freeze snapshot and evaluate invariants against a baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from plasmid_priority.governance import (
    build_freeze_snapshot,
    compare_invariants,
    load_freeze_contract,
    load_json,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "reports" / "freeze" / "baseline_freeze.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "freeze" / "current_freeze.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "config" / "freeze_contract.yaml",
    )
    parser.add_argument("--run-quality-checks", action="store_true")
    parser.add_argument("--promote-baseline", action="store_true")
    args = parser.parse_args(argv)

    snapshot = build_freeze_snapshot(
        project_root=args.project_root,
        run_quality_checks=args.run_quality_checks,
    )
    write_json(args.output, snapshot)

    contract = load_freeze_contract(args.contract)
    if args.baseline.exists():
        baseline = load_json(args.baseline)
        invariants = compare_invariants(baseline=baseline, candidate=snapshot, contract=contract)
        evaluation = {
            "baseline": str(args.baseline),
            "candidate": str(args.output),
            "results": [result.__dict__ for result in invariants],
            "rollback_required": any(result.rollback for result in invariants),
        }
        write_json(args.output.with_name("freeze_invariant_evaluation.json"), evaluation)

    if args.promote_baseline:
        write_json(args.baseline, snapshot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
