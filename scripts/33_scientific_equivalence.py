#!/usr/bin/env python3
"""Run scientific equivalence harness between two freeze snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from plasmid_priority.governance import (
    load_freeze_contract,
    load_json,
    scientific_equivalence,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "config" / "freeze_contract.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "freeze" / "scientific_equivalence.json",
    )
    args = parser.parse_args(argv)

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    contract = load_freeze_contract(args.contract)
    result = scientific_equivalence(baseline=baseline, candidate=candidate, contract=contract)
    write_json(args.output, result)
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
