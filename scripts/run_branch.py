#!/usr/bin/env python3
"""Unified branch CLI surface.

Example:
    python scripts/run_branch.py --branch geo_spread -- --research-models --jobs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plasmid_priority.pipeline.branch_runner import run_branch, supported_branches

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        required=True,
        choices=supported_branches(),
        help="Branch surface to run.",
    )
    parser.add_argument(
        "branch_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected branch CLI. Prefix with '--'.",
    )
    args = parser.parse_args(argv)

    forwarded = list(args.branch_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return int(run_branch(args.branch, branch_args=forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
