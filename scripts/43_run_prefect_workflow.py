#!/usr/bin/env python3
"""Run the Phase 4 Prefect DAG orchestration."""

from __future__ import annotations

import argparse
import sys

from plasmid_priority.pipeline.prefect_flow import (
    PREFECT_AVAILABLE,
    build_phase4_stage_plan,
    render_phase4_stage_plan,
    resolve_phase4_runtime_options,
    run_phase4_prefect_flow,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-fetch", action="store_true", help="Run external-data fetch stage first.")
    parser.add_argument("--release", action="store_true", help="Run release stage after reports.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved Phase 4 DAG and run underlying workflow modes in dry-run mode.",
    )
    args = parser.parse_args(argv)

    max_workers, data_root = resolve_phase4_runtime_options(
        max_workers=args.max_workers,
        data_root=args.data_root,
    )
    if args.dry_run:
        plan = build_phase4_stage_plan(include_fetch=args.include_fetch, run_release=args.release)
        print("Phase 4 stage plan:")
        for line in render_phase4_stage_plan(plan):
            print(f"  - {line}")
        return 0

    if not PREFECT_AVAILABLE:
        print(
            "Prefect is not installed. Install with: `uv pip install 'plasmid-priority[engineering]'`.",
            file=sys.stderr,
        )
        return 2

    run_phase4_prefect_flow(
        include_fetch=args.include_fetch,
        run_release=args.release,
        max_workers=max_workers,
        data_root=data_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
