#!/usr/bin/env python3
"""Run consolidated hardening audit summary.

This script generates a compact summary of all hardening audits:
- EPV (events-per-variable)
- Lead-time bias
- Missingness
- Schema validation status

Examples:
    # Run full hardening summary with default paths
    python scripts/run_hardening_summary.py

    # Output to specific file
    python scripts/run_hardening_summary.py --output-json summary.json

    # Generate markdown report
    python scripts/run_hardening_summary.py --markdown > hardening_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.reporting.hardening_summary import (
    build_hardening_audit_summary,
    format_hardening_summary_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate consolidated hardening audit summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Run with default paths, print JSON
  %(prog)s --markdown          # Output markdown report
  %(prog)s --output-json out.json   # Write JSON to file
        """,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="data/features/backbone_table.tsv",
        help="Path to backbone table TSV",
    )
    parser.add_argument(
        "--scored",
        type=str,
        default="data/scores/backbone_scored.tsv",
        help="Path to scored backbone table TSV",
    )
    parser.add_argument(
        "--harmonized",
        type=str,
        help="Path to harmonized plasmids TSV (optional)",
    )
    parser.add_argument(
        "--deduplicated",
        type=str,
        help="Path to deduplicated plasmids TSV (optional)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output markdown format instead of JSON",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        dest="output_json",
        help="Write JSON output to specified file",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--no-epv",
        action="store_true",
        help="Skip EPV audit",
    )
    parser.add_argument(
        "--no-lead-time",
        action="store_true",
        help="Skip lead-time bias audit",
    )
    parser.add_argument(
        "--no-missingness",
        action="store_true",
        help="Skip missingness audit",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return non-zero exit code if overall status is not 'ok' (fail, error, or concern)",
    )

    args = parser.parse_args(argv)
    context = build_context(PROJECT_ROOT)

    # Resolve paths relative to project root
    def resolve_path(path_str: str | None) -> str | None:
        if path_str is None:
            return None
        path = context.resolve_path(path_str)
        return str(path) if path.exists() else None

    backbone_path = resolve_path(args.backbone)
    scored_path = resolve_path(args.scored)
    harmonized_path = resolve_path(args.harmonized)
    deduplicated_path = resolve_path(args.deduplicated)

    # Load available tables
    backbone_table = None
    scored_table = None
    harmonized_table = None
    deduplicated_table = None

    if backbone_path:
        backbone_table = pd.read_csv(backbone_path, sep="\t")
        print(f"Loaded backbone table: {len(backbone_table)} rows", file=sys.stderr)
    else:
        print(f"Backbone table not found: {args.backbone}", file=sys.stderr)

    if scored_path:
        scored_table = pd.read_csv(scored_path, sep="\t")
        print(f"Loaded scored table: {len(scored_table)} rows", file=sys.stderr)
    else:
        print(f"Scored table not found: {args.scored}", file=sys.stderr)

    if harmonized_path:
        harmonized_table = pd.read_csv(harmonized_path, sep="\t")
        print(f"Loaded harmonized table: {len(harmonized_table)} rows", file=sys.stderr)

    if deduplicated_path:
        deduplicated_table = pd.read_csv(deduplicated_path, sep="\t")
        print(f"Loaded deduplicated table: {len(deduplicated_table)} rows", file=sys.stderr)

    if backbone_table is None and scored_table is None:
        print("Error: At least one of backbone or scored tables must be available.", file=sys.stderr)
        return 1

    # Build summary
    summary = build_hardening_audit_summary(
        backbone_table=backbone_table,
        scored_backbone_table=scored_table,
        harmonized_plasmids=harmonized_table,
        deduplicated_plasmids=deduplicated_table,
        include_schema_validation=not args.no_schema,
        include_epv=not args.no_epv,
        include_lead_time_bias=not args.no_lead_time,
        include_missingness=not args.no_missingness,
    )

    # Output
    json_output = json.dumps(summary, indent=2, default=str)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Wrote JSON to: {args.output_json}", file=sys.stderr)

    if args.markdown:
        output = format_hardening_summary_markdown(summary)
        print(output)
    else:
        print(json_output)

    # Return code based on overall status
    overall_status = summary.get("overall_status", "unknown")
    print(f"\nOverall hardening status: {overall_status}", file=sys.stderr)

    # Exit code: 0 for ok/skipped, 1 for fail/error/concern when --strict-exit is used
    if args.strict_exit and overall_status not in ("ok", "skipped"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
