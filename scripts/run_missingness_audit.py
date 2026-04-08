#!/usr/bin/env python3
"""Run missingness audit on backbone and scored tables.

This is a lightweight standalone script for developers to quickly
run missingness audits without running the full pipeline.

Examples:
    # Run audit with default paths (reads from data/features and data/scores)
    python scripts/run_missingness_audit.py

    # Run audit with custom paths
    python scripts/run_missingness_audit.py --backbone path/to/backbone.tsv --scored path/to/scored.tsv

    # Output to a specific directory
    python scripts/run_missingness_audit.py --output-dir /tmp/audit_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.validation.missingness import (
    audit_backbone_tables,
    format_missingness_report,
    print_backbone_audit_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run missingness audit on backbone tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use default data paths
  %(prog)s --print                   # Print report to stdout
  %(prog)s --backbone data.tsv       # Specify custom backbone table
  %(prog)s --scored scored.tsv       # Specify custom scored table
        """,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="Path to backbone table TSV (default: data/features/backbone_table.tsv)",
    )
    parser.add_argument(
        "--scored",
        type=str,
        help="Path to scored backbone table TSV (default: data/scores/backbone_scored.tsv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/audits",
        help="Directory to write audit artifacts (default: reports/audits)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_report",
        help="Print human-readable report to stdout",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON files, skip text reports",
    )

    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return non-zero exit code if audit status is not 'ok' (fail, error, or concern)",
    )

    args = parser.parse_args(argv)
    context = build_context(PROJECT_ROOT)

    # Resolve paths
    backbone_path = args.backbone
    scored_path = args.scored

    if backbone_path is None:
        backbone_path = str(context.resolve_path("data/features/backbone_table.tsv"))
    else:
        backbone_path = str(context.resolve_path(backbone_path))
    if scored_path is None:
        scored_path = str(context.resolve_path("data/scores/backbone_scored.tsv"))
    else:
        scored_path = str(context.resolve_path(scored_path))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Load data if available
    backbone_table = None
    scored_table = None

    if Path(backbone_path).exists():
        backbone_table = pd.read_csv(backbone_path, sep="\t")
        print(f"Loaded backbone table: {len(backbone_table)} rows from {backbone_path}")
    else:
        print(f"Backbone table not found: {backbone_path}")

    if Path(scored_path).exists():
        scored_table = pd.read_csv(scored_path, sep="\t")
        print(f"Loaded scored table: {len(scored_table)} rows from {scored_path}")
    else:
        print(f"Scored table not found: {scored_path}")

    if backbone_table is None and scored_table is None:
        print("Error: No tables found to audit. Run pipeline first or specify paths.")
        return 1

    # Run audit
    audit_result = audit_backbone_tables(
        backbone_table=backbone_table,
        scored_backbone_table=scored_table,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON artifact
    json_path = output_dir / "missingness_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit_result, f, indent=2)
    print(f"Wrote JSON: {json_path}")

    # Write per-table text reports
    if not args.json_only:
        if "backbone_table" in audit_result:
            txt_path = output_dir / "missingness_backbone_table.txt"
            with open(txt_path, "w") as f:
                f.write(format_missingness_report(audit_result["backbone_table"]))
            print(f"Wrote text report: {txt_path}")

        if "scored_backbone_table" in audit_result:
            txt_path = output_dir / "missingness_scored_backbone.txt"
            with open(txt_path, "w") as f:
                f.write(format_missingness_report(audit_result["scored_backbone_table"]))
            print(f"Wrote text report: {txt_path}")

    # Print to stdout if requested
    if args.print_report:
        print("\n" + "=" * 70)
        print_backbone_audit_report(audit_result)

    # Summary
    overall_status = audit_result.get("overall_status", "unknown")
    total_high = audit_result.get("high_missingness_columns_total", 0)

    print(f"\nAudit complete: status={overall_status}, high_missingness_columns={total_high}")

    # Exit code: 0 for ok, 1 for fail/error/concern when --strict-exit is used
    if args.strict_exit and overall_status not in ("ok", "skipped"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
