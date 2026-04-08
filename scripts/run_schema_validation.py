#!/usr/bin/env python3
"""Run schema validation on key pipeline tables.

This is a lightweight developer tool to validate table schemas
using Pandera.

Examples:
    # Validate all available tables with default paths
    python scripts/run_schema_validation.py

    # Validate specific tables
    python scripts/run_schema_validation.py --scored data/scores/backbone_scored.tsv

    # Output full JSON report
    python scripts/run_schema_validation.py --json > validation_report.json

Note:
    In environments where Pandera is unavailable, validation returns
    status='skipped' with explicit reason='pandera_not_installed'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.validation import (
    PANDERA_AVAILABLE,
    print_validation_report,
    validate_tables_from_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate table schemas with Pandera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Validate default paths
  %(prog)s --scored path/to/scored.tsv  # Validate specific table
  %(prog)s --json                       # Output JSON report
  %(prog)s --quiet                      # Exit code only (0=pass, 1=fail)

Note:
  If Pandera is unavailable in the active environment, validation returns
  skipped results with reason='pandera_not_installed'.
        """,
    )
    parser.add_argument(
        "--harmonized",
        type=str,
        help="Path to harmonized plasmids TSV",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="data/features/backbone_table.tsv",
        help="Path to backbone table TSV (default: data/features/backbone_table.tsv)",
    )
    parser.add_argument(
        "--scored",
        type=str,
        default="data/scores/backbone_scored.tsv",
        help="Path to scored backbone TSV (default: data/scores/backbone_scored.tsv)",
    )
    parser.add_argument(
        "--deduplicated",
        type=str,
        help="Path to deduplicated plasmids TSV",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of formatted report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output, return exit code only",
    )

    args = parser.parse_args(argv)
    context = build_context(PROJECT_ROOT)

    # Resolve paths relative to project root
    def resolve_path(path_str: str | None) -> str | None:
        if path_str is None:
            return None
        path = context.resolve_path(path_str)
        return str(path) if path.exists() else None

    harmonized_path = resolve_path(args.harmonized)
    backbone_path = resolve_path(args.backbone)
    scored_path = resolve_path(args.scored)
    deduplicated_path = resolve_path(args.deduplicated)

    if not args.quiet:
        if not PANDERA_AVAILABLE:
            print(
                "Note: Pandera unavailable in this environment; schema validation will be reported as skipped.\n",
                file=sys.stderr,
            )

        loaded = []
        if harmonized_path:
            loaded.append("harmonized")
        if backbone_path:
            loaded.append("backbone")
        if scored_path:
            loaded.append("scored")
        if deduplicated_path:
            loaded.append("deduplicated")

        if loaded:
            print(f"Validating tables: {', '.join(loaded)}", file=sys.stderr)
        else:
            print("Warning: No tables found to validate.", file=sys.stderr)

    # Run validation
    results = validate_tables_from_paths(
        harmonized_path=harmonized_path,
        backbones_path=backbone_path,
        scored_path=scored_path,
        deduplicated_path=deduplicated_path,
    )

    # Output
    if args.json:
        # Remove internal keys from JSON output
        clean_results = {k: v for k, v in results.items() if not k.startswith("_")}
        clean_results["_summary"] = results.get("_summary", {})
        print(json.dumps(clean_results, indent=2, default=str))
    elif not args.quiet:
        print_validation_report(results)

    # Exit code based on results
    summary = results.get("_summary", {})
    overall = summary.get("overall_status", "unknown")

    if not args.quiet:
        print(f"\nValidation status: {overall}", file=sys.stderr)

    # Return non-zero for fail/error, zero for pass/skipped
    if overall in ("fail", "error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
