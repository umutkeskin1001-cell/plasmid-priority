#!/usr/bin/env python3
"""Assemble the backbone-level outcome table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.features import build_backbone_table
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory
from plasmid_priority.validation.missingness import audit_backbone_tables, format_missingness_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build backbone table")
    parser.add_argument(
        "--audit-missingness",
        action="store_true",
        help="Run missingness audit and write artifacts to reports/audits/",
    )
    args = parser.parse_args()

    context = build_context(PROJECT_ROOT)
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    coherence_path = context.data_dir / "features/backbone_coherence.tsv"
    output_path = context.data_dir / "features/backbone_table.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "14_build_backbone_table") as run:
        run.record_input(backbones_path)
        run.record_input(coherence_path)
        run.record_output(output_path)

        pipeline = context.pipeline_settings
        records = read_tsv(backbones_path)
        coherence = read_tsv(coherence_path) if coherence_path.exists() else pd.DataFrame()
        backbone_assignment_mode = (
            records.get("backbone_assignment_mode", pd.Series(dtype=str))
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )
        explicit_mode = (
            str(backbone_assignment_mode.iloc[0]).strip()
            if not backbone_assignment_mode.empty
            else "training_only"
        )
        backbone_table = build_backbone_table(
            records,
            coherence,
            split_year=pipeline.split_year,
            new_country_threshold=pipeline.min_new_countries_for_spread,
            backbone_assignment_mode=explicit_mode,
        )
        backbone_table.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("backbone_table_rows", int(len(backbone_table)))

        # Light missingness audit (opt-in, non-invasive)
        if args.audit_missingness:
            audit_dir = context.reports_dir / "audits"
            ensure_directory(audit_dir)

            audit_result = audit_backbone_tables(backbone_table=backbone_table)

            # Write machine-readable JSON
            json_path = audit_dir / "missingness_backbone_table.json"
            with open(json_path, "w") as f:
                json.dump(audit_result.get("backbone_table", audit_result), f, indent=2)

            # Write human-readable report
            txt_path = audit_dir / "missingness_backbone_table.txt"
            with open(txt_path, "w") as f:
                if "backbone_table" in audit_result:
                    f.write(format_missingness_report(audit_result["backbone_table"]))
                else:
                    f.write("No backbone_table audit data available.\n")

            run.note(f"Missingness audit written to {audit_dir}")
            run.set_metric(
                "missingness_audit_status", audit_result.get("overall_status", "unknown")
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
