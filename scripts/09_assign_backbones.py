#!/usr/bin/env python3
"""Assign operational backbone IDs to plasmid records."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.backbone import assign_backbone_ids, assign_backbone_ids_training_only
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign operational backbone IDs to plasmid records.")
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Use all records for backbone assignment instead of the discovery-safe training-only default.",
    )
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    dedup_path = context.data_dir / "silver/plasmid_deduplicated.tsv"
    output_path = context.data_dir / "silver/plasmid_backbones.tsv"
    config_path = context.root / "config.yaml"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "09_assign_backbones") as run:
        run.record_input(dedup_path)
        run.record_input(config_path)
        run.record_output(output_path)
        records = read_tsv(dedup_path)
        if args.all_records:
            backbones = assign_backbone_ids(records)
            assignment_mode = "all_records"
        else:
            backbones = assign_backbone_ids_training_only(
                records,
                split_year=int(context.pipeline_settings.split_year),
            )
            assignment_mode = "training_only"
        backbones.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("plasmid_backbones_rows", int(len(backbones)))
        run.set_metric("backbone_count", int(backbones["backbone_id"].nunique()))
        run.set_metric("backbone_assignment_mode", assignment_mode)
        run.set_metric("split_year", int(context.pipeline_settings.split_year))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
