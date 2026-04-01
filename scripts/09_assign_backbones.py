#!/usr/bin/env python3
"""Assign operational backbone IDs to plasmid records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.backbone import assign_backbone_ids
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    dedup_path = context.root / "data/silver/plasmid_deduplicated.tsv"
    output_path = context.root / "data/silver/plasmid_backbones.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "09_assign_backbones") as run:
        run.record_input(dedup_path)
        run.record_output(output_path)
        records = read_tsv(dedup_path)
        backbones = assign_backbone_ids(records)
        backbones.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("plasmid_backbones_rows", int(len(backbones)))
        run.set_metric("backbone_count", int(backbones["backbone_id"].nunique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
