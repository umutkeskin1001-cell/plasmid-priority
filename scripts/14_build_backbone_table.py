#!/usr/bin/env python3
"""Assemble the backbone-level outcome table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.features import build_backbone_table
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    backbones_path = context.root / "data/silver/plasmid_backbones.tsv"
    coherence_path = context.root / "data/features/backbone_coherence.tsv"
    output_path = context.root / "data/features/backbone_table.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "14_build_backbone_table") as run:
        run.record_input(backbones_path)
        run.record_input(coherence_path)
        run.record_output(output_path)

        pipeline = context.pipeline_settings
        records = read_tsv(backbones_path)
        coherence = read_tsv(coherence_path) if coherence_path.exists() else pd.DataFrame()
        backbone_table = build_backbone_table(
            records,
            coherence,
            split_year=pipeline.split_year,
            new_country_threshold=pipeline.min_new_countries_for_spread,
        )
        backbone_table.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("backbone_table_rows", int(len(backbone_table)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
