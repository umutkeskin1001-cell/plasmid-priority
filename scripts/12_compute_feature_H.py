#!/usr/bin/env python3
"""Compute backbone-level host diversity feature H."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.features import compute_feature_h
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    output_path = context.data_dir / "features/feature_H.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "12_compute_feature_H") as run:
        run.record_input(backbones_path)
        run.record_output(output_path)
        pipeline = context.pipeline_settings
        records = read_tsv(backbones_path)
        feature_h = compute_feature_h(records, split_year=pipeline.split_year)
        feature_h.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("feature_h_rows", int(len(feature_h)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
