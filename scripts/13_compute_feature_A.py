#!/usr/bin/env python3
"""Compute backbone-level AMR feature A."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.features import build_training_canonical_table, compute_feature_a
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    canonical_input = context.data_dir / "features/training_canonical_table.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    output_path = context.data_dir / "features/feature_A.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "13_compute_feature_A") as run:
        for path in (canonical_input, backbones_path, amr_consensus_path):
            if path.exists():
                run.record_input(path)
        run.record_output(output_path)

        pipeline = context.pipeline_settings
        if canonical_input.exists():
            training_canonical = read_tsv(canonical_input)
        else:
            records = read_tsv(backbones_path)
            amr_consensus = read_tsv(amr_consensus_path)
            training_canonical = build_training_canonical_table(
                records,
                amr_consensus,
                split_year=pipeline.split_year,
            )
        feature_a = compute_feature_a(training_canonical)
        feature_a.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("feature_a_rows", int(len(feature_a)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
