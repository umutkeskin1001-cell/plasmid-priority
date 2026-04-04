#!/usr/bin/env python3
"""Compute backbone-level mobility feature T."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.features import build_training_canonical_table, compute_feature_t
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    backbones_path = context.root / "data/silver/plasmid_backbones.tsv"
    amr_consensus_path = context.root / "data/silver/plasmid_amr_consensus.tsv"
    canonical_output = context.root / "data/features/training_canonical_table.tsv"
    feature_output = context.root / "data/features/feature_T.tsv"
    ensure_directory(feature_output.parent)

    with ManagedScriptRun(context, "11_compute_feature_T") as run:
        for path in (backbones_path, amr_consensus_path):
            run.record_input(path)
        run.record_output(canonical_output)
        run.record_output(feature_output)

        pipeline = context.pipeline_settings
        records = read_tsv(backbones_path)
        amr_consensus = read_tsv(amr_consensus_path)
        training_canonical = build_training_canonical_table(
            records,
            amr_consensus,
            split_year=pipeline.split_year,
        )
        feature_t = compute_feature_t(training_canonical)
        training_canonical.to_csv(canonical_output, sep="\t", index=False)
        feature_t.to_csv(feature_output, sep="\t", index=False)

        run.set_rows_out("training_canonical_rows", int(len(training_canonical)))
        run.set_rows_out("feature_t_rows", int(len(feature_t)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
