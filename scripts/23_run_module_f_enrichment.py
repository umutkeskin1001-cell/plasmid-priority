#!/usr/bin/env python3
"""Run independent categorical enrichment analyses for visibility-positive backbones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_backbone_identity_table,
    build_module_f_enrichment_table,
    build_module_f_top_hits,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.root / "data/scores/backbone_scored.tsv"
    backbones_path = context.root / "data/silver/plasmid_backbones.tsv"
    amr_consensus_path = context.root / "data/silver/plasmid_amr_consensus.tsv"
    identity_output = context.root / "data/analysis/module_f_backbone_identity.tsv"
    enrichment_output = context.root / "data/analysis/module_f_enrichment.tsv"
    top_hits_output = context.root / "data/analysis/module_f_top_hits.tsv"
    ensure_directory(identity_output.parent)

    with ManagedScriptRun(context, "23_run_module_f_enrichment") as run:
        for path in (scored_path, backbones_path, amr_consensus_path):
            run.record_input(path)
        for path in (identity_output, enrichment_output, top_hits_output):
            run.record_output(path)

        scored = read_tsv(scored_path)
        backbones = read_tsv(backbones_path)
        amr_consensus = read_tsv(amr_consensus_path)

        identity = build_backbone_identity_table(scored, backbones, amr_consensus, split_year=2015)
        enrichment = build_module_f_enrichment_table(identity, label_column="spread_label", min_backbones=10)
        top_hits = build_module_f_top_hits(enrichment, q_threshold=0.05, max_per_group=3, max_total=20)

        identity.to_csv(identity_output, sep="\t", index=False)
        enrichment.to_csv(enrichment_output, sep="\t", index=False)
        top_hits.to_csv(top_hits_output, sep="\t", index=False)

        run.set_rows_out("module_f_identity_rows", int(len(identity)))
        run.set_rows_out("module_f_enrichment_rows", int(len(enrichment)))
        run.set_metric("module_f_significant_hits", int(top_hits["feature_value"].nunique()) if not top_hits.empty else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
