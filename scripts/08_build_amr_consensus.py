#!/usr/bin/env python3
"""Aggregate AMR hits into accession-level consensus summaries."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import build_amr_consensus
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    hits_path = context.data_dir / "silver/plasmid_amr_hits.tsv"
    output_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "08_build_amr_consensus") as run:
        run.record_input(hits_path)
        run.record_output(output_path)
        hits = read_tsv(hits_path)
        consensus = build_amr_consensus(hits)
        consensus.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("plasmid_amr_consensus_rows", int(len(consensus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
