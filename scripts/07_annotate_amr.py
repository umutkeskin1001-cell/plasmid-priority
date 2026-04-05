#!/usr/bin/env python3
"""Normalize accession-level AMR hits."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import build_amr_hits_table
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    amr_path = context.asset_path("plsdb_meta_tables_dir") / "amr.tsv"
    output_path = context.data_dir / "silver/plasmid_amr_hits.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "07_annotate_amr") as run:
        run.record_input(amr_path)
        run.record_output(output_path)
        amr_hits = build_amr_hits_table(str(amr_path))
        amr_hits.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("plasmid_amr_hits_rows", int(len(amr_hits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
