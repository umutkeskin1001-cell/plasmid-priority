#!/usr/bin/env python3
"""Write accession-level mobility annotations."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import build_mobility_table
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    dedup_path = context.root / "data/silver/plasmid_deduplicated.tsv"
    output_path = context.root / "data/silver/plasmid_mobility.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "06_annotate_mobility") as run:
        run.record_input(dedup_path)
        run.record_output(output_path)
        records = read_tsv(dedup_path)
        mobility = build_mobility_table(records)
        mobility.to_csv(output_path, sep="\t", index=False)
        run.set_rows_out("plasmid_mobility_rows", int(len(mobility)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
