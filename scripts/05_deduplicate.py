#!/usr/bin/env python3
"""Annotate stable canonical IDs for identical plasmid records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.dedup import annotate_canonical_ids
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    harmonized_path = context.data_dir / "silver/plasmid_harmonized.tsv"
    identical_path = context.asset_path("plsdb_meta_tables_dir") / "nucc_identical.csv"
    output_path = context.data_dir / "silver/plasmid_deduplicated.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "05_deduplicate") as run:
        run.record_input(harmonized_path)
        run.record_input(identical_path)
        run.record_output(output_path)

        harmonized = read_tsv(harmonized_path)
        identical = pd.read_csv(identical_path)
        deduplicated = annotate_canonical_ids(harmonized, identical)
        deduplicated.to_csv(output_path, sep="\t", index=False)

        run.set_rows_in("plasmid_harmonized_rows", int(len(harmonized)))
        run.set_rows_out("plasmid_deduplicated_rows", int(len(deduplicated)))
        run.set_metric("canonical_groups", int(deduplicated["canonical_id"].nunique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
