#!/usr/bin/env python3
"""Merge canonical metadata with typing and biosample context."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.harmonize.records import build_harmonized_plasmid_table
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    canonical_path = context.data_dir / "bronze/plsdb_canonical_metadata.tsv"
    typing_path = context.asset_path("plsdb_meta_tables_dir") / "typing.csv"
    biosample_path = context.asset_path("plsdb_meta_tables_dir") / "biosample.csv"
    plasmidfinder_path = context.asset_path("plsdb_meta_tables_dir") / "plasmidfinder.csv"
    output_path = context.data_dir / "silver/plasmid_harmonized.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "04_harmonize_metadata") as run:
        for path in (canonical_path, typing_path, biosample_path, plasmidfinder_path):
            run.record_input(path)
        run.record_output(output_path)

        harmonized = build_harmonized_plasmid_table(
            canonical_path,
            typing_path,
            biosample_path,
            plasmidfinder_path,
        )
        harmonized.to_csv(output_path, sep="\t", index=False)

        run.set_rows_out("plasmid_harmonized_rows", int(len(harmonized)))
        run.set_metric(
            "country_non_null", int(harmonized["country"].astype(str).str.len().gt(0).sum())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
