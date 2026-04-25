#!/usr/bin/env python3
"""Build the first bronze-layer metadata outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.harmonize import (
    build_plsdb_canonical_metadata,
    iter_refseq_inventory_rows,
    write_bronze_inventory,
)
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bronze-layer TSV outputs.",
    )
    parser.add_argument(
        "--refseq-limit",
        type=int,
        default=None,
        help="Optional record cap for the RefSeq raw inventory, useful for development smoke runs.",
    )
    return parser.parse_args()


def _check_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_cached_manifest(
    manifest_path: Path,
    input_paths: list[Path],
    *,
    refseq_limit: int | None,
) -> dict[str, object] | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("refseq_limit") != refseq_limit:
        return None
    if payload.get("input_signatures") != [_path_signature(path) for path in input_paths]:
        return None
    return payload  # type: ignore


def main() -> int:
    args = parse_args()
    context = build_context(PROJECT_ROOT)

    plsdb_metadata = context.asset_path("plsdb_metadata_tsv")
    taxonomy_csv = context.asset_path("plsdb_meta_tables_dir") / "taxonomy.csv"
    refseq_fasta = context.asset_path("refseq_plasmids_fasta")

    canonical_output = context.data_dir / "bronze/plsdb_canonical_metadata.tsv"
    inventory_output = context.data_dir / "bronze/bronze_plasmid_inventory.tsv"
    manifest_path = context.data_dir / "bronze/bronze_table_manifest.json"

    _check_overwrite(canonical_output, args.overwrite)
    _check_overwrite(inventory_output, args.overwrite)

    ensure_directory(canonical_output.parent)

    with ManagedScriptRun(context, "03_build_bronze_table") as run:
        for path in (plsdb_metadata, taxonomy_csv, refseq_fasta):
            run.record_input(path)
        run.record_output(canonical_output)
        run.record_output(inventory_output)
        run.note(
            "RefSeq rows are emitted as header-only raw inventory records until richer harmonization is implemented.",
        )
        cached = _load_cached_manifest(
            manifest_path,
            [plsdb_metadata, taxonomy_csv, refseq_fasta],
            refseq_limit=args.refseq_limit,
        )
        if cached is not None and canonical_output.exists() and inventory_output.exists():
            run.note(
                "Bronze metadata inputs unchanged; reusing existing canonical and inventory tables.",
            )
            plsdb_rows = int(cached.get("plsdb_canonical_metadata_rows", 0))  # type: ignore
            inventory_rows = int(cached.get("bronze_inventory_rows", 0))  # type: ignore
        else:
            plsdb_frame = build_plsdb_canonical_metadata(plsdb_metadata, taxonomy_csv)
            plsdb_frame.to_csv(canonical_output, sep="\t", index=False)

            refseq_rows = iter_refseq_inventory_rows(refseq_fasta, limit=args.refseq_limit)
            inventory_rows = write_bronze_inventory(plsdb_frame, refseq_rows, inventory_output)
            plsdb_rows = int(len(plsdb_frame))
            atomic_write_json(
                manifest_path,
                {
                    "refseq_limit": args.refseq_limit,
                    "input_signatures": [
                        _path_signature(path)
                        for path in [plsdb_metadata, taxonomy_csv, refseq_fasta]
                    ],
                    "plsdb_canonical_metadata_rows": plsdb_rows,
                    "bronze_inventory_rows": int(inventory_rows),
                },
            )

        run.set_rows_in("plsdb_metadata_rows", plsdb_rows)
        run.set_rows_out("plsdb_canonical_metadata_rows", plsdb_rows)
        run.set_rows_out("bronze_inventory_rows", inventory_rows)
        run.set_metric("refseq_limit", args.refseq_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
