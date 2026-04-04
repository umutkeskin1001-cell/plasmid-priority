#!/usr/bin/env python3
"""Build the derived bronze-layer combined plasmid FASTA."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.io import concatenate_fastas
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_cached_stats(
    manifest_path: Path, input_paths: list[Path], *, dry_run: bool
) -> dict[str, object] | None:
    if dry_run or not manifest_path.exists():
        return None
    try:
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cached_inputs = payload.get("input_signatures", [])
    expected_inputs = [_path_signature(path) for path in input_paths]
    if cached_inputs != expected_inputs:
        return None
    stats = payload.get("stats")
    return stats if isinstance(stats, dict) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output FASTA if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs and report counts without writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = build_context(PROJECT_ROOT)

    input_paths = [
        context.asset_path("plsdb_sequences_fasta"),
        context.asset_path("refseq_plasmids_fasta"),
    ]
    output_path = context.asset_path("bronze_all_plasmids_fasta")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    with ManagedScriptRun(context, "02_build_all_plasmids_fasta") as run:
        for path in input_paths:
            run.record_input(path)

        if args.dry_run:
            run.note("Dry-run mode enabled; output FASTA will not be written.")
        else:
            run.record_output(output_path)

        stats = _load_cached_stats(manifest_path, input_paths, dry_run=args.dry_run)
        if stats is not None and output_path.exists():
            run.note(
                "Input FASTA files unchanged; reusing existing combined FASTA and cached record statistics."
            )
        else:
            stats = concatenate_fastas(
                input_paths,
                output_path,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                atomic_write_json(
                    manifest_path,
                    {
                        "input_signatures": [_path_signature(path) for path in input_paths],
                        "output_path": str(output_path.resolve()),
                        "output_size": int(output_path.stat().st_size)
                        if output_path.exists()
                        else None,
                        "stats": stats,
                    },
                )
        run.set_metric("dry_run", args.dry_run)
        run.set_metric("record_count", stats["record_count"])
        run.set_metric("base_count", stats["base_count"])
        for input_file in stats["input_files"]:
            path = Path(str(input_file["path"]))
            run.set_rows_in(path.name, int(input_file["record_count"]))
        run.set_rows_out("combined_records", int(stats["record_count"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
