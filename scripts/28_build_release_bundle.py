#!/usr/bin/env python3
"""Build a compact release bundle from the curated report surface."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory, relative_path_str


RELEASE_FILES = (
    "reports/jury_brief.md",
    "reports/ozet_tr.md",
    "reports/tubitak_final_metrics.txt",
    "reports/core_tables/model_metrics.tsv",
    "reports/core_tables/model_selection_summary.tsv",
    "reports/core_tables/model_selection_scorecard.tsv",
    "reports/core_tables/candidate_portfolio.tsv",
    "reports/core_tables/candidate_evidence_matrix.tsv",
    "reports/core_tables/consensus_shortlist.tsv",
    "reports/core_figures/roc_curve.png",
    "reports/core_figures/pr_curve.png",
    "reports/core_figures/score_distribution.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    context = build_context()
    release_dir = ensure_directory(context.release_dir)
    bundle_dir = release_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_directory(bundle_dir)

    manifest_rows: list[dict[str, object]] = []

    with ManagedScriptRun(context, "28_build_release_bundle") as run:
        missing: list[str] = []
        for relpath in RELEASE_FILES:
            source = context.root / relpath
            if not source.exists():
                missing.append(relpath)
                continue
            destination = bundle_dir / relpath
            ensure_directory(destination.parent)
            shutil.copy2(source, destination)
            stat = source.stat()
            manifest_rows.append(
                {
                    "relative_path": relpath,
                    "size_bytes": int(stat.st_size),
                    "sha256": _sha256(source),
                }
            )
            run.record_input(source)
            run.record_output(destination)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "project_root": str(context.root),
            "bundle_root": relative_path_str(bundle_dir, context.root),
            "n_files": len(manifest_rows),
            "files": manifest_rows,
            "missing_files": missing,
        }
        manifest_path = release_dir / "plasmid_priority_release_manifest.json"
        atomic_write_json(manifest_path, manifest)
        run.record_output(manifest_path)

        readme_path = bundle_dir / "README.txt"
        readme_path.write_text(
            "\n".join(
                [
                    "Plasmid Priority Release Bundle",
                    "",
                    "This bundle contains the curated jury-facing report surface.",
                    "Core tables and figures are copied from the latest successful local build.",
                    "",
                    "See plasmid_priority_release_manifest.json for checksums.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run.record_output(readme_path)

        archive_base = release_dir / "plasmid_priority_release_bundle"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=bundle_dir))
        run.record_output(archive_path)
        run.set_metric("n_release_files", int(len(manifest_rows)))
        run.set_metric("n_missing_release_files", int(len(missing)))
        run.set_rows_out("bundle_files", int(len(manifest_rows)))
        if missing:
            run.warn("Some curated release files were missing and were omitted from the bundle.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
