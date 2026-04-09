#!/usr/bin/env python3
"""Build a compact release bundle from the curated report surface."""

from __future__ import annotations

import hashlib
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_artifact_provenance,
    load_provenance_json,
)
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    project_python_source_paths,
    relative_path_str,
)

RELEASE_FILES = (
    "reports/executive_summary.md",
    "reports/jury_brief.md",
    "reports/ozet_tr.md",
    "reports/pitch_notes.md",
    "reports/report_provenance.json",
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
    "reports/core_figures/temporal_design.png",
    "reports/core_figures/knownness_vs_oof_score_scatter.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version(project_root: Path) -> str:
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return "unknown"
    with pyproject_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return str(payload.get("project", {}).get("version", "unknown"))


def _build_release_info(context_root: Path) -> str:
    provenance_path = context_root / "reports/report_provenance.json"
    metrics_path = context_root / "reports/core_tables/model_metrics.tsv"
    version = _project_version(context_root)
    if not metrics_path.exists():
        lines = [
            "Plasmid Priority Release Bundle",
            f"Version: {version}",
            f"Generated on: {datetime.now().date().isoformat()}",
        ]
        if provenance_path.exists():
            provenance = load_provenance_json(provenance_path)
            lines.extend(
                [
                    f"Protocol ID: {provenance.get('protocol_id', 'unknown')}",
                    f"Protocol hash: {provenance.get('protocol_hash', 'unknown')}",
                    f"Code hash: {provenance.get('code_hash', 'unknown')}",
                    f"Input data hash: {provenance.get('input_data_hash', 'unknown')}",
                ]
            )
        return "\n".join(lines) + "\n"
    metrics = pd.read_csv(metrics_path, sep="\t")
    primary = metrics.loc[
        metrics["model_name"].astype(str) == "phylo_support_fusion_priority"
    ].head(1)
    if primary.empty:
        primary = metrics.sort_values("roc_auc", ascending=False).head(1)
    row = primary.iloc[0]
    permutation_text = "NA"
    selection_adjusted_text = "NA"
    permutation_p = pd.to_numeric(
        pd.Series([row.get("permutation_p_roc_auc")]), errors="coerce"
    ).iloc[0]
    selection_adjusted_p = pd.to_numeric(
        pd.Series([row.get("selection_adjusted_empirical_p_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.notna(permutation_p):
        permutation_text = (
            "< 0.001" if float(permutation_p) < 0.001 else f"= {float(permutation_p):.3f}"
        )
    if pd.notna(selection_adjusted_p):
        selection_adjusted_text = (
            "< 0.001"
            if float(selection_adjusted_p) < 0.001
            else f"= {float(selection_adjusted_p):.3f}"
        )
    n_permutations = pd.to_numeric(pd.Series([row.get("n_permutations", pd.NA)]), errors="coerce").iloc[0]
    n_permutations_text = int(n_permutations) if pd.notna(n_permutations) else 0
    lines = [
        "Plasmid Priority Release Bundle",
        f"Version: {version}",
        f"Generated on: {datetime.now().date().isoformat()}",
        f"Primary model: {row.get('model_name', 'unknown')}",
        f"ROC AUC: {float(row.get('roc_auc', float('nan'))):.4f} [{float(row.get('roc_auc_ci_lower', float('nan'))):.4f}–{float(row.get('roc_auc_ci_upper', float('nan'))):.4f}]",
        f"AP: {float(row.get('average_precision', float('nan'))):.4f} [{float(row.get('average_precision_ci_lower', float('nan'))):.4f}–{float(row.get('average_precision_ci_upper', float('nan'))):.4f}]",
        f"Selection-adjusted permutation p {selection_adjusted_text} (n={n_permutations_text})",
        f"Fixed-score permutation p {permutation_text} (n={n_permutations_text})",
    ]
    if provenance_path.exists():
        provenance = load_provenance_json(provenance_path)
        lines.extend(
            [
                f"Protocol ID: {provenance.get('protocol_id', 'unknown')}",
                f"Protocol hash: {provenance.get('protocol_hash', 'unknown')}",
                f"Code hash: {provenance.get('code_hash', 'unknown')}",
                f"Input data hash: {provenance.get('input_data_hash', 'unknown')}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    context = build_context()
    source_paths = project_python_source_paths(
        context.root,
        script_path=Path(__file__).resolve(),
    )
    release_dir = ensure_directory(context.release_dir)
    bundle_dir = release_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_directory(bundle_dir)

    manifest_rows: list[dict[str, object]] = []

    with ManagedScriptRun(context, "28_build_release_bundle") as run:
        provenance = build_artifact_provenance(
            protocol=context.protocol,
            artifact_family="release_bundle",
            input_paths=[
                context.root / relpath
                for relpath in RELEASE_FILES
                if (context.root / relpath).exists()
            ],
            source_paths=source_paths,
        )
        run.set_provenance(provenance)
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
            "provenance": provenance,
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

        release_info_path = bundle_dir / "RELEASE_INFO.txt"
        release_info_path.write_text(_build_release_info(context.root), encoding="utf-8")
        run.record_output(release_info_path)

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
