#!/usr/bin/env python3
"""Build a compact release bundle from the curated report surface."""

from __future__ import annotations

import hashlib
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory, relative_path_str

RELEASE_FILES = (
    "reports/executive_summary.md",
    "reports/headline_validation_summary.md",
    "reports/jury_brief.md",
    "reports/ozet_tr.md",
    "reports/pitch_notes.md",
    "reports/tubitak_final_metrics.txt",
    "reports/core_tables/headline_validation_summary.tsv",
    "reports/core_tables/model_metrics.tsv",
    "reports/core_tables/model_selection_summary.tsv",
    "reports/core_tables/model_selection_scorecard.tsv",
    "reports/core_tables/single_model_official_decision.tsv",
    "reports/diagnostic_tables/single_model_pareto_screen.tsv",
    "reports/diagnostic_tables/single_model_pareto_finalists.tsv",
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
    metrics_path = context_root / "reports/core_tables/model_metrics.tsv"
    single_model_decision_path = context_root / "reports/core_tables/single_model_official_decision.tsv"
    version = _project_version(context_root)
    if not metrics_path.exists():
        return (
            "\n".join(
                [
                    "Plasmid Priority Release Bundle",
                    f"Version: {version}",
                    f"Generated on: {datetime.now().date().isoformat()}",
                ]
            )
            + "\n"
        )
    metrics = pd.read_csv(metrics_path, sep="\t")
    decision_row = pd.Series(dtype=object)
    if single_model_decision_path.exists():
        try:
            decision_frame = pd.read_csv(single_model_decision_path, sep="\t")
        except EmptyDataError:
            decision_frame = pd.DataFrame()
        if not decision_frame.empty:
            decision_row = decision_frame.iloc[0]
    chosen_model_name = str(decision_row.get("official_model_name", "") or "").strip()
    primary = metrics.loc[metrics["model_name"].astype(str) == chosen_model_name].head(1)
    if primary.empty:
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
    n_permutations = pd.to_numeric(
        pd.Series([row.get("n_permutations", pd.NA)]), errors="coerce"
    ).iloc[0]
    n_permutations_text = int(n_permutations) if pd.notna(n_permutations) else 0
    decision_reason = str(decision_row.get("decision_reason", "") or "").strip()
    decision_status = str(
        decision_row.get("scientific_acceptance_status", "not_scored") or "not_scored"
    ).strip()
    return (
        "\n".join(
            [
                "Plasmid Priority Release Bundle",
                f"Version: {version}",
                f"Generated on: {datetime.now().date().isoformat()}",
                f"Primary model: {row.get('model_name', 'unknown')}",
                f"Single-model decision status: {decision_status}",
                f"Single-model decision reason: {decision_reason or 'not_reported'}",
                f"ROC AUC: {float(row.get('roc_auc', float('nan'))):.4f} [{float(row.get('roc_auc_ci_lower', float('nan'))):.4f}–{float(row.get('roc_auc_ci_upper', float('nan'))):.4f}]",
                f"AP: {float(row.get('average_precision', float('nan'))):.4f} [{float(row.get('average_precision_ci_lower', float('nan'))):.4f}–{float(row.get('average_precision_ci_upper', float('nan'))):.4f}]",
                f"Selection-adjusted permutation p {selection_adjusted_text} (n={n_permutations_text})",
                f"Fixed-score permutation p {permutation_text} (n={n_permutations_text})",
            ]
        )
        + "\n"
    )


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
