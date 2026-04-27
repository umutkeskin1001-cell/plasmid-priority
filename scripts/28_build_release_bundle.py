#!/usr/bin/env python3
"""Build a compact release bundle from the curated report surface."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from plasmid_priority.config import build_context
from plasmid_priority.protocol import ScientificProtocol, build_protocol_hash
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
    "reports/core_tables/official_model_scores.tsv",
    "reports/core_tables/official_model_scorecard.tsv",
    "reports/core_tables/official_candidate_decisions.tsv",
    "reports/core_tables/official_model_summary.json",
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
    # Scientific/reproducibility surface
    "docs/model_card.md",
    "docs/data_card.md",
    "docs/benchmark_contract.md",
    "docs/scientific_protocol.md",
    "docs/label_card_bundle.md",
    "docs/reproducibility_manifest.json",
    "reports/reviewer_pack/README.md",
    "reports/reviewer_pack/canonical_metadata.json",
    "reports/reviewer_pack/run_reproducibility.sh",
    # Performance dashboard
    "reports/performance/workflow_performance_dashboard.json",
    "reports/performance/workflow_performance_dashboard.md",
    "reports/release/release_readiness_report.json",
    "reports/release/release_readiness_report.md",
)

RELEASE_GLOB_FILES = ("reports/reviewer_pack/candidate_evidence_dossiers/*.md",)


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


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Caught suppressed exception: %s", exc, exc_info=True)
        return "unknown"
    return result.stdout.strip() or "unknown"


def _official_model_names(project_root: Path) -> list[str]:
    protocol_path = project_root / "reports/core_tables/benchmark_protocol.tsv"
    if not protocol_path.exists():
        return []
    try:
        protocol = pd.read_csv(protocol_path, sep="\t")
    except EmptyDataError:
        return []
    if (
        protocol.empty
        or "benchmark_role" not in protocol.columns
        or "model_name" not in protocol.columns
    ):
        return []
    official_roles = {
        "primary_benchmark",
        "governance_benchmark",
        "conservative_benchmark",
        "counts_baseline",
        "source_control",
    }
    return (
        protocol.loc[protocol["benchmark_role"].astype(str).isin(official_roles), "model_name"]
        .astype(str)
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def _release_protocol(context: Any) -> ScientificProtocol:
    config = context.config if isinstance(context.config, dict) else {}
    models = dict(config.get("models", {})) if isinstance(config.get("models", {}), dict) else {}
    pipeline = (
        dict(config.get("pipeline", {})) if isinstance(config.get("pipeline", {}), dict) else {}
    )
    models.setdefault("primary_model_name", "discovery_boosted")
    models.setdefault("primary_model_fallback", "parsimonious_priority")
    models.setdefault("conservative_model_name", "parsimonious_priority")
    models.setdefault("governance_model_name", "governance_linear")
    models.setdefault("governance_model_fallback", "support_synergy_priority")
    core_model_names = [
        str(name) for name in models.get("core_model_names", []) if str(name).strip()
    ]
    required_core_names = [
        str(models["primary_model_name"]),
        str(models["governance_model_name"]),
        str(models["conservative_model_name"]),
        "baseline_both",
    ]
    models["core_model_names"] = list(dict.fromkeys([*core_model_names, *required_core_names]))
    models.setdefault("research_model_names", [])
    models.setdefault("ablation_model_names", [])
    payload = {"pipeline": pipeline, "models": models}
    return ScientificProtocol.from_config(payload)


def _build_release_manifest(context: Any) -> dict[str, object]:
    protocol = _release_protocol(context)
    release_info_path = context.release_dir / "bundle" / "RELEASE_INFO.txt"
    decision_path = context.root / "reports/core_tables/single_model_official_decision.tsv"
    decision_status = "not_scored"
    if decision_path.exists():
        try:
            decision_frame = pd.read_csv(decision_path, sep="\t")
        except EmptyDataError:
            decision_frame = pd.DataFrame()
        if not decision_frame.empty:
            decision_status = str(
                decision_frame.iloc[0].get("scientific_acceptance_status", "not_scored")
                or "not_scored",
            ).strip()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(context.root),
        "bundle_root": relative_path_str(context.release_dir / "bundle", context.root),
        "n_files": 0,
        "files": [],
        "missing_files": [],
        "provenance": {
            "git_commit": _git_commit(context.root),
            "python_version": sys.version.split()[0],
            "project_version": _project_version(context.root),
            "lockfile_sha256": _sha256(context.root / "uv.lock")
            if (context.root / "uv.lock").exists()
            else "missing",
            "benchmark_contract_version": protocol.benchmark_contract_version,
            "benchmark_contract_hash": build_protocol_hash(protocol),
            "benchmark_scope": protocol.benchmark_scope,
            "scientific_acceptance_status": decision_status,
            "official_model_names": _official_model_names(context.root),
            "release_info_path": relative_path_str(release_info_path, context.root),
        },
    }


def _build_release_info(context_root: Path) -> str:
    metrics_path = context_root / "reports/core_tables/model_metrics.tsv"
    single_model_decision_path = (
        context_root / "reports/core_tables/single_model_official_decision.tsv"
    )
    official_summary_path = context_root / "reports/core_tables/official_model_summary.json"
    version = _project_version(context_root)
    if not metrics_path.exists():
        return (
            "\n".join(
                [
                    "Plasmid Priority Release Bundle",
                    f"Version: {version}",
                    f"Generated on: {datetime.now().date().isoformat()}",
                ],
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
        primary = metrics.loc[metrics["model_name"].astype(str) == "discovery_boosted"].head(1)
    if primary.empty:
        primary = metrics.sort_values("roc_auc", ascending=False).head(1)
    row = primary.iloc[0]
    permutation_text = "NA"
    selection_adjusted_text = "NA"
    permutation_p = pd.to_numeric(
        pd.Series([row.get("permutation_p_roc_auc")]),
        errors="coerce",
    ).iloc[0]
    selection_adjusted_p = pd.to_numeric(
        pd.Series([row.get("selection_adjusted_empirical_p_roc_auc")]),
        errors="coerce",
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
    fixed_score_n_permutations = pd.to_numeric(
        pd.Series([row.get("n_permutations", pd.NA)]),
        errors="coerce",
    ).iloc[0]
    selection_adjusted_n_permutations = pd.to_numeric(
        pd.Series([row.get("n_permutations_selection_adjusted", pd.NA)]),
        errors="coerce",
    ).iloc[0]
    fixed_score_n_permutations_text = (
        str(int(fixed_score_n_permutations)) if pd.notna(fixed_score_n_permutations) else "NA"
    )
    selection_adjusted_n_permutations_text = (
        str(int(selection_adjusted_n_permutations))
        if pd.notna(selection_adjusted_n_permutations)
        else "NA"
    )
    decision_reason = str(decision_row.get("decision_reason", "") or "").strip()
    decision_status = str(
        decision_row.get("scientific_acceptance_status", "not_scored") or "not_scored",
    ).strip()
    official_summary_lines: list[str] = []
    if official_summary_path.exists():
        try:
            official_summary = json.loads(official_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            official_summary = {}
        official_status = str(official_summary.get("official_model_family_status", "unknown"))
        decision_surface = str(official_summary.get("decision_surface", "unknown"))
        candidate_count = int(official_summary.get("candidate_count", 0) or 0)
        review_not_rank_count = int(official_summary.get("review_not_rank_count", 0) or 0)
        official_summary_lines = [
            f"Official model family status: {official_status}",
            f"Official decision surface: {decision_surface}",
            "Official candidate decisions: "
            f"{candidate_count} total; {review_not_rank_count} review_not_rank",
        ]
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
                "Selection-adjusted permutation p "
                f"{selection_adjusted_text} (n={selection_adjusted_n_permutations_text})",
                f"Fixed-score permutation p {permutation_text} (n={fixed_score_n_permutations_text})",
                *official_summary_lines,
            ],
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
        manifest = _build_release_manifest(context)
        copy_targets: list[Path] = []
        for relpath in RELEASE_FILES:
            copy_targets.append(context.root / relpath)
        for pattern in RELEASE_GLOB_FILES:
            copy_targets.extend(sorted(context.root.glob(pattern)))

        seen_relpaths: set[str] = set()
        for source in copy_targets:
            try:
                relpath = relative_path_str(source, context.root)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Caught suppressed exception: %s", exc, exc_info=True)
                continue
            if relpath in seen_relpaths:
                continue
            seen_relpaths.add(relpath)
            source = context.root / relpath
            if not source.exists():
                missing.append(relpath)
                continue
            destination = bundle_dir / relpath
            ensure_directory(destination.parent)
            shutil.copy2(source, destination)
            os.utime(destination, None)
            stat = source.stat()
            manifest_rows.append(
                {
                    "relative_path": relpath,
                    "size_bytes": int(stat.st_size),
                    "sha256": _sha256(source),
                },
            )
            run.record_input(source)
            run.record_output(destination)

        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["project_root"] = str(context.root)
        manifest["bundle_root"] = relative_path_str(bundle_dir, context.root)
        manifest["n_files"] = len(manifest_rows)
        manifest["files"] = manifest_rows
        manifest["missing_files"] = missing
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
                ],
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
