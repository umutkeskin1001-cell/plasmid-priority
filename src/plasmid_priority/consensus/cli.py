"""CLI entrypoint for the consensus branch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.consensus.calibration import (
    build_consensus_calibrated_prediction_table,
    build_consensus_calibration_summary,
)
from plasmid_priority.consensus.evaluate import (
    build_consensus_model_summary,
    build_consensus_prediction_table,
    evaluate_consensus_branch,
)
from plasmid_priority.consensus.provenance import build_consensus_run_provenance
from plasmid_priority.consensus.report import (
    build_consensus_report_card,
    format_consensus_report_markdown,
)
from plasmid_priority.consensus.specs import load_consensus_config, resolve_consensus_model_names
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.shared.data_inventory import build_branch_inventory
from plasmid_priority.shared.selection import select_branch_primary_model
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)


def _branch_paths(context_root: Path, data_root: Path) -> dict[str, Path]:
    branch_root = data_root / "consensus"
    return {
        "branch_root": branch_root,
        "analysis_dir": branch_root / "analysis",
        "runtime_dir": branch_root / "runtime",
        "inventory_dir": branch_root / "inventory",
        "geo_predictions_path": data_root
        / "geo_spread"
        / "analysis"
        / "geo_spread_calibrated_predictions.tsv",
        "bio_predictions_path": data_root
        / "bio_transfer"
        / "analysis"
        / "bio_transfer_calibrated_predictions.tsv",
        "clinical_predictions_path": data_root
        / "clinical_hazard"
        / "analysis"
        / "clinical_hazard_calibrated_predictions.tsv",
        "branch_config_path": branch_root / "runtime" / "config.yaml",
        "root_config_path": context_root / "config.yaml",
    }


def _sync_config_layers(
    context_root: Path,
    runtime_root: Path,
    config_paths: tuple[Path, ...],
) -> list[Path]:
    synced_paths: list[Path] = []
    for config_path in config_paths:
        destination = runtime_root / config_path.relative_to(context_root)
        ensure_directory(destination.parent)
        if config_path.exists():
            destination.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        synced_paths.append(destination)
    return synced_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the consensus branch.")
    parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    context = build_context(project_root)
    consensus_config = load_consensus_config(context.config)
    paths = _branch_paths(context.root, context.data_dir)
    ensure_directory(paths["analysis_dir"])
    ensure_directory(paths["inventory_dir"])
    ensure_directory(paths["branch_config_path"].parent)
    if paths["root_config_path"].exists():
        paths["branch_config_path"].write_text(
            paths["root_config_path"].read_text(encoding="utf-8"), encoding="utf-8"
        )
    config_snapshot_paths = _sync_config_layers(
        context.root, paths["runtime_dir"], context.config_paths
    )

    metrics_path = paths["analysis_dir"] / "consensus_metrics.json"
    summary_path = paths["analysis_dir"] / "consensus_model_summary.tsv"
    predictions_path = paths["analysis_dir"] / "consensus_predictions.tsv"
    breakdown_path = paths["analysis_dir"] / "consensus_branch_breakdown.tsv"
    watchlist_path = paths["analysis_dir"] / "consensus_review_watchlist.tsv"
    calibration_summary_path = paths["analysis_dir"] / "consensus_calibration_summary.tsv"
    calibrated_predictions_path = paths["analysis_dir"] / "consensus_calibrated_predictions.tsv"
    provenance_path = paths["analysis_dir"] / "consensus_provenance.json"
    report_card_path = paths["analysis_dir"] / "consensus_report_card.tsv"
    report_markdown_path = paths["analysis_dir"] / "consensus_report.md"
    manifest_path = paths["analysis_dir"] / "run_consensus_branch.manifest.json"
    used_inventory_path = paths["inventory_dir"] / "used_paths.tsv"
    unused_inventory_path = paths["inventory_dir"] / "unused_paths.tsv"
    inventory_summary_path = paths["inventory_dir"] / "summary.json"
    source_paths = project_python_source_paths(
        project_root,
        script_path=project_root / "scripts" / "run_consensus_branch.py",
    )
    model_names = resolve_consensus_model_names(context.config, include_research=True)
    cache_metadata = {
        "model_names": list(model_names),
        "jobs": int(os.cpu_count() or 1),
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "horizon_years": int(consensus_config.benchmark.horizon_years),
        },
    }

    with ManagedScriptRun(context, "run_consensus_branch") as run:
        for input_path in (
            paths["geo_predictions_path"],
            paths["bio_predictions_path"],
            paths["clinical_predictions_path"],
            *config_snapshot_paths,
        ):
            run.record_input(input_path)
        for output_path in (
            metrics_path,
            summary_path,
            predictions_path,
            breakdown_path,
            watchlist_path,
            calibration_summary_path,
            calibrated_predictions_path,
            provenance_path,
            report_card_path,
            report_markdown_path,
            used_inventory_path,
            unused_inventory_path,
            inventory_summary_path,
        ):
            run.record_output(output_path)
        if load_signature_manifest(
            manifest_path,
            input_paths=[
                paths["geo_predictions_path"],
                paths["bio_predictions_path"],
                paths["clinical_predictions_path"],
                *config_snapshot_paths,
            ],
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.set_metric("cache_hit", True)
            return 0

        geo = (
            read_tsv(paths["geo_predictions_path"])
            if paths["geo_predictions_path"].exists()
            else pd.DataFrame()
        )
        bio = (
            read_tsv(paths["bio_predictions_path"])
            if paths["bio_predictions_path"].exists()
            else pd.DataFrame()
        )
        clinical = (
            read_tsv(paths["clinical_predictions_path"])
            if paths["clinical_predictions_path"].exists()
            else pd.DataFrame()
        )
        from plasmid_priority.consensus.dataset import prepare_consensus_dataset

        dataset = prepare_consensus_dataset(geo, bio, clinical, config=context.config)
        results = evaluate_consensus_branch(
            dataset.table, model_names=model_names, config=context.config
        )
        summary = build_consensus_model_summary(results)
        predictions = build_consensus_prediction_table(results)
        calibration_summary = build_consensus_calibration_summary(
            results, scored=dataset.table, config=context.config
        )
        recommended_primary_model_name, selection_scorecard = select_branch_primary_model(
            results, calibration_summary=calibration_summary
        )
        calibrated_predictions = build_consensus_calibrated_prediction_table(
            results, scored=dataset.table, config=context.config
        )
        provenance = build_consensus_run_provenance(
            dataset.table,
            model_names=tuple(results),
            config=context.config,
            script_name="consensus/run_branch",
            source_paths=source_paths,
            recommended_primary_model_name=recommended_primary_model_name,
            calibration_summary=calibration_summary,
            predictions=predictions,
            calibrated_predictions=calibrated_predictions,
        )
        report_card = build_consensus_report_card(
            results,
            calibration_summary=calibration_summary,
            provenance=provenance,
            selection_scorecard=selection_scorecard,
        )
        report_markdown = format_consensus_report_markdown(report_card, provenance=provenance)
        used_inventory, unused_inventory, inventory_summary = build_branch_inventory(
            project_root,
            used_paths=[
                paths["geo_predictions_path"],
                paths["bio_predictions_path"],
                paths["clinical_predictions_path"],
                paths["branch_config_path"],
            ],
            data_root=context.data_dir,
        )
        metrics_payload = {
            "primary_model_name": recommended_primary_model_name
            or consensus_config.primary_model_name,
            "configured_primary_model_name": consensus_config.primary_model_name,
            "recommended_primary_model_name": recommended_primary_model_name,
            "model_names": list(model_names),
            "selection_scorecard": selection_scorecard.to_dict(orient="records")
            if not selection_scorecard.empty
            else [],
            **{
                name: {
                    **result.metrics,
                    "status": result.status,
                    "error_message": result.error_message,
                }
                for name, result in results.items()
            },
        }

        atomic_write_json(metrics_path, metrics_payload)
        summary.to_csv(summary_path, sep="\t", index=False)
        predictions.to_csv(predictions_path, sep="\t", index=False)
        build_consensus_prediction_table(results).to_csv(breakdown_path, sep="\t", index=False)
        watchlist = build_consensus_prediction_table(results)
        if "consensus_review_flag" in watchlist.columns:
            watchlist = watchlist.loc[
                watchlist["consensus_review_flag"].fillna(False).astype(bool)
            ].copy()
        watchlist.to_csv(watchlist_path, sep="\t", index=False)
        calibration_summary.to_csv(calibration_summary_path, sep="\t", index=False)
        calibrated_predictions.to_csv(calibrated_predictions_path, sep="\t", index=False)
        report_card.to_csv(report_card_path, sep="\t", index=False)
        report_markdown_path.write_text(report_markdown, encoding="utf-8")
        atomic_write_json(provenance_path, provenance)
        used_inventory.to_csv(used_inventory_path, sep="\t", index=False)
        unused_inventory.to_csv(unused_inventory_path, sep="\t", index=False)
        atomic_write_json(inventory_summary_path, inventory_summary)
        write_signature_manifest(
            manifest_path,
            input_paths=[
                paths["geo_predictions_path"],
                paths["bio_predictions_path"],
                paths["clinical_predictions_path"],
                paths["branch_config_path"],
            ],
            output_paths=[
                metrics_path,
                summary_path,
                predictions_path,
                breakdown_path,
                watchlist_path,
                calibration_summary_path,
                calibrated_predictions_path,
                provenance_path,
                report_card_path,
                report_markdown_path,
                used_inventory_path,
                unused_inventory_path,
                inventory_summary_path,
            ],
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        return 0
