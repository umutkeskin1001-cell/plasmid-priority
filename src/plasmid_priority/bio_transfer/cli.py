"""CLI entrypoint for the bio transfer branch."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from plasmid_priority.bio_transfer.calibration import (
    build_bio_transfer_calibrated_prediction_table,
    build_bio_transfer_calibration_summary,
)
from plasmid_priority.bio_transfer.dataset import prepare_bio_transfer_scored_table
from plasmid_priority.bio_transfer.evaluate import (
    build_bio_transfer_model_summary,
    build_bio_transfer_prediction_table,
    evaluate_bio_transfer_branch,
)
from plasmid_priority.bio_transfer.provenance import build_bio_transfer_run_provenance
from plasmid_priority.bio_transfer.report import (
    build_bio_transfer_report_card,
    format_bio_transfer_report_markdown,
)
from plasmid_priority.bio_transfer.specs import (
    load_bio_transfer_config,
    resolve_bio_transfer_model_names,
)
from plasmid_priority.config import build_context
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
from plasmid_priority.utils.managed_run import ManagedScriptRun


def _sync_file(source: Path, destination: Path) -> None:
    ensure_directory(destination.parent)
    if not source.exists():
        return
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if source_stat.st_size == destination_stat.st_size and int(source_stat.st_mtime) <= int(
            destination_stat.st_mtime
        ):
            return
    shutil.copy2(source, destination)


def _sync_config_layers(
    context_root: Path,
    runtime_root: Path,
    config_paths: tuple[Path, ...],
) -> list[Path]:
    synced_paths: list[Path] = []
    for config_path in config_paths:
        destination = runtime_root / config_path.relative_to(context_root)
        _sync_file(config_path, destination)
        synced_paths.append(destination)
    return synced_paths


def _branch_paths(context_root: Path, data_root: Path) -> dict[str, Path]:
    branch_root = data_root / "bio_transfer"
    return {
        "branch_root": branch_root,
        "analysis_dir": branch_root / "analysis",
        "input_dir": branch_root / "inputs",
        "runtime_dir": branch_root / "runtime",
        "inventory_dir": branch_root / "inventory",
        "legacy_scored_path": data_root / "scores" / "backbone_scored.tsv",
        "legacy_records_path": data_root / "silver" / "plasmid_harmonized.tsv",
        "branch_scored_path": branch_root / "inputs" / "backbone_scored.tsv",
        "branch_records_path": branch_root / "inputs" / "plasmid_harmonized.tsv",
        "branch_config_path": branch_root / "runtime" / "config.yaml",
        "root_config_path": context_root / "config.yaml",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bio transfer branch.")
    parser.add_argument(
        "--research-models", action="store_true", help="Also evaluate the research model."
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    context = build_context(project_root)
    bio_config = load_bio_transfer_config(context.config)
    paths = _branch_paths(context.root, context.data_dir)
    _sync_file(paths["legacy_scored_path"], paths["branch_scored_path"])
    _sync_file(paths["legacy_records_path"], paths["branch_records_path"])
    _sync_file(paths["root_config_path"], paths["branch_config_path"])
    config_snapshot_paths = _sync_config_layers(
        context.root, paths["runtime_dir"], context.config_paths
    )

    metrics_path = paths["analysis_dir"] / "bio_transfer_metrics.json"
    summary_path = paths["analysis_dir"] / "bio_transfer_model_summary.tsv"
    predictions_path = paths["analysis_dir"] / "bio_transfer_predictions.tsv"
    calibration_summary_path = paths["analysis_dir"] / "bio_transfer_calibration_summary.tsv"
    calibrated_predictions_path = paths["analysis_dir"] / "bio_transfer_calibrated_predictions.tsv"
    provenance_path = paths["analysis_dir"] / "bio_transfer_provenance.json"
    report_card_path = paths["analysis_dir"] / "bio_transfer_report_card.tsv"
    report_markdown_path = paths["analysis_dir"] / "bio_transfer_report.md"
    manifest_path = paths["analysis_dir"] / "run_bio_transfer_branch.manifest.json"
    used_inventory_path = paths["inventory_dir"] / "used_paths.tsv"
    unused_inventory_path = paths["inventory_dir"] / "unused_paths.tsv"
    inventory_summary_path = paths["inventory_dir"] / "summary.json"
    ensure_directory(metrics_path.parent)
    ensure_directory(paths["inventory_dir"])
    source_paths = project_python_source_paths(
        project_root,
        script_path=project_root / "scripts" / "run_bio_transfer_branch.py",
    )
    model_names = resolve_bio_transfer_model_names(
        context.config, include_research=bool(args.research_models)
    )
    cache_metadata = {
        "model_names": list(model_names),
        "research_models": bool(args.research_models),
        "jobs": int(args.jobs),
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "horizon_years": int(bio_config.benchmark.horizon_years),
        },
    }

    with ManagedScriptRun(context, "run_bio_transfer_branch") as run:
        run.record_input(paths["branch_scored_path"])
        run.record_input(paths["branch_records_path"])
        for config_path in config_snapshot_paths:
            run.record_input(config_path)
        for output_path in (
            metrics_path,
            summary_path,
            predictions_path,
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
                paths["branch_scored_path"],
                paths["branch_records_path"],
                *config_snapshot_paths,
            ],
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(paths["branch_scored_path"])
        records = (
            read_tsv(paths["branch_records_path"])
            if paths["branch_records_path"].exists()
            else pd.DataFrame()
        )
        prepared_scored = prepare_bio_transfer_scored_table(
            scored, config=context.config, records=records
        )
        results = evaluate_bio_transfer_branch(
            scored,
            model_names=model_names,
            n_jobs=max(int(args.jobs), 1),
            config=context.config,
            records=records,
        )
        summary = build_bio_transfer_model_summary(results)
        predictions = build_bio_transfer_prediction_table(results)
        calibration_summary = build_bio_transfer_calibration_summary(
            results, scored=prepared_scored, config=context.config
        )
        recommended_primary_model_name, selection_scorecard = select_branch_primary_model(
            results, calibration_summary=calibration_summary
        )
        calibrated_predictions = build_bio_transfer_calibrated_prediction_table(
            results, scored=prepared_scored, config=context.config
        )
        provenance = build_bio_transfer_run_provenance(
            prepared_scored,
            model_names=tuple(results),
            config=context.config,
            script_name="bio_transfer/run_branch",
            source_paths=source_paths,
            recommended_primary_model_name=recommended_primary_model_name,
            calibration_summary=calibration_summary,
            predictions=predictions,
            calibrated_predictions=calibrated_predictions,
        )
        report_card = build_bio_transfer_report_card(
            results,
            calibration_summary=calibration_summary,
            provenance=provenance,
            selection_scorecard=selection_scorecard,
        )
        report_markdown = format_bio_transfer_report_markdown(report_card, provenance=provenance)
        metrics_payload = {
            "primary_model_name": recommended_primary_model_name or bio_config.primary_model_name,
            "configured_primary_model_name": bio_config.primary_model_name,
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
        used_inventory, unused_inventory, inventory_summary = build_branch_inventory(
            project_root,
            used_paths=[
                paths["branch_scored_path"],
                paths["branch_records_path"],
                paths["branch_config_path"],
            ],
            data_root=context.data_dir,
        )

        atomic_write_json(metrics_path, metrics_payload)
        summary.to_csv(summary_path, sep="\t", index=False)
        predictions.to_csv(predictions_path, sep="\t", index=False)
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
                paths["branch_scored_path"],
                paths["branch_records_path"],
                paths["branch_config_path"],
            ],
            output_paths=[
                metrics_path,
                summary_path,
                predictions_path,
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
