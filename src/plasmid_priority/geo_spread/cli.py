"""CLI entrypoint for the geo spread branch."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.geo_spread.calibration import (
    build_geo_spread_calibrated_prediction_table,
    build_geo_spread_calibration_summary,
)
from plasmid_priority.geo_spread.evaluate import (
    build_geo_spread_model_summary,
    build_geo_spread_prediction_table,
    evaluate_geo_spread_branch,
)
from plasmid_priority.geo_spread.inventory import build_geo_spread_inventory
from plasmid_priority.geo_spread.provenance import build_geo_spread_run_provenance
from plasmid_priority.geo_spread.report import (
    build_geo_spread_report_card,
    format_geo_spread_report_markdown,
)
from plasmid_priority.geo_spread.select import select_geo_spread_primary_model
from plasmid_priority.geo_spread.specs import load_geo_spread_config, resolve_geo_spread_model_names
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
    branch_root = data_root / "geo_spread"
    return {
        "branch_root": branch_root,
        "analysis_dir": branch_root / "analysis",
        "input_dir": branch_root / "inputs",
        "runtime_dir": branch_root / "runtime",
        "inventory_dir": branch_root / "inventory",
        "legacy_scored_path": data_root / "scores" / "backbone_scored.tsv",
        "legacy_records_path": data_root / "silver" / "plasmid_backbones.tsv",
        "branch_scored_path": branch_root / "inputs" / "backbone_scored.tsv",
        "branch_records_path": branch_root / "inputs" / "plasmid_backbones.tsv",
        "branch_config_path": branch_root / "runtime" / "config.yaml",
        "root_config_path": context_root / "config.yaml",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the geo spread branch.")
    parser.add_argument(
        "--research-models",
        action="store_true",
        help="Also evaluate the research-only geo spread model.",
    )
    parser.add_argument(
        "--ablation-models",
        action="store_true",
        help="Also evaluate any geo spread ablation models.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers to use when evaluating the branch model surface.",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    context = build_context(project_root)
    geo_config = load_geo_spread_config(context.config)
    paths = _branch_paths(context.root, context.data_dir)
    _sync_file(paths["legacy_scored_path"], paths["branch_scored_path"])
    _sync_file(paths["legacy_records_path"], paths["branch_records_path"])
    _sync_file(paths["root_config_path"], paths["branch_config_path"])
    config_snapshot_paths = _sync_config_layers(
        context.root, paths["runtime_dir"], context.config_paths
    )

    metrics_path = paths["analysis_dir"] / "geo_spread_metrics.json"
    summary_path = paths["analysis_dir"] / "geo_spread_model_summary.tsv"
    predictions_path = paths["analysis_dir"] / "geo_spread_predictions.tsv"
    calibration_summary_path = paths["analysis_dir"] / "geo_spread_calibration_summary.tsv"
    calibrated_predictions_path = paths["analysis_dir"] / "geo_spread_calibrated_predictions.tsv"
    provenance_path = paths["analysis_dir"] / "geo_spread_provenance.json"
    report_card_path = paths["analysis_dir"] / "geo_spread_report_card.tsv"
    report_markdown_path = paths["analysis_dir"] / "geo_spread_report.md"
    manifest_path = paths["analysis_dir"] / "run_geo_spread_branch.manifest.json"
    used_inventory_path = paths["inventory_dir"] / "used_paths.tsv"
    unused_inventory_path = paths["inventory_dir"] / "unused_paths.tsv"
    inventory_summary_path = paths["inventory_dir"] / "summary.json"
    ensure_directory(metrics_path.parent)
    ensure_directory(paths["inventory_dir"])
    source_paths = project_python_source_paths(
        project_root,
        script_path=project_root / "scripts" / "geo_spread" / "run_branch.py",
    )
    model_names = resolve_geo_spread_model_names(
        context.config,
        include_research=bool(args.research_models),
        include_ablation=bool(args.ablation_models),
    )
    cache_metadata = {
        "model_names": list(model_names),
        "research_models": bool(args.research_models),
        "ablation_models": bool(args.ablation_models),
        "jobs": int(args.jobs),
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        },
    }

    with ManagedScriptRun(context, "run_geo_spread_branch") as run:
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
            run.note("Inputs, code, and config unchanged; reusing cached geo spread outputs.")
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(paths["branch_scored_path"])
        records = (
            read_tsv(paths["branch_records_path"])
            if paths["branch_records_path"].exists()
            else pd.DataFrame()
        )
        results = evaluate_geo_spread_branch(
            scored,
            model_names=model_names,
            n_jobs=max(int(args.jobs), 1),
            config=context.config,
            records=records,
        )
        summary = build_geo_spread_model_summary(results)
        predictions = build_geo_spread_prediction_table(results)
        calibration_summary = build_geo_spread_calibration_summary(
            results,
            scored=scored,
            config=context.config,
        )
        recommended_primary_model_name, selection_scorecard = select_geo_spread_primary_model(
            results,
            calibration_summary=calibration_summary,
        )
        calibrated_predictions = build_geo_spread_calibrated_prediction_table(
            results,
            scored=scored,
            config=context.config,
        )
        provenance = build_geo_spread_run_provenance(
            scored,
            model_names=tuple(results),
            config=context.config,
            script_name="geo_spread/run_branch",
            source_paths=source_paths,
            recommended_primary_model_name=recommended_primary_model_name,
            calibration_summary=calibration_summary,
            predictions=predictions,
            calibrated_predictions=calibrated_predictions,
        )
        report_card = build_geo_spread_report_card(
            results,
            calibration_summary=calibration_summary,
            provenance=provenance,
            selection_scorecard=selection_scorecard,
        )
        report_markdown = format_geo_spread_report_markdown(
            report_card,
            provenance=provenance,
        )
        best_predictive_model_name = (
            str(report_card.iloc[0]["model_name"]) if not report_card.empty else ""
        )
        per_model_metrics = {
            str(name): {
                **{str(key): value for key, value in cast(dict[str, Any], result.metrics).items()},
                "status": result.status,
                "error_message": result.error_message,
            }
            for name, result in results.items()
        }
        metrics_payload: dict[str, Any] = {
            "primary_model_name": recommended_primary_model_name or geo_config.primary_model_name,
            "configured_primary_model_name": geo_config.primary_model_name,
            "recommended_primary_model_name": recommended_primary_model_name,
            "best_predictive_model_name": best_predictive_model_name,
            "model_names": list(model_names),
            "selection_scorecard": (
                selection_scorecard.to_dict(orient="records")
                if not selection_scorecard.empty
                else []
            ),
        }
        metrics_payload.update(per_model_metrics)
        used_inventory, unused_inventory, inventory_summary = build_geo_spread_inventory(
            project_root,
            branch_data_root=paths["branch_root"],
            legacy_input_path=paths["legacy_scored_path"],
            legacy_records_path=paths["legacy_records_path"],
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
            metadata={**cache_metadata, "provenance_hash": provenance["run_signature"]},
        )
        run.set_rows_out("geo_spread_prediction_rows", int(len(predictions)))
        run.set_rows_out("geo_spread_calibrated_prediction_rows", int(len(calibrated_predictions)))
        run.set_rows_out("geo_spread_report_card_rows", int(len(report_card)))
        run.set_metric("models_run", len(results))
        run.set_metric("used_path_count", int(len(used_inventory)))
        run.set_metric("unused_path_count", int(len(unused_inventory)))
        run.set_metric("geo_spread_provenance_hash", provenance["run_signature"])
        run.set_metric("cache_hit", False)
    return 0
