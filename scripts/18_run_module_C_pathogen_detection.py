#!/usr/bin/env python3
"""Supportive descriptive analysis against Pathogen Detection metadata."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import get_active_model_names, get_primary_model_name
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_pathogen_detection_support,
    build_pathogen_strata_group_summary,
    build_pathogen_targets,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    materialize_recorded_paths,
    project_python_source_paths,
    write_signature_manifest,
)


def _build_pathogen_support_task(
    targets: pd.DataFrame,
    metadata_path: Path,
    dataset_name: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    detail, summary = build_pathogen_detection_support(targets, metadata_path)
    detail = detail.copy()
    summary = summary.copy()
    detail.insert(0, "pathogen_dataset", dataset_name)
    summary.insert(0, "pathogen_dataset", dataset_name)
    return dataset_name, detail, summary


def _run_pathogen_support(
    *,
    run: ManagedScriptRun,
    targets: pd.DataFrame,
    metadata_path: Path,
    detail_path: Path,
    summary_path: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run.record_input(metadata_path)
    detail, summary = build_pathogen_detection_support(targets, metadata_path)
    detail = detail.copy()
    summary = summary.copy()
    detail.insert(0, "pathogen_dataset", dataset_name)
    summary.insert(0, "pathogen_dataset", dataset_name)
    detail.to_csv(detail_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    run.record_output(detail_path)
    run.record_output(summary_path)
    run.set_rows_out(f"{dataset_name}_pathogen_detection_detail_rows", int(len(detail)))
    run.set_rows_out(f"{dataset_name}_pathogen_detection_summary_rows", int(len(summary)))
    return detail, summary


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    metrics_path = context.data_dir / "analysis/module_a_metrics.json"
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    config_paths = context_config_paths(context)
    manifest_path = context.data_dir / "analysis/18_run_module_c_pathogen_detection.manifest.json"
    pd_metadata_path = context.asset_path("pathogen_detection_metadata")
    pd_clinical_path = context.asset_path("pathogen_detection_clinical")
    pd_environmental_path = context.asset_path("pathogen_detection_environmental")
    detail_path = context.data_dir / "analysis/pathogen_detection_support.tsv"
    summary_path = context.data_dir / "analysis/pathogen_detection_group_summary.tsv"
    clinical_detail_path = context.data_dir / "analysis/pathogen_detection_clinical_support.tsv"
    clinical_summary_path = (
        context.data_dir / "analysis/pathogen_detection_clinical_group_summary.tsv"
    )
    environmental_detail_path = (
        context.data_dir / "analysis/pathogen_detection_environmental_support.tsv"
    )
    environmental_summary_path = (
        context.data_dir / "analysis/pathogen_detection_environmental_group_summary.tsv"
    )
    strata_summary_path = context.data_dir / "analysis/pathogen_detection_strata_group_summary.tsv"
    ensure_directory(detail_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/18_run_module_C_pathogen_detection.py",
    )
    dataset_specs = [
        ("combined", pd_metadata_path, detail_path, summary_path),
        ("clinical", pd_clinical_path, clinical_detail_path, clinical_summary_path),
        (
            "environmental",
            pd_environmental_path,
            environmental_detail_path,
            environmental_summary_path,
        ),
    ]
    active_specs = [
        (dataset_name, metadata_path, dataset_detail_path, dataset_summary_path)
        for dataset_name, metadata_path, dataset_detail_path, dataset_summary_path in dataset_specs
        if dataset_name == "combined" or metadata_path.exists()
    ]
    input_paths = [
        scored_path,
        backbones_path,
        amr_consensus_path,
        metrics_path,
        predictions_path,
        *config_paths,
        *(metadata_path for _, metadata_path, _, _ in active_specs),
    ]
    cache_metadata = {
        "active_pathogen_datasets": [dataset_name for dataset_name, *_ in active_specs],
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        },
    }

    with ManagedScriptRun(context, "18_run_module_C_pathogen_detection") as run:
        for path in input_paths:
            run.record_input(path)
        for _, _, dataset_detail_path, dataset_summary_path in active_specs:
            run.record_output(dataset_detail_path)
            run.record_output(dataset_summary_path)
        run.record_output(strata_summary_path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note(
                "Inputs, code, and config unchanged; reusing cached Pathogen Detection support outputs."
            )
            run.set_metric("cache_hit", True)
            return 0
        run.note(
            "Supportive descriptive analysis only. This module is not treated as independent validation."
        )
        run.note(
            "Clinical and environmental Pathogen Detection strata are analyzed separately when local metadata tables are present."
        )

        scored = read_tsv(scored_path)
        backbones = read_tsv(
            backbones_path,
            usecols=["backbone_id", "sequence_accession", "genus", "species"],
        )
        amr_consensus = read_tsv(amr_consensus_path)
        with metrics_path.open("r", encoding="utf-8") as handle:
            model_metrics = json.load(handle)
        predictions = read_tsv(predictions_path)
        primary_model_name = get_primary_model_name(get_active_model_names(model_metrics))
        primary_scores = predictions.loc[
            predictions["model_name"] == primary_model_name,
            ["backbone_id", "oof_prediction"],
        ].rename(columns={"oof_prediction": "primary_model_oof_prediction"})
        scored = scored.merge(primary_scores, on="backbone_id", how="left")
        eligible_count = int(scored["spread_label"].notna().sum())
        n_per_group = max(25, int(round(eligible_count * 0.25))) if eligible_count > 0 else 25
        run.note(
            f"Pathogen Detection contrasts use headline-model quartile extremes (top/bottom {n_per_group}) rather than a fixed top/bottom 100."
        )

        targets = build_pathogen_targets(
            scored,
            backbones,
            amr_consensus,
            n_per_group=n_per_group,
            score_column="primary_model_oof_prediction",
            eligible_only=True,
        )
        summary_frames: dict[str, pd.DataFrame] = {}

        for dataset_name, metadata_path, dataset_detail_path, dataset_summary_path in dataset_specs:
            if dataset_name != "combined" and not metadata_path.exists():
                run.warn(
                    f"Optional Pathogen Detection {dataset_name} table not found: {metadata_path}"
                )
                continue

        dataset_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        worker_override = os.environ.get("PLASMID_PRIORITY_PATHOGEN_WORKERS")
        if worker_override:
            try:
                configured_workers = max(1, int(worker_override))
            except ValueError:
                configured_workers = 3
        else:
            configured_workers = 3
        max_workers = min(len(active_specs), max(1, os.cpu_count() or 1), configured_workers)
        if max_workers <= 1:
            for dataset_name, metadata_path, _, _ in active_specs:
                _, detail, summary = _build_pathogen_support_task(
                    targets, metadata_path, dataset_name
                )
                dataset_results[dataset_name] = (detail, summary)
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            _build_pathogen_support_task, targets, metadata_path, dataset_name
                        ): (
                            dataset_name,
                            dataset_detail_path,
                            dataset_summary_path,
                        )
                        for dataset_name, metadata_path, dataset_detail_path, dataset_summary_path in active_specs
                    }
                    for future in as_completed(future_map):
                        dataset_name, _, _ = future_map[future]
                        result_name, detail, summary = future.result()
                        dataset_results[result_name] = (detail, summary)
            except (PermissionError, OSError, NotImplementedError):
                run.warn(
                    "ProcessPoolExecutor unavailable in this environment; falling back to serial Pathogen Detection support."
                )
                for dataset_name, metadata_path, _, _ in active_specs:
                    _, detail, summary = _build_pathogen_support_task(
                        targets, metadata_path, dataset_name
                    )
                    dataset_results[dataset_name] = (detail, summary)

        detail = pd.DataFrame()
        for dataset_name, _, dataset_detail_path, dataset_summary_path in active_specs:
            dataset_detail, dataset_summary = dataset_results[dataset_name]
            dataset_detail.to_csv(dataset_detail_path, sep="\t", index=False)
            dataset_summary.to_csv(dataset_summary_path, sep="\t", index=False)
            run.record_output(dataset_detail_path)
            run.record_output(dataset_summary_path)
            run.set_rows_out(
                f"{dataset_name}_pathogen_detection_detail_rows", int(len(dataset_detail))
            )
            run.set_rows_out(
                f"{dataset_name}_pathogen_detection_summary_rows", int(len(dataset_summary))
            )
            summary_frames[dataset_name] = dataset_summary.drop(columns=["pathogen_dataset"])
            if dataset_name == "combined":
                detail = dataset_detail

        strata_summary = build_pathogen_strata_group_summary(summary_frames)
        if not strata_summary.empty:
            strata_summary.to_csv(strata_summary_path, sep="\t", index=False)
            run.set_rows_out("pathogen_detection_strata_summary_rows", int(len(strata_summary)))

        run.set_metric("high_priority_targets", int((detail["priority_group"] == "high").sum()))
        run.set_metric("low_priority_targets", int((detail["priority_group"] == "low").sum()))
        write_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            output_paths=materialize_recorded_paths(context.root, run.output_files_written),
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
