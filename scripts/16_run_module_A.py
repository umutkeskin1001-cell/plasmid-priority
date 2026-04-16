#!/usr/bin/env python3
"""Run the primary retrospective modeling stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    assert_feature_columns_present,
    build_discovery_input_contract,
    get_conservative_model_name,
    get_module_a_model_names,
    get_official_model_names,
    get_primary_model_name,
    run_module_a,
    validate_discovery_input_contract,
)
from plasmid_priority.reporting import (
    ManagedScriptRun,
    augment_scored_with_structural_audit_features,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the primary retrospective modeling stack.")
    parser.add_argument(
        "--research-models",
        action="store_true",
        help="Also evaluate the research-only named models that are excluded from the default clean benchmark set.",
    )
    parser.add_argument(
        "--ablation-models",
        action="store_true",
        help="Also evaluate the T/H/A ablation models.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers to use when evaluating the model family.",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Run only the three jury-facing official models: discovery, governance, and baseline.",
    )
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    metrics_path = context.data_dir / "analysis/module_a_metrics.json"
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    manifest_path = context.data_dir / "analysis/16_run_module_a.manifest.json"
    config_paths = context_config_paths(context)
    raw_amr_path = context.data_dir / "raw/amr.tsv"
    mash_pairs_path = context.data_dir / "raw/plsdb_mashdb_sim.tsv"
    ensure_directory(metrics_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/16_run_module_A.py",
    )
    input_paths = [scored_path, *config_paths]
    for optional_input in (backbones_path, raw_amr_path, mash_pairs_path):
        if optional_input.exists():
            input_paths.append(optional_input)
    cache_metadata = {
        "research_models": bool(args.research_models),
        "ablation_models": bool(args.ablation_models),
        "official_only": bool(args.official_only),
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        },
    }

    with ManagedScriptRun(context, "16_run_module_A") as run:
        run.record_input(scored_path)
        for path in config_paths:
            run.record_input(path)
        run.record_output(metrics_path)
        run.record_output(predictions_path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached Module A outputs.")
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(scored_path)
        backbone_records = (
            read_tsv(
                backbones_path,
                usecols=["backbone_id", "sequence_accession", "resolved_year"],
            )
            if backbones_path.exists()
            else pd.DataFrame()
        )
        raw_amr = (
            read_tsv(
                raw_amr_path,
                usecols=["NUCCORE_ACC", "analysis_software_name", "gene_symbol", "drug_class"],
            )
            if raw_amr_path.exists()
            else pd.DataFrame()
        )
        mash_pairs = (
            read_tsv(
                mash_pairs_path,
                header=None,
                names=["source_accession", "target_accession"],
                usecols=[0, 1],
            )
            if mash_pairs_path.exists()
            else pd.DataFrame()
        )
        scored = augment_scored_with_structural_audit_features(
            scored,
            records=backbone_records,
            raw_amr=raw_amr,
            mash_pairs=mash_pairs,
            split_year=int(context.pipeline_settings.split_year),
        )
        model_names = get_module_a_model_names(
            include_research=args.research_models,
            include_ablations=args.ablation_models,
        )
        if args.official_only:
            model_names = get_official_model_names(model_names)
        required_columns = [
            column for model_name in model_names for column in MODULE_A_FEATURE_SETS[model_name]
        ]
        assert_feature_columns_present(
            scored,
            required_columns,
            label="Module A score input",
        )
        validate_discovery_input_contract(
            scored,
            model_names=model_names,
            contract=build_discovery_input_contract(int(context.pipeline_settings.split_year)),
            label="Module A score input",
        )
        results = run_module_a(
            scored,
            model_names=model_names,
            n_jobs=max(int(args.jobs), 1),
        )

        metrics_payload = {
            "primary_model_name": get_primary_model_name(list(results)),
            "conservative_model_name": get_conservative_model_name(list(results)),
        }
        metrics_payload.update(
            {
                name: {
                    **result.metrics,
                    "status": result.status,
                    "error_message": result.error_message,
                }
                for name, result in results.items()
            }
        )
        predictions = []
        for name, result in results.items():
            preds = result.predictions.copy()
            preds["model_name"] = name
            predictions.append(preds)
        prediction_table = pd.concat(predictions, ignore_index=True)

        atomic_write_json(metrics_path, metrics_payload)
        prediction_table.to_csv(predictions_path, sep="\t", index=False)
        write_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            output_paths=[metrics_path, predictions_path],
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        run.set_rows_out("module_a_prediction_rows", int(len(prediction_table)))
        run.set_metric("models_run", len(results))
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
