#!/usr/bin/env python3
"""Run the primary retrospective modeling stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

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
from plasmid_priority.validation import calibration_slope_intercept

COMPUTE_TIERS_PATH = PROJECT_ROOT / "config" / "model_compute_tiers.yaml"
DEFAULT_COMPUTE_TIERS: dict[str, dict[str, int]] = {
    "smoke": {"n_splits": 2, "n_repeats": 1},
    "dev": {"n_splits": 3, "n_repeats": 1},
    "model-refresh": {"n_splits": 5, "n_repeats": 1},
    "release-full": {"n_splits": 5, "n_repeats": 5},
}


def _load_compute_tiers(path: Path = COMPUTE_TIERS_PATH) -> dict[str, dict[str, int]]:
    if not path.exists():
        return dict(DEFAULT_COMPUTE_TIERS)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return dict(DEFAULT_COMPUTE_TIERS)
    tiers = payload.get("tiers", {}) if isinstance(payload, dict) else {}
    if not isinstance(tiers, dict):
        return dict(DEFAULT_COMPUTE_TIERS)
    merged = dict(DEFAULT_COMPUTE_TIERS)
    for name, values in tiers.items():
        if not isinstance(values, dict):
            continue
        n_splits = int(values.get("n_splits", merged.get(name, {}).get("n_splits", 5)))
        n_repeats = int(values.get("n_repeats", merged.get(name, {}).get("n_repeats", 1)))
        merged[str(name)] = {"n_splits": max(n_splits, 2), "n_repeats": max(n_repeats, 1)}
    return merged


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
    parser.add_argument(
        "--compute-tier",
        type=str,
        default=os.environ.get("PLASMID_PRIORITY_COMPUTE_TIER", "model-refresh"),
        help="Compute tier key from config/model_compute_tiers.yaml (smoke|dev|model-refresh|release-full).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeated stratified folds.",
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
                context.pipeline_settings.min_new_countries_for_spread,
            ),
        },
        "compute_tier": str(args.compute_tier),
        "seed": int(args.seed),
    }
    compute_tiers = _load_compute_tiers()
    tier = compute_tiers.get(str(args.compute_tier), compute_tiers["model-refresh"])
    n_splits = int(tier["n_splits"])
    n_repeats = int(tier["n_repeats"])

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
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=int(args.seed),
            n_jobs=max(int(args.jobs), 1),
        )

        metrics_payload = {
            "primary_model_name": get_primary_model_name(list(results)),
            "conservative_model_name": get_conservative_model_name(list(results)),
        }
        for name, result in results.items():
            result_metrics = {
                **result.metrics,
                "status": result.status,
                "error_message": result.error_message,
            }
            if not result.predictions.empty:
                calibration_frame = result.predictions.loc[
                    result.predictions["spread_label"].notna()
                    & result.predictions["oof_prediction"].notna()
                ].copy()
                if not calibration_frame.empty and calibration_frame["spread_label"].nunique() >= 2:
                    slope, intercept = calibration_slope_intercept(
                        calibration_frame["spread_label"].astype(int).to_numpy(),
                        calibration_frame["oof_prediction"].astype(float).to_numpy(),
                    )
                    result_metrics["calibration_slope"] = float(slope)
                    result_metrics["calibration_intercept"] = float(intercept)
                else:
                    result_metrics["calibration_slope"] = None
                    result_metrics["calibration_intercept"] = None
            else:
                result_metrics["calibration_slope"] = None
                result_metrics["calibration_intercept"] = None
            metrics_payload[name] = result_metrics  # type: ignore
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
        run.set_metric("n_splits", n_splits)
        run.set_metric("n_repeats", n_repeats)
        run.set_metric("compute_tier", str(args.compute_tier))
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
