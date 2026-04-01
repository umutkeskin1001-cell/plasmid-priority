#!/usr/bin/env python3
"""Run the primary retrospective modeling stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    assert_feature_columns_present,
    get_conservative_model_name,
    get_module_a_model_names,
    get_primary_model_name,
    run_module_a,
)
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import atomic_write_json, ensure_directory


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
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    scored_path = context.root / "data/scores/backbone_scored.tsv"
    metrics_path = context.root / "data/analysis/module_a_metrics.json"
    predictions_path = context.root / "data/analysis/module_a_predictions.tsv"
    ensure_directory(metrics_path.parent)

    with ManagedScriptRun(context, "16_run_module_A") as run:
        run.record_input(scored_path)
        run.record_output(metrics_path)
        run.record_output(predictions_path)

        scored = read_tsv(scored_path)
        model_names = get_module_a_model_names(
            include_research=args.research_models,
            include_ablations=args.ablation_models,
        )
        required_columns = [
            column
            for model_name in model_names
            for column in MODULE_A_FEATURE_SETS[model_name]
        ]
        assert_feature_columns_present(
            scored,
            required_columns,
            label="Module A score input",
        )
        results = run_module_a(
            scored,
            model_names=model_names,
        )

        metrics_payload = {
            "primary_model_name": get_primary_model_name(list(results)),
            "conservative_model_name": get_conservative_model_name(list(results)),
        }
        metrics_payload.update(
            {
                name: result.metrics
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
        run.set_rows_out("module_a_prediction_rows", int(len(prediction_table)))
        run.set_metric("models_run", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
