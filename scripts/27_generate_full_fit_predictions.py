#!/usr/bin/env python3
"""Generate full-fit contextual predictions for key models as upstream artifacts.

This script moves expensive model fitting work out of the report builder,
ensuring that report generation is a pure assembly/rendering stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import (
    build_discovery_input_contract,
    get_conservative_model_name,
    get_governance_model_name,
    validate_discovery_input_contract,
)
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.scoring import fit_full_model_predictions
from plasmid_priority.utils.dataframe import read_tsv, write_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    write_signature_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full-fit contextual predictions")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate predictions even if cached version exists",
    )
    args = parser.parse_args()

    context = build_context()
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    output_dir = context.data_dir / "analysis"
    ensure_directory(output_dir)

    config_paths = context_config_paths(context)
    manifest_path = context.reports_dir / "27_generate_full_fit_predictions.manifest.json"

    source_paths = [
        *config_paths,
        scored_path,
    ]

    with ManagedScriptRun(context, "27_generate_full_fit_predictions") as run:
        for path in source_paths:
            if path.exists():
                run.record_input(path)

        # Check cache
        cache_metadata = {}
        if not args.force:
            if load_signature_manifest(
                manifest_path,
                input_paths=[path for path in source_paths if path.exists()],
                source_paths=source_paths,
                metadata=cache_metadata,
            ):
                run.note("Inputs, code, and config unchanged; reusing cached full-fit predictions.")
                run.set_metric("cache_hit", True)
                return 0

        scored = read_tsv(scored_path)

        # Validate discovery contract
        primary_model_name = str(
            context.model_settings.get("primary_model_name", "discovery_12f_source")
        )
        conservative_model_name = str(
            get_conservative_model_name(context.config if isinstance(context.config, dict) else {})
        )
        governance_model_name = str(
            get_governance_model_name(context.config if isinstance(context.config, dict) else {})
        )

        validate_discovery_input_contract(
            scored,
            model_names=[primary_model_name],
            contract=build_discovery_input_contract(int(context.pipeline_settings.split_year)),
            label="Full-fit prediction input",
        )

        # Generate predictions for each model
        prediction_outputs = {
            primary_model_name: output_dir / "primary_model_full_fit_predictions.tsv",
            "baseline_both": output_dir / "baseline_both_full_fit_predictions.tsv",
            conservative_model_name: output_dir / "conservative_model_full_fit_predictions.tsv",
            governance_model_name: output_dir / "governance_model_full_fit_predictions.tsv",
        }

        for model_name, output_path in prediction_outputs.items():
            try:
                predictions = fit_full_model_predictions(scored, model_name=model_name)
                write_tsv(output_path, predictions)
                run.record_output(output_path)
                run.set_metric(f"{model_name}_n_predictions", len(predictions))
            except Exception as e:
                run.warn(f"Failed to generate predictions for {model_name}: {e}")

        # Write signature manifest
        write_signature_manifest(
            manifest_path,
            input_paths=[path for path in source_paths if path.exists()],
            source_paths=source_paths,
            metadata=cache_metadata,
        )

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
