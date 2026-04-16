#!/usr/bin/env python3
"""Run independent categorical enrichment analyses for visibility-positive backbones."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_backbone_identity_table,
    build_module_f_enrichment_table,
    build_module_f_top_hits,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    config_paths = context_config_paths(context)
    manifest_path = context.data_dir / "analysis/23_run_module_f_enrichment.manifest.json"
    identity_output = context.data_dir / "analysis/module_f_backbone_identity.tsv"
    enrichment_output = context.data_dir / "analysis/module_f_enrichment.tsv"
    top_hits_output = context.data_dir / "analysis/module_f_top_hits.tsv"
    ensure_directory(identity_output.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/23_run_module_f_enrichment.py",
    )
    input_paths = [scored_path, backbones_path, amr_consensus_path, *config_paths]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
    }

    with ManagedScriptRun(context, "23_run_module_f_enrichment") as run:
        for path in (scored_path, backbones_path, amr_consensus_path, *config_paths):
            run.record_input(path)
        for path in (identity_output, enrichment_output, top_hits_output):
            run.record_output(path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached Module F outputs.")
            run.set_metric("cache_hit", True)
            return 0

        pipeline = context.pipeline_settings
        scored = read_tsv(scored_path)
        backbones = read_tsv(backbones_path)
        amr_consensus = read_tsv(amr_consensus_path)

        identity = build_backbone_identity_table(
            scored,
            backbones,
            amr_consensus,
            split_year=pipeline.split_year,
        )
        enrichment = build_module_f_enrichment_table(
            identity, label_column="spread_label", min_backbones=10
        )
        top_hits = build_module_f_top_hits(
            enrichment, q_threshold=0.05, max_per_group=3, max_total=20
        )

        identity.to_csv(identity_output, sep="\t", index=False)
        enrichment.to_csv(enrichment_output, sep="\t", index=False)
        top_hits.to_csv(top_hits_output, sep="\t", index=False)
        write_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            output_paths=[identity_output, enrichment_output, top_hits_output],
            source_paths=source_paths,
            metadata=cache_metadata,
        )

        run.set_rows_out("module_f_identity_rows", int(len(identity)))
        run.set_rows_out("module_f_enrichment_rows", int(len(enrichment)))
        run.set_metric(
            "module_f_significant_hits",
            int(top_hits["feature_value"].nunique()) if not top_hits.empty else 0,
        )
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
