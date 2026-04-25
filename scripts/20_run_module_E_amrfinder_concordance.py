#!/usr/bin/env python3
"""Run a small AMRFinder probe and compare it with the provided AMR consensus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import get_active_model_names, get_primary_model_name
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_amrfinder_concordance_tables,
    parse_amrfinder_probe_report,
    run_amrfinder_probe,
    select_amrfinder_probe_panel,
    write_selected_fasta_records,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    materialize_recorded_paths,
    project_python_source_paths,
    write_signature_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=20,
        help="Number of high and low priority backbones to probe.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="AMRFinder threads for the probe run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    metrics_path = context.data_dir / "analysis/module_a_metrics.json"
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    config_paths = context_config_paths(context)
    manifest_path = (
        context.data_dir / "analysis/20_run_module_e_amrfinder_concordance.manifest.json"
    )
    all_plasmids_fasta = context.asset_path("bronze_all_plasmids_fasta")
    amrfinder_db_root = context.asset_path("amrfinder_db_dir")
    amrfinder_executable = shutil.which("amrfinder")

    probe_panel_path = context.data_dir / "analysis/amrfinder_probe_panel.tsv"
    probe_fasta_path = context.data_dir / "tmp/amrfinder_probe_panel.fasta"
    probe_hits_path = context.data_dir / "analysis/amrfinder_probe_hits.tsv"
    concordance_detail_path = context.data_dir / "analysis/amrfinder_concordance_detail.tsv"
    concordance_summary_path = context.data_dir / "analysis/amrfinder_concordance_summary.tsv"
    ensure_directory(probe_panel_path.parent)
    ensure_directory(probe_fasta_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/20_run_module_E_amrfinder_concordance.py",
    )
    input_paths = [
        scored_path,
        backbones_path,
        amr_consensus_path,
        metrics_path,
        predictions_path,
        *config_paths,
        all_plasmids_fasta,
        amrfinder_db_root,
    ]
    if amrfinder_executable is not None:
        input_paths.append(Path(amrfinder_executable))
    cache_metadata = {
        "amrfinder_available": amrfinder_executable is not None,
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread,
            ),
        },
        "probe_panel": {"n_per_group": int(args.n_per_group)},
    }

    with ManagedScriptRun(context, "20_run_module_E_amrfinder_concordance") as run:
        for path in input_paths:
            run.record_input(path)
        for path in (
            probe_panel_path,
            probe_hits_path,
            concordance_detail_path,
            concordance_summary_path,
        ):
            run.record_output(path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note(
                "Inputs, code, config, and AMRFinder environment unchanged; reusing cached concordance probe outputs.",
            )
            run.set_metric("cache_hit", True)
            return 0
        run.note(
            "AMRFinder probe is a concordance sanity check on a small representative panel, not a full resequencing annotation pass.",
        )

        scored = read_tsv(scored_path)
        backbones = read_tsv(
            backbones_path,
            usecols=[
                "backbone_id",
                "sequence_accession",
                "is_canonical_representative",
                "species",
                "genus",
            ],
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

        panel = select_amrfinder_probe_panel(
            scored,
            backbones,
            n_per_group=args.n_per_group,
            score_column="primary_model_oof_prediction",
            eligible_only=True,
        )
        panel.to_csv(probe_panel_path, sep="\t", index=False)
        run.set_rows_out("amrfinder_probe_panel_rows", int(len(panel)))

        extraction = write_selected_fasta_records(
            all_plasmids_fasta,
            panel["sequence_accession"].astype(str).tolist(),
            probe_fasta_path,
        )
        if extraction["missing"]:
            run.warn(
                f"AMRFinder probe FASTA missing {len(extraction['missing'])} selected accessions.",  # type: ignore
            )
        run.set_metric("amrfinder_probe_requested", int(extraction["requested"]))  # type: ignore
        run.set_metric("amrfinder_probe_found", int(extraction["found"]))  # type: ignore

        if amrfinder_executable is None:
            run.warn(
                "AMRFinder executable not found in PATH; skipping supportive concordance probe.",
            )
            # Write a valid TSV with headers instead of empty file to avoid boundary validation error
            pd.DataFrame(
                columns=[
                    "sequence_accession",
                    "amrfinder_gene_symbols",
                    "amrfinder_class_tokens",
                    "amrfinder_hit_count",
                ],
            ).to_csv(probe_hits_path, sep="\t", index=False)
            concordance_detail, concordance_summary = build_amrfinder_concordance_tables(
                panel.head(0),
                amr_consensus,
                pd.DataFrame(),
            )
        else:
            probe_result = run_amrfinder_probe(
                probe_fasta_path,
                probe_hits_path,
                amrfinder_db_root=amrfinder_db_root,
                threads=args.threads,
            )
            if probe_result["stderr"].strip():  # type: ignore
                run.note(probe_result["stderr"].strip())  # type: ignore

            amrfinder_probe = parse_amrfinder_probe_report(probe_hits_path)
            concordance_detail, concordance_summary = build_amrfinder_concordance_tables(
                panel,
                amr_consensus,
                amrfinder_probe,
            )
        concordance_detail.to_csv(concordance_detail_path, sep="\t", index=False)
        concordance_summary.to_csv(concordance_summary_path, sep="\t", index=False)
        run.set_rows_out("amrfinder_concordance_detail_rows", int(len(concordance_detail)))
        run.set_rows_out("amrfinder_concordance_summary_rows", int(len(concordance_summary)))
        if not concordance_summary.empty:
            overall = concordance_summary.loc[
                concordance_summary["priority_group"] == "overall"
            ].iloc[0]
            run.set_metric("amrfinder_mean_gene_jaccard", float(overall["mean_gene_jaccard"]))
            run.set_metric("amrfinder_mean_class_jaccard", float(overall["mean_class_jaccard"]))
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
