#!/usr/bin/env python3
"""Exploratory AMR composition comparison for high vs low priority backbones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)


def _explode_classes(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["amr_drug_classes"] = working["amr_drug_classes"].fillna("")
    working["amr_drug_classes"] = working["amr_drug_classes"].str.split(",")
    working = working.explode("amr_drug_classes")
    working["amr_drug_classes"] = working["amr_drug_classes"].fillna("").astype(str).str.strip()
    return working.loc[working["amr_drug_classes"] != ""]


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    config_path = context.root / "config.yaml"
    manifest_path = context.data_dir / "analysis/17_run_module_b.manifest.json"
    output_path = context.data_dir / "analysis/module_b_amr_class_comparison.tsv"
    ensure_directory(output_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/17_run_module_B.py",
    )
    input_paths = [scored_path, backbones_path, amr_consensus_path, config_path]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
    }

    with ManagedScriptRun(context, "17_run_module_B") as run:
        for path in (scored_path, backbones_path, amr_consensus_path, config_path):
            run.record_input(path)
        run.record_output(output_path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached Module B output.")
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(scored_path)
        accessions = read_tsv(
            backbones_path,
            usecols=["sequence_accession", "backbone_id"],
        )
        amr = read_tsv(amr_consensus_path)

        low_cut = scored["priority_index"].quantile(0.10)
        high_cut = scored["priority_index"].quantile(0.90)
        selected = scored.loc[
            scored["priority_index"].le(low_cut) | scored["priority_index"].ge(high_cut),
            ["backbone_id", "priority_index"],
        ].copy()
        selected["priority_group"] = (
            selected["priority_index"].ge(high_cut).map({True: "high", False: "low"})
        )

        merged = accessions.merge(
            selected[["backbone_id", "priority_group"]], on="backbone_id", how="inner"
        )
        merged = merged.merge(amr, on="sequence_accession", how="left")
        exploded = _explode_classes(merged)

        totals = merged.groupby("priority_group")["sequence_accession"].nunique().to_dict()
        summary = (
            exploded.groupby(["priority_group", "amr_drug_classes"])["sequence_accession"]
            .nunique()
            .reset_index(name="n_accessions")
        )
        summary["group_total_accessions"] = summary["priority_group"].map(totals)
        summary["prevalence"] = summary["n_accessions"] / summary["group_total_accessions"]
        pivot = summary.pivot(
            index="amr_drug_classes", columns="priority_group", values="prevalence"
        ).fillna(0.0)
        pivot["prevalence_delta_high_minus_low"] = pivot.get("high", 0.0) - pivot.get("low", 0.0)
        output = pivot.reset_index().sort_values("prevalence_delta_high_minus_low", ascending=False)
        output.to_csv(output_path, sep="\t", index=False)
        write_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            output_paths=[output_path],
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        run.set_rows_out("module_b_rows", int(len(output)))
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
