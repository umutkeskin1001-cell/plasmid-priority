#!/usr/bin/env python3
"""Supportive analyses using WHO MIA, CARD, and MOB-suite reference datasets."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import get_active_model_names, get_primary_model_name
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_card_support,
    build_mobsuite_support,
    build_priority_backbone_support_frame,
    build_who_mia_reference_catalog,
    build_who_mia_support,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    materialize_recorded_paths,
    project_python_source_paths,
    write_signature_manifest,
)


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    metrics_path = context.data_dir / "analysis/module_a_metrics.json"
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    config_paths = context_config_paths(context)
    manifest_path = context.data_dir / "analysis/19_run_module_d_external_support.manifest.json"
    card_archive_path = context.asset_path("card_archive")
    mobsuite_tar_path = context.asset_path("mobsuite_db_tar")
    who_text_path = context.asset_path("who_mia_text")

    card_detail_path = context.data_dir / "analysis/card_gene_support.tsv"
    card_summary_path = context.data_dir / "analysis/card_group_summary.tsv"
    card_family_path = context.data_dir / "analysis/card_gene_family_comparison.tsv"
    card_mechanism_path = context.data_dir / "analysis/card_mechanism_comparison.tsv"
    mobsuite_detail_path = context.data_dir / "analysis/mobsuite_host_range_support.tsv"
    mobsuite_summary_path = context.data_dir / "analysis/mobsuite_host_range_group_summary.tsv"
    who_detail_path = context.data_dir / "analysis/who_mia_support.tsv"
    who_summary_path = context.data_dir / "analysis/who_mia_group_summary.tsv"
    who_category_path = context.data_dir / "analysis/who_mia_category_comparison.tsv"
    who_reference_path = context.data_dir / "analysis/who_mia_reference_catalog.tsv"
    ensure_directory(card_detail_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/19_run_module_D_external_support.py",
    )
    input_paths = [
        scored_path,
        backbones_path,
        amr_consensus_path,
        metrics_path,
        predictions_path,
        *config_paths,
    ]
    if who_text_path.exists():
        input_paths.append(who_text_path)
    if card_archive_path.exists():
        input_paths.append(card_archive_path)
    if mobsuite_tar_path.exists():
        input_paths.append(mobsuite_tar_path)
    cache_metadata = {
        "optional_inputs": {
            "card_archive": card_archive_path.exists(),
            "mobsuite_db_tar": mobsuite_tar_path.exists(),
            "who_mia_text": who_text_path.exists(),
        },
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread,
            ),
        },
    }

    with ManagedScriptRun(context, "19_run_module_D_external_support") as run:
        for path in input_paths:
            run.record_input(path)
        for path in (who_detail_path, who_summary_path, who_category_path):
            run.record_output(path)
        if who_text_path.exists():
            run.record_output(who_reference_path)
        if card_archive_path.exists():
            for path in (
                card_detail_path,
                card_summary_path,
                card_family_path,
                card_mechanism_path,
            ):
                run.record_output(path)
        if mobsuite_tar_path.exists():
            for path in (mobsuite_detail_path, mobsuite_summary_path):
                run.record_output(path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached external support outputs.")
            run.set_metric("cache_hit", True)
            return 0
        run.note(
            "WHO MIA, CARD, and MOB-suite outputs are supportive descriptive layers and are not used for model training.",
        )

        scored = read_tsv(scored_path)
        backbones = read_tsv(
            backbones_path,
            usecols=[
                "backbone_id",
                "sequence_accession",
                "genus",
                "species",
                "primary_replicon",
                "replicon_types",
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
        eligible_count = int(scored["spread_label"].notna().sum())
        n_per_group = max(25, int(round(eligible_count * 0.25))) if eligible_count > 0 else 25
        run.note(
            f"External support contrasts use headline-model quartile extremes (top/bottom {n_per_group}) rather than a fixed top/bottom 100.",
        )

        priority_backbones = build_priority_backbone_support_frame(
            scored,
            backbones,
            amr_consensus,
            n_per_group=n_per_group,
            score_column="primary_model_oof_prediction",
            eligible_only=True,
        )
        run.set_rows_in("external_support_priority_backbones", int(len(priority_backbones)))

        who_detail, who_summary, who_category = build_who_mia_support(priority_backbones)
        who_detail.to_csv(who_detail_path, sep="\t", index=False)
        who_summary.to_csv(who_summary_path, sep="\t", index=False)
        who_category.to_csv(who_category_path, sep="\t", index=False)
        run.set_rows_out("who_mia_support_rows", int(len(who_detail)))
        run.set_metric(
            "who_hpecia_high_backbones",
            int(who_detail.loc[who_detail["priority_group"] == "high", "who_mia_any_hpecia"].sum()),
        )
        run.set_metric(
            "who_hpecia_low_backbones",
            int(who_detail.loc[who_detail["priority_group"] == "low", "who_mia_any_hpecia"].sum()),
        )
        if who_text_path.exists():
            who_reference = build_who_mia_reference_catalog(who_text_path)
            who_reference.to_csv(who_reference_path, sep="\t", index=False)
            run.set_rows_out("who_mia_reference_rows", int(len(who_reference)))
            run.set_metric(
                "who_mia_reference_classes_present",
                int(who_reference["reference_class_present_in_text"].fillna(False).sum()),
            )
            missing_reference = int(
                (~who_reference["reference_class_present_in_text"].fillna(False)).sum(),
            )
            if missing_reference > 0:
                run.warn(
                    f"WHO MIA reference text did not contain {missing_reference} curated class labels by exact normalized string match.",
                )
        else:
            run.warn(f"Optional WHO MIA text extraction not found: {who_text_path}")

        if card_archive_path.exists():
            card_detail, card_summary, card_family, card_mechanism = build_card_support(
                priority_backbones,
                card_archive_path,
            )
            card_detail.to_csv(card_detail_path, sep="\t", index=False)
            card_summary.to_csv(card_summary_path, sep="\t", index=False)
            card_family.to_csv(card_family_path, sep="\t", index=False)
            card_mechanism.to_csv(card_mechanism_path, sep="\t", index=False)
            run.set_rows_out("card_support_rows", int(len(card_detail)))
            run.set_metric(
                "card_high_backbones_with_support",
                int(
                    card_detail.loc[
                        card_detail["priority_group"] == "high",
                        "card_any_support",
                    ].sum(),
                ),
            )
            run.set_metric(
                "card_low_backbones_with_support",
                int(
                    card_detail.loc[
                        card_detail["priority_group"] == "low",
                        "card_any_support",
                    ].sum(),
                ),
            )
        else:
            run.warn(f"Optional CARD archive not found: {card_archive_path}")

        if mobsuite_tar_path.exists():
            mobsuite_detail, mobsuite_summary = build_mobsuite_support(
                priority_backbones,
                mobsuite_tar_path,
            )
            mobsuite_detail.to_csv(mobsuite_detail_path, sep="\t", index=False)
            mobsuite_summary.to_csv(mobsuite_summary_path, sep="\t", index=False)
            run.set_rows_out("mobsuite_support_rows", int(len(mobsuite_detail)))
            run.set_metric(
                "mobsuite_high_backbones_with_literature_support",
                int(
                    mobsuite_detail.loc[
                        mobsuite_detail["priority_group"] == "high",
                        "mobsuite_any_literature_support",
                    ].sum(),
                ),
            )
            run.set_metric(
                "mobsuite_low_backbones_with_literature_support",
                int(
                    mobsuite_detail.loc[
                        mobsuite_detail["priority_group"] == "low",
                        "mobsuite_any_literature_support",
                    ].sum(),
                ),
            )
        else:
            run.warn(f"Optional MOB-suite archive not found: {mobsuite_tar_path}")

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
