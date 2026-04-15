#!/usr/bin/env python3
"""Build advanced validation audits from scored backbones and raw metadata tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    build_coefficient_stability_table,
    get_conservative_model_name,
    get_primary_model_name,
)
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.reporting.advanced_audits import (
    build_amr_uncertainty_table,
    build_count_outcome_alignment,
    build_counterfactual_shortlist_comparison,
    build_country_missingness_bounds,
    build_country_upload_propensity,
    build_duplicate_completeness_change_audit,
    build_event_timing_outcomes,
    build_exposure_adjusted_event_table,
    build_exposure_adjusted_outcome_audit,
    build_geographic_jump_distance_table,
    build_knownness_matched_validation,
    build_macro_region_jump_table,
    build_mash_similarity_graph_table,
    build_matched_stratum_propensity_audit,
    build_metadata_quality_table,
    build_missingness_sensitivity_performance,
    build_nonlinear_deconfounding_audit,
    build_operational_risk_dictionary,
    build_ordinal_outcome_alignment,
    build_secondary_outcome_performance,
    build_weighted_outcome_audit,
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
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    adaptive_predictions_path = context.data_dir / "analysis/adaptive_gated_predictions.tsv"
    config_path = context.root / "config.yaml"
    manifest_path = context.data_dir / "analysis/27_run_advanced_audits.manifest.json"
    raw_dir = context.asset_path("plsdb_meta_tables_dir")
    assembly_path = raw_dir / "assembly.csv"
    biosample_path = raw_dir / "biosample.csv"
    nucc_identical_path = raw_dir / "nucc_identical.csv"
    changes_path = raw_dir / "changes.tsv"
    raw_amr_path = raw_dir / "amr.tsv"
    mash_pairs_path = raw_dir / "plsdb_mashdb_sim.tsv"

    knownness_matched_output = context.data_dir / "analysis/knownness_matched_validation.tsv"
    matched_propensity_output = context.data_dir / "analysis/matched_stratum_propensity_audit.tsv"
    nonlinear_output = context.data_dir / "analysis/nonlinear_deconfounding_audit.tsv"
    country_propensity_output = context.data_dir / "analysis/country_upload_propensity.tsv"
    macro_region_jump_output = context.data_dir / "analysis/macro_region_jump_outcome.tsv"
    secondary_outcome_output = context.data_dir / "analysis/secondary_outcome_performance.tsv"
    weighted_outcome_output = context.data_dir / "analysis/weighted_country_outcome_audit.tsv"
    count_outcome_output = context.data_dir / "analysis/new_country_count_audit.tsv"
    metadata_quality_output = context.data_dir / "analysis/metadata_quality_summary.tsv"
    event_timing_output = context.data_dir / "analysis/event_timing_outcomes.tsv"
    exposure_event_output = context.data_dir / "analysis/exposure_adjusted_event_outcomes.tsv"
    exposure_outcome_audit_output = (
        context.data_dir / "analysis/exposure_adjusted_outcome_audit.tsv"
    )
    ordinal_outcome_output = context.data_dir / "analysis/ordinal_outcome_audit.tsv"
    country_missingness_bounds_output = context.data_dir / "analysis/country_missingness_bounds.tsv"
    country_missingness_sensitivity_output = (
        context.data_dir / "analysis/country_missingness_sensitivity.tsv"
    )
    geographic_jump_output = context.data_dir / "analysis/geographic_jump_distance_outcome.tsv"
    duplicate_quality_output = context.data_dir / "analysis/duplicate_completeness_change_audit.tsv"
    amr_uncertainty_output = context.data_dir / "analysis/amr_uncertainty_summary.tsv"
    mash_graph_output = context.data_dir / "analysis/mash_similarity_graph.tsv"
    counterfactual_output = context.data_dir / "analysis/counterfactual_shortlist_comparison.tsv"
    operational_risk_output = context.data_dir / "analysis/operational_risk_dictionary.tsv"
    coefficient_stability_cv_output = context.data_dir / "analysis/coefficient_stability_cv.tsv"
    ensure_directory(knownness_matched_output.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/27_run_advanced_audits.py",
    )
    input_paths = [
        scored_path,
        backbones_path,
        predictions_path,
        adaptive_predictions_path,
        assembly_path,
        biosample_path,
        nucc_identical_path,
        changes_path,
        raw_amr_path,
        mash_pairs_path,
        config_path,
    ]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
    }

    with ManagedScriptRun(context, "27_run_advanced_audits") as run:
        for path in (
            scored_path,
            backbones_path,
            predictions_path,
            adaptive_predictions_path,
            assembly_path,
            biosample_path,
            nucc_identical_path,
            changes_path,
            raw_amr_path,
            mash_pairs_path,
            config_path,
        ):
            if path.exists():
                run.record_input(path)
        for path in (
            knownness_matched_output,
            matched_propensity_output,
            nonlinear_output,
            country_propensity_output,
            macro_region_jump_output,
            secondary_outcome_output,
            weighted_outcome_output,
            count_outcome_output,
            metadata_quality_output,
            event_timing_output,
            exposure_event_output,
            exposure_outcome_audit_output,
            ordinal_outcome_output,
            country_missingness_bounds_output,
            country_missingness_sensitivity_output,
            geographic_jump_output,
            duplicate_quality_output,
            amr_uncertainty_output,
            mash_graph_output,
            counterfactual_output,
            operational_risk_output,
            coefficient_stability_cv_output,
        ):
            run.record_output(path)
        if load_signature_manifest(
            manifest_path,
            input_paths=[path for path in input_paths if path.exists()],
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached advanced audits.")
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(scored_path)
        backbones = read_tsv(backbones_path)
        predictions = read_tsv(predictions_path)
        if adaptive_predictions_path.exists():
            adaptive = read_tsv(adaptive_predictions_path)
            if {"backbone_id", "adaptive_prediction", "spread_label", "model_name"} <= set(
                adaptive.columns
            ):
                adaptive_predictions = adaptive[
                    ["backbone_id", "adaptive_prediction", "spread_label", "model_name"]
                ].rename(columns={"adaptive_prediction": "oof_prediction"})
                adaptive_predictions["visibility_expansion_label"] = adaptive_predictions[
                    "spread_label"
                ]
                predictions = pd.concat(
                    [predictions, adaptive_predictions[predictions.columns]],
                    ignore_index=True,
                )

        primary_model_name = get_primary_model_name(
            predictions["model_name"].astype(str).unique().tolist()
        )
        conservative_model_name = get_conservative_model_name(
            predictions["model_name"].astype(str).unique().tolist()
        )
        model_names = [
            name
            for name in [
                primary_model_name,
                conservative_model_name,
                "natural_auc_priority",
                "knownness_robust_priority",
                "support_calibrated_priority",
                "support_synergy_priority",
                "host_transfer_synergy_priority",
                "ecology_clinical_priority",
                "contextual_bio_priority",
                "adaptive_natural_priority",
                "adaptive_knownness_robust_priority",
                "adaptive_knownness_blend_priority",
                "adaptive_support_calibrated_blend_priority",
                "adaptive_support_synergy_blend_priority",
                "adaptive_host_transfer_synergy_blend_priority",
                "adaptive_threat_architecture_blend_priority",
                "baseline_both",
            ]
            if name in set(predictions["model_name"].astype(str))
        ]

        country_propensity = build_country_upload_propensity(backbones)
        country_propensity.to_csv(country_propensity_output, sep="\t", index=False)

        macro_region_jump = build_macro_region_jump_table(backbones, country_propensity)
        macro_region_jump.to_csv(macro_region_jump_output, sep="\t", index=False)

        event_timing = build_event_timing_outcomes(backbones)
        event_timing.to_csv(event_timing_output, sep="\t", index=False)

        exposure_adjusted = build_exposure_adjusted_event_table(
            backbones,
            country_propensity,
        )
        exposure_adjusted.to_csv(exposure_event_output, sep="\t", index=False)

        geographic_jump = build_geographic_jump_distance_table(backbones)
        geographic_jump.to_csv(geographic_jump_output, sep="\t", index=False)

        combined_secondary = macro_region_jump.merge(event_timing, on="backbone_id", how="outer")
        combined_secondary = combined_secondary.merge(
            geographic_jump[["backbone_id", "long_jump_label"]],
            on="backbone_id",
            how="left",
        )

        secondary_outcomes = build_secondary_outcome_performance(
            predictions,
            combined_secondary,
            outcome_columns=[
                "macro_region_jump_label",
                "host_family_jump_label",
                "host_order_jump_label",
                "event_within_1y_label",
                "event_within_3y_label",
                "event_within_5y_label",
                "three_countries_within_3y_label",
                "three_countries_within_5y_label",
                "long_jump_label",
            ],
            model_names=model_names,
        )
        secondary_outcomes.to_csv(secondary_outcome_output, sep="\t", index=False)

        weighted_outcome = build_weighted_outcome_audit(
            predictions,
            macro_region_jump,
            model_names=model_names,
        )
        weighted_outcome.to_csv(weighted_outcome_output, sep="\t", index=False)

        count_outcome = build_count_outcome_alignment(
            predictions,
            macro_region_jump,
            count_column="n_new_countries_recomputed",
            model_names=model_names,
        )
        count_outcome.to_csv(count_outcome_output, sep="\t", index=False)

        exposure_adjusted_outcome = build_exposure_adjusted_outcome_audit(
            predictions,
            exposure_adjusted,
            model_names=model_names,
        )
        exposure_adjusted_outcome.to_csv(exposure_outcome_audit_output, sep="\t", index=False)

        ordinal_outcome = build_ordinal_outcome_alignment(
            predictions,
            event_timing,
            ordinal_column="spread_severity_bin",
            model_names=model_names,
        )
        ordinal_outcome.to_csv(ordinal_outcome_output, sep="\t", index=False)

        matched_model_names = sorted(set(predictions["model_name"].astype(str)))
        knownness_matched = build_knownness_matched_validation(
            scored,
            predictions,
            model_names=matched_model_names,
        )
        knownness_matched.to_csv(knownness_matched_output, sep="\t", index=False)

        matched_propensity = build_matched_stratum_propensity_audit(
            scored,
            predictions,
            model_names=matched_model_names,
        )
        matched_propensity.to_csv(matched_propensity_output, sep="\t", index=False)

        nonlinear = build_nonlinear_deconfounding_audit(scored)
        nonlinear.to_csv(nonlinear_output, sep="\t", index=False)

        assembly = pd.read_csv(
            assembly_path,
            usecols=[
                "NUCCORE_UID",
                "ASSEMBLY_Status",
                "ASSEMBLY_coverage",
                "ASSEMBLY_SeqReleaseDate",
            ],
        )
        biosample = pd.read_csv(
            biosample_path,
            usecols=["NUCCORE_UID", "BIOSAMPLE_pathogenicity", "DISEASE_tags", "ECOSYSTEM_tags"],
        )
        nucc_identical = pd.read_csv(
            nucc_identical_path,
            usecols=["NUCCORE_ACC", "NUCCORE_Completeness", "NUCCORE_DuplicatedEntry"],
        )
        metadata_quality = build_metadata_quality_table(
            backbones, scored, assembly, biosample, nucc_identical
        )
        metadata_quality.to_csv(metadata_quality_output, sep="\t", index=False)

        country_missingness_bounds = build_country_missingness_bounds(backbones)
        country_missingness_bounds.to_csv(country_missingness_bounds_output, sep="\t", index=False)

        country_missingness_sensitivity = build_missingness_sensitivity_performance(
            predictions,
            country_missingness_bounds,
            model_names=model_names,
        )
        country_missingness_sensitivity.to_csv(
            country_missingness_sensitivity_output, sep="\t", index=False
        )

        changes = read_tsv(changes_path, usecols=["NUCCORE_ACC", "Flag", "Comment"])
        duplicate_quality = build_duplicate_completeness_change_audit(
            backbones, nucc_identical, changes
        )
        duplicate_quality.to_csv(duplicate_quality_output, sep="\t", index=False)

        raw_amr = read_tsv(
            raw_amr_path,
            usecols=["NUCCORE_ACC", "analysis_software_name", "gene_symbol", "drug_class"],
        )
        amr_uncertainty = build_amr_uncertainty_table(backbones, raw_amr)
        amr_uncertainty.to_csv(amr_uncertainty_output, sep="\t", index=False)

        mash_pairs = read_tsv(
            mash_pairs_path,
            header=None,
            names=["source_accession", "target_accession"],
            usecols=[0, 1],
        )
        mash_graph = build_mash_similarity_graph_table(backbones, mash_pairs)
        mash_graph.to_csv(mash_graph_output, sep="\t", index=False)

        counterfactual = build_counterfactual_shortlist_comparison(
            scored,
            predictions,
            primary_model_name=primary_model_name,
            baseline_model_name="baseline_both",
        )
        counterfactual.to_csv(counterfactual_output, sep="\t", index=False)

        operational_risk = build_operational_risk_dictionary(
            predictions,
            combined_secondary,
            scored=scored,
            model_names=matched_model_names,
        )
        operational_risk.to_csv(operational_risk_output, sep="\t", index=False)

        if primary_model_name in MODULE_A_FEATURE_SETS:
            coefficient_stability_cv = build_coefficient_stability_table(
                scored,
                model_name=primary_model_name,
                columns=MODULE_A_FEATURE_SETS[primary_model_name],
                n_splits=5,
                n_repeats=1,
                include_fold_coefficients=True,
            )
        else:
            coefficient_stability_cv = pd.DataFrame()
        coefficient_stability_cv.to_csv(coefficient_stability_cv_output, sep="\t", index=False)

        run.set_rows_out("country_upload_propensity_rows", int(len(country_propensity)))
        run.set_rows_out("macro_region_jump_rows", int(len(macro_region_jump)))
        run.set_rows_out("secondary_outcome_rows", int(len(secondary_outcomes)))
        run.set_rows_out("weighted_outcome_rows", int(len(weighted_outcome)))
        run.set_rows_out("count_outcome_rows", int(len(count_outcome)))
        run.set_rows_out("knownness_matched_rows", int(len(knownness_matched)))
        run.set_rows_out("matched_propensity_rows", int(len(matched_propensity)))
        run.set_rows_out("nonlinear_deconfounding_rows", int(len(nonlinear)))
        run.set_rows_out("metadata_quality_rows", int(len(metadata_quality)))
        run.set_rows_out("event_timing_rows", int(len(event_timing)))
        run.set_rows_out("exposure_adjusted_rows", int(len(exposure_adjusted)))
        run.set_rows_out("exposure_adjusted_audit_rows", int(len(exposure_adjusted_outcome)))
        run.set_rows_out("ordinal_outcome_rows", int(len(ordinal_outcome)))
        run.set_rows_out("country_missingness_rows", int(len(country_missingness_bounds)))
        run.set_rows_out(
            "country_missingness_sensitivity_rows", int(len(country_missingness_sensitivity))
        )
        run.set_rows_out("geographic_jump_rows", int(len(geographic_jump)))
        run.set_rows_out("duplicate_quality_rows", int(len(duplicate_quality)))
        run.set_rows_out("amr_uncertainty_rows", int(len(amr_uncertainty)))
        run.set_rows_out("mash_graph_rows", int(len(mash_graph)))
        run.set_rows_out("counterfactual_rows", int(len(counterfactual)))
        run.set_rows_out("operational_risk_rows", int(len(operational_risk)))
        run.set_rows_out("coefficient_stability_cv_rows", int(len(coefficient_stability_cv)))
        write_signature_manifest(
            manifest_path,
            input_paths=[path for path in input_paths if path.exists()],
            output_paths=materialize_recorded_paths(context.root, run.output_files_written),
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        run.set_metric("cache_hit", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
