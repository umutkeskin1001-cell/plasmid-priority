#!/usr/bin/env python3
"""Assemble final summary tables and report metadata."""

# mypy: ignore-errors

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.cache import stable_hash
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    annotate_knownness_metadata,
    build_discovery_input_contract,
    build_standardized_coefficient_table,
    get_active_model_names,
    get_conservative_model_name,
    get_governance_model_name,
    get_primary_model_name,
    validate_discovery_input_contract,
)
from plasmid_priority.reporting import (
    ManagedScriptRun,
    annotate_candidate_explanation_fields,
    build_amrfinder_coverage_table,
    build_backbone_identity_table,
    build_benchmark_protocol_table,
    build_blocked_holdout_summary,
    build_candidate_dossier_table,
    build_candidate_portfolio_table,
    build_candidate_risk_table,
    build_candidate_signature_context,
    build_candidate_universe_table,
    build_component_floor_diagnostics,
    build_confirmatory_cohort_summary,
    build_consensus_candidate_ranking,
    build_consensus_shortlist,
    build_decision_yield_table,
    build_false_negative_audit,
    build_frozen_scientific_acceptance_audit,
    build_h_feature_diagnostics,
    build_model_family_summary,
    build_model_selection_scorecard,
    build_model_simplicity_summary,
    build_module_f_enrichment_table,
    build_module_f_top_hits,
    build_multiverse_stability_table,
    build_novelty_margin_summary,
    build_official_benchmark_panel,
    build_pathogen_group_comparison,
    build_primary_model_selection_summary,
    build_report_overview_table,
    build_score_axis_summary,
    build_score_distribution_diagnostics,
    build_temporal_drift_summary,
    build_threshold_flip_table,
    build_threshold_utility_table,
    validate_report_artifact,
)
from plasmid_priority.reporting.cache import ReportCache
from plasmid_priority.reporting.figures import generate_all_figures
from plasmid_priority.reporting.narrative_utils import (
    benchmark_scope_note as _benchmark_scope_note,
)
from plasmid_priority.reporting.narrative_utils import (
    blocked_holdout_summary_text as _blocked_holdout_summary_text,
)
from plasmid_priority.reporting.narrative_utils import (
    blocked_holdout_summary_text_tr as _blocked_holdout_summary_text_tr,
)
from plasmid_priority.reporting.narrative_utils import (
    candidate_stability_summary_text as _candidate_stability_summary_text,
)
from plasmid_priority.reporting.narrative_utils import (
    country_missingness_summary_text as _country_missingness_summary_text,
)
from plasmid_priority.reporting.narrative_utils import (
    format_interval as _format_interval,
)
from plasmid_priority.reporting.narrative_utils import (
    format_pvalue as _format_pvalue,
)
from plasmid_priority.reporting.narrative_utils import (
    governance_watch_label as _governance_watch_label,
)
from plasmid_priority.reporting.narrative_utils import (
    pretty_report_model_label as _pretty_report_model_label,
)
from plasmid_priority.reporting.narrative_utils import (
    rolling_temporal_summary as _rolling_temporal_summary,
)
from plasmid_priority.reporting.narrative_utils import (
    select_confirmatory_row as _select_confirmatory_row,
)
from plasmid_priority.reporting.narrative_utils import (
    strict_acceptance_status as _strict_acceptance_status,
)
from plasmid_priority.reporting.narrative_utils import (
    summarize_false_negative_audit as _summarize_false_negative_audit,
)
from plasmid_priority.reporting.narrative_utils import (
    top_sign_stable_features as _top_sign_stable_features,
)
from plasmid_priority.reporting.report_build_helpers import (
    brier_skill_score as _brier_skill_score,
)
from plasmid_priority.reporting.report_build_helpers import (
    build_spatial_holdout_summary as _build_spatial_holdout_summary,
)
from plasmid_priority.reporting.report_build_helpers import (
    format_scorecard_rank_text as _format_scorecard_rank_text,
)
from plasmid_priority.reporting.report_build_helpers import (
    metrics_to_frame as _metrics_to_frame,
)
from plasmid_priority.reporting.report_build_helpers import (
    official_report_model_names as _official_report_model_names,
)
from plasmid_priority.reporting.report_build_helpers import (
    read_if_exists as _read_if_exists,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    build_l2_sensitivity_table as _build_l2_sensitivity_table,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    build_outcome_robustness_grid as _build_outcome_robustness_grid,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    build_threshold_sensitivity_table as _build_threshold_sensitivity_table,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    build_weighting_sensitivity_table as _build_weighting_sensitivity_table,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    dominant_non_empty as _dominant_non_empty,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    humanize_action_tier as _humanize_action_tier,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    humanize_candidate_reason as _humanize_candidate_reason,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    humanize_evidence_tier as _humanize_evidence_tier,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    humanize_portfolio_track as _humanize_portfolio_track,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    humanize_taxon_label as _humanize_taxon_label,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    top_likely_amr_genes as _top_likely_amr_genes,
)
from plasmid_priority.reporting.report_candidate_helpers import (
    top_public_health_amr_classes as _top_public_health_amr_classes,
)
from plasmid_priority.reporting.report_pipeline_helpers import (
    build_table_router,
    register_default_report_outputs,
)
from plasmid_priority.reporting.report_surface_helpers import (
    attach_official_benchmark_context as _attach_official_benchmark_context,
)
from plasmid_priority.reporting.report_surface_helpers import (
    build_official_benchmark_context as _build_official_benchmark_context,
)
from plasmid_priority.reporting.report_surface_helpers import (
    primary_baseline_delta_text as _primary_baseline_delta_text,
)
from plasmid_priority.reporting.report_surface_helpers import (
    prune_duplicate_table_artifacts as _prune_duplicate_table_artifacts,
)
from plasmid_priority.reporting.report_surface_helpers import (
    prune_shadowed_report_tables as _prune_shadowed_report_tables,
)
from plasmid_priority.reporting.report_surface_helpers import (
    select_summary_candidate_briefs as _select_summary_candidate_briefs,
)
from plasmid_priority.utils import benchmark_runtime
from plasmid_priority.utils.dataframe import coalescing_left_merge, read_parquet, read_tsv
from plasmid_priority.utils.files import (
    load_signature_manifest,
    materialize_recorded_paths,
    path_signature,
    project_python_source_paths,
    write_signature_manifest,
)
from plasmid_priority.utils.numeric_ops import copy_frame, fill0, int0, to_numeric_series


def _build_report_model_metrics(
    model_metrics: pd.DataFrame,
    *,
    calibration_metrics: pd.DataFrame | None = None,
    permutation_summary: pd.DataFrame | None = None,
    selection_adjusted_permutation_summary: pd.DataFrame | None = None,
    comparison_table: pd.DataFrame | None = None,
    confirmatory_cohort_summary: pd.DataFrame | None = None,
    spatial_holdout_summary: pd.DataFrame | None = None,
    primary_model_name: str | None = None,
    governance_model_name: str | None = None,
    baseline_model_name: str = "baseline_both",
) -> pd.DataFrame:
    report_metrics = model_metrics.pipe(copy_frame)
    calibration_metrics = calibration_metrics if calibration_metrics is not None else pd.DataFrame()
    permutation_summary = permutation_summary if permutation_summary is not None else pd.DataFrame()
    selection_adjusted_permutation_summary = (
        selection_adjusted_permutation_summary
        if selection_adjusted_permutation_summary is not None
        else pd.DataFrame()
    )
    comparison_table = comparison_table if comparison_table is not None else pd.DataFrame()
    confirmatory_cohort_summary = (
        confirmatory_cohort_summary if confirmatory_cohort_summary is not None else pd.DataFrame()
    )
    spatial_holdout_summary = (
        spatial_holdout_summary if spatial_holdout_summary is not None else pd.DataFrame()
    )

    if not calibration_metrics.empty:
        keep = [
            column
            for column in [
                "model_name",
                "ece",
                "expected_calibration_error",
                "max_calibration_error",
                "scientific_acceptance_failed_criteria",
            ]
            if column in calibration_metrics.columns
        ]
        if keep:
            report_metrics = report_metrics.merge(
                calibration_metrics[keep].drop_duplicates("model_name"),
                on="model_name",
                how="left",
            )

    if not permutation_summary.empty:
        keep = [
            column
            for column in ["model_name", "n_permutations", "empirical_p_roc_auc"]
            if column in permutation_summary.columns
        ]
        if keep:
            permutation_payload = (
                permutation_summary[keep]
                .drop_duplicates("model_name")
                .rename(columns={"empirical_p_roc_auc": "permutation_p_roc_auc"})
            )
            report_metrics = report_metrics.merge(permutation_payload, on="model_name", how="left")

    if not selection_adjusted_permutation_summary.empty:
        keep = [
            column
            for column in [
                "model_name",
                "n_permutations",
                "selection_adjusted_empirical_p_roc_auc",
                "selection_adjusted_empirical_p_average_precision",
                "n_models_in_scope",
                "modal_selected_model_name",
                "modal_selected_model_share",
            ]
            if column in selection_adjusted_permutation_summary.columns
        ]
        if keep:
            report_metrics = report_metrics.merge(
                selection_adjusted_permutation_summary[keep].drop_duplicates("model_name"),
                on="model_name",
                how="left",
                suffixes=("", "_selection_adjusted"),
            )

    if not comparison_table.empty and primary_model_name:
        baseline_delta = comparison_table.loc[
            comparison_table.get("primary_model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(str(primary_model_name))
            & comparison_table.get("comparison_model_name", pd.Series(dtype=str))
            .astype(str)
            .eq("baseline_both")
        ].head(1)
        if not baseline_delta.empty:
            delta = baseline_delta.iloc[0]
            report_metrics["delta_vs_baseline_roc_auc"] = np.nan
            report_metrics["delta_vs_baseline_ci_lower"] = np.nan
            report_metrics["delta_vs_baseline_ci_upper"] = np.nan
            primary_mask = report_metrics["model_name"].astype(str).eq(str(primary_model_name))
            report_metrics.loc[primary_mask, "delta_vs_baseline_roc_auc"] = float(
                delta["delta_roc_auc"],
            )
            report_metrics.loc[primary_mask, "delta_vs_baseline_ci_lower"] = float(
                delta["delta_roc_auc_ci_lower"],
            )
            report_metrics.loc[primary_mask, "delta_vs_baseline_ci_upper"] = float(
                delta["delta_roc_auc_ci_upper"],
            )

    if not spatial_holdout_summary.empty:
        keep = [
            column
            for column in [
                "model_name",
                "spatial_holdout_roc_auc",
                "spatial_holdout_regions",
                "spatial_holdout_n_backbones",
                "best_spatial_holdout_region",
                "best_spatial_holdout_region_roc_auc",
                "worst_spatial_holdout_region",
                "worst_spatial_holdout_region_roc_auc",
            ]
            if column in spatial_holdout_summary.columns
        ]
        if keep:
            report_metrics = report_metrics.merge(
                spatial_holdout_summary[keep].drop_duplicates("model_name"),
                on="model_name",
                how="left",
            )

    if not confirmatory_cohort_summary.empty and primary_model_name:
        confirmatory_primary = confirmatory_cohort_summary.loc[
            confirmatory_cohort_summary.get("cohort_name", pd.Series(dtype=str))
            .astype(str)
            .eq("confirmatory_internal")
            & confirmatory_cohort_summary.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(str(primary_model_name))
            & confirmatory_cohort_summary.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
        ].head(1)
        if not confirmatory_primary.empty:
            confirmatory = confirmatory_primary.iloc[0]
            confirmatory_row: dict[str, object] = {
                column: np.nan for column in report_metrics.columns
            }
            confirmatory_row.update(
                {
                    "model_name": "internal_high_integrity_subset_primary_model",
                    "status": "ok",
                    "roc_auc": float(confirmatory.get("roc_auc", np.nan)),
                    "average_precision": float(confirmatory.get("average_precision", np.nan)),
                    "brier_score": float(confirmatory.get("brier_score", np.nan)),
                    "positive_prevalence": float(confirmatory.get("positive_prevalence", np.nan)),
                    "n_backbones": int(confirmatory.get("n_backbones", 0)),
                    "n_positive": int(confirmatory.get("n_positive", 0)),
                    "ece": float(
                        confirmatory.get(
                            "ece",
                            confirmatory.get("expected_calibration_error", np.nan),
                        ),
                    ),
                    "expected_calibration_error": float(
                        confirmatory.get("expected_calibration_error", np.nan),
                    ),
                    "max_calibration_error": float(
                        confirmatory.get("max_calibration_error", np.nan),
                    ),
                    "share_of_primary_eligible": float(
                        confirmatory.get("share_of_primary_eligible", np.nan),
                    ),
                },
            )
            report_metrics = pd.concat(
                [report_metrics, pd.DataFrame([confirmatory_row])],
                ignore_index=True,
                sort=False,
            )

    report_metrics["brier_skill_score"] = [
        _brier_skill_score(
            row.get("brier_score", np.nan),
            row.get("positive_prevalence", np.nan),
        )
        for row in report_metrics.to_dict(orient="records")
    ]

    official_names = set(
        _official_report_model_names(
            primary_model_name=primary_model_name,
            governance_model_name=governance_model_name,
            baseline_model_name=baseline_model_name,
        ),
    )
    internal_audit_names = {"internal_high_integrity_subset_primary_model"}
    report_metrics["report_visibility"] = np.where(
        report_metrics["model_name"].astype(str).isin(official_names),
        "official",
        np.where(
            report_metrics["model_name"].astype(str).isin(internal_audit_names),
            "internal_audit",
            "audit_only",
        ),
    )
    display_order = {
        name: index
        for index, name in enumerate(
            [
                *(
                    _official_report_model_names(
                        primary_model_name=primary_model_name,
                        governance_model_name=governance_model_name,
                        baseline_model_name=baseline_model_name,
                    )
                ),
                "internal_high_integrity_subset_primary_model",
            ],
        )
    }
    report_metrics["_display_order"] = (
        report_metrics["model_name"].astype(str).map(lambda value: display_order.get(value, 999))
    )
    report_metrics = report_metrics.sort_values(
        ["_display_order", "roc_auc", "model_name"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop(columns="_display_order")

    leading_columns = [
        "model_name",
        "status",
        "roc_auc",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "average_precision",
        "average_precision_ci_lower",
        "average_precision_ci_upper",
        "brier_score",
        "brier_skill_score",
        "brier_score_ci_lower",
        "brier_score_ci_upper",
        "ece",
        "expected_calibration_error",
        "max_calibration_error",
        "selection_adjusted_empirical_p_roc_auc",
        "permutation_p_roc_auc",
        "n_permutations",
        "delta_vs_baseline_roc_auc",
        "delta_vs_baseline_ci_lower",
        "delta_vs_baseline_ci_upper",
        "spatial_holdout_roc_auc",
        "spatial_holdout_regions",
        "spatial_holdout_n_backbones",
        "best_spatial_holdout_region",
        "best_spatial_holdout_region_roc_auc",
        "worst_spatial_holdout_region",
        "worst_spatial_holdout_region_roc_auc",
        "n_backbones",
        "n_positive",
        "positive_prevalence",
        "share_of_primary_eligible",
        "report_visibility",
    ]
    trailing_columns = [
        column for column in report_metrics.columns if column not in leading_columns
    ]
    ordered_columns = [
        column for column in leading_columns if column in report_metrics.columns
    ] + trailing_columns
    return report_metrics[ordered_columns]


def _build_candidate_brief_table(
    candidate_portfolio: pd.DataFrame,
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    *,
    model_selection_summary: pd.DataFrame | None = None,
    decision_yield: pd.DataFrame | None = None,
    split_year: int = 2015,
) -> pd.DataFrame:
    if candidate_portfolio.empty or backbones.empty:
        return pd.DataFrame()
    candidate_portfolio = candidate_portfolio.loc[
        :,
        ~candidate_portfolio.columns.duplicated(),
    ].pipe(copy_frame)
    merged = (
        backbones.merge(
            amr_consensus[["sequence_accession", "amr_gene_symbols", "amr_drug_classes"]],
            on="sequence_accession",
            how="left",
        )
        if not amr_consensus.empty
        else backbones.pipe(copy_frame)
    )
    years = to_numeric_series(merged["resolved_year"]).pipe(int0)
    merged = merged.assign(resolved_year_int=years)
    model_selection_summary = (
        model_selection_summary if model_selection_summary is not None else pd.DataFrame()
    )
    decision_yield = decision_yield if decision_yield is not None else pd.DataFrame()
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )

    def _decision_yield_lookup(prefix: str, top_k: int) -> tuple[float, float]:
        if decision_yield.empty or "model_name" not in decision_yield.columns:
            return (np.nan, np.nan)
        match = decision_yield.loc[
            decision_yield["model_name"].astype(str).eq(str(prefix))
            & to_numeric_series(
                decision_yield.get("top_k", pd.Series(np.nan, index=decision_yield.index)),
                errors="coerce",
            )
            .astype("Int64")
            .eq(int(top_k))
        ].head(1)
        if match.empty:
            return (np.nan, np.nan)
        row = match.iloc[0]
        return (
            float(row.get("precision_at_k", np.nan)),
            float(row.get("recall_at_k", np.nan)),
        )

    official_primary_model = str(selection_row.get("published_primary_model", "") or "").strip()
    official_governance_model = str(selection_row.get("governance_primary_model", "") or "").strip()
    official_conservative_model = str(
        selection_row.get("conservative_model_name", "") or "",
    ).strip()

    rows: list[dict[str, object]] = []
    # Preserve first-seen order while guaranteeing unique categorical categories.
    candidate_ids = list(dict.fromkeys(candidate_portfolio["backbone_id"].astype(str).tolist()))
    for row in candidate_portfolio.to_dict(orient="records"):
        backbone_id = str(row["backbone_id"])
        frame = merged.loc[merged["backbone_id"].astype(str) == backbone_id].pipe(copy_frame)
        if frame.empty:
            continue
        training = frame.loc[frame["resolved_year_int"] <= split_year].pipe(copy_frame)
        testing = frame.loc[
            (frame["resolved_year_int"] > split_year) & (frame["resolved_year_int"] <= 2023)
        ].pipe(copy_frame)
        training_countries = sorted(
            {value for value in training["country"].fillna("").astype(str).str.strip() if value},
        )
        testing_countries = sorted(
            {value for value in testing["country"].fillna("").astype(str).str.strip() if value},
        )
        new_countries = sorted(set(testing_countries) - set(training_countries))
        summary_context = training if not training.empty else frame
        dominant_species = _dominant_non_empty(summary_context["species"])
        dominant_genus = _dominant_non_empty(summary_context["genus"])
        primary_replicon = _dominant_non_empty(summary_context["primary_replicon"])
        dominant_source = _dominant_non_empty(summary_context["record_origin"])
        top_amr_classes = _top_public_health_amr_classes(
            summary_context.get("amr_drug_classes", pd.Series(dtype=str)),
        )
        top_amr_genes = _top_likely_amr_genes(
            summary_context.get("amr_gene_symbols", pd.Series(dtype=str)),
        )
        source_support_tier = str(row.get("source_support_tier", ""))
        evidence_tier = str(
            row.get("evidence_tier", row.get("candidate_confidence_tier", "")) or "",
        )
        action_tier = str(row.get("action_tier", row.get("recommended_monitoring_tier", "")) or "")
        consensus_membership = bool(row.get("in_consensus_top50", False))
        enriched_signatures = str(row.get("module_f_enriched_signatures", "") or "")
        species_or_genus = _humanize_taxon_label(dominant_species or dominant_genus or backbone_id)
        new_country_count = int(row.get("n_new_countries", len(new_countries)) or 0)
        primary_driver_axis = _humanize_candidate_reason(row.get("primary_driver_axis", ""))
        secondary_driver_axis = _humanize_candidate_reason(row.get("secondary_driver_axis", ""))
        mechanistic_rationale = _humanize_candidate_reason(row.get("mechanistic_rationale", ""))
        monitoring_rationale = _humanize_candidate_reason(row.get("monitoring_rationale", ""))
        operational_risk_score = to_numeric_series(
            pd.Series([row.get("operational_risk_score", np.nan)]),
        ).iloc[0]
        macro_jump_risk = to_numeric_series(
            pd.Series([row.get("risk_macro_region_jump_3y", np.nan)]),
        ).iloc[0]
        event_within_3y_risk = to_numeric_series(
            pd.Series([row.get("risk_event_within_3y", np.nan)]),
        ).iloc[0]
        three_countries_5y_risk = to_numeric_series(
            pd.Series([row.get("risk_three_countries_within_5y", np.nan)]),
        ).iloc[0]
        risk_uncertainty = to_numeric_series(
            pd.Series([row.get("risk_uncertainty", np.nan)]),
        ).iloc[0]
        risk_decision_tier = str(row.get("risk_decision_tier", "") or "").strip()
        candidate_confidence_score = to_numeric_series(
            pd.Series([row.get("candidate_confidence_score", np.nan)]),
        ).iloc[0]
        candidate_explanation_summary = str(row.get("candidate_explanation_summary", "") or "")
        multiverse_stability_score = to_numeric_series(
            pd.Series([row.get("multiverse_stability_score", np.nan)]),
        ).iloc[0]
        multiverse_stability_tier = str(row.get("multiverse_stability_tier", "") or "").strip()
        bootstrap_top10_value = to_numeric_series(
            pd.Series([row.get("bootstrap_top_10_frequency", np.nan)]),
        ).iloc[0]
        variant_top10_value = to_numeric_series(
            pd.Series([row.get("variant_top_10_frequency", np.nan)]),
        ).iloc[0]
        track_value = str(row.get("portfolio_track", "") or "")
        track_en = _humanize_portfolio_track(track_value, language="en") or "candidate context"
        track_tr = _humanize_portfolio_track(track_value, language="tr") or "aday baglami"
        evidence_tier_en = _humanize_evidence_tier(evidence_tier, language="en") or "unspecified"
        evidence_tier_tr = _humanize_evidence_tier(evidence_tier, language="tr") or "belirtilmedi"
        action_tier_en = _humanize_action_tier(action_tier, language="en") or "unassigned"
        action_tier_tr = _humanize_action_tier(action_tier, language="tr") or "atanmadi"
        consensus_en = "yes" if consensus_membership else "no"
        consensus_tr = "evet" if consensus_membership else "hayir"
        novelty_note_en = (
            " It is not the main published-primary shortlist and should be treated as a separate exploratory watchlist."
            if track_value == "novel_signal"
            else ""
        )
        novelty_note_tr = (
            " Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir."
            if track_value == "novel_signal"
            else ""
        )
        mechanistic_note_en = (
            f" Mechanistic emphasis: {primary_driver_axis or 'mixed signal'}"
            + (f" with secondary {secondary_driver_axis}" if secondary_driver_axis else "")
            + (f"; profile tags: {mechanistic_rationale}." if mechanistic_rationale else ".")
        )
        mechanistic_note_tr = (
            f" Mekanistik agirlik: {primary_driver_axis or 'karma sinyal'}"
            + (f"; ikincil eksen {secondary_driver_axis}" if secondary_driver_axis else "")
            + (f"; profil etiketleri: {mechanistic_rationale}." if mechanistic_rationale else ".")
        )
        monitoring_note_en = (
            f" Monitoring rationale: {monitoring_rationale}." if monitoring_rationale else ""
        )
        monitoring_note_tr = (
            f" Izlem gerekcesi: {monitoring_rationale}." if monitoring_rationale else ""
        )
        risk_note_en = ""
        if pd.notna(operational_risk_score):
            risk_note_en = (
                f" Operational risk dictionary: overall `{float(operational_risk_score):.2f}`, "
                f"3-year event `{float(event_within_3y_risk):.2f}`"
                if pd.notna(event_within_3y_risk)
                else f" Operational risk dictionary: overall `{float(operational_risk_score):.2f}`"
            )
            if pd.notna(macro_jump_risk):
                risk_note_en += f", macro-region jump `{float(macro_jump_risk):.2f}`"
            if pd.notna(three_countries_5y_risk):
                risk_note_en += (
                    f", 5-year >=3-country spread `{float(three_countries_5y_risk):.2f}`"
                )
            if risk_decision_tier:
                risk_note_en += f"; decision tier `{risk_decision_tier}`"
            if pd.notna(risk_uncertainty):
                risk_note_en += f"; uncertainty `{float(risk_uncertainty):.2f}`"
            risk_note_en += "."
        risk_note_tr = ""
        if pd.notna(operational_risk_score):
            risk_note_tr = (
                f" Operasyonel risk sozlugu: genel `{float(operational_risk_score):.2f}`, "
                f"3 yil icinde olay `{float(event_within_3y_risk):.2f}`"
                if pd.notna(event_within_3y_risk)
                else f" Operasyonel risk sozlugu: genel `{float(operational_risk_score):.2f}`"
            )
            if pd.notna(macro_jump_risk):
                risk_note_tr += f", makro-bolge sicrama `{float(macro_jump_risk):.2f}`"
            if pd.notna(three_countries_5y_risk):
                risk_note_tr += f", 5 yilda >=3 ulke yayilim `{float(three_countries_5y_risk):.2f}`"
            if risk_decision_tier:
                risk_note_tr += f"; karar seviyesi `{risk_decision_tier}`"
        if pd.notna(risk_uncertainty):
            risk_note_tr += f"; belirsizlik `{float(risk_uncertainty):.2f}`"
        risk_note_tr += "."
        uncertainty_review_tier = str(row.get("uncertainty_review_tier", "") or "").strip().lower()
        if not uncertainty_review_tier or uncertainty_review_tier == "nan":
            uncertainty_review_tier = "clear"
        uncertainty_review_labels_en = {
            "clear": "clear",
            "review": "review",
            "abstain": "abstain",
        }
        uncertainty_review_labels_tr = {
            "clear": "temiz",
            "review": "inceleme",
            "abstain": "cekimser",
        }
        uncertainty_review_note_en = f" Uncertainty review tier: {uncertainty_review_labels_en.get(uncertainty_review_tier, uncertainty_review_tier)}."
        uncertainty_review_note_tr = f" Belirsizlik inceleme seviyesi: {uncertainty_review_labels_tr.get(uncertainty_review_tier, uncertainty_review_tier)}."
        confidence_note_en = (
            f" Confidence score: {float(candidate_confidence_score):.2f}."
            if pd.notna(candidate_confidence_score)
            else ""
        )
        confidence_note_tr = (
            f" Guven skoru: {float(candidate_confidence_score):.2f}."
            if pd.notna(candidate_confidence_score)
            else ""
        )
        primary_top10_precision, primary_top10_recall = _decision_yield_lookup(
            official_primary_model,
            10,
        )
        primary_top25_precision, primary_top25_recall = _decision_yield_lookup(
            official_primary_model,
            25,
        )
        governance_top10_precision, governance_top10_recall = _decision_yield_lookup(
            official_governance_model,
            10,
        )
        governance_top25_precision, governance_top25_recall = _decision_yield_lookup(
            official_governance_model,
            25,
        )
        conservative_top10_precision, conservative_top10_recall = _decision_yield_lookup(
            official_conservative_model,
            10,
        )
        decision_yield_note_en = ""
        if pd.notna(primary_top10_precision) or pd.notna(governance_top10_precision):
            decision_yield_bits: list[str] = []
            if pd.notna(primary_top10_precision):
                decision_yield_bits.append(
                    f"primary top-10 precision {float(primary_top10_precision):.2f}",
                )
            if pd.notna(primary_top25_precision):
                decision_yield_bits.append(
                    f"primary top-25 recall {float(primary_top25_recall):.2f}",
                )
            if pd.notna(governance_top10_precision):
                decision_yield_bits.append(
                    f"governance top-10 precision {float(governance_top10_precision):.2f}",
                )
            if pd.notna(conservative_top10_precision):
                decision_yield_bits.append(
                    f"conservative top-10 precision {float(conservative_top10_precision):.2f}",
                )
            decision_yield_note_en = (
                " Official benchmark yield: " + "; ".join(decision_yield_bits) + "."
            )
        decision_yield_note_tr = ""
        if pd.notna(primary_top10_precision) or pd.notna(governance_top10_precision):
            decision_yield_bits_tr: list[str] = []
            if pd.notna(primary_top10_precision):
                decision_yield_bits_tr.append(
                    f"birincil top-10 isabet {float(primary_top10_precision):.2f}",
                )
            if pd.notna(primary_top25_recall):
                decision_yield_bits_tr.append(
                    f"birincil top-25 yakalama {float(primary_top25_recall):.2f}",
                )
            if pd.notna(governance_top10_precision):
                decision_yield_bits_tr.append(
                    f"yonetisim top-10 isabet {float(governance_top10_precision):.2f}",
                )
            if pd.notna(conservative_top10_precision):
                decision_yield_bits_tr.append(
                    f"konservatif top-10 isabet {float(conservative_top10_precision):.2f}",
                )
            decision_yield_note_tr = (
                " Resmi benchmark verimi: " + "; ".join(decision_yield_bits_tr) + "."
            )
        if pd.notna(multiverse_stability_score):
            stability_note_en = (
                f" Rank stability: {multiverse_stability_tier or 'multiverse'} "
                f"{float(multiverse_stability_score):.2f}."
            )
            stability_note_tr = (
                f" Sira kararliligi: {multiverse_stability_tier or 'multiverse'} "
                f"{float(multiverse_stability_score):.2f}."
            )
        else:
            stability_bits_en: list[str] = []
            stability_bits_tr: list[str] = []
            if pd.notna(bootstrap_top10_value):
                stability_bits_en.append(f"bootstrap top-10 {float(bootstrap_top10_value):.2f}")
                stability_bits_tr.append(f"bootstrap top-10 {float(bootstrap_top10_value):.2f}")
            if pd.notna(variant_top10_value):
                stability_bits_en.append(f"variant top-10 {float(variant_top10_value):.2f}")
                stability_bits_tr.append(f"varyant top-10 {float(variant_top10_value):.2f}")
            stability_note_en = (
                f" Rank stability: {', '.join(stability_bits_en)}." if stability_bits_en else ""
            )
            stability_note_tr = (
                f" Sira kararliligi: {', '.join(stability_bits_tr)}." if stability_bits_tr else ""
            )
        primary_decision_utility_score = to_numeric_series(
            pd.Series([row.get("official_primary_decision_utility_score", np.nan)]),
            errors="coerce",
        ).iloc[0]
        primary_decision_threshold = to_numeric_series(
            pd.Series([row.get("official_primary_optimal_decision_threshold", np.nan)]),
            errors="coerce",
        ).iloc[0]
        utility_note_en = (
            f" Decision utility: {float(primary_decision_utility_score):.2f} at threshold "
            f"{float(primary_decision_threshold):.2f}."
            if pd.notna(primary_decision_utility_score) and pd.notna(primary_decision_threshold)
            else ""
        )
        utility_note_tr = (
            f" Karar faydasi: {float(primary_decision_utility_score):.2f}; esik "
            f"{float(primary_decision_threshold):.2f}."
            if pd.notna(primary_decision_utility_score) and pd.notna(primary_decision_threshold)
            else ""
        )
        summary_en = (
            f"{backbone_id} is dominated by {species_or_genus}; training-period support is {int(row.get('member_count_train', 0) or 0)} records across "
            f"{int(row.get('n_countries_train', 0) or 0)} countries, and after 2015 it appears in {new_country_count} new countries. "
            f"For review, treat it as {track_en}. Evidence tier: {evidence_tier_en}; monitoring tier: {action_tier_en}; multi-model consensus top-50: {consensus_en}."
            f"{novelty_note_en}{mechanistic_note_en} {monitoring_note_en}{risk_note_en}{uncertainty_review_note_en}{confidence_note_en}{stability_note_en}{decision_yield_note_en}{utility_note_en}"
            f"{' Case summary: ' + candidate_explanation_summary + '.' if candidate_explanation_summary else ''}"
            f" Main public-health AMR classes: {top_amr_classes or 'none detected'}."
        )
        summary_tr = (
            f"{backbone_id} omurgasi agirlikli olarak {species_or_genus} ile iliskilidir; egitim doneminde {int(row.get('member_count_train', 0) or 0)} kayit ve "
            f"{int(row.get('n_countries_train', 0) or 0)} ulke destegi vardir, 2015 sonrasinda ise {new_country_count} yeni ulkede gorulmustur. "
            f"Juri yorumu icin bu aday {track_tr} olarak ele alinmalidir. Kanit seviyesi {evidence_tier_tr}; izlem duzeyi {action_tier_tr}; coklu model uzlasi top-50 durumu: {consensus_tr}."
            f"{novelty_note_tr}{mechanistic_note_tr} {monitoring_note_tr}{risk_note_tr}{uncertainty_review_note_tr}{confidence_note_tr}{stability_note_tr}{decision_yield_note_tr}{utility_note_tr}"
            f"{' Vaka ozeti: ' + candidate_explanation_summary + '.' if candidate_explanation_summary else ''}"
            f" Baskin halk sagligi odakli AMR siniflari: {top_amr_classes or 'tespit edilmedi'}."
        )
        rows.append(
            {
                "portfolio_track": row.get("portfolio_track"),
                "track_rank": row.get("track_rank"),
                "backbone_id": backbone_id,
                "official_primary_model": official_primary_model,
                "official_conservative_model": official_conservative_model,
                "official_governance_model": official_governance_model,
                "official_benchmark_panel_size": to_numeric_series(
                    pd.Series([row.get("official_benchmark_panel_size", np.nan)]),
                ).iloc[0],
                "official_primary_top_10_precision": primary_top10_precision,
                "official_primary_top_10_recall": primary_top10_recall,
                "official_primary_top_25_precision": primary_top25_precision,
                "official_primary_top_25_recall": primary_top25_recall,
                "official_primary_decision_utility_score": to_numeric_series(
                    pd.Series([row.get("official_primary_decision_utility_score", np.nan)]),
                    errors="coerce",
                ).iloc[0],
                "official_primary_optimal_decision_threshold": to_numeric_series(
                    pd.Series([row.get("official_primary_optimal_decision_threshold", np.nan)]),
                    errors="coerce",
                ).iloc[0],
                "official_governance_top_10_precision": governance_top10_precision,
                "official_governance_top_10_recall": governance_top10_recall,
                "official_governance_top_25_precision": governance_top25_precision,
                "official_governance_top_25_recall": governance_top25_recall,
                "official_governance_decision_utility_score": to_numeric_series(
                    pd.Series([row.get("official_governance_decision_utility_score", np.nan)]),
                    errors="coerce",
                ).iloc[0],
                "official_governance_optimal_decision_threshold": to_numeric_series(
                    pd.Series([row.get("official_governance_optimal_decision_threshold", np.nan)]),
                    errors="coerce",
                ).iloc[0],
                "official_conservative_top_10_precision": conservative_top10_precision,
                "official_conservative_top_10_recall": conservative_top10_recall,
                "official_conservative_decision_utility_score": to_numeric_series(
                    pd.Series([row.get("official_conservative_decision_utility_score", np.nan)]),
                    errors="coerce",
                ).iloc[0],
                "official_conservative_optimal_decision_threshold": to_numeric_series(
                    pd.Series(
                        [row.get("official_conservative_optimal_decision_threshold", np.nan)],
                    ),
                    errors="coerce",
                ).iloc[0],
                "dominant_genus": dominant_genus,
                "dominant_species": dominant_species,
                "primary_replicon": primary_replicon,
                "dominant_record_origin": dominant_source,
                "source_support_tier": source_support_tier,
                "training_country_count": int(len(training_countries)),
                "training_country_examples": ",".join(training_countries[:5]),
                "new_country_count_post_2015": int(len(new_countries)),
                "new_country_examples_post_2015": ",".join(new_countries[:5]),
                "bootstrap_top_10_frequency": bootstrap_top10_value,
                "variant_top_10_frequency": variant_top10_value,
                "first_year_observed": int(
                    frame["resolved_year_int"].loc[frame["resolved_year_int"] > 0].min(),
                )
                if (frame["resolved_year_int"] > 0).any()
                else np.nan,
                "last_year_observed": int(
                    frame["resolved_year_int"].loc[frame["resolved_year_int"] > 0].max(),
                )
                if (frame["resolved_year_int"] > 0).any()
                else np.nan,
                "top_amr_classes": top_amr_classes,
                "top_amr_genes": top_amr_genes,
                "module_f_enriched_signatures": enriched_signatures,
                "primary_driver_axis": row.get("primary_driver_axis", ""),
                "secondary_driver_axis": row.get("secondary_driver_axis", ""),
                "mechanistic_rationale": row.get("mechanistic_rationale", ""),
                "monitoring_rationale": row.get("monitoring_rationale", ""),
                "candidate_confidence_score": candidate_confidence_score,
                "multiverse_stability_score": multiverse_stability_score,
                "multiverse_stability_tier": multiverse_stability_tier,
                "candidate_explanation_summary": candidate_explanation_summary,
                "low_candidate_confidence_risk": bool(
                    row.get("low_candidate_confidence_risk", False),
                ),
                "operational_risk_score": operational_risk_score,
                "risk_macro_region_jump_3y": macro_jump_risk,
                "risk_event_within_3y": event_within_3y_risk,
                "risk_three_countries_within_5y": three_countries_5y_risk,
                "risk_uncertainty": risk_uncertainty,
                "risk_decision_tier": risk_decision_tier,
                "uncertainty_review_tier": uncertainty_review_tier,
                "candidate_summary_en": summary_en,
                "candidate_summary_tr": summary_tr,
            },
        )
    brief = pd.DataFrame(rows)
    if brief.empty:
        return brief
    order = pd.Categorical(brief["backbone_id"], categories=candidate_ids, ordered=True)
    return (
        brief.assign(_order=order)
        .sort_values(["portfolio_track", "track_rank", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _add_visibility_alias(frame: pd.DataFrame) -> pd.DataFrame:
    if (
        frame.empty
        or "spread_label" not in frame.columns
        or "visibility_expansion_label" in frame.columns
    ):
        return frame
    aliased = frame.pipe(copy_frame)
    aliased["visibility_expansion_label"] = aliased["spread_label"]
    return aliased


def _deduplicate_backbone_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per backbone_id while keeping the strongest-ranked row first."""
    if frame.empty or "backbone_id" not in frame.columns:
        return frame
    working = frame.pipe(copy_frame)
    working["backbone_id"] = working["backbone_id"].astype(str)
    if not working["backbone_id"].duplicated().any():
        return working

    sort_columns: list[str] = []
    ascending: list[bool] = []

    if "portfolio_track" in working.columns:
        sort_columns.append("portfolio_track")
        ascending.append(True)
    if "track_rank" in working.columns:
        working["_track_rank_order"] = to_numeric_series(working["track_rank"]).fillna(
            np.inf,
        )
        sort_columns.append("_track_rank_order")
        ascending.append(True)
    if "candidate_confidence_score" in working.columns:
        working["_candidate_confidence_order"] = to_numeric_series(
            working["candidate_confidence_score"],
        ).fillna(-np.inf)
        sort_columns.append("_candidate_confidence_order")
        ascending.append(False)
    if "multiverse_stability_score" in working.columns:
        working["_multiverse_stability_order"] = to_numeric_series(
            working["multiverse_stability_score"],
        ).fillna(-np.inf)
        sort_columns.append("_multiverse_stability_order")
        ascending.append(False)

    sort_columns.append("backbone_id")
    ascending.append(True)

    deduped = (
        working.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates("backbone_id", keep="first")
        .drop(
            columns=[
                column
                for column in (
                    "_track_rank_order",
                    "_candidate_confidence_order",
                    "_multiverse_stability_order",
                )
                if column in working.columns
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )
    return deduped


def _build_candidate_evidence_matrix(
    candidate_portfolio: pd.DataFrame,
    candidate_briefs: pd.DataFrame,
    candidate_threshold_flip: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_portfolio.empty:
        return pd.DataFrame()
    matrix = candidate_portfolio.pipe(copy_frame)
    if not candidate_briefs.empty:
        brief_columns = [
            column
            for column in [
                "backbone_id",
                "dominant_genus",
                "dominant_species",
                "top_amr_classes",
                "top_amr_genes",
            ]
            if column in candidate_briefs.columns
        ]
        if brief_columns:
            matrix = coalescing_left_merge(
                matrix,
                candidate_briefs[brief_columns],
                on="backbone_id",
            )
    if not candidate_threshold_flip.empty:
        flip_columns = [
            column
            for column in [
                "backbone_id",
                "threshold_flip_count",
                "eligible_for_threshold_audit",
                "default_threshold",
                "spread_label_default",
                "label_ge_2",
                "label_ge_3",
                "label_ge_4",
            ]
            if column in candidate_threshold_flip.columns
        ]
        if flip_columns:
            matrix = coalescing_left_merge(
                matrix,
                candidate_threshold_flip[flip_columns],
                on="backbone_id",
            )
    preferred_columns = [
        "portfolio_track",
        "track_rank",
        "backbone_id",
        "official_primary_model",
        "official_conservative_model",
        "official_governance_model",
        "official_benchmark_panel_size",
        "official_primary_top_10_precision",
        "official_primary_top_10_recall",
        "official_primary_top_25_precision",
        "official_primary_top_25_recall",
        "official_primary_decision_utility_score",
        "official_primary_optimal_decision_threshold",
        "official_governance_top_10_precision",
        "official_governance_top_10_recall",
        "official_governance_top_25_precision",
        "official_governance_top_25_recall",
        "official_governance_decision_utility_score",
        "official_governance_optimal_decision_threshold",
        "official_conservative_top_10_precision",
        "official_conservative_top_10_recall",
        "official_conservative_decision_utility_score",
        "official_conservative_optimal_decision_threshold",
        "evidence_tier",
        "action_tier",
        "false_positive_risk_tier",
        "risk_flag_count",
        "consensus_rank",
        "consensus_support_count",
        "rank_disagreement_primary_vs_conservative",
        "candidate_confidence_score",
        "candidate_explanation_summary",
        "low_candidate_confidence_risk",
        "multiverse_stability_score",
        "multiverse_stability_tier",
        "primary_model_candidate_score",
        "conservative_model_candidate_score",
        "baseline_both_candidate_score",
        "novelty_margin_vs_baseline",
        "operational_risk_score",
        "risk_spread_probability",
        "risk_spread_severity",
        "risk_macro_region_jump_3y",
        "risk_event_within_3y",
        "risk_three_countries_within_5y",
        "risk_uncertainty",
        "risk_decision_tier",
        "risk_abstain_flag",
        "bootstrap_top_10_frequency",
        "variant_top_10_frequency",
        "external_support_modalities_count",
        "source_support_tier",
        "module_f_enriched_signature_count",
        "top_amr_classes",
        "top_amr_genes",
        "n_new_countries",
        "spread_label",
        "default_threshold",
        "spread_label_default",
        "threshold_flip_count",
        "eligible_for_threshold_audit",
        "label_ge_2",
        "label_ge_3",
        "label_ge_4",
        "dominant_species",
        "dominant_genus",
    ]
    available = [column for column in preferred_columns if column in matrix.columns]
    return matrix[available].reset_index(drop=True)


def _build_operational_risk_watchlist(
    operational_risk_dictionary: pd.DataFrame,
    *,
    primary_model_name: str,
    candidate_portfolio: pd.DataFrame | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    keep_columns = [
        "operational_risk_rank",
        "backbone_id",
        "model_name",
        "portfolio_track",
        "track_rank",
        "candidate_confidence_tier",
        "recommended_monitoring_tier",
        "false_positive_risk_tier",
        "consensus_rank",
        "operational_risk_score",
        "risk_spread_probability",
        "risk_spread_severity",
        "risk_macro_region_jump_3y",
        "risk_event_within_3y",
        "risk_three_countries_within_5y",
        "risk_uncertainty",
        "risk_uncertainty_quantile",
        "risk_component_std",
        "risk_abstain_flag",
        "risk_decision_tier",
        "risk_route_context",
        "knownness_score",
        "knownness_half",
        "source_band",
        "member_count_band",
        "country_count_band",
    ]
    if operational_risk_dictionary.empty:
        return pd.DataFrame(columns=keep_columns)
    working = operational_risk_dictionary.loc[
        operational_risk_dictionary.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(primary_model_name)
    ].pipe(copy_frame)
    if working.empty:
        return pd.DataFrame(columns=keep_columns)
    if candidate_portfolio is not None and not candidate_portfolio.empty:
        portfolio_columns = [
            column
            for column in [
                "backbone_id",
                "portfolio_track",
                "track_rank",
                "candidate_confidence_tier",
                "recommended_monitoring_tier",
                "consensus_rank",
                "false_positive_risk_tier",
            ]
            if column in candidate_portfolio.columns
        ]
        if portfolio_columns:
            working = coalescing_left_merge(
                working,
                candidate_portfolio[portfolio_columns].drop_duplicates("backbone_id"),
                on="backbone_id",
            )
    decision_order = {"action": 0, "review": 1, "abstain": 2}
    working["risk_decision_tier"] = (
        working.get(
            "risk_decision_tier",
            pd.Series("review", index=working.index, dtype=object),
        )
        .fillna("review")
        .astype(str)
    )
    working["_decision_order"] = (
        working["risk_decision_tier"].map(decision_order).fillna(3).astype(int)
    )
    tier_targets = {
        "action": max(1, int(round(top_k * 0.45))),
        "review": max(1, int(round(top_k * 0.35))),
        "abstain": max(1, top_k - int(round(top_k * 0.45)) - int(round(top_k * 0.35))),
    }
    selected_frames: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    for tier in ("action", "review", "abstain"):
        tier_rows = working.loc[working["risk_decision_tier"].eq(tier)].pipe(copy_frame)
        if tier_rows.empty:
            continue
        tier_rows = tier_rows.sort_values(
            ["operational_risk_score", "risk_uncertainty", "backbone_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        take_n = min(len(tier_rows), tier_targets.get(tier, 0))
        if take_n > 0:
            chosen = tier_rows.head(take_n).pipe(copy_frame)
            selected_ids.update(chosen["backbone_id"].astype(str).tolist())
            selected_frames.append(chosen)
    remaining_slots = max(int(top_k) - sum(len(frame) for frame in selected_frames), 0)
    if remaining_slots > 0:
        remainder = working.loc[~working["backbone_id"].astype(str).isin(selected_ids)].pipe(
            copy_frame
        )
        if not remainder.empty:
            remainder = remainder.sort_values(
                ["_decision_order", "operational_risk_score", "risk_uncertainty", "backbone_id"],
                ascending=[True, False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            selected_frames.append(remainder.head(remaining_slots))
    if selected_frames:
        ordered = pd.concat(selected_frames, ignore_index=True, sort=False)
    else:
        ordered = working.sort_values(
            ["_decision_order", "operational_risk_score", "risk_uncertainty", "backbone_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    ordered["operational_risk_rank"] = np.arange(1, len(ordered) + 1)
    ordered = ordered.drop(columns="_decision_order")
    available_columns = [column for column in keep_columns if column in ordered.columns]
    return ordered[available_columns].reset_index(drop=True)


def _build_candidate_multiverse_stability(
    candidate_stability: pd.DataFrame,
    candidate_threshold_flip: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_stability.empty:
        return pd.DataFrame()
    base = candidate_stability.pipe(copy_frame)
    threshold_cols = [
        column
        for column in [
            "backbone_id",
            "threshold_flip_count",
            "eligible_for_threshold_audit",
        ]
        if column in candidate_threshold_flip.columns
    ]
    if threshold_cols:
        base = coalescing_left_merge(
            base,
            candidate_threshold_flip[threshold_cols].drop_duplicates("backbone_id"),
            on="backbone_id",
        )
    frequency_columns = [
        "bootstrap_top_10_frequency",
        "bootstrap_top_25_frequency",
        "variant_top_10_frequency",
        "variant_top_25_frequency",
    ]
    for column in frequency_columns:
        if column in base.columns:
            base[column] = to_numeric_series(base[column]).pipe(fill0)
    threshold_flip = to_numeric_series(
        base.get("threshold_flip_count", pd.Series(np.nan, index=base.index)),
    )
    eligible_threshold = (
        base.get("eligible_for_threshold_audit", pd.Series(False, index=base.index))
        .fillna(False)
        .astype(bool)
    )
    threshold_robustness = pd.Series(np.nan, index=base.index, dtype=float)
    if eligible_threshold.any():
        max_flip = max(float(threshold_flip.loc[eligible_threshold].max()), 1.0)
        threshold_robustness.loc[eligible_threshold] = (
            1.0 - threshold_flip.loc[eligible_threshold].fillna(max_flip) / max_flip
        ).clip(lower=0.0, upper=1.0)
    base["threshold_robustness_score"] = threshold_robustness
    component_scores: list[pd.Series] = []
    component_active: list[pd.Series] = []
    if "bootstrap_top_25_frequency" in base.columns:
        component_scores.append(base["bootstrap_top_25_frequency"].astype(float))
        component_active.append(pd.Series(True, index=base.index, dtype=bool))
    if "variant_top_25_frequency" in base.columns:
        component_scores.append(base["variant_top_25_frequency"].astype(float))
        component_active.append(pd.Series(True, index=base.index, dtype=bool))
    if threshold_cols:
        component_scores.append(base["threshold_robustness_score"].pipe(fill0).astype(float))
        component_active.append(eligible_threshold)
    if component_scores:
        score_sum = pd.Series(0.0, index=base.index, dtype=float)
        active_count = pd.Series(0, index=base.index, dtype=int)
        for score, active in zip(component_scores, component_active, strict=False):
            score_sum = score_sum + score.where(active, 0.0)
            active_count = active_count + active.astype(int)
        base["multiverse_component_count"] = active_count.astype(int)
        base["multiverse_stability_score"] = np.where(
            active_count > 0,
            score_sum / active_count,
            np.nan,
        )
    else:
        base["multiverse_component_count"] = 0
        base["multiverse_stability_score"] = np.nan
    base["multiverse_stability_tier"] = np.select(
        [
            base["multiverse_stability_score"].pipe(fill0) >= 0.80,
            base["multiverse_stability_score"].pipe(fill0) >= 0.55,
        ],
        ["stable", "moderately_stable"],
        default="fragile",
    )
    preferred = [
        "backbone_id",
        "multiverse_stability_score",
        "multiverse_stability_tier",
        "bootstrap_top_10_frequency",
        "bootstrap_top_25_frequency",
        "variant_top_10_frequency",
        "variant_top_25_frequency",
        "threshold_flip_count",
        "threshold_robustness_score",
        "multiverse_component_count",
        "bootstrap_rank_std",
        "variant_rank_std",
        "coherence_score",
        "knownness_score",
        "primary_model_candidate_score",
    ]
    available = [column for column in preferred if column in base.columns]
    return (
        base[available]
        .sort_values(
            ["multiverse_stability_score", "primary_model_candidate_score"],
            ascending=[False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def _build_candidate_case_studies(
    candidate_briefs: pd.DataFrame,
    *,
    per_track: int = 3,
) -> pd.DataFrame:
    selected = _select_summary_candidate_briefs(candidate_briefs, per_track=per_track)
    if selected.empty:
        return pd.DataFrame()
    preferred_columns = [
        "portfolio_track",
        "track_rank",
        "backbone_id",
        "official_primary_model",
        "official_conservative_model",
        "official_governance_model",
        "official_benchmark_panel_size",
        "official_primary_top_10_precision",
        "official_primary_top_10_recall",
        "official_primary_top_25_precision",
        "official_primary_top_25_recall",
        "official_primary_decision_utility_score",
        "official_primary_optimal_decision_threshold",
        "official_governance_top_10_precision",
        "official_governance_top_10_recall",
        "official_governance_top_25_precision",
        "official_governance_top_25_recall",
        "official_governance_decision_utility_score",
        "official_governance_optimal_decision_threshold",
        "official_conservative_top_10_precision",
        "official_conservative_top_10_recall",
        "official_conservative_decision_utility_score",
        "official_conservative_optimal_decision_threshold",
        "dominant_genus",
        "dominant_species",
        "primary_replicon",
        "dominant_record_origin",
        "source_support_tier",
        "top_amr_classes",
        "top_amr_genes",
        "module_f_enriched_signatures",
        "primary_driver_axis",
        "secondary_driver_axis",
        "mechanistic_rationale",
        "monitoring_rationale",
        "candidate_confidence_score",
        "candidate_explanation_summary",
        "low_candidate_confidence_risk",
        "bootstrap_top_10_frequency",
        "variant_top_10_frequency",
        "multiverse_stability_score",
        "multiverse_stability_tier",
        "operational_risk_score",
        "risk_macro_region_jump_3y",
        "risk_event_within_3y",
        "risk_three_countries_within_5y",
        "risk_uncertainty",
        "risk_decision_tier",
        "uncertainty_review_tier",
        "candidate_summary_en",
        "candidate_summary_tr",
    ]
    available = [column for column in preferred_columns if column in selected.columns]
    return selected[available].reset_index(drop=True)


def _build_headline_validation_summary(
    model_metrics: pd.DataFrame,
    *,
    primary_model_name: str,
    governance_model_name: str,
    single_model_official_decision: pd.DataFrame | None = None,
    single_model_pareto_finalists: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary_specs = [
        ("discovery_primary", primary_model_name),
        ("governance_watch_only", governance_model_name),
        ("counts_baseline", "baseline_both"),
        ("internal_high_integrity_subset", "internal_high_integrity_subset_primary_model"),
    ]
    single_model_official_decision = (
        single_model_official_decision
        if single_model_official_decision is not None
        else pd.DataFrame()
    )
    single_model_pareto_finalists = (
        single_model_pareto_finalists
        if single_model_pareto_finalists is not None
        else pd.DataFrame()
    )
    single_model_decision_row = (
        single_model_official_decision.iloc[0]
        if not single_model_official_decision.empty
        else pd.Series(dtype=object)
    )
    single_model_name = str(single_model_decision_row.get("official_model_name", "") or "").strip()
    if single_model_name:
        summary_specs.append(("single_model_pareto_official", single_model_name))
    for summary_label, model_name in summary_specs:
        row = model_metrics.loc[model_metrics["model_name"].astype(str).eq(str(model_name))].head(1)
        if row.empty and summary_label != "single_model_pareto_official":
            continue
        metric_row = row.iloc[0] if not row.empty else pd.Series(dtype=object)
        single_model_row = (
            single_model_decision_row
            if summary_label == "single_model_pareto_official"
            else pd.Series(dtype=object)
        )
        single_model_finalist_row = pd.Series(dtype=object)
        if (
            summary_label == "single_model_pareto_official"
            and not single_model_pareto_finalists.empty
        ):
            finalist_match = single_model_pareto_finalists.loc[
                single_model_pareto_finalists.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            ].head(1)
            if not finalist_match.empty:
                single_model_finalist_row = finalist_match.iloc[0]
        rows.append(
            {
                "summary_label": summary_label,
                "model_name": str(model_name),
                "roc_auc": float(
                    single_model_finalist_row.get(
                        "roc_auc",
                        single_model_row.get("roc_auc", metric_row.get("roc_auc", np.nan)),
                    ),
                ),
                "roc_auc_ci": _format_interval(
                    metric_row.get("roc_auc_ci_lower"),
                    metric_row.get("roc_auc_ci_upper"),
                ),
                "average_precision": float(
                    single_model_finalist_row.get(
                        "average_precision",
                        single_model_row.get(
                            "average_precision",
                            metric_row.get("average_precision", np.nan),
                        ),
                    ),
                ),
                "average_precision_ci": _format_interval(
                    metric_row.get("average_precision_ci_lower"),
                    metric_row.get("average_precision_ci_upper"),
                ),
                "brier_score": float(
                    single_model_finalist_row.get(
                        "brier_score",
                        metric_row.get("brier_score", np.nan),
                    ),
                ),
                "brier_skill_score": float(
                    single_model_finalist_row.get(
                        "brier_skill_score",
                        metric_row.get("brier_skill_score", np.nan),
                    ),
                ),
                "ece": float(single_model_finalist_row.get("ece", metric_row.get("ece", np.nan))),
                "max_calibration_error": float(
                    single_model_finalist_row.get(
                        "max_calibration_error",
                        metric_row.get("max_calibration_error", np.nan),
                    ),
                ),
                "scientific_acceptance_status": str(
                    single_model_row.get(
                        "scientific_acceptance_status",
                        single_model_finalist_row.get(
                            "scientific_acceptance_status",
                            metric_row.get("scientific_acceptance_status", "not_scored"),
                        ),
                    ),
                ),
                "scientific_acceptance_failed_criteria": str(
                    single_model_row.get(
                        "scientific_acceptance_failed_criteria",
                        single_model_finalist_row.get(
                            "scientific_acceptance_failed_criteria",
                            metric_row.get("scientific_acceptance_failed_criteria", "not_scored"),
                        ),
                    ),
                ),
                "selection_adjusted_empirical_p_roc_auc": float(
                    to_numeric_series(
                        pd.Series(
                            [
                                single_model_finalist_row.get(
                                    "selection_adjusted_empirical_p_roc_auc",
                                    metric_row.get("selection_adjusted_empirical_p_roc_auc"),
                                ),
                            ],
                        ),
                        errors="coerce",
                    ).iloc[0],
                )
                if pd.notna(
                    to_numeric_series(
                        pd.Series(
                            [
                                single_model_finalist_row.get(
                                    "selection_adjusted_empirical_p_roc_auc",
                                    metric_row.get("selection_adjusted_empirical_p_roc_auc"),
                                ),
                            ],
                        ),
                        errors="coerce",
                    ).iloc[0],
                )
                else np.nan,
                "permutation_p_roc_auc": float(
                    to_numeric_series(
                        pd.Series([metric_row.get("permutation_p_roc_auc")]),
                    ).iloc[0],
                )
                if pd.notna(
                    to_numeric_series(
                        pd.Series([metric_row.get("permutation_p_roc_auc")]),
                    ).iloc[0],
                )
                else np.nan,
                "delta_vs_baseline_roc_auc": float(
                    metric_row.get("delta_vs_baseline_roc_auc", np.nan),
                ),
                "delta_vs_baseline_ci": _format_interval(
                    metric_row.get("delta_vs_baseline_ci_lower"),
                    metric_row.get("delta_vs_baseline_ci_upper"),
                ),
                "spatial_holdout_roc_auc": float(
                    single_model_finalist_row.get(
                        "spatial_holdout_roc_auc",
                        metric_row.get("spatial_holdout_roc_auc", np.nan),
                    ),
                ),
                "n_backbones": int(metric_row.get("n_backbones", 0))
                if pd.notna(metric_row.get("n_backbones"))
                else 0,
                "n_positive": int(metric_row.get("n_positive", 0))
                if pd.notna(metric_row.get("n_positive"))
                else 0,
                "decision_reason": str(single_model_row.get("decision_reason", "")),
                "selected_from_n_finalists": int(
                    single_model_row.get("selected_from_n_finalists", 0),
                )
                if pd.notna(single_model_row.get("selected_from_n_finalists"))
                else 0,
                "failure_severity": float(single_model_row.get("failure_severity", np.nan)),
                "weighted_objective_score": float(
                    single_model_row.get("weighted_objective_score", np.nan),
                ),
                "screen_fit_seconds": float(single_model_row.get("screen_fit_seconds", np.nan)),
                "compute_efficiency_score": float(
                    single_model_row.get("compute_efficiency_score", np.nan),
                ),
            },
        )
    return pd.DataFrame(rows)


def _write_headline_validation_summary(
    output_path: Path,
    summary_table: pd.DataFrame,
    *,
    primary_model_name: str,
    governance_model_name: str,
    rolling_temporal: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
) -> None:
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = variant_consistency if variant_consistency is not None else pd.DataFrame()
    primary_status_row = summary_table.loc[
        summary_table["summary_label"].astype(str).eq("discovery_primary")
    ].head(1)
    primary_status = (
        _strict_acceptance_status(primary_status_row.iloc[0])
        if not primary_status_row.empty
        else "not_scored"
    )
    benchmark_scope_note = _benchmark_scope_note(primary_status)
    discovery_primary_label = (
        "Discovery primary benchmark"
        if primary_status == "pass"
        else "Discovery primary benchmark candidate"
    )
    lines = [
        "# Headline Validation Summary",
        "",
        "This is the canonical one-page validation surface for jury review.",
        "",
        f"- {discovery_primary_label}: `{primary_model_name}`",
        f"- {_governance_watch_label()}: `{governance_model_name}`",
        "- Baseline comparator: `baseline_both`",
        f"- {benchmark_scope_note}",
        "- Permutation entries below include the selection-adjusted official-model null; the fixed-score label-permutation audit is retained only as an exploratory appendix diagnostic.",
        "- The explicit leakage canary is exported separately in `future_sentinel_audit.tsv`.",
        "- The frozen acceptance audit is exported separately in `frozen_scientific_acceptance_audit.tsv`.",
        "- The nonlinear deconfounding audit is exported separately in `nonlinear_deconfounding_audit.tsv`.",
        "- Calibration metrics below are fixed-bin diagnostics: ECE, max calibration error, calibration slope, and calibration intercept are reported with their binning semantics made explicit.",
        "- Alternative endpoint audits are exported separately in `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv`.",
        "- The prospective freeze audits are exported separately in `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv`.",
        "- The graph, counterfactual, geographic-jump, and AMR-uncertainty diagnostics are exported separately in `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv`.",
        "- Frozen scientific acceptance combines matched-knownness, source holdout, spatial holdout, calibration, selection-adjusted null, and leakage review.",
        "",
        "| Surface | Model | ROC AUC | ROC AUC 95% CI | AP | AP 95% CI | Brier | Brier Skill | Fixed-bin ECE | Fixed-bin Max CE | Calibration Slope | Calibration Intercept | Frozen Acceptance | Frozen Acceptance Reason | Selection-adjusted p | Fixed-score p | Delta vs baseline | Spatial holdout AUC | n | Positives |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_table.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("summary_label", "")),
                    str(row.get("model_name", "")),
                    f"{float(row.get('roc_auc', np.nan)):.3f}"
                    if pd.notna(row.get("roc_auc"))
                    else "NA",
                    str(row.get("roc_auc_ci", "NA")),
                    f"{float(row.get('average_precision', np.nan)):.3f}"
                    if pd.notna(row.get("average_precision"))
                    else "NA",
                    str(row.get("average_precision_ci", "NA")),
                    f"{float(row.get('brier_score', np.nan)):.3f}"
                    if pd.notna(row.get("brier_score"))
                    else "NA",
                    f"{float(row.get('brier_skill_score', np.nan)):.3f}"
                    if pd.notna(row.get("brier_skill_score"))
                    else "NA",
                    f"{float(row.get('ece', np.nan)):.3f}" if pd.notna(row.get("ece")) else "NA",
                    f"{float(row.get('max_calibration_error', np.nan)):.3f}"
                    if pd.notna(row.get("max_calibration_error"))
                    else "NA",
                    f"{float(row.get('calibration_slope', np.nan)):.3f}"
                    if pd.notna(row.get("calibration_slope"))
                    else "NA",
                    f"{float(row.get('calibration_intercept', np.nan)):.3f}"
                    if pd.notna(row.get("calibration_intercept"))
                    else "NA",
                    str(row.get("scientific_acceptance_status", "not_scored")),
                    str(row.get("scientific_acceptance_failed_criteria", "not_scored")),
                    _format_pvalue(row.get("selection_adjusted_empirical_p_roc_auc")),
                    _format_pvalue(row.get("permutation_p_roc_auc")),
                    (
                        f"{float(row.get('delta_vs_baseline_roc_auc', np.nan)):.3f} "
                        f"({row.get('delta_vs_baseline_ci', 'NA')})"
                    )
                    if pd.notna(row.get("delta_vs_baseline_roc_auc"))
                    else "NA",
                    f"{float(row.get('spatial_holdout_roc_auc', np.nan)):.3f}"
                    if pd.notna(row.get("spatial_holdout_roc_auc"))
                    else "NA",
                    str(int(row.get("n_backbones", 0))),
                    str(int(row.get("n_positive", 0))),
                ],
            )
            + " |",
        )
    single_model_row = summary_table.loc[
        summary_table.get("summary_label", pd.Series(dtype=str))
        .astype(str)
        .eq("single_model_pareto_official")
    ].head(1)
    if not single_model_row.empty:
        row = single_model_row.iloc[0]
        decision_reason = str(row.get("decision_reason", "") or "").strip() or "not_reported"
        selected_from_n_finalists = to_numeric_series(
            pd.Series([row.get("selected_from_n_finalists", np.nan)]),
            errors="coerce",
        ).iloc[0]
        lines.extend(
            [
                "",
                "## Single-Model Pareto Decision",
                "",
                (
                    f"- Official single-model candidate: `{str(row.get('model_name', '')).strip()}`; "
                    f"status `{str(row.get('scientific_acceptance_status', 'not_scored'))}`; "
                    f"reason `{decision_reason}`."
                ),
            ],
        )
        if pd.notna(selected_from_n_finalists) and int(selected_from_n_finalists) > 0:
            lines.append(
                f"- Selected from `{int(selected_from_n_finalists)}` Pareto finalists after finalist-heavy audit.",
            )
        weighted_objective = to_numeric_series(
            pd.Series([row.get("weighted_objective_score", np.nan)]),
        ).iloc[0]
        failure_severity = to_numeric_series(
            pd.Series([row.get("failure_severity", np.nan)]),
        ).iloc[0]
        screen_fit_seconds = to_numeric_series(
            pd.Series([row.get("screen_fit_seconds", np.nan)]),
        ).iloc[0]
        if pd.notna(weighted_objective) and pd.notna(failure_severity):
            lines.append(
                f"- Weighted objective `{float(weighted_objective):.3f}` with failure severity `{float(failure_severity):.3f}`.",
            )
        if pd.notna(screen_fit_seconds):
            lines.append(
                f"- Full Stage A screen time for the winning candidate family row was `{float(screen_fit_seconds):.2f}` seconds.",
            )
    rolling_summary = _rolling_temporal_summary(
        rolling_temporal if rolling_temporal is not None else pd.DataFrame(),
    )
    if rolling_summary:
        lines.extend(
            [
                "",
                "## Rolling-Origin Validation",
                "",
                (
                    f"Nested rolling-origin validation spans outer split years {rolling_summary['split_year_min']}"
                    f" to {rolling_summary['split_year_max']} across horizons {rolling_summary['horizon_values']}"
                    f" years and assignment modes {rolling_summary['assignment_modes']}."
                ),
                (
                    f"Across {rolling_summary['n_rows']} successful outer-split rows, ROC AUC mean "
                    f"{rolling_summary['roc_auc_mean']:.3f} (range {rolling_summary['roc_auc_min']:.3f}"
                    f" to {rolling_summary['roc_auc_max']:.3f}) and AP mean {rolling_summary['average_precision_mean']:.3f}"
                    f" (range {rolling_summary['average_precision_min']:.3f} to {rolling_summary['average_precision_max']:.3f})."
                ),
                (
                    f"Mean Brier score across the successful outer splits is {rolling_summary['brier_score_mean']:.3f}."
                ),
            ],
        )
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    if blocked_holdout_text:
        lines.extend(
            [
                "",
                "## Blocked Holdout Audit",
                "",
                f"- {blocked_holdout_text}",
            ],
        )
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    if country_missingness_text:
        lines.extend(
            [
                "",
                "## Country Missingness",
                "",
                f"- {country_missingness_text}",
            ],
        )
    if rank_stability_text or variant_consistency_text:
        lines.extend(
            [
                "",
                "## Ranking Stability",
                "",
            ],
        )
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _attach_single_model_decision_summary(
    model_selection_summary: pd.DataFrame,
    single_model_official_decision: pd.DataFrame | None,
    *,
    published_primary_model: str,
) -> pd.DataFrame:
    if model_selection_summary.empty or single_model_official_decision is None:
        return model_selection_summary.pipe(copy_frame)
    if single_model_official_decision.empty:
        return model_selection_summary.pipe(copy_frame)
    decision_row = single_model_official_decision.iloc[0]
    working = model_selection_summary.pipe(copy_frame)
    working.loc[:, "single_model_official_model"] = str(
        decision_row.get("official_model_name", "") or "",
    ).strip()
    working.loc[:, "single_model_official_decision_reason"] = str(
        decision_row.get("decision_reason", "") or "",
    ).strip()
    working.loc[:, "single_model_official_scientific_acceptance_status"] = str(
        decision_row.get("scientific_acceptance_status", "not_scored") or "not_scored",
    ).strip()
    working.loc[:, "single_model_official_failed_criteria"] = str(
        decision_row.get("scientific_acceptance_failed_criteria", "") or "",
    ).strip()
    working.loc[:, "single_model_official_failure_severity"] = float(
        decision_row.get("failure_severity", np.nan),
    )
    working.loc[:, "single_model_official_roc_auc"] = float(decision_row.get("roc_auc", np.nan))
    working.loc[:, "single_model_official_average_precision"] = float(
        decision_row.get("average_precision", np.nan),
    )
    working.loc[:, "single_model_official_weighted_objective_score"] = float(
        decision_row.get("weighted_objective_score", np.nan),
    )
    working.loc[:, "single_model_official_screen_fit_seconds"] = float(
        decision_row.get("screen_fit_seconds", np.nan),
    )
    working.loc[:, "single_model_official_compute_efficiency_score"] = float(
        decision_row.get("compute_efficiency_score", np.nan),
    )
    working.loc[:, "single_model_selected_from_n_finalists"] = int(
        to_numeric_series(
            pd.Series([decision_row.get("selected_from_n_finalists", np.nan)]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0],
    )
    working.loc[:, "single_model_official_matches_published_primary"] = bool(
        str(decision_row.get("official_model_name", "") or "").strip()
        == str(published_primary_model or "").strip(),
    )
    return working


def _write_executive_summary(
    output_path: Path,
    *,
    primary_model_name: str,
    governance_model_name: str,
    baseline_model_name: str,
    model_metrics: pd.DataFrame,
    confirmatory_cohort_summary: pd.DataFrame,
    false_negative_audit: pd.DataFrame,
    candidate_case_studies: pd.DataFrame,
    blocked_holdout_summary: pd.DataFrame | None = None,
    rolling_temporal: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
) -> None:
    metrics = model_metrics.set_index("model_name", drop=False)
    primary = metrics.loc[str(primary_model_name)]
    governance = (
        metrics.loc[str(governance_model_name)]
        if str(governance_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    baseline = (
        metrics.loc[str(baseline_model_name)]
        if str(baseline_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = variant_consistency if variant_consistency is not None else pd.DataFrame()
    false_negative_count, top_drivers = _summarize_false_negative_audit(false_negative_audit)
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    ).rstrip(".")
    primary_status = _strict_acceptance_status(primary)
    benchmark_scope_note = _benchmark_scope_note(primary_status)
    headline_label = (
        "accepted headline benchmark"
        if primary_status == "pass"
        else "conditional benchmark candidate"
    )
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    lines = [
        "# Executive Summary",
        "",
        "Plasmid Priority is a retrospective surveillance ranking framework for plasmid backbone classes. It does not claim causal spread prediction; it asks whether pre-2016 genomic signals are associated with post-2015 international visibility increase.",
        "",
        f"The Seer ({headline_label}): `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"The Guard (governance watch-only): `{_pretty_report_model_label(governance_model_name)}` | ROC AUC `{float(governance.get('roc_auc', np.nan)):.3f}` | AP `{float(governance.get('average_precision', np.nan)):.3f}`.",
        f"The Baseline: `{_pretty_report_model_label(baseline_model_name)}` | ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}` | AP `{float(baseline.get('average_precision', np.nan)):.3f}`.",
        "",
        f"Benchmark scope: {benchmark_scope_note}",
        "Calibration note: fixed-bin ECE, max calibration error, calibration slope, and calibration intercept are reported explicitly, rather than being treated as uninterpreted summary numbers.",
        "",
        "## Method Overview",
        "",
        "```text",
        "Raw Data (PLSDB + RefSeq + Pathogen Detection)",
        "                |",
        "                v",
        "      Harmonization and Deduplication",
        "                |",
        "                v",
        "      Backbone Assignment (MOB-suite style)",
        "                |",
        "                v",
        "       Temporal Split: <=2015 | >2015",
        "                |",
        "                v",
        "        T / H / A Feature Extraction",
        "                |",
        "                v",
        "   L2-Regularized Logistic Regression (OOF)",
        "          /                           \\",
        "         v                             v",
        " Discovery Track          Governance Watch-only Track",
        "                \\               /",
        "                 v             v",
        "            Candidate Portfolio + Risk Tiers",
        "```",
        "",
        "## Validation Posture",
        "",
        "- No external validation claim is made.",
        "- Validation is framed as temporal holdout, source holdout, knownness-matched auditing, and an internal high-integrity subset audit.",
        f"- False-negative audit: `{false_negative_count}` later positives remain outside the practical shortlist; dominant miss drivers are `{top_drivers}`.",
    ]
    rolling_summary = _rolling_temporal_summary(
        rolling_temporal if rolling_temporal is not None else pd.DataFrame(),
    )
    if rolling_summary:
        lines.append(
            f"- Rolling-origin validation: outer split years {rolling_summary['split_year_min']} to {rolling_summary['split_year_max']} across horizons {rolling_summary['horizon_values']} with assignment modes {rolling_summary['assignment_modes']}; ROC AUC mean {rolling_summary['roc_auc_mean']:.3f} (range {rolling_summary['roc_auc_min']:.3f} to {rolling_summary['roc_auc_max']:.3f}).",
        )
    if not confirmatory_primary.empty:
        lines.append(
            f"- Internal high-integrity subset audit: `{int(confirmatory_primary['n_backbones'])}` backbones | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`.",
        )
    if country_missingness_text:
        lines.extend(
            [
                "",
                "## Country Missingness",
                "",
                f"- {country_missingness_text}",
            ],
        )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    if rank_stability_text or variant_consistency_text:
        lines.extend(
            [
                "",
                "## Ranking Stability",
                "",
            ],
        )
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    lines.extend(
        [
            "",
            "## Release Surface",
            "",
            f"- `{int(len(candidate_case_studies))}` case studies are exported in `candidate_case_studies.tsv`.",
            "- Jury-facing narrative lives in `jury_brief.md` and `ozet_tr.md`.",
            "- `frozen_scientific_acceptance_audit.tsv` records the headline acceptance gate across matched-knownness, source holdout, spatial holdout, calibration, and leakage review.",
            f"- Blocked holdout audit is exported in `blocked_holdout_summary.tsv`{f': {blocked_holdout_text}' if blocked_holdout_text else ''}.",
            "- `nonlinear_deconfounding_audit.tsv` records the nonlinear deconfounding check used to keep knownness residualization transparent.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv` record the alternative-endpoint stress tests for ordinal, exposure-adjusted, and macro-region jump outcomes.",
            "- `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv` record the quasi-prospective freeze surface used to check whether the shortlist survives a forward-looking holdout.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv` record the leakage canary, graph audit, counterfactual shortlist comparison, geographic-jump diagnostic, and AMR-uncertainty summary.",
            "- Candidate rank stability is exported in `candidate_rank_stability.tsv` and model-variant consistency is exported in `candidate_variant_consistency.tsv`.",
            "- `calibration_threshold_summary.png` is exported as a compact calibration/threshold diagnostic when threshold-sensitivity data are available.",
            "- Figures in `reports/core_figures/` are presentation-ready.",
        ],
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return

    metrics = model_metrics.set_index("model_name", drop=False)
    primary = metrics.loc[str(primary_model_name)]
    governance = (
        metrics.loc[str(governance_model_name)]
        if str(governance_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    baseline = (
        metrics.loc[str(baseline_model_name)]
        if str(baseline_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, top_drivers = _summarize_false_negative_audit(false_negative_audit)
    lines = [
        "# Executive Summary",
        "",
        "Primary claim: the project is a retrospective plasmid-backbone surveillance ranking system, not a mechanistic spread oracle.",
        "",
        "Official models:",
        f"- The Seer (discovery): `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"- The Guard (governance): `{_pretty_report_model_label(governance_model_name)}` | ROC AUC `{float(governance.get('roc_auc', np.nan)):.3f}` | AP `{float(governance.get('average_precision', np.nan)):.3f}`.",
        f"- The Baseline: `{_pretty_report_model_label(baseline_model_name)}` | ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}` | AP `{float(baseline.get('average_precision', np.nan)):.3f}`.",
        "",
        "Validation posture:",
        "- No external validation claim is made. The confirmatory layer is an internal high-integrity cohort plus strict temporal, source, and knownness stress tests.",
    ]
    if not confirmatory_primary.empty:
        lines.append(
            f"- Internal high-integrity subset audit: `{int(confirmatory_primary['n_backbones'])}` backbones | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}` | share `{float(confirmatory_primary['share_of_primary_eligible']):.2%}` of primary-eligible backbones.",
        )
    lines.extend(
        [
            f"- False-negative audit: `{false_negative_count}` missed positives beyond the practical shortlist; dominant miss drivers: `{top_drivers}`.",
            "",
            "Case studies included in this release:",
            f"- `{int(len(candidate_case_studies))}` short biological case studies exported to `candidate_case_studies.tsv` for jury review.",
        ],
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pitch_notes(
    output_path: Path,
    *,
    primary_model_name: str,
    governance_model_name: str,
    model_metrics: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame,
    coefficient_stability_cv: pd.DataFrame,
    confirmatory_cohort_summary: pd.DataFrame,
    false_negative_audit: pd.DataFrame,
) -> None:
    metrics = model_metrics.set_index("model_name", drop=False)
    primary = (
        metrics.loc[str(primary_model_name)]
        if str(primary_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    baseline = (
        metrics.loc["baseline_both"]
        if "baseline_both" in metrics.index
        else pd.Series(dtype=object)
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, top_drivers = _summarize_false_negative_audit(false_negative_audit)
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    stable_features = _top_sign_stable_features(coefficient_stability_cv)
    lines = [
        "# Pitch Notes",
        "",
        "## Olası Jüri Soruları ve Yanıtları",
        "",
        '**S: "Model sadece zaten iyi bilinen büyük backbone\'ları mı buluyor?"**',
        (
            f"C: `baseline_both` ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}` üretirken "
            f"`{primary_model_name}` ROC AUC `{float(primary.get('roc_auc', np.nan)):.3f}` üretiyor. "
            f"Delta `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`. "
            f"Eşleştirilmiş knownness/source strata audit'inde de ana model "
            f"`{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}` vs baseline `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}`."
        )
        if not matched_primary.empty and not matched_baseline.empty
        else (
            f"C: `baseline_both` ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}`, "
            f"`{primary_model_name}` ROC AUC `{float(primary.get('roc_auc', np.nan)):.3f}`. "
            f"Delta `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`; bu, biyolojik sinyalin yalnızca popülerlik sayacı olmadığını gösterir."
        ),
        "",
        '**S: "Tüm modeller strict testi kaybediyorsa metodoloji geçerli mi?"**',
        "C: Evet. Strict matched-knownness/source-holdout testi en zor alt kohortu izole eder. Burada başarısız olmak metodolojinin çöktüğünü değil, mevcut veri yoğunluğunun bu alt dilimde sınırlı olduğunu gösterir. Bu kısıt raporda proaktif olarak açıkça belirtilir.",
        "",
        '**S: "29 özellik, 989 örnek; overfit değil mi?"**',
        f"C: Ana model L2 düzenlemeli lojistik regresyondur ve OOF tahminlerle değerlendirilir. Katsayı kararlılığı için 5-fold CV özeti ayrı verilir; en kararlı örnek sinyaller: `{stable_features}`.",
        "",
        '**S: "Bu gerçek bir tahmin sistemi mi, yoksa retrospektif analiz mi?"**',
        "C: Bu çalışma kasıtlı olarak retrospektiftir. Soru şudur: eğitim dönemindeki genomik sinyaller, sonraki dönemdeki coğrafi görünürlük artışıyla ilişkili mi? Prospektif klinik erken uyarı iddiası yapılmaz.",
        "",
        '**S: "Governance modeli neden discovery modelinden ayrı?"**',
        f"C: Discovery modeli `{_pretty_report_model_label(primary_model_name)}` ayırma gücünü optimize eder. Governance watch-only modeli `{_pretty_report_model_label(governance_model_name)}` ise kalibrasyon, belirsizlik ve abstention davranışını öne çıkarır; en yüksek AUC'u kovalamaz.",
    ]
    if not confirmatory_primary.empty:
        lines.extend(
            [
                "",
                "## Ek Not",
                "",
                f"- Confirmatory cohort sonucu: `{int(confirmatory_primary['n_backbones'])}` backbone üzerinde ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}`.",
                f"- False-negative audit: `{false_negative_count}` missed positive; baskın nedenler `{top_drivers}`.",
            ],
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return


def _write_turkish_summary(
    output_path: Path,
    *,
    primary_model_name: str,
    conservative_model_name: str,
    model_metrics: pd.DataFrame,
    candidate_briefs: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    decision_yield: pd.DataFrame,
    model_selection_scorecard: pd.DataFrame | None = None,
    model_selection_summary: pd.DataFrame,
    knownness_summary: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame | None = None,
    source_balance_resampling: pd.DataFrame,
    novelty_specialist_metrics: pd.DataFrame,
    adaptive_gated_metrics: pd.DataFrame,
    gate_consistency_audit: pd.DataFrame | None = None,
    secondary_outcome_performance: pd.DataFrame | None = None,
    weighted_country_outcome: pd.DataFrame | None = None,
    count_outcome_audit: pd.DataFrame | None = None,
    metadata_quality_summary: pd.DataFrame | None = None,
    operational_risk_watchlist: pd.DataFrame | None = None,
    confirmatory_cohort_summary: pd.DataFrame | None = None,
    false_negative_audit: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
    outcome_threshold: int,
) -> None:
    knownness_matched_validation = (
        knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    )
    weighted_country_outcome = (
        weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    )
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    operational_risk_watchlist = (
        operational_risk_watchlist if operational_risk_watchlist is not None else pd.DataFrame()
    )
    confirmatory_cohort_summary = (
        confirmatory_cohort_summary if confirmatory_cohort_summary is not None else pd.DataFrame()
    )
    false_negative_audit = (
        false_negative_audit if false_negative_audit is not None else pd.DataFrame()
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = variant_consistency if variant_consistency is not None else pd.DataFrame()
    primary = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(primary_model_name)
    ].iloc[0]
    conservative = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(conservative_model_name)
    ].iloc[0]
    baseline = model_metrics.loc[model_metrics["model_name"].astype(str) == "baseline_both"].iloc[0]
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )
    governance_model_name = str(selection_row.get("governance_primary_model", "") or "").strip()
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, false_negative_drivers = _summarize_false_negative_audit(
        false_negative_audit,
    )
    blocked_holdout_text = _blocked_holdout_summary_text_tr(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    weighted_row = (
        weighted_country_outcome.loc[
            weighted_country_outcome.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in weighted_country_outcome.columns
        else pd.DataFrame()
    )
    count_row = (
        count_outcome_audit.loc[
            count_outcome_audit.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in count_outcome_audit.columns
        else pd.DataFrame()
    )
    selected_briefs = _select_summary_candidate_briefs(candidate_briefs, per_track=2)
    lines = [
        "# Proje Özeti",
        "",
        "## Temel Mesaj",
        "",
        "Plasmid Priority, plasmid omurga sınıflarını eğitim dönemindeki genomik sinyallerle puanlayan ve bu puanların daha sonraki coğrafi görünürlük artışıyla ilişkili olup olmadığını test eden retrospektif bir analiz hattıdır.",
        "",
        f"Ana model `{primary_model_name}` için ROC AUC `{float(primary['roc_auc']):.3f}`, AP `{float(primary['average_precision']):.3f}` ve Brier Skill Score `{float(primary.get('brier_skill_score', np.nan)):.3f}` olarak raporlanır. Mevcut permütasyon denetimi sabit-skor label-permutation audit'i olduğu için model-seçim-düzeltilmiş anlamlılık iddiası olarak değil, keşifsel sinyal kontrolü olarak okunmalıdır. Sayım temelli karşılaştırma modeli `{float(baseline['roc_auc']):.3f}` ROC AUC üretir; ana modelin bu taban modele karşı kazancı `{_primary_baseline_delta_text(model_metrics, primary_model_name)}` düzeyindedir.",
        "",
        "## Model Seçimi",
        "",
        "Model seçimi tek bir metriğe göre yapılmamıştır. Birlikte okunan ölçütler şunlardır:",
        "",
        "1. Genel ROC AUC ve AP.",
        "2. Düşük bilinirlik yarısındaki performans.",
        "3. Eşleştirilmiş bilinirlik/kaynak katmanlarındaki performans.",
        "4. Kaynak dışlama denetimi ve diğer sağlamlık analizleri.",
        "5. Pratik kısa liste verimi.",
        "",
        f"Bu nedenle discovery hattında `{primary_model_name}` korunur; governance watch-only hattında ise `{governance_model_name or 'ayrı governance watch-only modeli'}` daha temkinli yorum katmanı olarak ele alınır.",
        "",
        "## Metodoloji",
        "",
        "- Ham veri PLSDB, RefSeq ve Pathogen Detection metadata kaynaklarından gelir.",
        "- Kayıtlar harmonize edilir, yinelenenler ayıklanır ve omurga sınıfı ataması yapılır.",
        "- Zaman ayrımı `<=2015` eğitim ve `>2015` sonuç penceresi olacak şekilde kuruludur.",
        "- T, H ve A eksenleri yalnızca eğitim döneminden hesaplanır.",
        f"- Yayılım etiketi, test döneminde en az `{int(outcome_threshold)}` yeni ülke görülmesiyle tanımlanır.",
        "- Değerlendirme OOF lojistik regresyon tahminleri üzerinden yapılır.",
        "",
        "## Türkiye Bağlamı",
        "",
        "WHO'nun 2025 GLASS özeti, antibiyotik direnci yükünün özellikle Güneydoğu Asya ve Doğu Akdeniz bölgelerinde yüksek olduğunu vurgular. Türkiye, AMR sürveyansı açısından bu bölgesel baskının doğrudan önemli olduğu bir ülkedir.",
        "",
        "ECDC ve WHO Europe çerçevelerinde karbapenem dirençli *Klebsiella pneumoniae* ile GSBL/ESBL üreten *Escherichia coli*, Enterobacterales kaynaklı hastane yükünün temel başlıkları arasında yer almaktadır. Bu nedenle bu çalışmanın ürettiği omurga sınıfı önceliklendirmesi, Türkiye'de genomik AMR sürveyansına doğrudan uyarlanabilecek bir kanıt-konsept çerçevesi sunar.",
        "",
        "Bu proje klinik karar desteği vermez; ancak Türkiye'de ulusal veya kurumsal genomik sürveyans akışlarında hangi omurga sınıflarının önce incelenmesi gerektiğini sistematikleştirebilir.",
        "",
        "## Ana Bulgular",
        "",
        f"- Ana model: ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"- Koruyucu model: ROC AUC `{float(conservative['roc_auc']):.3f}` | AP `{float(conservative['average_precision']):.3f}`.",
        f"- Baseline model: ROC AUC `{float(baseline['roc_auc']):.3f}` | AP `{float(baseline['average_precision']):.3f}`.",
        f"- Yanlış negatif incelemesi: kısa liste dışında kalan `{false_negative_count}` pozitif vardır; baskın nedenler `{false_negative_drivers}`.",
    ]
    if not confirmatory_primary.empty:
        lines.append(
            f"- İç yüksek-bütünlük alt-küme denetimi: `{int(confirmatory_primary['n_backbones'])}` omurga sınıfı | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`.",
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Eşleştirilmiş bilinirlik/kaynak katmanları denetimi: ana model `{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}`, taban model `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}`.",
        )
    if not weighted_row.empty:
        lines.append(
            f"- Ağırlıklı yeni ülke yükü ile ilişki: Spearman ρ `{float(weighted_row.iloc[0]['spearman_corr']):.3f}`.",
        )
    if not count_row.empty:
        lines.append(
            f"- Ham yeni ülke sayısı ile ilişki: Spearman ρ `{float(count_row.iloc[0]['spearman_corr']):.3f}` {_format_interval(count_row.iloc[0].get('spearman_ci_lower'), count_row.iloc[0].get('spearman_ci_upper'))}.",
        )
    spatial_auc = to_numeric_series(
        pd.Series([primary.get("spatial_holdout_roc_auc")]),
    ).iloc[0]
    if pd.notna(spatial_auc):
        lines.append(f"- Mekânsal holdout denetimi: ağırlıklı ROC AUC `{float(spatial_auc):.3f}`.")
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="tr",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="tr",
    )
    if rank_stability_text or variant_consistency_text:
        lines.extend(
            [
                "",
                "## Sıralama Kararlılığı",
                "",
            ],
        )
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    lines.extend(
        [
            "",
            "## Sınırlılıklar",
            "",
            "- Bu çalışma retrospektiftir; prospektif erken uyarı sistemi iddiası taşımaz.",
            "- Sonuç değişkeni doğrudan biyolojik yayılım değil, sonraki dönem ülke görünürlüğü artışıdır.",
            "- En sıkı bilinirlik eşleştirmeli kaynak dışlama testi en zor alt kohorttur; bu katmanda tüm modeller temkinli yorumlanmalıdır.",
            "- Fırsat yanlılığı tamamen giderilemez: daha erken görülen omurgaların daha uzun takip penceresi vardır.",
            "- `risk_uncertainty` bir güven aralığı değildir; risk-bileşeni uyumsuzluğu, karar sınırına yakınlık ve düşük bilinirlik cezasından türetilen operasyonel bir belirsizlik skorudur.",
            "",
            "## Örnek Adaylar",
            "",
        ],
    )
    track_labels = {
        "established_high_risk": "yerleşik yüksek risk kısa listesi",
        "novel_signal": "erken-sinyal izleme hattı",
    }
    support_labels = {
        "cross_source_supported": "çok kaynaklı destek",
        "single_source_supported": "tek kaynaklı destek",
        "limited_support": "sınırlı destek",
    }
    risk_labels = {
        "action": "eylem",
        "review": "inceleme",
        "abstain": "çekimser",
    }
    for row in selected_briefs.itertuples(index=False):
        backbone_id = str(getattr(row, "backbone_id", "NA"))
        species = str(getattr(row, "dominant_species", "") or "").replace("_", " ").strip()
        if not species or species.lower() == "nan":
            species = (
                str(getattr(row, "dominant_genus", "") or "").replace("_", " ").strip()
                or "belirtilmemiş baskın takson"
            )
        replicon = str(getattr(row, "primary_replicon", "") or "").strip()
        replicon_text = (
            f", baskın replikon `{replicon}`" if replicon and replicon.lower() != "nan" else ""
        )
        track = track_labels.get(str(getattr(row, "portfolio_track", "")), "izlem listesi")
        support_tier = support_labels.get(
            str(getattr(row, "source_support_tier", "")),
            "destek düzeyi belirtilmemiş",
        )
        amr_classes = str(getattr(row, "top_amr_classes", "") or "").strip()
        amr_text = (
            amr_classes
            if amr_classes and amr_classes.lower() != "nan"
            else "belirgin AMR sınıfı sinyali yok"
        )
        risk_tier = risk_labels.get(str(getattr(row, "risk_decision_tier", "")), "belirsiz")
        operational_risk = to_numeric_series(
            pd.Series([getattr(row, "operational_risk_score", np.nan)]),
        ).iloc[0]
        uncertainty = to_numeric_series(
            pd.Series([getattr(row, "risk_uncertainty", np.nan)]),
        ).iloc[0]
        risk_parts = []
        if pd.notna(operational_risk):
            risk_parts.append(f"genel risk `{float(operational_risk):.2f}`")
        if pd.notna(uncertainty):
            risk_parts.append(f"belirsizlik `{float(uncertainty):.2f}`")
        risk_text = ", ".join(risk_parts) if risk_parts else "risk özeti mevcut değil"
        lines.append(
            f"- `{backbone_id}`: baskın tür `{species}`{replicon_text}; bu aday `{track}` içinde değerlendirilir. "
            f"Kaynak desteği `{support_tier}`, operasyonel karar katmanı `{risk_tier}` ve {risk_text}. "
            f"Öne çıkan AMR sınıfları: {amr_text}.",
        )
    lines.extend(
        [
            "",
            "## Sürüm Yüzeyi",
            "",
            "- `frozen_scientific_acceptance_audit.tsv`: doğrulayıcı kabul katmanını; eşleştirilmiş bilinirlik, kaynak dışlama, mekânsal holdout, kalibrasyon ve leakage incelemesini raporlar.",
            f"- `blocked_holdout_summary.tsv`: {blocked_holdout_text or 'iç kaynak/bölge stres testi olarak raporlanır.'}",
            "- `nonlinear_deconfounding_audit.tsv`: knownness residualization için kullanılan doğrusal olmayan karıştırma denetimini raporlar.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv` ve `macro_region_jump_outcome.tsv`: sırasal, maruziyet-düzeltilmiş ve makro-bölge sıçrama sonuçları için alternatif sonuç stres testlerini raporlar.",
            "- `prospective_candidate_freeze.tsv` ve `annual_candidate_freeze_summary.tsv`: kısa listenin ileriye dönük bir holdout üzerinde ayakta kalıp kalmadığını kontrol eden quasi-prospective freeze yüzeyini raporlar.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv` ve `amr_uncertainty_summary.tsv`: leakage canary, graph denetimi, counterfactual shortlist comparison, geographic-jump tanısı ve AMR belirsizlik özetini raporlar.",
            "- `country_missingness_bounds.tsv` ve `country_missingness_sensitivity.tsv`: ülke eksikliği varsayımlarına göre etiket ve performans duyarlılığını raporlar.",
            "- `candidate_rank_stability.tsv` ve `candidate_variant_consistency.tsv`: aday sıralama kararlılığını ve model-varyant tutarlılığını raporlar.",
            "- `calibration_threshold_summary.png`: kalibrasyon ve eşik duyarlılığı için kompakt tanı grafiğidir.",
            "- `jury_brief.md` ve `ozet_tr.md`: jüriye dönük anlatının dağıtım yüzeyleri.",
        ],
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return


def _is_amrfinder_reportable(coverage: pd.DataFrame) -> bool:
    if coverage.empty or "priority_group" not in coverage.columns:
        return False
    groups = coverage.loc[coverage["priority_group"].isin(["high", "low"])].pipe(copy_frame)
    if len(groups) < 2:
        return False
    if int(groups["n_sequences"].sum()) < 30:
        return False
    evaluable = groups["nonempty_concordance_evaluable_fraction"].pipe(fill0)
    return bool(
        (groups["n_sequences"].pipe(int0) >= 10).all() and (evaluable >= 0.5).all(),
    )


def _write_jury_brief(
    output_path: Path,
    *,
    primary_model_name: str,
    conservative_model_name: str,
    model_metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    dropout_table: pd.DataFrame,
    scored: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    decision_yield: pd.DataFrame,
    model_selection_scorecard: pd.DataFrame | None = None,
    model_selection_summary: pd.DataFrame,
    knownness_summary: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame | None = None,
    source_balance_resampling: pd.DataFrame,
    novelty_specialist_metrics: pd.DataFrame,
    adaptive_gated_metrics: pd.DataFrame,
    gate_consistency_audit: pd.DataFrame | None = None,
    secondary_outcome_performance: pd.DataFrame | None = None,
    weighted_country_outcome: pd.DataFrame | None = None,
    count_outcome_audit: pd.DataFrame | None = None,
    metadata_quality_summary: pd.DataFrame | None = None,
    operational_risk_watchlist: pd.DataFrame | None = None,
    confirmatory_cohort_summary: pd.DataFrame | None = None,
    false_negative_audit: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
    outcome_threshold: int,
) -> None:
    knownness_matched_validation = (
        knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    )
    weighted_country_outcome = (
        weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    )
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    operational_risk_watchlist = (
        operational_risk_watchlist if operational_risk_watchlist is not None else pd.DataFrame()
    )
    confirmatory_cohort_summary = (
        confirmatory_cohort_summary if confirmatory_cohort_summary is not None else pd.DataFrame()
    )
    false_negative_audit = (
        false_negative_audit if false_negative_audit is not None else pd.DataFrame()
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = variant_consistency if variant_consistency is not None else pd.DataFrame()
    primary = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(primary_model_name)
    ].iloc[0]
    conservative = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(conservative_model_name)
    ].iloc[0]
    baseline = model_metrics.loc[model_metrics["model_name"].astype(str) == "baseline_both"].iloc[0]
    source = model_metrics.loc[model_metrics["model_name"].astype(str) == "source_only"].head(1)
    source = source.iloc[0] if not source.empty else pd.Series(dtype=object)
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )
    governance_model_name = str(selection_row.get("governance_primary_model", "") or "").strip()
    governance_scorecard_row = (
        model_selection_scorecard.loc[
            model_selection_scorecard.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(governance_model_name)
        ].head(1)
        if governance_model_name
        and model_selection_scorecard is not None
        and not model_selection_scorecard.empty
        else pd.DataFrame()
    )
    primary_scorecard_row = (
        model_selection_scorecard.loc[
            model_selection_scorecard.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(primary_model_name)
        ].head(1)
        if model_selection_scorecard is not None and not model_selection_scorecard.empty
        else pd.DataFrame()
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, false_negative_drivers = _summarize_false_negative_audit(
        false_negative_audit,
    )
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    weighted_row = (
        weighted_country_outcome.loc[
            weighted_country_outcome.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in weighted_country_outcome.columns
        else pd.DataFrame()
    )
    count_row = (
        count_outcome_audit.loc[
            count_outcome_audit.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in count_outcome_audit.columns
        else pd.DataFrame()
    )
    operational_action_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("action")
            .sum(),
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    operational_review_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("review")
            .sum(),
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    operational_abstain_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("abstain")
            .sum(),
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    strongest_overall = model_metrics.sort_values(
        ["roc_auc", "average_precision"],
        ascending=False,
    ).iloc[0]
    primary_vs_strongest_overlap_25 = selection_row.get(
        "primary_vs_strongest_top_25_overlap_count",
        np.nan,
    )
    primary_vs_strongest_overlap_50 = selection_row.get(
        "primary_vs_strongest_top_50_overlap_count",
        np.nan,
    )
    primary_selection_text = "Model selection was not driven by a single metric. We jointly considered ROC AUC, average precision, lower-knownness behavior, matched-knownness/source performance, source holdout robustness, and practical shortlist yield."
    if not primary_scorecard_row.empty:
        scorecard_total_models = (
            len(model_selection_scorecard) if model_selection_scorecard is not None else 0
        )
        primary_selection_text += (
            " In the current scorecard, the headline model ranks "
            f"{_format_scorecard_rank_text(primary_scorecard_row.iloc[0], total_models=scorecard_total_models, track_label_hint='discovery')}."
        )
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    lines = [
        "# Jury Brief",
        "",
        "## Core Claim",
        "",
        "This framework retrospectively prioritizes plasmid backbone surveillance units using pre-2016 genomic and ecological features, then tests whether those same backbone classes later show multi-country visibility increase.",
        "",
        "## Formal Hypotheses",
        "",
        "- **H0 (null)**: A <=2015 T/H/A-derived priority signal has no discriminative association with post-2015 multi-country visibility expansion.",
        "- **H1 (alternative)**: The same priority signal is positively associated with post-2015 multi-country visibility expansion.",
        "- **Significance criterion**: empirical permutation p-value below the predeclared threshold for the current headline model.",
        "",
        "## Current Benchmark",
        "",
        f"- Discovery benchmark: `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"- Counts-only baseline: `baseline_both` | ROC AUC `{float(baseline['roc_auc']):.3f}` | AP `{float(baseline['average_precision']):.3f}`.",
        f"- Conservative benchmark: `{_pretty_report_model_label(conservative_model_name)}` | ROC AUC `{float(conservative['roc_auc']):.3f}` | AP `{float(conservative['average_precision']):.3f}`.",
        f"- Source-only control: `source_only` | ROC AUC `{float(source.get('roc_auc', np.nan)):.3f}`.",
        f"- Strongest audited metric model: `{_pretty_report_model_label(str(strongest_overall['model_name']))}` | ROC AUC `{float(strongest_overall['roc_auc']):.3f}` | AP `{float(strongest_overall['average_precision']):.3f}`.",
        (
            f"- Governance watch-only: `{_pretty_report_model_label(governance_model_name)}` | ROC AUC `{float(governance_scorecard_row.iloc[0].get('roc_auc', np.nan)):.3f}` | AP `{float(governance_scorecard_row.iloc[0].get('average_precision', np.nan)):.3f}` | strict `{'pass' if bool(governance_scorecard_row.iloc[0].get('strict_knownness_acceptance_flag', False)) else 'fail'}`."
            if not governance_scorecard_row.empty
            else "- Governance watch-only: no distinct governance candidate could be resolved from the current scorecard."
        ),
        f"- Selection-adjusted official-model permutation audit for the headline ROC AUC: `p {_format_pvalue(primary.get('selection_adjusted_empirical_p_roc_auc'))}`; the older fixed-score label-permutation entry is retained only as an exploratory appendix diagnostic.",
        f"- Delta vs counts-only baseline: `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`.",
        "",
        "## Primary Model Selection Rationale",
        "",
        primary_selection_text,
        "",
        "Operationally, the headline model is preferred because it keeps the strongest balance between discrimination and shortlist usefulness. In this refresh, the primary model also preserves a top-10 precision of 1.0 while remaining clearly above the counts-only baseline in matched-knownness auditing.",
        "Governance track logic is kept separate from discovery-track optimization even when the shortlisted candidates partially overlap.",
        "",
        "## Strict Test Interpretation",
        "",
        "The strict matched-knownness/source-holdout test isolates the hardest low-knownness and low-support slice of the dataset. No current model fully passes this acceptance rule.",
        "",
        "This should be interpreted as a data-limited regime, not as evidence that the entire methodology collapses. The primary dataset contains 989 eligible backbone classes, but the strict slice is materially smaller and therefore noisier. We report this limitation explicitly instead of hiding it behind the stronger overall metrics.",
        "",
        "## Decision Readout",
        "",
        f"- Outcome definition: later visibility in at least `{int(outcome_threshold)}` new countries.",
        f"- Operational watchlist mix: `{operational_action_count}` action + `{operational_review_count}` review + `{operational_abstain_count}` abstain rows.",
        f"- False-negative audit: `{false_negative_count}` later positives remain outside the practical shortlist; dominant miss drivers are `{false_negative_drivers}`.",
        "- `operational_risk_watchlist.tsv` is the calibrated deployment-facing table for the current shortlist.",
        f"- Current operational watchlist mix: `{operational_action_count}` action + `{operational_review_count}` review + `{operational_abstain_count}` abstain rows.",
        "- This remains a shortlist-prioritization benchmark rather than an exhaustive detector for every later positive backbone.",
    ]
    if not confirmatory_primary.empty:
        lines.append(
            f"- Internal high-integrity subset audit: `{int(confirmatory_primary['n_backbones'])}` higher-integrity backbones | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`.",
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Matched knownness/source strata: primary `{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}` vs baseline `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}` weighted ROC AUC.",
        )
    if not count_row.empty:
        lines.append(
            f"- Raw later new-country count alignment: Spearman ρ `{float(count_row.iloc[0]['spearman_corr']):.3f}` {_format_interval(count_row.iloc[0].get('spearman_ci_lower'), count_row.iloc[0].get('spearman_ci_upper'))}.",
        )
    if not weighted_row.empty:
        lines.append(
            f"- Weighted new-country burden alignment: Spearman ρ `{float(weighted_row.iloc[0]['spearman_corr']):.3f}`.",
        )
    spatial_auc = to_numeric_series(
        pd.Series([primary.get("spatial_holdout_roc_auc")]),
    ).iloc[0]
    if pd.notna(spatial_auc):
        spatial_regions = to_numeric_series(
            pd.Series([primary.get("spatial_holdout_regions")]),
        ).iloc[0]
        worst_region = str(primary.get("worst_spatial_holdout_region", "") or "").strip()
        worst_region_auc = to_numeric_series(
            pd.Series([primary.get("worst_spatial_holdout_region_roc_auc")]),
        ).iloc[0]
        region_text = (
            f" across `{int(spatial_regions)}` held-out dominant regions"
            if pd.notna(spatial_regions)
            else ""
        )
        if worst_region and pd.notna(worst_region_auc):
            region_text += (
                f"; hardest region `{worst_region}` at ROC AUC `{float(worst_region_auc):.3f}`"
            )
        lines.append(
            f"- Spatial holdout audit: weighted ROC AUC `{float(spatial_auc):.3f}`{region_text}.",
        )
    if blocked_holdout_text:
        lines.extend(
            [
                "",
                "## Blocked Holdout Audit",
                "",
                f"- {blocked_holdout_text} This is an internal source/region stress test, not external validation.",
            ],
        )
    if country_missingness_text:
        lines.extend(
            [
                "",
                "## Country Missingness",
                "",
                f"- {country_missingness_text}",
            ],
        )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    if rank_stability_text or variant_consistency_text:
        lines.extend(
            [
                "",
                "## Ranking Stability",
                "",
            ],
        )
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    if pd.notna(primary_vs_strongest_overlap_25) or pd.notna(primary_vs_strongest_overlap_50):
        overlap_bits: list[str] = []
        if pd.notna(primary_vs_strongest_overlap_25):
            overlap_bits.append(f"top-25 overlap: `{int(primary_vs_strongest_overlap_25)}/25`")
        if pd.notna(primary_vs_strongest_overlap_50):
            overlap_bits.append(f"top-50 overlap: `{int(primary_vs_strongest_overlap_50)}/50`")
        lines.append(
            f"- Discovery shortlist agreement with the strongest audited metric model: {'; '.join(overlap_bits)}.",
        )
    lines.extend(
        [
            "- A knownness-gated audit model (`adaptive_natural_priority`) remains useful for lower-knownness stress testing but is not the headline benchmark.",
            "- Observed host-diversity terms should be interpreted cautiously because they partly behave like sampling saturation / knownness signals.",
            "- Supportive external layers are descriptive context only; AMRFinder is optional and not required for the headline benchmark.",
            "- Only three models are official in the jury-facing narrative: discovery, governance watch-only, and baseline.",
            "",
            "## Zero-Floor Component Behavior",
            "",
            "When a backbone lacks direct evidence for a component, the normalized contribution is explicitly allowed to stay at zero rather than being imputed upward by unrelated metadata support.",
            "",
            "## OLS Residual Approach",
            "",
            "Knownness-sensitive H-support terms are residualized against knownness proxies so that the retained signal is not a disguised count effect.",
        ],
    )
    lines.extend(
        [
            "",
            "## Turkey Context",
            "",
            "WHO's 2025 GLASS summary highlights a high antibiotic-resistance burden in the Eastern Mediterranean region. For Türkiye, this makes Enterobacterales-focused genomic surveillance directly relevant.",
            "",
            "Within the ECDC/WHO Europe surveillance framing, carbapenem-resistant *Klebsiella pneumoniae* and ESBL-producing *Escherichia coli* remain core public-health concerns. A backbone-level prioritization system is therefore operationally meaningful for Turkish genomic AMR surveillance, even though this project does not claim clinical decision support.",
            "",
            "## Interpretation Guardrails",
            "",
            "- No external validation claim is made.",
            "- T, H and A features are computed only from `resolved_year <= 2015` rows.",
            "- The outcome is later country visibility increase, not direct biological fitness or transmission proof.",
            "- Opportunity bias is a declared limitation: backbones seen earlier have longer time-at-risk.",
            "",
            "## Release Surface",
            "",
            "- `frozen_scientific_acceptance_audit.tsv` records the headline acceptance gate across matched-knownness, source holdout, spatial holdout, calibration, and leakage review.",
            "- `blocked_holdout_summary.tsv` records the blocked source/region stress test used for the internal audit layer.",
            "- `nonlinear_deconfounding_audit.tsv` records the nonlinear deconfounding check used to keep knownness residualization transparent.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv` record the alternative-endpoint stress tests for ordinal, exposure-adjusted, and macro-region jump outcomes.",
            "- `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv` record the quasi-prospective freeze surface used to check whether the shortlist survives a forward-looking holdout.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv` record the leakage canary, graph audit, counterfactual shortlist comparison, geographic-jump diagnostic, and AMR-uncertainty summary.",
            "- `candidate_rank_stability.tsv` and `candidate_variant_consistency.tsv` record backbone-level ranking stability across bootstrap and model-variant audits.",
            "- `calibration_threshold_summary.png` captures the compact calibration/threshold view used in slide decks.",
            "- `reports/core_figures/` contains the rest of the presentation-ready figure pack.",
        ],
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return


def _export_support_appendix_tables(
    *,
    run: ManagedScriptRun,
    final_tables_dir: object,
    module_b_path: Path,
    module_c_path: Path,
    module_c_group_path: Path,
    module_c_clinical_path: Path,
    module_c_clinical_group_path: Path,
    module_c_environmental_path: Path,
    module_c_environmental_group_path: Path,
    module_c_strata_group_path: Path,
    who_detail_path: Path,
    who_summary_path: Path,
    who_category_path: Path,
    who_reference_path: Path,
    card_detail_path: Path,
    card_summary_path: Path,
    card_family_path: Path,
    card_mechanism_path: Path,
    mobsuite_detail_path: Path,
    mobsuite_summary_path: Path,
    amrfinder_probe_panel_path: Path,
    amrfinder_probe_hits_path: Path,
    amrfinder_detail_path: Path,
    amrfinder_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    if module_b_path.exists():
        _read_if_exists(module_b_path).to_csv(
            final_tables_dir / "module_b_amr_class_comparison.tsv",
            sep="\t",
            index=False,
        )

    module_c = _read_if_exists(module_c_path)
    module_c_group = _read_if_exists(module_c_group_path)
    module_c_clinical = _read_if_exists(module_c_clinical_path)
    module_c_clinical_group = _read_if_exists(module_c_clinical_group_path)
    module_c_environmental = _read_if_exists(module_c_environmental_path)
    module_c_environmental_group = _read_if_exists(module_c_environmental_group_path)
    module_c_strata_group = _read_if_exists(module_c_strata_group_path)
    if not module_c.empty:
        module_c.to_csv(final_tables_dir / "pathogen_detection_support.tsv", sep="\t", index=False)
        run.record_output(final_tables_dir / "pathogen_detection_support.tsv")
    if not module_c_group.empty:
        module_c_group.to_csv(
            final_tables_dir / "pathogen_detection_group_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_group_summary.tsv")
    if not module_c_clinical.empty:
        module_c_clinical.to_csv(
            final_tables_dir / "pathogen_detection_clinical_support.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_clinical_support.tsv")
    if not module_c_clinical_group.empty:
        module_c_clinical_group.to_csv(
            final_tables_dir / "pathogen_detection_clinical_group_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_clinical_group_summary.tsv")
    if not module_c_environmental.empty:
        module_c_environmental.to_csv(
            final_tables_dir / "pathogen_detection_environmental_support.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_environmental_support.tsv")
    if not module_c_environmental_group.empty:
        module_c_environmental_group.to_csv(
            final_tables_dir / "pathogen_detection_environmental_group_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_environmental_group_summary.tsv")
    if not module_c_strata_group.empty:
        module_c_strata_group.to_csv(
            final_tables_dir / "pathogen_detection_strata_group_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "pathogen_detection_strata_group_summary.tsv")

    who_detail = _read_if_exists(who_detail_path)
    who_summary = _read_if_exists(who_summary_path)
    who_category = _read_if_exists(who_category_path)
    who_reference = _read_if_exists(who_reference_path)
    card_detail = _read_if_exists(card_detail_path)
    card_summary = _read_if_exists(card_summary_path)
    card_family = _read_if_exists(card_family_path)
    card_mechanism = _read_if_exists(card_mechanism_path)
    mobsuite_detail = _read_if_exists(mobsuite_detail_path)
    mobsuite_summary = _read_if_exists(mobsuite_summary_path)
    amrfinder_probe_panel = _read_if_exists(amrfinder_probe_panel_path)
    amrfinder_probe_hits = _read_if_exists(amrfinder_probe_hits_path)
    amrfinder_detail = _read_if_exists(amrfinder_detail_path)
    amrfinder_summary = _read_if_exists(amrfinder_summary_path)
    amrfinder_coverage = build_amrfinder_coverage_table(amrfinder_summary)
    amrfinder_reportable = _is_amrfinder_reportable(amrfinder_coverage)
    amrfinder_sequences_total = (
        int(amrfinder_coverage["n_sequences"].sum()) if not amrfinder_coverage.empty else 0
    )
    if amrfinder_sequences_total == 0:
        amrfinder_reason = "AMRFinder executable/report unavailable; appendix-only probe skipped"
    elif not amrfinder_reportable:
        amrfinder_reason = "probe panel too small or imbalanced; keep as appendix-only sanity check"
    else:
        amrfinder_reason = "coverage acceptable for descriptive appendix reporting"
    amrfinder_summary_table = pd.DataFrame(
        [
            {
                "reportable_in_main_report": bool(amrfinder_reportable),
                "reason": amrfinder_reason,
                "n_sequences_total": amrfinder_sequences_total,
                "high_group_sequences": int(
                    amrfinder_coverage.loc[
                        amrfinder_coverage["priority_group"] == "high",
                        "n_sequences",
                    ].sum(),
                )
                if not amrfinder_coverage.empty
                else 0,
                "low_group_sequences": int(
                    amrfinder_coverage.loc[
                        amrfinder_coverage["priority_group"] == "low",
                        "n_sequences",
                    ].sum(),
                )
                if not amrfinder_coverage.empty
                else 0,
            },
        ],
    )
    amrfinder_summary_table.to_csv(
        final_tables_dir / "amrfinder_coverage_summary.tsv",
        sep="\t",
        index=False,
    )
    run.record_output(final_tables_dir / "amrfinder_coverage_summary.tsv")
    if not who_detail.empty:
        who_detail.to_csv(final_tables_dir / "who_mia_support.tsv", sep="\t", index=False)
        run.record_output(final_tables_dir / "who_mia_support.tsv")
    if not who_summary.empty:
        who_summary.to_csv(final_tables_dir / "who_mia_group_summary.tsv", sep="\t", index=False)
        run.record_output(final_tables_dir / "who_mia_group_summary.tsv")
    if not who_category.empty:
        who_category.to_csv(
            final_tables_dir / "who_mia_category_comparison.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "who_mia_category_comparison.tsv")
    if not who_reference.empty:
        who_reference.to_csv(
            final_tables_dir / "who_mia_reference_catalog.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "who_mia_reference_catalog.tsv")
    if not card_detail.empty:
        card_detail.to_csv(final_tables_dir / "card_gene_support.tsv", sep="\t", index=False)
        run.record_output(final_tables_dir / "card_gene_support.tsv")
    if not card_summary.empty:
        card_summary.to_csv(final_tables_dir / "card_group_summary.tsv", sep="\t", index=False)
        run.record_output(final_tables_dir / "card_group_summary.tsv")
    if not card_family.empty:
        card_family.to_csv(
            final_tables_dir / "card_gene_family_comparison.tsv",
            sep="\t",
            index=False,
        )
    run.record_output(final_tables_dir / "card_gene_family_comparison.tsv")
    if not card_mechanism.empty:
        card_mechanism.to_csv(
            final_tables_dir / "card_mechanism_comparison.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "card_mechanism_comparison.tsv")
    if not mobsuite_detail.empty:
        mobsuite_detail.to_csv(
            final_tables_dir / "mobsuite_host_range_support.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "mobsuite_host_range_support.tsv")
    if not mobsuite_summary.empty:
        mobsuite_summary.to_csv(
            final_tables_dir / "mobsuite_host_range_group_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "mobsuite_host_range_group_summary.tsv")
    if amrfinder_reportable:
        if not amrfinder_probe_panel.empty:
            amrfinder_probe_panel.to_csv(
                final_tables_dir / "amrfinder_probe_panel.tsv",
                sep="\t",
                index=False,
            )
            run.record_output(final_tables_dir / "amrfinder_probe_panel.tsv")
        if not amrfinder_probe_hits.empty:
            amrfinder_probe_hits.to_csv(
                final_tables_dir / "amrfinder_probe_hits.tsv",
                sep="\t",
                index=False,
            )
            run.record_output(final_tables_dir / "amrfinder_probe_hits.tsv")
        if not amrfinder_detail.empty:
            amrfinder_detail.to_csv(
                final_tables_dir / "amrfinder_concordance_detail.tsv",
                sep="\t",
                index=False,
            )
            run.record_output(final_tables_dir / "amrfinder_concordance_detail.tsv")
    if not amrfinder_summary.empty:
        amrfinder_summary.to_csv(
            final_tables_dir / "amrfinder_concordance_summary.tsv",
            sep="\t",
            index=False,
        )
        run.record_output(final_tables_dir / "amrfinder_concordance_summary.tsv")
    else:
        for stale_name in (
            "amrfinder_probe_panel.tsv",
            "amrfinder_probe_hits.tsv",
            "amrfinder_concordance_detail.tsv",
            "amrfinder_concordance_summary.tsv",
        ):
            stale_path = final_tables_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    pathogen_detail_frames = [
        frame for frame in (module_c, module_c_clinical, module_c_environmental) if not frame.empty
    ]
    build_pathogen_group_comparison(
        pd.concat(pathogen_detail_frames, ignore_index=True)
        if pathogen_detail_frames
        else pd.DataFrame(),
    ).to_csv(final_tables_dir / "pathogen_detection_group_comparison.tsv", sep="\t", index=False)
    return (
        module_c,
        who_detail,
        card_detail,
        mobsuite_detail,
        amrfinder_detail,
        amrfinder_reportable,
    )


def _write_routed_table_exports(
    *,
    final_tables_dir: object,
    family_summary_path: Path,
    model_metrics: pd.DataFrame,
    source_validation: pd.DataFrame,
    calibration: pd.DataFrame,
    family_summary: pd.DataFrame,
    subgroup_performance: pd.DataFrame,
    comparison_table: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    blocked_holdout_calibration_summary: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    coefficient_stability_cv: pd.DataFrame,
    dropout_table: pd.DataFrame,
    source_balance_resampling: pd.DataFrame,
    group_holdout: pd.DataFrame,
    blocked_holdout_summary: pd.DataFrame,
    permutation_detail: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    selection_adjusted_permutation_detail: pd.DataFrame,
    selection_adjusted_permutation_summary: pd.DataFrame,
    negative_control: pd.DataFrame,
    future_sentinel_audit: pd.DataFrame,
    logistic_impl: pd.DataFrame,
    logistic_convergence: pd.DataFrame,
    simplicity_summary: pd.DataFrame,
    knownness_summary: pd.DataFrame,
    knownness_strata: pd.DataFrame,
    country_quality: pd.DataFrame,
    purity_atlas: pd.DataFrame,
    assignment_confidence: pd.DataFrame,
    incremental_value: pd.DataFrame,
    novelty_specialist_metrics: pd.DataFrame,
    novelty_specialist_predictions: pd.DataFrame,
    adaptive_gated_metrics: pd.DataFrame,
    adaptive_gated_predictions: pd.DataFrame,
    gate_consistency_audit: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame,
    matched_propensity_audit: pd.DataFrame,
    nonlinear_deconfounding: pd.DataFrame,
    operational_risk_dictionary: pd.DataFrame,
    country_upload_propensity: pd.DataFrame,
    macro_region_jump: pd.DataFrame,
    secondary_outcome_performance: pd.DataFrame,
    weighted_country_outcome: pd.DataFrame,
    count_outcome_audit: pd.DataFrame,
    metadata_quality_summary: pd.DataFrame,
    event_timing_outcomes: pd.DataFrame,
    exposure_adjusted_event: pd.DataFrame,
    exposure_adjusted_outcome_audit: pd.DataFrame,
    ordinal_outcome_audit: pd.DataFrame,
    country_missingness_bounds: pd.DataFrame,
    country_missingness_sensitivity: pd.DataFrame,
    geographic_jump: pd.DataFrame,
    duplicate_quality: pd.DataFrame,
    amr_uncertainty: pd.DataFrame,
    mash_graph: pd.DataFrame,
    counterfactual_shortlist: pd.DataFrame,
    rolling_temporal: pd.DataFrame,
    rolling_assignment_diagnostics: pd.DataFrame,
) -> None:
    routed_table_exports: dict[str, pd.DataFrame] = {
        "model_metrics.tsv": model_metrics,
        "source_stratified_consistency.tsv": source_validation,
        "calibration_table.tsv": calibration,
        "model_family_summary.tsv": family_summary,
        "model_subgroup_performance.tsv": subgroup_performance,
        "model_comparison_summary.tsv": comparison_table,
        "calibration_metrics.tsv": calibration_metrics,
        "blocked_holdout_calibration_summary.tsv": blocked_holdout_calibration_summary,
        "primary_model_coefficients.tsv": coefficient_table,
        "primary_model_coefficient_stability.tsv": coefficient_stability,
        "coefficient_stability_cv.tsv": coefficient_stability_cv,
        "feature_dropout_importance.tsv": dropout_table,
        "source_balance_resampling.tsv": source_balance_resampling,
        "group_holdout_performance.tsv": group_holdout,
        "blocked_holdout_summary.tsv": blocked_holdout_summary,
        "permutation_null_distribution.tsv": permutation_detail,
        "permutation_null_summary.tsv": permutation_summary,
        "selection_adjusted_permutation_null_distribution.tsv": selection_adjusted_permutation_detail,
        "selection_adjusted_permutation_null_summary.tsv": selection_adjusted_permutation_summary,
        "negative_control_audit.tsv": negative_control,
        "future_sentinel_audit.tsv": future_sentinel_audit,
        "logistic_implementation_audit.tsv": logistic_impl,
        "logistic_convergence_audit.tsv": logistic_convergence,
        "model_simplicity_summary.tsv": simplicity_summary,
        "knownness_audit_summary.tsv": knownness_summary,
        "knownness_stratified_performance.tsv": knownness_strata,
        "country_quality_summary.tsv": country_quality,
        "backbone_purity_atlas.tsv": purity_atlas,
        "assignment_confidence_summary.tsv": assignment_confidence,
        "incremental_value_over_baseline.tsv": incremental_value,
        "novelty_specialist_metrics.tsv": novelty_specialist_metrics,
        "novelty_specialist_predictions.tsv": novelty_specialist_predictions,
        "adaptive_gated_metrics.tsv": adaptive_gated_metrics,
        "adaptive_gated_predictions.tsv": adaptive_gated_predictions,
        "gate_consistency_audit.tsv": gate_consistency_audit,
        "knownness_matched_validation.tsv": knownness_matched_validation,
        "matched_stratum_propensity_audit.tsv": matched_propensity_audit,
        "nonlinear_deconfounding_audit.tsv": nonlinear_deconfounding,
        "operational_risk_dictionary_full.tsv": operational_risk_dictionary,
        "country_upload_propensity.tsv": country_upload_propensity,
        "macro_region_jump_outcome.tsv": macro_region_jump,
        "secondary_outcome_performance.tsv": secondary_outcome_performance,
        "weighted_country_outcome_audit.tsv": weighted_country_outcome,
        "new_country_count_audit.tsv": count_outcome_audit,
        "metadata_quality_summary.tsv": metadata_quality_summary,
        "event_timing_outcomes.tsv": event_timing_outcomes,
        "exposure_adjusted_event_outcomes.tsv": exposure_adjusted_event,
        "exposure_adjusted_outcome_audit.tsv": exposure_adjusted_outcome_audit,
        "ordinal_outcome_audit.tsv": ordinal_outcome_audit,
        "country_missingness_bounds.tsv": country_missingness_bounds,
        "country_missingness_sensitivity.tsv": country_missingness_sensitivity,
        "geographic_jump_distance_outcome.tsv": geographic_jump,
        "duplicate_completeness_change_audit.tsv": duplicate_quality,
        "amr_uncertainty_summary.tsv": amr_uncertainty,
        "mash_similarity_graph.tsv": mash_graph,
        "counterfactual_shortlist_comparison.tsv": counterfactual_shortlist,
        "rolling_temporal_validation.tsv": rolling_temporal,
        "rolling_assignment_diagnostics.tsv": rolling_assignment_diagnostics,
    }
    for file_name, frame in routed_table_exports.items():
        frame.to_csv(final_tables_dir / file_name, sep="\t", index=False)
    family_summary.to_csv(family_summary_path, sep="\t", index=False)


def _build_candidate_stability_bundle(
    *,
    top: pd.DataFrame,
    top_bio: pd.DataFrame,
    top_primary: pd.DataFrame,
    prospective_freeze: pd.DataFrame,
    scored: pd.DataFrame,
    rank_stability: pd.DataFrame,
    variant_consistency: pd.DataFrame,
    who_detail: pd.DataFrame,
    card_detail: pd.DataFrame,
    mobsuite_detail: pd.DataFrame,
    module_c: pd.DataFrame,
    predictions: pd.DataFrame,
    primary_model_name: str,
    contextual_prediction_frames: dict[str, pd.DataFrame],
    knownness_summary: pd.DataFrame,
    metadata_quality_summary: pd.DataFrame,
    macro_region_jump: pd.DataFrame,
    operational_risk_dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_operational_risk = pd.DataFrame()
    if not operational_risk_dictionary.empty:
        primary_operational_risk = operational_risk_dictionary.loc[
            operational_risk_dictionary.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(primary_model_name)
        ].pipe(copy_frame)
    if not primary_operational_risk.empty:
        risk_columns = [
            column
            for column in [
                "backbone_id",
                "risk_spread_probability",
                "risk_spread_severity",
                "risk_macro_region_jump_3y",
                "risk_event_within_3y",
                "risk_three_countries_within_5y",
                "operational_risk_score",
                "risk_component_std",
                "risk_uncertainty",
                "risk_abstain_flag",
                "risk_decision_tier",
                "risk_route_context",
                "knownness_score",
                "knownness_half",
                "knownness_quartile",
                "source_band",
                "member_count_band",
                "country_count_band",
            ]
            if column in primary_operational_risk.columns
        ]
        primary_operational_risk = primary_operational_risk[risk_columns].drop_duplicates(
            "backbone_id",
        )

    candidate_context_frames = [
        top.pipe(copy_frame),
        top_bio.pipe(copy_frame),
        top_primary.pipe(copy_frame),
    ]
    if not prospective_freeze.empty and "backbone_id" in prospective_freeze.columns:
        freeze_ids = prospective_freeze["backbone_id"].astype(str).unique().tolist()
        freeze_rows = scored.loc[scored["backbone_id"].astype(str).isin(freeze_ids)].pipe(
            copy_frame
        )
        if not freeze_rows.empty:
            candidate_context_frames.append(freeze_rows)
    candidate_context = (
        pd.concat(candidate_context_frames, ignore_index=True)
        .drop_duplicates(subset=["backbone_id"], keep="first")
        .sort_values("priority_index", ascending=False)
        .reset_index(drop=True)
    )
    candidate_context = candidate_context.loc[candidate_context["spread_label"].notna()].pipe(
        copy_frame
    )
    candidate_stability = candidate_context.pipe(copy_frame)
    if not rank_stability.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            rank_stability.drop(columns=["base_priority_index"], errors="ignore"),
            on="backbone_id",
        )
    if not variant_consistency.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            variant_consistency.drop(columns=["base_priority_index"], errors="ignore"),
            on="backbone_id",
        )
    if not who_detail.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            who_detail[
                [
                    "backbone_id",
                    "who_mia_any_support",
                    "who_mia_any_hpecia",
                    "who_mia_mapped_fraction",
                ]
            ],
            on="backbone_id",
        )
    if not card_detail.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            card_detail[["backbone_id", "card_any_support", "card_match_fraction"]],
            on="backbone_id",
        )
    if not mobsuite_detail.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            mobsuite_detail[
                [
                    "backbone_id",
                    "mobsuite_any_literature_support",
                    "mobsuite_any_cluster_support",
                ]
            ],
            on="backbone_id",
        )
    if not module_c.empty:
        pd_combined = module_c.loc[
            module_c["pathogen_dataset"] == "combined",
            ["backbone_id", "pd_any_support", "pd_matching_fraction"],
        ]
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            pd_combined,
            on="backbone_id",
        )
    primary_predictions = predictions.loc[
        predictions["model_name"] == primary_model_name,
        ["backbone_id", "oof_prediction"],
    ].pipe(copy_frame)
    primary_predictions = primary_predictions.rename(
        columns={"oof_prediction": "primary_model_oof_prediction"},
    )
    candidate_stability = coalescing_left_merge(
        candidate_stability,
        primary_predictions,
        on="backbone_id",
    )
    baseline_predictions = predictions.loc[
        predictions["model_name"] == "baseline_both",
        ["backbone_id", "oof_prediction"],
    ].pipe(copy_frame)
    baseline_predictions = baseline_predictions.rename(
        columns={"oof_prediction": "baseline_both_oof_prediction"},
    )
    candidate_stability = coalescing_left_merge(
        candidate_stability,
        baseline_predictions,
        on="backbone_id",
    )
    primary_oof = candidate_stability.get(
        "primary_model_oof_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    baseline_oof = candidate_stability.get(
        "baseline_both_oof_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    candidate_stability["novelty_margin_vs_baseline"] = primary_oof.fillna(
        0.0,
    ) - baseline_oof.pipe(fill0)
    for prediction_frame in contextual_prediction_frames.values():
        if prediction_frame.empty:
            continue
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            prediction_frame,
            on="backbone_id",
        )
    knownness_columns = ["backbone_id", "knownness_score", "knownness_half"]
    knownness_meta_all = pd.DataFrame(columns=knownness_columns)
    knownness_source_columns = [
        "backbone_id",
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "refseq_share_train",
    ]
    if not knownness_summary.empty and set(knownness_source_columns).issubset(scored.columns):
        knownness_meta_all = annotate_knownness_metadata(
            scored[knownness_source_columns].pipe(copy_frame),
        )
    eligible_backbones = predictions.loc[
        predictions["model_name"] == primary_model_name,
        ["backbone_id"],
    ].drop_duplicates()
    knownness_meta_eligible = eligible_backbones.merge(
        scored[
            [
                "backbone_id",
                "log1p_member_count_train",
                "log1p_n_countries_train",
                "refseq_share_train",
            ]
        ],
        on="backbone_id",
        how="left",
    )
    knownness_meta_eligible = annotate_knownness_metadata(knownness_meta_eligible)
    candidate_stability = coalescing_left_merge(
        candidate_stability,
        knownness_meta_all[knownness_columns],
        on="backbone_id",
    )
    if not metadata_quality_summary.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            metadata_quality_summary[
                [
                    column
                    for column in [
                        "backbone_id",
                        "metadata_quality_score",
                        "metadata_quality_tier",
                        "country_coverage_fraction",
                        "duplicate_fraction",
                    ]
                    if column in metadata_quality_summary.columns
                ]
            ],
            on="backbone_id",
        )
    if not macro_region_jump.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            macro_region_jump[
                [
                    column
                    for column in [
                        "backbone_id",
                        "n_new_macro_regions",
                        "macro_region_jump_label",
                        "n_new_host_families",
                        "host_family_jump_label",
                        "n_new_host_orders",
                        "host_order_jump_label",
                        "weighted_new_country_burden",
                        "rarity_weighted_new_country_burden",
                    ]
                    if column in macro_region_jump.columns
                ]
            ],
            on="backbone_id",
        )
    if not primary_operational_risk.empty:
        candidate_stability = coalescing_left_merge(
            candidate_stability,
            primary_operational_risk,
            on="backbone_id",
        )

    primary_oof = candidate_stability.get(
        "primary_model_oof_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    primary_full = candidate_stability.get(
        "primary_model_full_fit_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    conservative_oof = candidate_stability.get(
        "conservative_model_oof_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    conservative_full = candidate_stability.get(
        "conservative_model_full_fit_prediction",
        pd.Series(np.nan, index=candidate_stability.index, dtype=float),
    ).astype(float)
    candidate_stability["primary_model_candidate_score"] = primary_oof.fillna(primary_full)
    candidate_stability["conservative_model_candidate_score"] = conservative_oof.fillna(
        conservative_full,
    )
    candidate_stability["high_confidence_candidate"] = (
        candidate_stability["priority_index"].notna()
        & candidate_stability["coherence_score"].pipe(fill0).ge(0.5)
        & candidate_stability["bootstrap_top_k_frequency"].pipe(fill0).ge(0.7)
        & candidate_stability["variant_top_k_frequency"].pipe(fill0).ge(0.6)
    )
    consensus_candidates = build_consensus_candidate_ranking(
        candidate_stability,
        primary_score_column="primary_model_candidate_score",
        conservative_score_column="conservative_model_candidate_score",
        top_k=50,
    )
    return (
        candidate_stability,
        knownness_meta_eligible,
        consensus_candidates,
        primary_operational_risk,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_report_mode = os.environ.get("PLASMID_PRIORITY_REPORT_MODE", "report-full").strip()
    default_report_mode = (
        env_report_mode
        if env_report_mode in {"report-fast", "report-full", "report-diff"}
        else "report-full"
    )
    parser = argparse.ArgumentParser(description="Build report artifacts and figures.")
    parser.add_argument(
        "--report-mode",
        choices=("report-fast", "report-full", "report-diff"),
        default=default_report_mode,
        help="report-fast skips heavy figure rendering; report-diff redraws only changed figures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_mode = str(args.report_mode)
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    scored_parquet_path = context.data_dir / "scores/backbone_scored.parquet"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_consensus_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    metrics_path = context.data_dir / "analysis/module_a_metrics.json"
    sensitivity_path = context.data_dir / "analysis/sensitivity_summary.json"
    module_b_path = context.data_dir / "analysis/module_b_amr_class_comparison.tsv"
    module_c_path = context.data_dir / "analysis/pathogen_detection_support.tsv"
    module_c_group_path = context.data_dir / "analysis/pathogen_detection_group_summary.tsv"
    module_c_clinical_path = context.data_dir / "analysis/pathogen_detection_clinical_support.tsv"
    module_c_clinical_group_path = (
        context.data_dir / "analysis/pathogen_detection_clinical_group_summary.tsv"
    )
    module_c_environmental_path = (
        context.data_dir / "analysis/pathogen_detection_environmental_support.tsv"
    )
    module_c_environmental_group_path = (
        context.data_dir / "analysis/pathogen_detection_environmental_group_summary.tsv"
    )
    module_c_strata_group_path = (
        context.data_dir / "analysis/pathogen_detection_strata_group_summary.tsv"
    )
    who_detail_path = context.data_dir / "analysis/who_mia_support.tsv"
    who_summary_path = context.data_dir / "analysis/who_mia_group_summary.tsv"
    who_category_path = context.data_dir / "analysis/who_mia_category_comparison.tsv"
    who_reference_path = context.data_dir / "analysis/who_mia_reference_catalog.tsv"
    card_detail_path = context.data_dir / "analysis/card_gene_support.tsv"
    card_summary_path = context.data_dir / "analysis/card_group_summary.tsv"
    card_family_path = context.data_dir / "analysis/card_gene_family_comparison.tsv"
    card_mechanism_path = context.data_dir / "analysis/card_mechanism_comparison.tsv"
    mobsuite_detail_path = context.data_dir / "analysis/mobsuite_host_range_support.tsv"
    mobsuite_summary_path = context.data_dir / "analysis/mobsuite_host_range_group_summary.tsv"
    amrfinder_probe_panel_path = context.data_dir / "analysis/amrfinder_probe_panel.tsv"
    amrfinder_probe_hits_path = context.data_dir / "analysis/amrfinder_probe_hits.tsv"
    amrfinder_detail_path = context.data_dir / "analysis/amrfinder_concordance_detail.tsv"
    amrfinder_summary_path = context.data_dir / "analysis/amrfinder_concordance_summary.tsv"
    source_validation_path = context.data_dir / "analysis/source_stratified_consistency.tsv"
    calibration_path = context.data_dir / "analysis/calibration_table.tsv"
    family_summary_path = context.data_dir / "analysis/model_family_summary.tsv"
    subgroup_performance_path = context.data_dir / "analysis/model_subgroup_performance.tsv"
    comparison_path = context.data_dir / "analysis/model_comparison_summary.tsv"
    calibration_metrics_path = context.data_dir / "analysis/calibration_metrics.tsv"
    blocked_holdout_calibration_path = (
        context.data_dir / "analysis/blocked_holdout_calibration_summary.tsv"
    )
    coefficient_path = context.data_dir / "analysis/primary_model_coefficients.tsv"
    coefficient_stability_path = (
        context.data_dir / "analysis/primary_model_coefficient_stability.tsv"
    )
    coefficient_stability_cv_path = context.data_dir / "analysis/coefficient_stability_cv.tsv"
    dropout_path = context.data_dir / "analysis/feature_dropout_importance.tsv"
    source_balance_resampling_path = context.data_dir / "analysis/source_balance_resampling.tsv"
    group_holdout_path = context.data_dir / "analysis/group_holdout_performance.tsv"
    permutation_detail_path = context.data_dir / "analysis/permutation_null_distribution.tsv"
    permutation_summary_path = context.data_dir / "analysis/permutation_null_summary.tsv"
    selection_adjusted_permutation_detail_path = (
        context.data_dir / "analysis/selection_adjusted_permutation_null_distribution.tsv"
    )
    selection_adjusted_permutation_summary_path = (
        context.data_dir / "analysis/selection_adjusted_permutation_null_summary.tsv"
    )
    negative_control_path = context.data_dir / "analysis/negative_control_audit.tsv"
    future_sentinel_path = context.data_dir / "analysis/future_sentinel_audit.tsv"
    logistic_impl_path = context.data_dir / "analysis/logistic_implementation_audit.tsv"
    logistic_convergence_path = context.data_dir / "analysis/logistic_convergence_audit.tsv"
    simplicity_path = context.data_dir / "analysis/model_simplicity_summary.tsv"
    knownness_summary_path = context.data_dir / "analysis/knownness_audit_summary.tsv"
    knownness_strata_path = context.data_dir / "analysis/knownness_stratified_performance.tsv"
    country_quality_path = context.data_dir / "analysis/country_quality_summary.tsv"
    purity_atlas_path = context.data_dir / "analysis/backbone_purity_atlas.tsv"
    assignment_confidence_path = context.data_dir / "analysis/assignment_confidence_summary.tsv"
    incremental_value_path = context.data_dir / "analysis/incremental_value_over_baseline.tsv"
    novelty_specialist_metrics_path = context.data_dir / "analysis/novelty_specialist_metrics.tsv"
    novelty_specialist_predictions_path = (
        context.data_dir / "analysis/novelty_specialist_predictions.tsv"
    )
    adaptive_gated_metrics_path = context.data_dir / "analysis/adaptive_gated_metrics.tsv"
    adaptive_gated_predictions_path = context.data_dir / "analysis/adaptive_gated_predictions.tsv"
    gate_consistency_audit_path = context.data_dir / "analysis/gate_consistency_audit.tsv"
    knownness_matched_validation_path = (
        context.data_dir / "analysis/knownness_matched_validation.tsv"
    )
    matched_propensity_audit_path = (
        context.data_dir / "analysis/matched_stratum_propensity_audit.tsv"
    )
    nonlinear_deconfounding_path = context.data_dir / "analysis/nonlinear_deconfounding_audit.tsv"
    operational_risk_dictionary_path = context.data_dir / "analysis/operational_risk_dictionary.tsv"
    country_upload_propensity_path = context.data_dir / "analysis/country_upload_propensity.tsv"
    macro_region_jump_path = context.data_dir / "analysis/macro_region_jump_outcome.tsv"
    secondary_outcome_performance_path = (
        context.data_dir / "analysis/secondary_outcome_performance.tsv"
    )
    weighted_country_outcome_path = context.data_dir / "analysis/weighted_country_outcome_audit.tsv"
    count_outcome_audit_path = context.data_dir / "analysis/new_country_count_audit.tsv"
    metadata_quality_summary_path = context.data_dir / "analysis/metadata_quality_summary.tsv"
    event_timing_outcomes_path = context.data_dir / "analysis/event_timing_outcomes.tsv"
    exposure_adjusted_event_path = (
        context.data_dir / "analysis/exposure_adjusted_event_outcomes.tsv"
    )
    exposure_adjusted_outcome_audit_path = (
        context.data_dir / "analysis/exposure_adjusted_outcome_audit.tsv"
    )
    ordinal_outcome_audit_path = context.data_dir / "analysis/ordinal_outcome_audit.tsv"
    country_missingness_bounds_path = context.data_dir / "analysis/country_missingness_bounds.tsv"
    country_missingness_sensitivity_path = (
        context.data_dir / "analysis/country_missingness_sensitivity.tsv"
    )
    geographic_jump_path = context.data_dir / "analysis/geographic_jump_distance_outcome.tsv"
    duplicate_quality_path = context.data_dir / "analysis/duplicate_completeness_change_audit.tsv"
    amr_uncertainty_path = context.data_dir / "analysis/amr_uncertainty_summary.tsv"
    mash_graph_path = context.data_dir / "analysis/mash_similarity_graph.tsv"
    counterfactual_shortlist_path = (
        context.data_dir / "analysis/counterfactual_shortlist_comparison.tsv"
    )
    module_f_identity_path = context.data_dir / "analysis/module_f_backbone_identity.tsv"
    module_f_enrichment_path = context.data_dir / "analysis/module_f_enrichment.tsv"
    module_f_top_hits_path = context.data_dir / "analysis/module_f_top_hits.tsv"
    rolling_temporal_path = context.data_dir / "analysis/rolling_temporal_validation.tsv"
    rolling_assignment_diagnostic_path = (
        context.data_dir / "analysis/rolling_assignment_diagnostics.tsv"
    )
    rank_stability_path = context.data_dir / "analysis/candidate_rank_stability.tsv"
    variant_consistency_path = context.data_dir / "analysis/candidate_variant_consistency.tsv"
    prospective_freeze_path = context.data_dir / "analysis/prospective_candidate_freeze.tsv"
    annual_freeze_summary_path = context.data_dir / "analysis/annual_candidate_freeze_summary.tsv"
    predictions_path = context.data_dir / "analysis/module_a_predictions.tsv"
    single_model_pareto_screen_path = context.data_dir / "analysis/single_model_pareto_screen.tsv"
    single_model_pareto_finalists_path = (
        context.data_dir / "analysis/single_model_pareto_finalists.tsv"
    )
    single_model_official_decision_path = (
        context.data_dir / "analysis/single_model_official_decision.tsv"
    )
    analysis_dir = context.data_dir / "analysis"

    core_dir, diag_dir, final_tables_dir = build_table_router(context.reports_dir, analysis_dir)

    figures_dir = context.reports_dir / "_figure_router"
    jury_brief_path = context.reports_dir / "jury_brief.md"
    turkish_summary_path = context.reports_dir / "ozet_tr.md"
    executive_summary_path = context.reports_dir / "executive_summary.md"
    pitch_notes_path = context.reports_dir / "pitch_notes.md"
    headline_summary_path = context.reports_dir / "headline_validation_summary.md"
    stale_turkiye_context_path = diag_dir / "turkiye_candidate_context.tsv"
    config_paths = context_config_paths(context)
    manifest_path = context.reports_dir / "24_build_reports.manifest.json"
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/24_build_reports.py",
    )
    input_paths = [
        *config_paths,
        scored_path,
        backbones_path,
        amr_consensus_path,
        metrics_path,
        sensitivity_path,
        module_b_path,
        module_c_path,
        module_c_group_path,
        module_c_clinical_path,
        module_c_clinical_group_path,
        module_c_environmental_path,
        module_c_environmental_group_path,
        module_c_strata_group_path,
        who_detail_path,
        who_summary_path,
        who_category_path,
        card_detail_path,
        card_summary_path,
        card_family_path,
        card_mechanism_path,
        mobsuite_detail_path,
        mobsuite_summary_path,
        amrfinder_probe_panel_path,
        amrfinder_probe_hits_path,
        amrfinder_detail_path,
        amrfinder_summary_path,
        source_validation_path,
        calibration_path,
        family_summary_path,
        # Full-fit prediction artifacts (upstream science outputs)
        context.data_dir / "analysis" / "primary_model_full_fit_predictions.tsv",
        context.data_dir / "analysis" / "baseline_both_full_fit_predictions.tsv",
        context.data_dir / "analysis" / "conservative_model_full_fit_predictions.tsv",
        context.data_dir / "analysis" / "governance_model_full_fit_predictions.tsv",
        subgroup_performance_path,
        comparison_path,
        calibration_metrics_path,
        blocked_holdout_calibration_path,
        coefficient_path,
        coefficient_stability_path,
        dropout_path,
        source_balance_resampling_path,
        group_holdout_path,
        permutation_detail_path,
        permutation_summary_path,
        selection_adjusted_permutation_detail_path,
        selection_adjusted_permutation_summary_path,
        negative_control_path,
        future_sentinel_path,
        logistic_impl_path,
        logistic_convergence_path,
        simplicity_path,
        knownness_summary_path,
        knownness_strata_path,
        country_quality_path,
        purity_atlas_path,
        assignment_confidence_path,
        incremental_value_path,
        novelty_specialist_metrics_path,
        novelty_specialist_predictions_path,
        adaptive_gated_metrics_path,
        adaptive_gated_predictions_path,
        gate_consistency_audit_path,
        knownness_matched_validation_path,
        matched_propensity_audit_path,
        nonlinear_deconfounding_path,
        operational_risk_dictionary_path,
        country_upload_propensity_path,
        macro_region_jump_path,
        secondary_outcome_performance_path,
        weighted_country_outcome_path,
        count_outcome_audit_path,
        metadata_quality_summary_path,
        event_timing_outcomes_path,
        exposure_adjusted_event_path,
        exposure_adjusted_outcome_audit_path,
        ordinal_outcome_audit_path,
        country_missingness_bounds_path,
        country_missingness_sensitivity_path,
        geographic_jump_path,
        duplicate_quality_path,
        amr_uncertainty_path,
        mash_graph_path,
        counterfactual_shortlist_path,
        module_f_identity_path,
        module_f_enrichment_path,
        module_f_top_hits_path,
        rolling_temporal_path,
        rolling_assignment_diagnostic_path,
        rank_stability_path,
        variant_consistency_path,
        prospective_freeze_path,
        annual_freeze_summary_path,
        predictions_path,
        single_model_pareto_screen_path,
        single_model_pareto_finalists_path,
        single_model_official_decision_path,
    ]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread,
            ),
        },
        "report_mode": report_mode,
    }
    report_cache = ReportCache(context.reports_dir / ".cache")
    input_signatures = [path_signature(path) for path in input_paths if path.exists()]
    source_signatures = [path_signature(path) for path in source_paths if path.exists()]
    report_key = report_cache.build_report_key(
        report_name="24_build_reports",
        input_hashes=input_signatures,
        config_hash=stable_hash(cache_metadata),
        protocol_hash=stable_hash(context.config),
        code_hash=stable_hash(source_signatures),
        mode=report_mode,
    )

    with benchmark_runtime("24_build_reports_total"):
        with ManagedScriptRun(context, "24_build_reports") as run:
            for path in input_paths:
                if path.exists():
                    run.record_input(path)
            register_default_report_outputs(
                run,
                final_tables_dir=final_tables_dir,
                core_dir=core_dir,
                jury_brief_path=jury_brief_path,
                turkish_summary_path=turkish_summary_path,
                executive_summary_path=executive_summary_path,
                pitch_notes_path=pitch_notes_path,
                headline_summary_path=headline_summary_path,
                family_summary_path=family_summary_path,
            )
            report_expected_outputs = [
                jury_brief_path,
                turkish_summary_path,
                executive_summary_path,
                pitch_notes_path,
                headline_summary_path,
            ]
            if load_signature_manifest(
                manifest_path,
                input_paths=[path for path in input_paths if path.exists()],
                source_paths=source_paths,
                metadata=cache_metadata,
            ):
                run.note("Inputs, code, and config unchanged; reusing cached report outputs.")
                run.set_metric("cache_hit", True)
                return 0
            if report_cache.is_report_current(
                report_name="24_build_reports",
                report_key=report_key,
                outputs=report_expected_outputs,
            ):
                run.note("Report cache hit: key and core outputs unchanged.")
                run.set_metric("cache_hit", True)
                return 0

            if scored_parquet_path.exists():
                scored = annotate_knownness_metadata(read_parquet(scored_parquet_path))
                run.note("Loaded scored backbone table from Parquet via DuckDB.")
                run.set_metric("scored_input_engine", "duckdb_parquet")
            else:
                scored = annotate_knownness_metadata(read_tsv(scored_path))
                run.set_metric("scored_input_engine", "tsv")
            pipeline = context.pipeline_settings
            backbones = read_tsv(backbones_path)
            amr_consensus = _read_if_exists(amr_consensus_path)
            model_metrics = _metrics_to_frame(metrics_path)
            predictions = read_tsv(predictions_path)
            calibration = read_tsv(calibration_path)
            source_validation = read_tsv(source_validation_path)
            active_model_names = get_active_model_names(model_metrics)
            active_model_name_set = set(active_model_names)
            primary_model_name = get_primary_model_name(active_model_names)
            conservative_model_name = get_conservative_model_name(active_model_names)
            governance_model_name = get_governance_model_name(active_model_names)
            validate_discovery_input_contract(
                scored,
                model_names=[primary_model_name],
                contract=build_discovery_input_contract(int(pipeline.split_year)),
                label="Report score input",
            )

            contextual_prediction_frames: dict[str, pd.DataFrame] = {}
            seen_contextual_models: set[str] = set()
            for model_name, output_column in (
                (primary_model_name, "primary_model_full_fit_prediction"),
                ("baseline_both", "baseline_both_full_fit_prediction"),
                (conservative_model_name, "conservative_model_full_fit_prediction"),
                (governance_model_name, "governance_model_full_fit_prediction"),
            ):
                if model_name in seen_contextual_models:
                    continue
                if model_name not in active_model_name_set:
                    continue
                if output_column in contextual_prediction_frames:
                    continue
                seen_contextual_models.add(model_name)
                # Read full-fit predictions from upstream artifact instead of computing them
                prediction_path = (
                    context.data_dir / "analysis" / f"{model_name}_full_fit_predictions.tsv"
                )
                if prediction_path.exists():
                    contextual_prediction_frames[output_column] = read_tsv(prediction_path).rename(
                        columns={
                            "prediction": output_column,
                            "prediction_posterior_mean": f"{output_column}_posterior_mean",
                            "prediction_std": f"{output_column}_std",
                            "prediction_ci_lower": f"{output_column}_ci_lower",
                            "prediction_ci_upper": f"{output_column}_ci_upper",
                        },
                    )
                else:
                    # Fallback to empty dataframe if file doesn't exist
                    contextual_prediction_frames[output_column] = pd.DataFrame(
                        columns=[
                            "backbone_id",
                            output_column,
                            f"{output_column}_posterior_mean",
                            f"{output_column}_std",
                            f"{output_column}_ci_lower",
                            f"{output_column}_ci_upper",
                        ],
                    )
            family_summary = read_tsv(family_summary_path)
            if (
                family_summary.empty
                or "model_name" not in family_summary.columns
                or "evidence_role" not in family_summary.columns
                or "model_track" not in family_summary.columns
                or "track_summary" not in family_summary.columns
                or primary_model_name not in set(family_summary["model_name"].astype(str))
                or (
                    "bio_residual_synergy_priority" in active_model_name_set
                    and "bio_residual_synergy_priority"
                    not in set(family_summary["model_name"].astype(str))
                )
            ):
                family_summary = build_model_family_summary(model_metrics)
            subgroup_performance = read_tsv(subgroup_performance_path)
            comparison_table = read_tsv(comparison_path)
            calibration_metrics = read_tsv(calibration_metrics_path)
            blocked_holdout_calibration_summary = _read_if_exists(blocked_holdout_calibration_path)
            coefficient_table = read_tsv(coefficient_path)
            coefficient_stability = read_tsv(coefficient_stability_path)
            coefficient_stability_cv = _read_if_exists(coefficient_stability_cv_path)
            dropout_table = read_tsv(dropout_path)
            source_balance_resampling = read_tsv(source_balance_resampling_path)
            if stale_turkiye_context_path.exists():
                stale_turkiye_context_path.unlink()
            group_holdout = _read_if_exists(group_holdout_path)
            blocked_holdout_summary = build_blocked_holdout_summary(group_holdout)
            spatial_holdout_summary = _build_spatial_holdout_summary(group_holdout)
            permutation_detail = _read_if_exists(permutation_detail_path)
            permutation_summary = _read_if_exists(permutation_summary_path)
            selection_adjusted_permutation_detail = _read_if_exists(
                selection_adjusted_permutation_detail_path,
            )
            selection_adjusted_permutation_summary = _read_if_exists(
                selection_adjusted_permutation_summary_path,
            )
            negative_control = _read_if_exists(negative_control_path)
            future_sentinel_audit = _read_if_exists(future_sentinel_path)
            logistic_impl = _read_if_exists(logistic_impl_path)
            logistic_convergence = _read_if_exists(logistic_convergence_path)
            simplicity_summary = _read_if_exists(simplicity_path)
            knownness_summary = _read_if_exists(knownness_summary_path)
            knownness_strata = _read_if_exists(knownness_strata_path)
            knownness_matched_validation = _read_if_exists(knownness_matched_validation_path)
            matched_propensity_audit = _read_if_exists(matched_propensity_audit_path)
            country_quality = _read_if_exists(country_quality_path)
            country_upload_propensity = _read_if_exists(country_upload_propensity_path)
            purity_atlas = _read_if_exists(purity_atlas_path)
            assignment_confidence = _read_if_exists(assignment_confidence_path)
            incremental_value = _read_if_exists(incremental_value_path)
            novelty_specialist_metrics = _read_if_exists(novelty_specialist_metrics_path)
            novelty_specialist_predictions = _read_if_exists(novelty_specialist_predictions_path)
            adaptive_gated_metrics = _read_if_exists(adaptive_gated_metrics_path)
            adaptive_gated_predictions = _read_if_exists(adaptive_gated_predictions_path)
            gate_consistency_audit = _read_if_exists(gate_consistency_audit_path)
            nonlinear_deconfounding = _read_if_exists(nonlinear_deconfounding_path)
            operational_risk_dictionary = _read_if_exists(operational_risk_dictionary_path)
            macro_region_jump = _read_if_exists(macro_region_jump_path)
            secondary_outcome_performance = _read_if_exists(secondary_outcome_performance_path)
            weighted_country_outcome = _read_if_exists(weighted_country_outcome_path)
            count_outcome_audit = _read_if_exists(count_outcome_audit_path)
            metadata_quality_summary = _read_if_exists(metadata_quality_summary_path)
            event_timing_outcomes = _read_if_exists(event_timing_outcomes_path)
            exposure_adjusted_event = _read_if_exists(exposure_adjusted_event_path)
            exposure_adjusted_outcome_audit = _read_if_exists(exposure_adjusted_outcome_audit_path)
            ordinal_outcome_audit = _read_if_exists(ordinal_outcome_audit_path)
            country_missingness_bounds = _read_if_exists(country_missingness_bounds_path)
            country_missingness_sensitivity = _read_if_exists(country_missingness_sensitivity_path)
            geographic_jump = _read_if_exists(geographic_jump_path)
            duplicate_quality = _read_if_exists(duplicate_quality_path)
            amr_uncertainty = _read_if_exists(amr_uncertainty_path)
            mash_graph = _read_if_exists(mash_graph_path)
            counterfactual_shortlist = _read_if_exists(counterfactual_shortlist_path)
            module_f_identity = _read_if_exists(module_f_identity_path)
            module_f_enrichment = _read_if_exists(module_f_enrichment_path)
            module_f_top_hits = _read_if_exists(module_f_top_hits_path)
            if module_f_identity.empty and not backbones.empty and not amr_consensus.empty:
                module_f_identity = build_backbone_identity_table(
                    scored,
                    backbones,
                    amr_consensus,
                    split_year=pipeline.split_year,
                )
            if module_f_enrichment.empty and not module_f_identity.empty:
                module_f_enrichment = build_module_f_enrichment_table(
                    module_f_identity,
                    label_column="spread_label",
                    min_backbones=10,
                )
            if module_f_top_hits.empty and not module_f_enrichment.empty:
                module_f_top_hits = build_module_f_top_hits(
                    module_f_enrichment,
                    q_threshold=0.05,
                    max_per_group=3,
                    max_total=20,
                )
            rolling_temporal = _read_if_exists(rolling_temporal_path)
            rolling_assignment_diagnostics = _read_if_exists(rolling_assignment_diagnostic_path)
            rank_stability = _read_if_exists(rank_stability_path)
            variant_consistency = _read_if_exists(variant_consistency_path)
            prospective_freeze = _read_if_exists(prospective_freeze_path)
            annual_freeze_summary = _read_if_exists(annual_freeze_summary_path)
            single_model_pareto_screen = _read_if_exists(single_model_pareto_screen_path)
            single_model_pareto_finalists = _read_if_exists(single_model_pareto_finalists_path)
            single_model_official_decision = _read_if_exists(single_model_official_decision_path)
            if annual_freeze_summary.empty and not rolling_temporal.empty:
                rolling_ok = rolling_temporal.loc[
                    rolling_temporal.get(
                        "status",
                        pd.Series("", index=rolling_temporal.index),
                    ).astype(str)
                    == "ok"
                ].pipe(copy_frame)
                if not rolling_ok.empty:
                    annual_freeze_summary = rolling_ok.pipe(copy_frame)
                    if "horizon_years" not in annual_freeze_summary.columns:
                        annual_freeze_summary["horizon_years"] = annual_freeze_summary[
                            "test_year_end"
                        ].astype(int) - annual_freeze_summary["split_year"].astype(int)
                    annual_freeze_summary["n_candidates"] = 25
                    annual_freeze_summary["n_positive_candidates"] = np.nan
                    annual_freeze_summary["precision_at_25"] = np.nan
                    annual_freeze_summary["mean_n_new_countries"] = np.nan
                    annual_freeze_summary["top_backbone_id"] = ""
            with sensitivity_path.open("r", encoding="utf-8") as handle:
                sensitivity = json.load(handle)
            scored["operational_priority_index"] = scored.get(
                "operational_priority_index",
                scored.get("priority_index", 0.0),
            ).fillna(scored.get("priority_index", 0.0))
            top = (
                scored.loc[scored["spread_label"].notna()]
                .sort_values("operational_priority_index", ascending=False)
                .head(100)
            )
            top_operational_backlog = (
                scored.loc[scored["member_count_train"].pipe(int0) > 0]
                .sort_values(
                    "operational_priority_index",
                    ascending=False,
                )
                .head(100)
            )
            top_bio = (
                scored.loc[scored["spread_label"].notna()]
                .sort_values("bio_priority_index", ascending=False)
                .head(100)
            )
            top_bio_backlog = (
                scored.loc[scored["member_count_train"].pipe(int0) > 0]
                .sort_values(
                    "bio_priority_index",
                    ascending=False,
                )
                .head(100)
            )
            top_primary = scored.loc[scored["spread_label"].notna()].pipe(copy_frame)
            if primary_model_name in set(predictions["model_name"].astype(str)):
                top_primary = top_primary.merge(
                    predictions.loc[
                        predictions["model_name"] == primary_model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                ).sort_values("primary_model_oof_prediction", ascending=False)
            top_primary = top_primary.head(100)
            top = _add_visibility_alias(top)
            top_operational_backlog = _add_visibility_alias(top_operational_backlog)
            top_bio = _add_visibility_alias(top_bio)
            top_bio_backlog = _add_visibility_alias(top_bio_backlog)
            top_primary = _add_visibility_alias(top_primary)
            top_primary.to_csv(
                final_tables_dir / "top_primary_candidates.tsv",
                sep="\t",
                index=False,
            )
            _write_routed_table_exports(
                final_tables_dir=final_tables_dir,
                family_summary_path=family_summary_path,
                model_metrics=model_metrics,
                source_validation=source_validation,
                calibration=calibration,
                family_summary=family_summary,
                subgroup_performance=subgroup_performance,
                comparison_table=comparison_table,
                calibration_metrics=calibration_metrics,
                blocked_holdout_calibration_summary=blocked_holdout_calibration_summary,
                coefficient_table=coefficient_table,
                coefficient_stability=coefficient_stability,
                coefficient_stability_cv=coefficient_stability_cv,
                dropout_table=dropout_table,
                source_balance_resampling=source_balance_resampling,
                group_holdout=group_holdout,
                blocked_holdout_summary=blocked_holdout_summary,
                permutation_detail=permutation_detail,
                permutation_summary=permutation_summary,
                selection_adjusted_permutation_detail=selection_adjusted_permutation_detail,
                selection_adjusted_permutation_summary=selection_adjusted_permutation_summary,
                negative_control=negative_control,
                future_sentinel_audit=future_sentinel_audit,
                logistic_impl=logistic_impl,
                logistic_convergence=logistic_convergence,
                simplicity_summary=simplicity_summary,
                knownness_summary=knownness_summary,
                knownness_strata=knownness_strata,
                country_quality=country_quality,
                purity_atlas=purity_atlas,
                assignment_confidence=assignment_confidence,
                incremental_value=incremental_value,
                novelty_specialist_metrics=novelty_specialist_metrics,
                novelty_specialist_predictions=novelty_specialist_predictions,
                adaptive_gated_metrics=adaptive_gated_metrics,
                adaptive_gated_predictions=adaptive_gated_predictions,
                gate_consistency_audit=gate_consistency_audit,
                knownness_matched_validation=knownness_matched_validation,
                matched_propensity_audit=matched_propensity_audit,
                nonlinear_deconfounding=nonlinear_deconfounding,
                operational_risk_dictionary=operational_risk_dictionary,
                country_upload_propensity=country_upload_propensity,
                macro_region_jump=macro_region_jump,
                secondary_outcome_performance=secondary_outcome_performance,
                weighted_country_outcome=weighted_country_outcome,
                count_outcome_audit=count_outcome_audit,
                metadata_quality_summary=metadata_quality_summary,
                event_timing_outcomes=event_timing_outcomes,
                exposure_adjusted_event=exposure_adjusted_event,
                exposure_adjusted_outcome_audit=exposure_adjusted_outcome_audit,
                ordinal_outcome_audit=ordinal_outcome_audit,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                geographic_jump=geographic_jump,
                duplicate_quality=duplicate_quality,
                amr_uncertainty=amr_uncertainty,
                mash_graph=mash_graph,
                counterfactual_shortlist=counterfactual_shortlist,
                rolling_temporal=rolling_temporal,
                rolling_assignment_diagnostics=rolling_assignment_diagnostics,
            )
            multiverse_stability = build_multiverse_stability_table(
                rank_stability,
                variant_consistency,
            )
            multiverse_stability.to_csv(
                final_tables_dir / "candidate_multiverse_stability.tsv",
                sep="\t",
                index=False,
            )
            module_f_identity.to_csv(
                final_tables_dir / "module_f_backbone_identity.tsv",
                sep="\t",
                index=False,
            )
            module_f_enrichment.to_csv(
                final_tables_dir / "module_f_enrichment.tsv",
                sep="\t",
                index=False,
            )
            module_f_top_hits.to_csv(
                final_tables_dir / "module_f_top_hits.tsv",
                sep="\t",
                index=False,
            )
            rank_stability.to_csv(
                final_tables_dir / "candidate_rank_stability.tsv",
                sep="\t",
                index=False,
            )
            variant_consistency.to_csv(
                final_tables_dir / "candidate_variant_consistency.tsv",
                sep="\t",
                index=False,
            )
            prospective_freeze_export = (
                prospective_freeze.loc[prospective_freeze["spread_label"].notna()].pipe(copy_frame)
                if not prospective_freeze.empty and "spread_label" in prospective_freeze.columns
                else prospective_freeze.pipe(copy_frame)
            )
            prospective_freeze_export = _add_visibility_alias(prospective_freeze_export)
            prospective_freeze_export.to_csv(
                final_tables_dir / "prospective_candidate_freeze.tsv",
                sep="\t",
                index=False,
            )
            annual_freeze_summary.to_csv(
                final_tables_dir / "annual_candidate_freeze_summary.tsv",
                sep="\t",
                index=False,
            )

            sensitivity_rows = []
            for variant, metrics in sensitivity.items():
                row = {"variant": variant}
                row.update(metrics)
                sensitivity_rows.append(row)
            pd.DataFrame(sensitivity_rows).to_csv(
                final_tables_dir / "sensitivity_summary.tsv",
                sep="\t",
                index=False,
            )
            threshold_sensitivity = _build_threshold_sensitivity_table(
                sensitivity,
                default_threshold=pipeline.min_new_countries_for_spread,
            )
            threshold_sensitivity.to_csv(
                final_tables_dir / "threshold_sensitivity_summary.tsv",
                sep="\t",
                index=False,
            )
            l2_sensitivity = _build_l2_sensitivity_table(sensitivity)
            l2_sensitivity.to_csv(
                final_tables_dir / "l2_sensitivity_summary.tsv",
                sep="\t",
                index=False,
            )
            weighting_sensitivity = _build_weighting_sensitivity_table(sensitivity)
            weighting_sensitivity.to_csv(
                final_tables_dir / "weighting_sensitivity_summary.tsv",
                sep="\t",
                index=False,
            )

            (
                module_c,
                who_detail,
                card_detail,
                mobsuite_detail,
                amrfinder_detail,
                amrfinder_reportable,
            ) = _export_support_appendix_tables(
                run=run,
                final_tables_dir=final_tables_dir,
                module_b_path=module_b_path,
                module_c_path=module_c_path,
                module_c_group_path=module_c_group_path,
                module_c_clinical_path=module_c_clinical_path,
                module_c_clinical_group_path=module_c_clinical_group_path,
                module_c_environmental_path=module_c_environmental_path,
                module_c_environmental_group_path=module_c_environmental_group_path,
                module_c_strata_group_path=module_c_strata_group_path,
                who_detail_path=who_detail_path,
                who_summary_path=who_summary_path,
                who_category_path=who_category_path,
                who_reference_path=who_reference_path,
                card_detail_path=card_detail_path,
                card_summary_path=card_summary_path,
                card_family_path=card_family_path,
                card_mechanism_path=card_mechanism_path,
                mobsuite_detail_path=mobsuite_detail_path,
                mobsuite_summary_path=mobsuite_summary_path,
                amrfinder_probe_panel_path=amrfinder_probe_panel_path,
                amrfinder_probe_hits_path=amrfinder_probe_hits_path,
                amrfinder_detail_path=amrfinder_detail_path,
                amrfinder_summary_path=amrfinder_summary_path,
            )

            (
                candidate_stability,
                knownness_meta_eligible,
                consensus_candidates,
                primary_operational_risk,
            ) = _build_candidate_stability_bundle(
                top=top,
                top_bio=top_bio,
                top_primary=top_primary,
                prospective_freeze=prospective_freeze,
                scored=scored,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
                who_detail=who_detail,
                card_detail=card_detail,
                mobsuite_detail=mobsuite_detail,
                module_c=module_c,
                predictions=predictions,
                primary_model_name=primary_model_name,
                contextual_prediction_frames=contextual_prediction_frames,
                knownness_summary=knownness_summary,
                metadata_quality_summary=metadata_quality_summary,
                macro_region_jump=macro_region_jump,
                operational_risk_dictionary=operational_risk_dictionary,
            )

            scored_for_h = scored.pipe(copy_frame)
            if primary_model_name in set(predictions["model_name"].astype(str)):
                scored_for_h = scored_for_h.merge(
                    predictions.loc[
                        predictions["model_name"] == primary_model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                )
            h_feature_diagnostics = build_h_feature_diagnostics(
                scored_for_h,
                model_metrics=model_metrics,
                coefficient_table=coefficient_table,
                dropout_table=dropout_table,
                mobsuite_detail=mobsuite_detail,
                score_column="primary_model_oof_prediction",
            )
            h_feature_diagnostics.to_csv(
                final_tables_dir / "h_feature_diagnostics.tsv",
                sep="\t",
                index=False,
            )

            score_axis_summary = build_score_axis_summary(
                scored,
                predictions,
                primary_model_name=primary_model_name,
                baseline_model_name="baseline_both",
            )
            score_axis_summary.to_csv(
                final_tables_dir / "score_axis_summary.tsv",
                sep="\t",
                index=False,
            )

            score_distribution_diagnostics = build_score_distribution_diagnostics(scored)
            score_distribution_diagnostics.to_csv(
                final_tables_dir / "score_distribution_diagnostics.tsv",
                sep="\t",
                index=False,
            )

            component_floor_diagnostics = build_component_floor_diagnostics(scored)
            component_floor_diagnostics.to_csv(
                final_tables_dir / "component_floor_diagnostics.tsv",
                sep="\t",
                index=False,
            )

            temporal_drift_summary = build_temporal_drift_summary(backbones)
            temporal_drift_summary.to_csv(
                final_tables_dir / "temporal_drift_summary.tsv",
                sep="\t",
                index=False,
            )

            if simplicity_summary.empty:
                simplicity_summary = build_model_simplicity_summary(
                    model_metrics,
                    predictions,
                    primary_model_name=primary_model_name,
                    conservative_model_name=conservative_model_name,
                )
            simplicity_summary.to_csv(
                final_tables_dir / "model_simplicity_summary.tsv",
                sep="\t",
                index=False,
            )

            outcome_robustness = _build_outcome_robustness_grid(
                sensitivity,
                rolling_temporal,
                default_threshold=pipeline.min_new_countries_for_spread,
            )
            outcome_robustness.to_csv(
                final_tables_dir / "outcome_robustness_grid.tsv",
                sep="\t",
                index=False,
            )

            dossier_base = (
                consensus_candidates.head(25).pipe(copy_frame)
                if not consensus_candidates.empty
                else (
                    prospective_freeze.head(25).pipe(copy_frame)
                    if not prospective_freeze.empty
                    else candidate_stability.head(25).pipe(copy_frame)
                )
            )
            candidate_dossiers = build_candidate_dossier_table(
                dossier_base,
                candidate_stability=candidate_stability,
                predictions=predictions,
                primary_model_name=primary_model_name,
                conservative_model_name=conservative_model_name,
                who_detail=who_detail,
                card_detail=card_detail,
                mobsuite_detail=mobsuite_detail,
                pathogen_support=module_c,
                amrfinder_detail=amrfinder_detail,
            )
            candidate_risk = build_candidate_risk_table(candidate_dossiers)
            if not candidate_risk.empty:
                candidate_dossiers = coalescing_left_merge(
                    candidate_dossiers,
                    candidate_risk[
                        ["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]
                    ],
                    on="backbone_id",
                )
            candidate_dossiers.to_csv(
                final_tables_dir / "candidate_dossiers.tsv",
                sep="\t",
                index=False,
            )
            candidate_risk.to_csv(
                final_tables_dir / "candidate_risk_flags.tsv",
                sep="\t",
                index=False,
            )
            novelty_watchlist = scored.pipe(copy_frame)
            if primary_model_name in set(predictions["model_name"].astype(str)):
                novelty_watchlist = novelty_watchlist.merge(
                    predictions.loc[
                        predictions["model_name"] == primary_model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                )
            if "baseline_both" in set(predictions["model_name"].astype(str)):
                novelty_watchlist = novelty_watchlist.merge(
                    predictions.loc[
                        predictions["model_name"] == "baseline_both",
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "baseline_both_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                )
            if not knownness_meta_eligible.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    knownness_meta_eligible[
                        ["backbone_id", "knownness_score", "knownness_half"]
                    ].drop_duplicates("backbone_id"),
                    on="backbone_id",
                    how="left",
                )
            if not novelty_specialist_predictions.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    novelty_specialist_predictions[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "novelty_specialist_prediction",
                                "knownness_quartile",
                                "training_cohort",
                            ]
                            if column in novelty_specialist_predictions.columns
                        ]
                    ].drop_duplicates("backbone_id"),
                    on="backbone_id",
                    how="left",
                )
            if not primary_operational_risk.empty:
                novelty_watchlist = coalescing_left_merge(
                    novelty_watchlist,
                    primary_operational_risk,
                    on="backbone_id",
                )
            if not who_detail.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    who_detail[["backbone_id", "who_mia_any_support"]],
                    on="backbone_id",
                    how="left",
                )
            if not card_detail.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    card_detail[["backbone_id", "card_any_support"]],
                    on="backbone_id",
                    how="left",
                )
            if not mobsuite_detail.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    mobsuite_detail[["backbone_id", "mobsuite_any_literature_support"]],
                    on="backbone_id",
                    how="left",
                )
            if not module_c.empty:
                novelty_watchlist = novelty_watchlist.merge(
                    module_c.loc[
                        module_c["pathogen_dataset"] == "combined",
                        ["backbone_id", "pd_any_support"],
                    ],
                    on="backbone_id",
                    how="left",
                )
            for column in (
                "who_mia_any_support",
                "card_any_support",
                "mobsuite_any_literature_support",
                "pd_any_support",
            ):
                if column in novelty_watchlist.columns:
                    novelty_watchlist[column] = novelty_watchlist[column].fillna(False).astype(bool)
            novelty_watchlist["external_support_modalities_count"] = (
                novelty_watchlist[
                    [
                        column
                        for column in (
                            "who_mia_any_support",
                            "card_any_support",
                            "mobsuite_any_literature_support",
                            "pd_any_support",
                        )
                        if column in novelty_watchlist.columns
                    ]
                ]
                .sum(axis=1)
                .astype(int)
            )
            novelty_watchlist["primary_model_candidate_score"] = novelty_watchlist[
                "primary_model_oof_prediction"
            ].astype(float)
            novelty_watchlist["baseline_both_candidate_score"] = novelty_watchlist[
                "baseline_both_oof_prediction"
            ].astype(float)
            novelty_watchlist["candidate_prediction_source"] = "oof"
            novelty_watchlist["eligible_for_oof"] = True
            novelty_watchlist["novelty_margin_vs_baseline"] = np.nan
            novelty_watchlist_margin_mask = (
                novelty_watchlist["primary_model_oof_prediction"].notna()
                & novelty_watchlist["baseline_both_oof_prediction"].notna()
            )
            novelty_watchlist.loc[novelty_watchlist_margin_mask, "novelty_margin_vs_baseline"] = (
                novelty_watchlist.loc[
                    novelty_watchlist_margin_mask,
                    "primary_model_oof_prediction",
                ].astype(float)
                - novelty_watchlist.loc[
                    novelty_watchlist_margin_mask,
                    "baseline_both_oof_prediction",
                ].astype(float)
            )
            if "novelty_specialist_prediction" not in novelty_watchlist.columns:
                novelty_watchlist["novelty_specialist_prediction"] = 0.0
            for prediction_frame in contextual_prediction_frames.values():
                if prediction_frame.empty:
                    continue
                novelty_watchlist = novelty_watchlist.merge(
                    prediction_frame,
                    on="backbone_id",
                    how="left",
                )
            conservative_oof_column = None
            if conservative_model_name in set(predictions["model_name"].astype(str)):
                conservative_oof_column = "conservative_model_oof_prediction"
            if conservative_oof_column is not None:
                novelty_watchlist = novelty_watchlist.merge(
                    predictions.loc[
                        predictions["model_name"] == conservative_model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": conservative_oof_column}),
                    on="backbone_id",
                    how="left",
                )
            if "knownness_half" not in novelty_watchlist.columns:
                for candidate in ("knownness_half_x", "knownness_half_y"):
                    if candidate in novelty_watchlist.columns:
                        novelty_watchlist["knownness_half"] = novelty_watchlist[candidate]
                        break
            if (
                "knownness_half" not in novelty_watchlist.columns
                and "knownness_score" in novelty_watchlist.columns
            ):
                knownness_values = to_numeric_series(
                    novelty_watchlist["knownness_score"],
                )
                valid_knownness = knownness_values.notna()
                if valid_knownness.any():
                    median_knownness = float(knownness_values.loc[valid_knownness].median())
                    novelty_watchlist["knownness_half"] = np.where(
                        knownness_values.le(median_knownness),
                        "lower_half",
                        "upper_half",
                    )
                else:
                    novelty_watchlist["knownness_half"] = "lower_half"
            if "knownness_half" not in novelty_watchlist.columns:
                novelty_watchlist["knownness_half"] = "lower_half"
            novelty_watchlist = novelty_watchlist.loc[
                novelty_watchlist["spread_label"].notna()
                & novelty_watchlist["member_count_train"].pipe(int0).gt(0)
                & novelty_watchlist["knownness_half"].fillna("").eq("lower_half")
            ].pipe(copy_frame)
            novelty_watchlist = novelty_watchlist.loc[
                novelty_watchlist["novelty_margin_vs_baseline"].pipe(fill0).gt(0)
                & (
                    novelty_watchlist["member_count_train"].pipe(int0).ge(2)
                    | novelty_watchlist["external_support_modalities_count"]
                    .fillna(0)
                    .astype(int)
                    .gt(0)
                )
            ].pipe(copy_frame)
            novelty_watchlist = novelty_watchlist.sort_values(
                [
                    "novelty_specialist_prediction",
                    "novelty_margin_vs_baseline",
                    "primary_model_oof_prediction",
                    "priority_index",
                ],
                ascending=[False, False, False, False],
            ).head(25)
            if (
                conservative_oof_column is not None
                or "conservative_model_full_fit_prediction" in novelty_watchlist.columns
            ):
                conservative_oof = novelty_watchlist.get(
                    conservative_oof_column,
                    pd.Series(np.nan, index=novelty_watchlist.index, dtype=float),
                ).astype(float)
                conservative_full = novelty_watchlist.get(
                    "conservative_model_full_fit_prediction",
                    pd.Series(np.nan, index=novelty_watchlist.index, dtype=float),
                ).astype(float)
                novelty_watchlist["conservative_model_candidate_score"] = conservative_oof.fillna(
                    conservative_full,
                )
                novelty_watchlist = _add_visibility_alias(novelty_watchlist)
                if "knownness_half" not in novelty_watchlist.columns:
                    for candidate in ("knownness_half_x", "knownness_half_y"):
                        if candidate in novelty_watchlist.columns:
                            novelty_watchlist["knownness_half"] = novelty_watchlist[candidate]
                            break
            if (
                "knownness_half" not in novelty_watchlist.columns
                and "knownness_score" in novelty_watchlist.columns
            ):
                knownness_values = to_numeric_series(
                    novelty_watchlist["knownness_score"],
                )
                valid_knownness = knownness_values.notna()
                if valid_knownness.any():
                    median_knownness = float(knownness_values.loc[valid_knownness].median())
                    novelty_watchlist["knownness_half"] = np.where(
                        knownness_values.le(median_knownness),
                        "lower_half",
                        "upper_half",
                    )
                else:
                    novelty_watchlist["knownness_half"] = "lower_half"
            if "knownness_half" not in novelty_watchlist.columns:
                novelty_watchlist["knownness_half"] = "lower_half"
            novelty_watchlist.to_csv(
                final_tables_dir / "novelty_watchlist.tsv",
                sep="\t",
                index=False,
            )

            novelty_frontier = scored.loc[scored["spread_label"].notna()].pipe(copy_frame)
            if primary_model_name in set(predictions["model_name"].astype(str)):
                novelty_frontier = novelty_frontier.merge(
                    predictions.loc[
                        predictions["model_name"] == primary_model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                )
            if "baseline_both" in set(predictions["model_name"].astype(str)):
                novelty_frontier = novelty_frontier.merge(
                    predictions.loc[
                        predictions["model_name"] == "baseline_both",
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": "baseline_both_oof_prediction"}),
                    on="backbone_id",
                    how="left",
                )
            if not knownness_meta_eligible.empty:
                novelty_frontier = novelty_frontier.merge(
                    knownness_meta_eligible[
                        ["backbone_id", "knownness_score", "knownness_half"]
                    ].drop_duplicates("backbone_id"),
                    on="backbone_id",
                    how="left",
                )
            if not novelty_specialist_predictions.empty:
                novelty_frontier = novelty_frontier.merge(
                    novelty_specialist_predictions[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "novelty_specialist_prediction",
                                "knownness_quartile",
                            ]
                            if column in novelty_specialist_predictions.columns
                        ]
                    ].drop_duplicates("backbone_id"),
                    on="backbone_id",
                    how="left",
                )
            novelty_frontier["novelty_margin_vs_baseline"] = np.nan
            novelty_frontier_margin_mask = (
                novelty_frontier["primary_model_oof_prediction"].notna()
                & novelty_frontier["baseline_both_oof_prediction"].notna()
            )
            novelty_frontier.loc[novelty_frontier_margin_mask, "novelty_margin_vs_baseline"] = (
                novelty_frontier.loc[
                    novelty_frontier_margin_mask,
                    "primary_model_oof_prediction",
                ].astype(float)
                - novelty_frontier.loc[
                    novelty_frontier_margin_mask,
                    "baseline_both_oof_prediction",
                ].astype(float)
            )
            if "novelty_specialist_prediction" not in novelty_frontier.columns:
                novelty_frontier["novelty_specialist_prediction"] = 0.0
            novelty_frontier = _add_visibility_alias(novelty_frontier)

            novelty_margin_summary = build_novelty_margin_summary(
                predictions,
                scored,
                primary_model_name=primary_model_name,
                baseline_model_name="baseline_both",
            )
            novelty_margin_summary.to_csv(
                final_tables_dir / "novelty_margin_summary.tsv",
                sep="\t",
                index=False,
            )

            candidate_portfolio = build_candidate_portfolio_table(
                candidate_dossiers,
                novelty_watchlist,
                established_n=10,
                novel_n=10,
            )
            if not candidate_portfolio.empty:
                if not consensus_candidates.empty:
                    candidate_portfolio = coalescing_left_merge(
                        candidate_portfolio,
                        consensus_candidates[
                            [
                                column
                                for column in [
                                    "backbone_id",
                                    "consensus_rank",
                                    "consensus_candidate_score",
                                    "consensus_support_count",
                                    "primary_rank",
                                    "conservative_rank",
                                    "rank_disagreement_primary_vs_conservative",
                                ]
                                if column in consensus_candidates.columns
                            ]
                        ],
                        on="backbone_id",
                    )
            if not candidate_stability.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    candidate_stability[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "coherence_score",
                                "bootstrap_top_k_frequency",
                                "bootstrap_top_10_frequency",
                                "bootstrap_top_25_frequency",
                            ]
                            if column in candidate_stability.columns
                        ]
                    ],
                    on="backbone_id",
                )
            if not variant_consistency.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    variant_consistency[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "variant_top_k_frequency",
                                "variant_top_10_frequency",
                                "variant_top_25_frequency",
                            ]
                            if column in variant_consistency.columns
                        ]
                    ],
                    on="backbone_id",
                )
            portfolio_risk = build_candidate_risk_table(candidate_portfolio)
            if not portfolio_risk.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    portfolio_risk[
                        ["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]
                    ],
                    on="backbone_id",
                )
            candidate_portfolio["in_consensus_top50"] = candidate_portfolio.get(
                "consensus_rank",
                pd.Series(np.nan, index=candidate_portfolio.index),
            ).notna()
            bootstrap_top10 = candidate_portfolio.get(
                "bootstrap_top_10_frequency",
                pd.Series(0.0, index=candidate_portfolio.index, dtype=float),
            ).pipe(fill0)
            risk_tier = (
                candidate_portfolio.get(
                    "false_positive_risk_tier",
                    pd.Series("unknown", index=candidate_portfolio.index, dtype=object),
                )
                .fillna("unknown")
                .astype(str)
            )
            candidate_portfolio["recommended_monitoring_tier"] = np.select(
                [
                    (bootstrap_top10 >= 0.80) & risk_tier.isin(["low", "medium"]),
                    (bootstrap_top10 >= 0.50) & risk_tier.ne("high"),
                ],
                ["core_surveillance", "extended_watchlist"],
                default="low_confidence_backlog",
            )
            candidate_portfolio["evidence_tier"] = candidate_portfolio.get(
                "candidate_confidence_tier",
                pd.Series("unknown", index=candidate_portfolio.index),
            ).fillna("unknown")
            candidate_portfolio["action_tier"] = candidate_portfolio["recommended_monitoring_tier"]
            candidate_signature_context = build_candidate_signature_context(
                candidate_portfolio,
                module_f_identity,
                module_f_enrichment,
                q_threshold=0.05,
                max_signatures_per_candidate=5,
            )
            if not candidate_signature_context.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    candidate_signature_context,
                    on="backbone_id",
                )
                candidate_dossiers = coalescing_left_merge(
                    candidate_dossiers,
                    candidate_signature_context,
                    on="backbone_id",
                )
            candidate_portfolio = _deduplicate_backbone_rows(candidate_portfolio)
            candidate_portfolio = _add_visibility_alias(candidate_portfolio)
            high_confidence_export = candidate_stability.loc[
                candidate_stability["high_confidence_candidate"].fillna(False)
            ].pipe(copy_frame)
            if "spread_label" in high_confidence_export.columns:
                high_confidence_export = high_confidence_export.loc[
                    high_confidence_export["spread_label"].fillna(0).astype(float) >= 1.0
                ].pipe(copy_frame)
            if not candidate_risk.empty:
                high_confidence_export = coalescing_left_merge(
                    high_confidence_export,
                    candidate_risk[
                        ["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]
                    ],
                    on="backbone_id",
                )
            high_confidence_export = high_confidence_export.loc[
                high_confidence_export["false_positive_risk_tier"].fillna("high").ne("high")
            ].pipe(copy_frame)
            high_confidence_export = _add_visibility_alias(high_confidence_export)

            candidate_universe = build_candidate_universe_table(
                scored=scored,
                consensus_candidates=consensus_candidates,
                candidate_dossiers=candidate_dossiers,
                candidate_portfolio=candidate_portfolio,
                novelty_watchlist=novelty_watchlist,
                prospective_freeze=prospective_freeze_export,
                high_confidence_candidates=high_confidence_export,
                candidate_risk=candidate_risk,
            )
            candidate_universe = _add_visibility_alias(candidate_universe)

            candidate_threshold_flip = build_threshold_flip_table(
                scored,
                candidate_ids=candidate_universe["backbone_id"].astype(str).tolist()
                if not candidate_universe.empty
                else None,
                default_threshold=pipeline.min_new_countries_for_spread,
            )
            candidate_threshold_flip.to_csv(
                final_tables_dir / "candidate_threshold_flip.tsv",
                sep="\t",
                index=False,
            )
            candidate_multiverse_stability = _build_candidate_multiverse_stability(
                candidate_stability,
                candidate_threshold_flip,
            )
            candidate_multiverse_stability.to_csv(
                final_tables_dir / "candidate_multiverse_stability.tsv",
                sep="\t",
                index=False,
            )
            if not candidate_multiverse_stability.empty:
                merge_columns = [
                    column
                    for column in [
                        "backbone_id",
                        "multiverse_stability_score",
                        "multiverse_stability_tier",
                        "threshold_robustness_score",
                    ]
                    if column in candidate_multiverse_stability.columns
                ]
            if not candidate_portfolio.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    candidate_multiverse_stability[merge_columns],
                    on="backbone_id",
                )
            if not candidate_dossiers.empty:
                candidate_dossiers = coalescing_left_merge(
                    candidate_dossiers,
                    candidate_multiverse_stability[merge_columns],
                    on="backbone_id",
                )
            exposure_merge_columns = [
                column
                for column in [
                    "backbone_id",
                    "new_country_rate_per_year",
                    "weighted_new_country_rate_per_year",
                    "rarity_weighted_new_country_rate_per_year",
                    "first_event_speed_score",
                    "third_event_speed_score",
                    "fast_or_broad_expansion_label",
                ]
                if column in exposure_adjusted_event.columns
            ]
            if len(exposure_merge_columns) > 1:
                exposure_payload = exposure_adjusted_event[exposure_merge_columns].drop_duplicates(
                    "backbone_id",
                )
                if not candidate_portfolio.empty:
                    candidate_portfolio = coalescing_left_merge(
                        candidate_portfolio,
                        exposure_payload,
                        on="backbone_id",
                    )
                if not candidate_dossiers.empty:
                    candidate_dossiers = coalescing_left_merge(
                        candidate_dossiers,
                        exposure_payload,
                        on="backbone_id",
                    )
            if not primary_operational_risk.empty:
                if not candidate_portfolio.empty:
                    candidate_portfolio = coalescing_left_merge(
                        candidate_portfolio,
                        primary_operational_risk,
                        on="backbone_id",
                    )
                if not candidate_dossiers.empty:
                    candidate_dossiers = coalescing_left_merge(
                        candidate_dossiers,
                        primary_operational_risk,
                        on="backbone_id",
                    )
            candidate_dossiers = annotate_candidate_explanation_fields(candidate_dossiers)
            candidate_portfolio = annotate_candidate_explanation_fields(candidate_portfolio)
            candidate_dossiers = _deduplicate_backbone_rows(candidate_dossiers)
            candidate_portfolio = _deduplicate_backbone_rows(candidate_portfolio)
            operational_risk_watchlist = _build_operational_risk_watchlist(
                operational_risk_dictionary,
                primary_model_name=primary_model_name,
                candidate_portfolio=candidate_portfolio,
                top_k=50,
            )
            consensus_shortlist = build_consensus_shortlist(
                consensus_candidates,
                candidate_portfolio,
                candidate_multiverse_stability,
                top_k=25,
            )
            consensus_shortlist = _add_visibility_alias(consensus_shortlist)
            consensus_shortlist.to_csv(
                final_tables_dir / "consensus_shortlist.tsv",
                sep="\t",
                index=False,
            )

            decision_yield = build_decision_yield_table(
                predictions,
                model_names=list(
                    dict.fromkeys(
                        [
                            primary_model_name,
                            conservative_model_name,
                            governance_model_name,
                            "knownness_robust_priority",
                            "host_transfer_synergy_priority",
                            "threat_architecture_priority",
                            "natural_auc_priority",
                            "baseline_both",
                            "evidence_aware_priority",
                        ],
                    ),
                ),
            )
            decision_yield.to_csv(
                final_tables_dir / "decision_yield_summary.tsv",
                sep="\t",
                index=False,
            )
            threshold_utility_summary = build_threshold_utility_table(
                predictions,
                model_names=list(
                    dict.fromkeys(
                        [
                            primary_model_name,
                            conservative_model_name,
                            governance_model_name,
                            "knownness_robust_priority",
                            "host_transfer_synergy_priority",
                            "threat_architecture_priority",
                            "natural_auc_priority",
                            "baseline_both",
                            "evidence_aware_priority",
                        ],
                    ),
                ),
            )
            threshold_utility_summary.to_csv(
                final_tables_dir / "threshold_utility_summary.tsv",
                sep="\t",
                index=False,
            )
            stable_adaptive_model_names = (
                gate_consistency_audit.loc[
                    gate_consistency_audit.get("gate_consistency_tier", pd.Series(dtype=str))
                    .astype(str)
                    .eq("stable"),
                    "model_name",
                ].drop_duplicates()
                if not gate_consistency_audit.empty
                and "gate_consistency_tier" in gate_consistency_audit.columns
                else pd.Series(dtype=str)
            )
            adaptive_has_model_name = (
                not adaptive_gated_metrics.empty and "model_name" in adaptive_gated_metrics.columns
            )
            adaptive_has_roc_auc = "roc_auc" in adaptive_gated_metrics.columns
            if adaptive_has_model_name:
                preferred_adaptive_metrics = adaptive_gated_metrics.loc[
                    adaptive_gated_metrics["model_name"].isin(stable_adaptive_model_names)
                ].pipe(copy_frame)
            if adaptive_has_roc_auc and not preferred_adaptive_metrics.empty:
                preferred_adaptive_metrics = preferred_adaptive_metrics.sort_values(
                    "roc_auc",
                    ascending=False,
                )
            if not preferred_adaptive_metrics.empty:
                best_adaptive_model_name = str(preferred_adaptive_metrics.iloc[0]["model_name"])
            elif adaptive_has_roc_auc:
                best_adaptive_model_name = str(
                    adaptive_gated_metrics.sort_values("roc_auc", ascending=False).iloc[0][
                        "model_name"
                    ],
                )
            elif not adaptive_gated_metrics.empty:
                best_adaptive_model_name = str(adaptive_gated_metrics.iloc[0]["model_name"])
            else:
                best_adaptive_model_name = "adaptive_natural_priority"
            decision_budget_curve = build_decision_yield_table(
                predictions,
                model_names=list(
                    dict.fromkeys(
                        [
                            primary_model_name,
                            conservative_model_name,
                            governance_model_name,
                            "baseline_both",
                            "natural_auc_priority",
                            "threat_architecture_priority",
                            "contextual_bio_priority",
                            best_adaptive_model_name,
                        ],
                    ),
                ),
                top_ks=tuple(range(5, 105, 5)),
            )
            decision_budget_curve.to_csv(
                final_tables_dir / "decision_budget_curve.tsv",
                sep="\t",
                index=False,
            )
            false_negative_audit = build_false_negative_audit(
                scored,
                predictions,
                primary_model_name=primary_model_name,
                metadata_quality=metadata_quality_summary,
                candidate_threshold_flip=candidate_threshold_flip,
                shortlist_cutoffs=(25, 50),
                top_n=50,
            )
            false_negative_audit.to_csv(
                final_tables_dir / "false_negative_audit.tsv",
                sep="\t",
                index=False,
            )
            confirmatory_cohort_summary = build_confirmatory_cohort_summary(
                scored,
                predictions,
                model_names=list(
                    dict.fromkeys(
                        [
                            primary_model_name,
                            governance_model_name,
                            "baseline_both",
                        ],
                    ),
                ),
                metadata_quality=metadata_quality_summary,
            )
            confirmatory_cohort_summary.to_csv(
                final_tables_dir / "confirmatory_cohort_summary.tsv",
                sep="\t",
                index=False,
            )
            spatial_holdout_summary.to_csv(
                core_dir / "spatial_holdout_summary.tsv",
                sep="\t",
                index=False,
            )
            report_model_metrics = _build_report_model_metrics(
                model_metrics,
                calibration_metrics=calibration_metrics,
                permutation_summary=permutation_summary,
                selection_adjusted_permutation_summary=selection_adjusted_permutation_summary,
                comparison_table=comparison_table,
                confirmatory_cohort_summary=confirmatory_cohort_summary,
                spatial_holdout_summary=spatial_holdout_summary,
                primary_model_name=primary_model_name,
                governance_model_name=governance_model_name,
            )

            model_selection_scorecard = build_model_selection_scorecard(
                report_model_metrics,
                predictions,
                scored,
                knownness_matched_validation=knownness_matched_validation,
                group_holdout=group_holdout,
                single_model_finalist_audit=single_model_pareto_finalists,
                model_names=active_model_names,
            )
            acceptance_columns = [
                column
                for column in [
                    "model_name",
                    "ece",
                    "selection_adjusted_empirical_p_roc_auc",
                    "matched_knownness_weighted_roc_auc",
                    "knownness_matched_gap",
                    "source_holdout_weighted_roc_auc",
                    "source_holdout_gap",
                    "spatial_holdout_roc_auc",
                    "spatial_holdout_gap",
                    "scientific_acceptance_scored",
                    "scientific_acceptance_flag",
                    "scientific_acceptance_status",
                    "scientific_acceptance_failed_criteria",
                    "matched_knownness_gate_pass",
                    "source_holdout_gate_pass",
                    "spatial_holdout_gate_pass",
                    "calibration_gate_pass",
                    "selection_adjusted_gate_pass",
                    "leakage_review_gate_pass",
                    "spatial_holdout_gap",
                ]
                if column in model_selection_scorecard.columns
            ]
            if acceptance_columns:
                report_model_metrics = report_model_metrics.merge(
                    model_selection_scorecard[acceptance_columns].drop_duplicates("model_name"),
                    on="model_name",
                    how="left",
                    suffixes=("", "_scorecard"),
                )
                for column in [
                    "ece",
                    "selection_adjusted_empirical_p_roc_auc",
                    "matched_knownness_weighted_roc_auc",
                    "knownness_matched_gap",
                    "source_holdout_weighted_roc_auc",
                    "source_holdout_gap",
                    "spatial_holdout_roc_auc",
                    "spatial_holdout_gap",
                    "scientific_acceptance_scored",
                    "scientific_acceptance_flag",
                    "scientific_acceptance_status",
                    "scientific_acceptance_failed_criteria",
                    "matched_knownness_gate_pass",
                    "source_holdout_gate_pass",
                    "spatial_holdout_gate_pass",
                    "calibration_gate_pass",
                    "selection_adjusted_gate_pass",
                    "leakage_review_gate_pass",
                ]:
                    scorecard_column = f"{column}_scorecard"
                    if scorecard_column not in report_model_metrics.columns:
                        continue
                    if column in report_model_metrics.columns:
                        report_model_metrics[column] = report_model_metrics[
                            scorecard_column
                        ].combine_first(report_model_metrics[column])
                    else:
                        report_model_metrics[column] = report_model_metrics[scorecard_column]
                    report_model_metrics = report_model_metrics.drop(columns=scorecard_column)
            report_model_metrics.to_csv(
                final_tables_dir / "model_metrics.tsv",
                sep="\t",
                index=False,
            )
            headline_validation_summary = _build_headline_validation_summary(
                report_model_metrics,
                primary_model_name=primary_model_name,
                governance_model_name=governance_model_name,
                single_model_official_decision=single_model_official_decision,
                single_model_pareto_finalists=single_model_pareto_finalists,
            )
            headline_validation_summary.to_csv(
                core_dir / "headline_validation_summary.tsv",
                sep="\t",
                index=False,
            )
            single_model_pareto_screen.to_csv(
                final_tables_dir / "single_model_pareto_screen.tsv",
                sep="\t",
                index=False,
            )
            single_model_pareto_finalists.to_csv(
                final_tables_dir / "single_model_pareto_finalists.tsv",
                sep="\t",
                index=False,
            )
            single_model_official_decision.to_csv(
                final_tables_dir / "single_model_official_decision.tsv",
                sep="\t",
                index=False,
            )

            core_model_coefficients = pd.DataFrame()
            core_heatmap_models = [
                "discovery_boosted",
                "parsimonious_priority",
                "structured_signal_priority",
                "support_synergy_priority",
                "governance_linear",
            ]
            coefficient_frames: list[pd.DataFrame] = []
            for model_name in core_heatmap_models:
                if model_name not in MODULE_A_FEATURE_SETS or model_name not in set(
                    model_metrics["model_name"].astype(str),
                ):
                    continue
                coefficient_frames.append(
                    build_standardized_coefficient_table(
                        scored,
                        model_name=model_name,
                        columns=MODULE_A_FEATURE_SETS[model_name],
                    ),
                )
            if coefficient_frames:
                core_model_coefficients = pd.concat(coefficient_frames, ignore_index=True)
                core_model_coefficients.to_csv(
                    core_dir / "core_model_coefficients.tsv",
                    sep="\t",
                    index=False,
                )

            model_selection_scorecard_export = model_selection_scorecard.pipe(copy_frame)
            if (
                "selection_rank" in model_selection_scorecard_export.columns
                and model_selection_scorecard_export["selection_rank"].isna().any()
            ):
                fallback_ranks = pd.Series(
                    np.arange(1, len(model_selection_scorecard_export) + 1),
                    index=model_selection_scorecard_export.index,
                    dtype="Int64",
                )
                model_selection_scorecard_export["selection_rank"] = (
                    model_selection_scorecard_export["selection_rank"].combine_first(
                        fallback_ranks
                    )
                )
            model_selection_scorecard_export.to_csv(
                final_tables_dir / "model_selection_scorecard.tsv",
                sep="\t",
                index=False,
            )
            frozen_scientific_acceptance_audit = build_frozen_scientific_acceptance_audit(
                model_selection_scorecard,
            )
            frozen_scientific_acceptance_audit.to_csv(
                core_dir / "frozen_scientific_acceptance_audit.tsv",
                sep="\t",
                index=False,
            )

            model_selection_summary = build_primary_model_selection_summary(
                model_metrics,
                primary_model_name=primary_model_name,
                conservative_model_name=conservative_model_name,
                governance_model_name=governance_model_name,
                predictions=predictions,
                decision_yield=decision_yield,
                blocked_holdout_calibration_summary=blocked_holdout_calibration_summary,
                family_summary=family_summary,
                simplicity_summary=simplicity_summary,
                model_selection_scorecard=model_selection_scorecard,
            )
            model_selection_summary = _attach_single_model_decision_summary(
                model_selection_summary,
                single_model_official_decision,
                published_primary_model=primary_model_name,
            )
            model_selection_summary.to_csv(
                final_tables_dir / "model_selection_summary.tsv",
                sep="\t",
                index=False,
            )
            benchmark_protocol = build_benchmark_protocol_table(
                model_metrics,
                model_selection_summary,
                adaptive_gated_metrics=adaptive_gated_metrics,
                gate_consistency_audit=gate_consistency_audit,
                model_selection_scorecard=model_selection_scorecard,
            )
            benchmark_protocol.to_csv(
                final_tables_dir / "benchmark_protocol.tsv",
                sep="\t",
                index=False,
            )
            official_benchmark_panel = build_official_benchmark_panel(benchmark_protocol)
            official_benchmark_panel.to_csv(
                final_tables_dir / "official_benchmark_panel.tsv",
                sep="\t",
                index=False,
            )
            official_benchmark_context = _build_official_benchmark_context(
                model_selection_summary,
                decision_yield,
                benchmark_protocol=benchmark_protocol,
            )
            candidate_universe = _attach_official_benchmark_context(
                candidate_universe,
                official_benchmark_context,
            )
            candidate_dossiers = _attach_official_benchmark_context(
                candidate_dossiers,
                official_benchmark_context,
            )
            candidate_portfolio = _attach_official_benchmark_context(
                candidate_portfolio,
                official_benchmark_context,
            )
            candidate_portfolio = _deduplicate_backbone_rows(candidate_portfolio)
            candidate_universe.to_csv(
                final_tables_dir / "candidate_universe.tsv",
                sep="\t",
                index=False,
            )

            candidate_dossiers.to_csv(
                final_tables_dir / "candidate_dossiers.tsv",
                sep="\t",
                index=False,
            )
            candidate_portfolio.to_csv(
                final_tables_dir / "candidate_portfolio.tsv",
                sep="\t",
                index=False,
            )
            operational_risk_watchlist.to_csv(
                final_tables_dir / "operational_risk_watchlist.tsv",
                sep="\t",
                index=False,
            )
            candidate_briefs = _build_candidate_brief_table(
                candidate_portfolio,
                backbones,
                amr_consensus,
                model_selection_summary=model_selection_summary,
                decision_yield=decision_yield,
                split_year=pipeline.split_year,
            )
            candidate_case_studies = _build_candidate_case_studies(candidate_briefs, per_track=3)
            candidate_case_studies.to_csv(
                final_tables_dir / "candidate_case_studies.tsv",
                sep="\t",
                index=False,
            )
            report_overview = build_report_overview_table(
                model_selection_summary=model_selection_summary,
                decision_yield=decision_yield,
                threshold_utility_summary=threshold_utility_summary,
                candidate_portfolio=candidate_portfolio,
                candidate_case_studies=candidate_case_studies,
                false_negative_audit=false_negative_audit,
            )
            report_overview.to_csv(final_tables_dir / "report_overview.tsv", sep="\t", index=False)
            candidate_evidence_matrix = _build_candidate_evidence_matrix(
                candidate_portfolio,
                candidate_briefs,
                candidate_threshold_flip,
            )
            candidate_evidence_matrix.to_csv(
                final_tables_dir / "candidate_evidence_matrix.tsv",
                sep="\t",
                index=False,
            )
            if not candidate_briefs.empty:
                candidate_stability = coalescing_left_merge(
                    candidate_stability,
                    candidate_briefs[["backbone_id", "dominant_genus", "dominant_species"]],
                    on="backbone_id",
                )
            validate_report_artifact(
                model_selection_scorecard,
                artifact_name="model_selection_scorecard",
                required_columns=(
                    "model_name",
                    "selection_rank",
                    "selection_composite_score",
                    "decision_utility_score",
                ),
                unique_key="model_name",
            )
            validate_report_artifact(
                candidate_portfolio,
                artifact_name="candidate_portfolio",
                required_columns=(
                    "backbone_id",
                    "portfolio_track",
                    "candidate_confidence_score",
                    "multiverse_stability_score",
                ),
                unique_key="backbone_id",
                probability_columns=(
                    "candidate_confidence_score",
                    "multiverse_stability_score",
                    "bootstrap_top_10_frequency",
                    "variant_top_10_frequency",
                ),
            )
            validate_report_artifact(
                candidate_case_studies,
                artifact_name="candidate_case_studies",
                required_columns=(
                    "backbone_id",
                    "candidate_summary_en",
                    "candidate_summary_tr",
                ),
                unique_key="backbone_id",
                probability_columns=("candidate_confidence_score", "multiverse_stability_score"),
            )
            validate_report_artifact(
                threshold_utility_summary,
                artifact_name="threshold_utility_summary",
                required_columns=(
                    "model_name",
                    "optimal_threshold",
                    "optimal_threshold_utility_per_sample",
                ),
                unique_key="model_name",
                probability_columns=("optimal_threshold",),
            )
            _prune_duplicate_table_artifacts(core_dir, diag_dir, final_tables_dir.core_files)
            _prune_shadowed_report_tables(
                core_dir,
                diag_dir,
                analysis_dir,
                preserve_file_names={
                    "single_model_pareto_screen.tsv",
                    "single_model_pareto_finalists.tsv",
                    "single_model_official_decision.tsv",
                },
            )
            _write_turkish_summary(
                turkish_summary_path,
                primary_model_name=primary_model_name,
                conservative_model_name=conservative_model_name,
                model_metrics=report_model_metrics,
                candidate_briefs=candidate_briefs,
                candidate_portfolio=candidate_portfolio,
                decision_yield=decision_yield,
                model_selection_scorecard=model_selection_scorecard,
                model_selection_summary=model_selection_summary,
                knownness_summary=knownness_summary,
                knownness_matched_validation=knownness_matched_validation,
                source_balance_resampling=source_balance_resampling,
                novelty_specialist_metrics=novelty_specialist_metrics,
                adaptive_gated_metrics=adaptive_gated_metrics,
                gate_consistency_audit=gate_consistency_audit,
                secondary_outcome_performance=secondary_outcome_performance,
                weighted_country_outcome=weighted_country_outcome,
                count_outcome_audit=count_outcome_audit,
                metadata_quality_summary=metadata_quality_summary,
                operational_risk_watchlist=operational_risk_watchlist,
                confirmatory_cohort_summary=confirmatory_cohort_summary,
                false_negative_audit=false_negative_audit,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
                outcome_threshold=pipeline.min_new_countries_for_spread,
            )
            _write_executive_summary(
                executive_summary_path,
                primary_model_name=primary_model_name,
                governance_model_name=governance_model_name,
                baseline_model_name="baseline_both",
                model_metrics=report_model_metrics,
                confirmatory_cohort_summary=confirmatory_cohort_summary,
                false_negative_audit=false_negative_audit,
                candidate_case_studies=candidate_case_studies,
                rolling_temporal=rolling_temporal,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
            )
            _write_pitch_notes(
                pitch_notes_path,
                primary_model_name=primary_model_name,
                governance_model_name=governance_model_name,
                model_metrics=report_model_metrics,
                knownness_matched_validation=knownness_matched_validation,
                coefficient_stability_cv=coefficient_stability_cv,
                confirmatory_cohort_summary=confirmatory_cohort_summary,
                false_negative_audit=false_negative_audit,
            )
            _write_headline_validation_summary(
                headline_summary_path,
                headline_validation_summary,
                primary_model_name=primary_model_name,
                governance_model_name=governance_model_name,
                rolling_temporal=rolling_temporal,
                blocked_holdout_summary=blocked_holdout_summary,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
            )
            legacy_detailed_summary = context.reports_dir / "tubitak_detayli_proje_ozeti_tr.txt"
            if legacy_detailed_summary.exists():
                legacy_detailed_summary.unlink()

            if report_mode == "report-fast":
                run.note("report-fast mode: skipped heavy figure rendering.")
                figure_paths = []
            else:
                figure_paths = generate_all_figures(
                    scored=scored,
                    predictions=predictions,
                    calibration=calibration,
                    threshold_sensitivity=threshold_sensitivity,
                    model_metrics=report_model_metrics,
                    coefficient_table=coefficient_table,
                    coefficient_stability=coefficient_stability,
                    dropout_table=dropout_table,
                    candidate_stability=candidate_stability,
                    candidate_portfolio=candidate_portfolio,
                    false_negative_audit=false_negative_audit,
                    core_model_coefficients=core_model_coefficients,
                    governance_model_name=governance_model_name,
                    figures_dir=figures_dir,
                    primary_model_name=primary_model_name,
                    report_cache=report_cache,
                    report_mode=report_mode,
                )
            for figure_path in figure_paths:
                run.record_output(Path(figure_path))
            if not amrfinder_reportable:
                stale_figure = figures_dir / "amrfinder_concordance.png"
                if stale_figure.exists():
                    stale_figure.unlink()

            _write_jury_brief(
                jury_brief_path,
                primary_model_name=primary_model_name,
                conservative_model_name=conservative_model_name,
                model_metrics=report_model_metrics,
                family_summary=family_summary,
                dropout_table=dropout_table,
                scored=scored,
                candidate_portfolio=candidate_portfolio,
                decision_yield=decision_yield,
                model_selection_scorecard=model_selection_scorecard,
                model_selection_summary=model_selection_summary,
                knownness_summary=knownness_summary,
                knownness_matched_validation=knownness_matched_validation,
                source_balance_resampling=source_balance_resampling,
                novelty_specialist_metrics=novelty_specialist_metrics,
                adaptive_gated_metrics=adaptive_gated_metrics,
                gate_consistency_audit=gate_consistency_audit,
                secondary_outcome_performance=secondary_outcome_performance,
                weighted_country_outcome=weighted_country_outcome,
                count_outcome_audit=count_outcome_audit,
                metadata_quality_summary=metadata_quality_summary,
                operational_risk_watchlist=operational_risk_watchlist,
                confirmatory_cohort_summary=confirmatory_cohort_summary,
                false_negative_audit=false_negative_audit,
                blocked_holdout_summary=blocked_holdout_summary,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
                outcome_threshold=pipeline.min_new_countries_for_spread,
            )
            run.set_rows_out("top_backbones_rows", int(len(top)))
            run.set_metric("figure_count", len(figure_paths))
            write_signature_manifest(
                manifest_path,
                input_paths=[path for path in input_paths if path.exists()],
                output_paths=materialize_recorded_paths(context.root, run.output_files_written),
                source_paths=source_paths,
                metadata=cache_metadata,
            )
            report_cache.put_report(
                report_name="24_build_reports",
                report_key=report_key,
                outputs=materialize_recorded_paths(context.root, run.output_files_written),
            )
            run.set_metric("cache_hit", False)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
