#!/usr/bin/env python3
"""Build source-stratified and calibration-oriented validation tables."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    NOVELTY_SPECIALIST_FEATURES,
    NOVELTY_SPECIALIST_FIT_CONFIG,
    annotate_knownness_metadata,
    assert_feature_columns_present,
    build_coefficient_stability_table,
    build_discovery_input_contract,
    build_feature_dropout_audit,
    build_logistic_convergence_audit,
    build_standardized_coefficient_table,
    evaluate_feature_columns,
    evaluate_model_name,
    get_conservative_model_name,
    get_governance_model_name,
    get_official_model_names,
    get_primary_model_name,
    validate_discovery_input_contract,
)
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_calibration_metric_table,
    build_future_sentinel_audit,
    build_gate_consistency_audit,
    build_group_holdout_performance,
    build_knownness_audit_tables,
    build_logistic_implementation_audit,
    build_model_comparison_table,
    build_model_family_summary,
    build_model_simplicity_summary,
    build_model_subgroup_performance,
    build_negative_control_audit,
    build_permutation_null_tables,
    build_selection_adjusted_permutation_null,
    build_source_balance_resampling_table,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    materialize_recorded_paths,
    project_python_source_paths,
    write_signature_manifest,
)
from plasmid_priority.utils.geography import (
    build_country_quality_summary,
    dominant_macro_region_table,
)
from plasmid_priority.validation import (
    average_precision,
    average_precision_enrichment,
    average_precision_lift,
    bootstrap_intervals,
    brier_score,
    positive_prevalence,
    roc_auc_score,
)


def _dominant_training_value_table(
    records: pd.DataFrame,
    source_column: str,
    *,
    output_column: str | None = None,
    split_year: int = 2015,
) -> pd.DataFrame:
    training = records.loc[
        pd.to_numeric(records["resolved_year"], errors="coerce").fillna(0).astype(int) <= split_year
    ].copy()
    output_column = output_column or source_column
    if training.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])
    values = training[source_column].fillna("").astype(str).str.strip()
    nonempty = training.loc[values.ne("")].copy()
    nonempty[output_column] = values.loc[values.ne("")]
    if nonempty.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])
    counts = (
        nonempty.groupby(["backbone_id", output_column], as_index=False)
        .size()
        .sort_values(
            ["backbone_id", "size", output_column], ascending=[True, False, True], kind="mergesort"
        )
        .drop_duplicates("backbone_id", keep="first")[["backbone_id", output_column]]
        .reset_index(drop=True)
    )
    return counts


def _normalize_primary_replicon_family(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    token = re.split(r"[,;/|\s]+", text, maxsplit=1)[0]
    if token.startswith("Inc") and len(token) > 3:
        suffix = token[3:]
        for char in suffix:
            if char.isalpha():
                return f"Inc{char.upper()}"
        return "Inc"
    return re.split(r"[-_/]", token, maxsplit=1)[0]


def _dominant_training_amr_class_table(
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    *,
    split_year: int = 2015,
    output_column: str = "dominant_amr_class_train",
) -> pd.DataFrame:
    training = backbones.loc[
        pd.to_numeric(backbones["resolved_year"], errors="coerce").fillna(0).astype(int)
        <= split_year
    ].copy()
    if training.empty or amr_consensus.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])
    merged = training.merge(
        amr_consensus[["sequence_accession", "amr_drug_classes"]],
        on="sequence_accession",
        how="left",
    )
    exploded = merged.assign(
        amr_token=merged["amr_drug_classes"].fillna("").astype(str).str.split(",")
    ).explode("amr_token")
    exploded["amr_token"] = exploded["amr_token"].fillna("").astype(str).str.strip()
    exploded = exploded.loc[exploded["amr_token"].ne(""), ["backbone_id", "amr_token"]].copy()
    if exploded.empty:
        return pd.DataFrame(columns=["backbone_id", output_column])
    counts = (
        exploded.groupby(["backbone_id", "amr_token"], as_index=False)
        .size()
        .sort_values(
            ["backbone_id", "size", "amr_token"], ascending=[True, False, False], kind="mergesort"
        )
        .drop_duplicates("backbone_id", keep="first")
        .rename(columns={"amr_token": output_column})[["backbone_id", output_column]]
        .reset_index(drop=True)
    )
    counts["backbone_id"] = counts["backbone_id"].astype(str)
    return counts


def _recommended_cv_splits(frame: pd.DataFrame, *, default_splits: int = 5) -> int | None:
    labels = frame["spread_label"].dropna().astype(int)
    if labels.empty or labels.nunique() < 2:
        return None
    max_splits = int(min(default_splits, labels.sum(), (labels == 0).sum()))
    return max_splits if max_splits >= 2 else None


def _top_k_precision_recall(
    y_true: np.ndarray, y_score: np.ndarray, *, top_k: int
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or top_k <= 0:
        return float("nan"), float("nan")
    top_k = min(int(top_k), len(y_true))
    order = np.argsort(-y_score, kind="mergesort")[:top_k]
    selected = y_true[order]
    positives = max(int((y_true == 1).sum()), 1)
    true_positives = int((selected == 1).sum())
    return float(true_positives / top_k), float(true_positives / positives)


def _summarize_prediction_frame(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    model_name: str,
    extra: dict[str, object] | None = None,
) -> pd.DataFrame:
    valid = frame.loc[frame["spread_label"].notna() & frame[prediction_column].notna()].copy()
    if valid.empty or valid["spread_label"].astype(int).nunique() < 2:
        row = {"model_name": model_name, "status": "skipped_insufficient_label_variation"}
        if extra:
            row.update(extra)
        return pd.DataFrame([row])
    y = valid["spread_label"].astype(int).to_numpy()
    preds = valid[prediction_column].astype(float).to_numpy()
    precision_at_10, recall_at_10 = _top_k_precision_recall(y, preds, top_k=10)
    precision_at_25, recall_at_25 = _top_k_precision_recall(y, preds, top_k=25)
    metrics = {
        "roc_auc": roc_auc_score(y, preds),
        "average_precision": average_precision(y, preds),
        "positive_prevalence": positive_prevalence(y),
        "average_precision_lift": average_precision_lift(y, preds),
        "average_precision_enrichment": average_precision_enrichment(y, preds),
        "brier_score": brier_score(y, preds),
        "precision_at_top_10": precision_at_10,
        "recall_at_top_10": recall_at_10,
        "precision_at_top_25": precision_at_25,
        "recall_at_top_25": recall_at_25,
        "n_backbones": int(len(valid)),
        "n_positive": int((y == 1).sum()),
    }
    intervals = bootstrap_intervals(
        y,
        preds,
        {
            "roc_auc": roc_auc_score,
            "average_precision": average_precision,
            "brier_score": brier_score,
        },
    )
    metrics["roc_auc_ci_lower"] = intervals["roc_auc"]["lower"]
    metrics["roc_auc_ci_upper"] = intervals["roc_auc"]["upper"]
    metrics["average_precision_ci_lower"] = intervals["average_precision"]["lower"]
    metrics["average_precision_ci_upper"] = intervals["average_precision"]["upper"]
    metrics["brier_score_ci_lower"] = intervals["brier_score"]["lower"]
    metrics["brier_score_ci_upper"] = intervals["brier_score"]["upper"]
    row = {"model_name": model_name, "status": "ok"}
    if extra:
        row.update(extra)
    row.update(metrics)
    return pd.DataFrame([row])


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.root / "data/scores/backbone_scored.tsv"
    backbones_path = context.root / "data/silver/plasmid_backbones.tsv"
    amr_consensus_path = context.root / "data/silver/plasmid_amr_consensus.tsv"
    metrics_path = context.root / "data/analysis/module_a_metrics.json"
    module_a_predictions = context.root / "data/analysis/module_a_predictions.tsv"
    config_path = context.root / "config.yaml"
    manifest_path = context.root / "data/analysis/21_run_validation.manifest.json"
    source_output = context.root / "data/analysis/source_stratified_consistency.tsv"
    calibration_output = context.root / "data/analysis/calibration_table.tsv"
    subgroup_output = context.root / "data/analysis/model_subgroup_performance.tsv"
    family_summary_output = context.root / "data/analysis/model_family_summary.tsv"
    comparison_output = context.root / "data/analysis/model_comparison_summary.tsv"
    calibration_metrics_output = context.root / "data/analysis/calibration_metrics.tsv"
    coefficients_output = context.root / "data/analysis/primary_model_coefficients.tsv"
    coefficient_stability_output = (
        context.root / "data/analysis/primary_model_coefficient_stability.tsv"
    )
    dropout_output = context.root / "data/analysis/feature_dropout_importance.tsv"
    source_balance_resampling_output = context.root / "data/analysis/source_balance_resampling.tsv"
    group_holdout_output = context.root / "data/analysis/group_holdout_performance.tsv"
    permutation_detail_output = context.root / "data/analysis/permutation_null_distribution.tsv"
    permutation_summary_output = context.root / "data/analysis/permutation_null_summary.tsv"
    selection_adjusted_permutation_detail_output = (
        context.root / "data/analysis/selection_adjusted_permutation_null_distribution.tsv"
    )
    selection_adjusted_permutation_summary_output = (
        context.root / "data/analysis/selection_adjusted_permutation_null_summary.tsv"
    )
    negative_control_output = context.root / "data/analysis/negative_control_audit.tsv"
    logistic_impl_output = context.root / "data/analysis/logistic_implementation_audit.tsv"
    logistic_convergence_output = context.root / "data/analysis/logistic_convergence_audit.tsv"
    simplicity_output = context.root / "data/analysis/model_simplicity_summary.tsv"
    knownness_summary_output = context.root / "data/analysis/knownness_audit_summary.tsv"
    knownness_strata_output = context.root / "data/analysis/knownness_stratified_performance.tsv"
    country_quality_output = context.root / "data/analysis/country_quality_summary.tsv"
    purity_atlas_output = context.root / "data/analysis/backbone_purity_atlas.tsv"
    assignment_confidence_output = context.root / "data/analysis/assignment_confidence_summary.tsv"
    incremental_value_output = context.root / "data/analysis/incremental_value_over_baseline.tsv"
    novelty_specialist_metrics_output = (
        context.root / "data/analysis/novelty_specialist_metrics.tsv"
    )
    novelty_specialist_predictions_output = (
        context.root / "data/analysis/novelty_specialist_predictions.tsv"
    )
    adaptive_gated_metrics_output = context.root / "data/analysis/adaptive_gated_metrics.tsv"
    adaptive_gated_predictions_output = (
        context.root / "data/analysis/adaptive_gated_predictions.tsv"
    )
    gate_consistency_output = context.root / "data/analysis/gate_consistency_audit.tsv"
    future_sentinel_output = context.root / "data/analysis/future_sentinel_audit.tsv"
    ensure_directory(source_output.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/21_run_validation.py",
    )
    input_paths = [
        scored_path,
        backbones_path,
        amr_consensus_path,
        metrics_path,
        module_a_predictions,
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

    with ManagedScriptRun(context, "21_run_validation") as run:
        for path in (
            scored_path,
            backbones_path,
            amr_consensus_path,
            metrics_path,
            module_a_predictions,
            config_path,
        ):
            run.record_input(path)
        run.record_output(source_output)
        run.record_output(calibration_output)
        run.record_output(subgroup_output)
        run.record_output(family_summary_output)
        run.record_output(comparison_output)
        run.record_output(calibration_metrics_output)
        run.record_output(coefficients_output)
        run.record_output(coefficient_stability_output)
        run.record_output(dropout_output)
        run.record_output(source_balance_resampling_output)
        run.record_output(group_holdout_output)
        run.record_output(permutation_detail_output)
        run.record_output(permutation_summary_output)
        run.record_output(selection_adjusted_permutation_detail_output)
        run.record_output(selection_adjusted_permutation_summary_output)
        run.record_output(negative_control_output)
        run.record_output(logistic_impl_output)
        run.record_output(logistic_convergence_output)
        run.record_output(simplicity_output)
        run.record_output(knownness_summary_output)
        run.record_output(knownness_strata_output)
        run.record_output(country_quality_output)
        run.record_output(purity_atlas_output)
        run.record_output(assignment_confidence_output)
        run.record_output(incremental_value_output)
        run.record_output(novelty_specialist_metrics_output)
        run.record_output(novelty_specialist_predictions_output)
        run.record_output(adaptive_gated_metrics_output)
        run.record_output(adaptive_gated_predictions_output)
        run.record_output(gate_consistency_output)
        run.record_output(future_sentinel_output)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached validation outputs.")
            run.set_metric("cache_hit", True)
            return 0

        scored = read_tsv(scored_path).copy()
        newest_upstream_mtime = scored_path.stat().st_mtime
        stale_outputs = [
            str(path.name)
            for path in (metrics_path, module_a_predictions)
            if path.exists() and path.stat().st_mtime < newest_upstream_mtime
        ]
        if stale_outputs:
            stale_text = ", ".join(f"`{name}`" for name in stale_outputs)
            raise RuntimeError(
                f"Module A outputs {stale_text} are older than `backbone_scored.tsv`. "
                "Rerun `python3 scripts/16_run_module_A.py` before validation."
            )
        required_columns = [
            column
            for model_name in [
                "baseline_both",
                "bio_clean_priority",
                "natural_auc_priority",
                "knownness_robust_priority",
                "host_transfer_synergy_priority",
            ]
            for column in MODULE_A_FEATURE_SETS[model_name]
        ]
        required_columns.extend(NOVELTY_SPECIALIST_FEATURES)
        assert_feature_columns_present(
            scored,
            required_columns,
            label="Validation score input",
        )
        validate_discovery_input_contract(
            scored,
            model_names=get_official_model_names(MODULE_A_FEATURE_SETS.keys()),
            contract=build_discovery_input_contract(int(context.pipeline_settings.split_year)),
            label="Validation score input",
        )
        backbones = read_tsv(
            backbones_path,
            usecols=[
                "backbone_id",
                "resolved_year",
                "genus",
                "country",
                "TAXONOMY_family",
                "TAXONOMY_order",
                "predicted_mobility",
                "primary_replicon",
                "sequence_accession",
            ],
        )
        backbones["primary_replicon_family"] = (
            backbones["primary_replicon"]
            .fillna("")
            .astype(str)
            .map(_normalize_primary_replicon_family)
        )
        amr_consensus = read_tsv(
            amr_consensus_path,
            usecols=["sequence_accession", "amr_drug_classes"],
        )
        pipeline = context.pipeline_settings
        scored = scored.assign(
            dominant_source=scored["refseq_share_train"]
            .ge(0.5)
            .map({True: "refseq_leaning", False: "insd_leaning"})
        )
        dominant_genus = _dominant_training_value_table(
            backbones, "genus", output_column="dominant_genus_train"
        )
        dominant_region = dominant_macro_region_table(
            backbones,
            split_year=pipeline.split_year,
        )
        dominant_host_family = _dominant_training_value_table(
            backbones, "TAXONOMY_family", output_column="dominant_host_family_train"
        )
        dominant_host_order = _dominant_training_value_table(
            backbones, "TAXONOMY_order", output_column="dominant_host_order_train"
        )
        dominant_mobility = _dominant_training_value_table(
            backbones, "predicted_mobility", output_column="dominant_mobility_train"
        )
        dominant_primary_replicon = _dominant_training_value_table(
            backbones, "primary_replicon", output_column="dominant_primary_replicon_train"
        )
        dominant_primary_replicon_family = _dominant_training_value_table(
            backbones,
            "primary_replicon_family",
            output_column="dominant_primary_replicon_family_train",
        )
        dominant_amr_class = _dominant_training_amr_class_table(
            backbones,
            amr_consensus,
            split_year=pipeline.split_year,
        )
        scored = scored.merge(dominant_genus, on="backbone_id", how="left")
        scored = scored.merge(dominant_region, on="backbone_id", how="left")
        scored = scored.merge(dominant_host_family, on="backbone_id", how="left")
        scored = scored.merge(dominant_host_order, on="backbone_id", how="left")
        scored = scored.merge(dominant_mobility, on="backbone_id", how="left")
        scored = scored.merge(dominant_primary_replicon, on="backbone_id", how="left")
        scored = scored.merge(dominant_primary_replicon_family, on="backbone_id", how="left")
        scored = scored.merge(dominant_amr_class, on="backbone_id", how="left")
        country_quality = build_country_quality_summary(
            backbones,
            split_year=pipeline.split_year,
        )
        country_quality.to_csv(country_quality_output, sep="\t", index=False)
        purity_columns = [
            "backbone_id",
            "member_count_train",
            "n_countries_train",
            "spread_label",
            "backbone_purity_score",
            "genus_purity_train",
            "family_purity_train",
            "mobility_purity_train",
            "replicon_purity_train",
            "mean_n_replicon_types_train",
            "multi_replicon_fraction_train",
            "primary_replicon_diversity_train",
            "replicon_architecture_norm",
            "assignment_primary_fraction",
            "assignment_confidence_score",
            "mash_neighbor_distance_train_mean",
        ]
        purity_atlas = scored[
            [column for column in purity_columns if column in scored.columns]
        ].copy()
        purity_atlas.to_csv(purity_atlas_output, sep="\t", index=False)
        if "assignment_confidence_score" in scored.columns:
            assignment_summary = scored[
                [
                    "backbone_id",
                    "assignment_confidence_score",
                    "member_count_train",
                    "n_countries_train",
                    "spread_label",
                ]
            ].copy()
            assignment_summary["assignment_confidence_score"] = assignment_summary[
                "assignment_confidence_score"
            ].fillna(0.0)
            assignment_summary["member_count_train"] = (
                assignment_summary["member_count_train"].fillna(0).astype(int)
            )
            assignment_summary["n_countries_train"] = (
                assignment_summary["n_countries_train"].fillna(0).astype(int)
            )
            assignment_summary["eligible_for_outcome"] = assignment_summary["spread_label"].notna()
            assignment_summary["assignment_confidence_tier"] = pd.cut(
                assignment_summary["assignment_confidence_score"].fillna(0.0),
                bins=[-0.01, 0.60, 0.90, 1.01],
                labels=["fallback_leaning", "mixed", "primary_cluster_leaning"],
            ).astype(str)
            assignment_summary = assignment_summary.groupby(
                "assignment_confidence_tier", as_index=False
            ).agg(
                n_backbones=("backbone_id", "nunique"),
                n_eligible_backbones=("eligible_for_outcome", "sum"),
                mean_assignment_confidence=("assignment_confidence_score", "mean"),
                mean_member_count_train=("member_count_train", "mean"),
                mean_n_countries_train=("n_countries_train", "mean"),
                positive_prevalence=("spread_label", "mean"),
            )
            assignment_summary["positive_prevalence"] = np.where(
                assignment_summary["n_eligible_backbones"].fillna(0).astype(int) > 0,
                assignment_summary["positive_prevalence"],
                0.0,
            )
        else:
            assignment_summary = pd.DataFrame(
                columns=[
                    "assignment_confidence_tier",
                    "n_backbones",
                    "n_eligible_backbones",
                    "mean_assignment_confidence",
                    "mean_member_count_train",
                    "mean_n_countries_train",
                    "positive_prevalence",
                ]
            )
        assignment_summary.to_csv(assignment_confidence_output, sep="\t", index=False)
        source_table = (
            scored.groupby("dominant_source")
            .agg(
                n_backbones=("backbone_id", "nunique"),
                mean_priority_index=("priority_index", "mean"),
                mean_spread_label=("spread_label", "mean"),
                mean_coherence=("coherence_score", "mean"),
            )
            .reset_index()
        )
        source_table.to_csv(source_output, sep="\t", index=False)

        predictions = read_tsv(module_a_predictions)
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            model_metrics = pd.DataFrame(
                [
                    {"model_name": name, **metrics}
                    for name, metrics in payload.items()
                    if isinstance(metrics, dict) and "roc_auc" in metrics
                ]
            )
        primary_model_name = get_primary_model_name(predictions["model_name"].unique().tolist())
        conservative_model_name = get_conservative_model_name(
            predictions["model_name"].unique().tolist()
        )
        governance_model_name = get_governance_model_name(
            predictions["model_name"].unique().tolist()
        )
        knownness_meta = annotate_knownness_metadata(
            scored.loc[scored["spread_label"].notna()].copy()
        )

        novelty_specialist_rows: list[dict[str, object]] = []
        novelty_cohorts = {
            "lower_half_knownness": knownness_meta["knownness_half"].astype(str).eq("lower_half"),
        }
        novelty_comparison_models: list[str] = []
        for model_name in [
            primary_model_name,
            conservative_model_name,
            "natural_auc_priority",
            "host_transfer_synergy_priority",
            "baseline_both",
        ]:
            if model_name in MODULE_A_FEATURE_SETS and model_name not in novelty_comparison_models:
                novelty_comparison_models.append(model_name)
        for cohort_name, cohort_mask in novelty_cohorts.items():
            cohort = knownness_meta.loc[cohort_mask].copy()
            n_splits = _recommended_cv_splits(cohort)
            if n_splits is None:
                novelty_specialist_rows.append(
                    {
                        "cohort_name": cohort_name,
                        "model_name": "novelty_specialist_priority",
                        "feature_family": "novelty_specialist",
                        "n_features": int(len(NOVELTY_SPECIALIST_FEATURES)),
                        "n_backbones": int(len(cohort)),
                        "n_positive": int(cohort["spread_label"].fillna(0).sum()),
                        "status": "skipped_insufficient_label_variation",
                    }
                )
                continue
            for model_name in novelty_comparison_models:
                result = evaluate_model_name(
                    cohort,
                    model_name=model_name,
                    n_splits=n_splits,
                    n_repeats=5,
                    seed=42,
                    include_ci=False,
                )
                row = {
                    "cohort_name": cohort_name,
                    "model_name": model_name,
                    "feature_family": "named_model",
                    "n_features": int(len(MODULE_A_FEATURE_SETS[model_name])),
                    "n_splits": int(n_splits),
                    "n_repeats": 5,
                    "status": "ok",
                }
                row.update(result.metrics)
                novelty_specialist_rows.append(row)

            specialist_result = evaluate_feature_columns(
                cohort,
                columns=NOVELTY_SPECIALIST_FEATURES,
                label="novelty_specialist_priority",
                n_splits=n_splits,
                n_repeats=5,
                seed=42,
                fit_config=NOVELTY_SPECIALIST_FIT_CONFIG,
                include_ci=False,
            )
            specialist_row = {
                "cohort_name": cohort_name,
                "model_name": "novelty_specialist_priority",
                "feature_family": "novelty_specialist",
                "n_features": int(len(NOVELTY_SPECIALIST_FEATURES)),
                "n_splits": int(n_splits),
                "n_repeats": 5,
                "status": "ok",
            }
            specialist_row.update(specialist_result.metrics)
            novelty_specialist_rows.append(specialist_row)
        novelty_specialist_metrics = pd.DataFrame(novelty_specialist_rows)
        novelty_specialist_metrics.to_csv(novelty_specialist_metrics_output, sep="\t", index=False)

        lower_half_train = knownness_meta.loc[
            knownness_meta["knownness_half"].astype(str).eq("lower_half")
        ].copy()
        lower_half_splits = _recommended_cv_splits(lower_half_train)

        novelty_specialist_oof = pd.DataFrame(
            columns=["backbone_id", "novelty_specialist_prediction"]
        )

        if lower_half_splits is not None:
            specialist_oof_result = evaluate_feature_columns(
                lower_half_train,
                columns=NOVELTY_SPECIALIST_FEATURES,
                label="novelty_specialist_priority",
                n_splits=lower_half_splits,
                n_repeats=5,
                seed=42,
                fit_config=NOVELTY_SPECIALIST_FIT_CONFIG,
                include_ci=False,
            )
            novelty_specialist_oof = specialist_oof_result.predictions.rename(
                columns={"oof_prediction": "novelty_specialist_prediction"}
            )[["backbone_id", "novelty_specialist_prediction"]].copy()

        novelty_specialist_predictions = knownness_meta[
            [
                "backbone_id",
                "spread_label",
                "knownness_score",
                "knownness_half",
                "knownness_quartile",
            ]
        ].copy()
        novelty_specialist_predictions = novelty_specialist_predictions.merge(
            novelty_specialist_oof,
            on="backbone_id",
            how="left",
        )
        novelty_specialist_predictions["model_name"] = "novelty_specialist_priority"
        novelty_specialist_predictions["training_cohort"] = "lower_half_knownness"
        novelty_specialist_predictions["prediction_protocol"] = "oof_lower_half_cv"
        for model_name, column_name in [
            (primary_model_name, "primary_model_oof_prediction"),
            ("baseline_both", "baseline_both_oof_prediction"),
            ("natural_auc_priority", "natural_auc_oof_prediction"),
            ("host_transfer_synergy_priority", "host_transfer_synergy_oof_prediction"),
        ]:
            if model_name in predictions["model_name"].astype(str).unique():
                novelty_specialist_predictions = novelty_specialist_predictions.merge(
                    predictions.loc[
                        predictions["model_name"] == model_name,
                        ["backbone_id", "oof_prediction"],
                    ].rename(columns={"oof_prediction": column_name}),
                    on="backbone_id",
                    how="left",
                )
        if {"primary_model_oof_prediction", "baseline_both_oof_prediction"} <= set(
            novelty_specialist_predictions.columns
        ):
            novelty_specialist_predictions["novelty_margin_vs_baseline"] = np.nan
            novelty_margin_mask = (
                novelty_specialist_predictions["primary_model_oof_prediction"].notna()
                & novelty_specialist_predictions["baseline_both_oof_prediction"].notna()
            )
            novelty_specialist_predictions.loc[
                novelty_margin_mask, "novelty_margin_vs_baseline"
            ] = novelty_specialist_predictions.loc[
                novelty_margin_mask, "primary_model_oof_prediction"
            ].astype(float) - novelty_specialist_predictions.loc[
                novelty_margin_mask, "baseline_both_oof_prediction"
            ].astype(float)
        novelty_specialist_predictions.to_csv(
            novelty_specialist_predictions_output, sep="\t", index=False
        )

        adaptive_gated_predictions = pd.DataFrame(
            columns=[
                "backbone_id",
                "adaptive_prediction",
                "spread_label",
                "knownness_score",
                "knownness_half",
                "knownness_quartile",
                "model_name",
                "base_model_name",
                "specialist_model_name",
                "gating_rule",
                "prediction_source",
                "specialist_weight_lower_half",
                "base_oof_prediction",
                "novelty_specialist_prediction",
                "upper_half_route_prediction",
                "lower_half_route_prediction",
            ]
        )
        adaptive_gated_metrics = pd.DataFrame(
            columns=[
                "model_name",
                "status",
                "base_model_name",
                "specialist_model_name",
                "gating_rule",
                "specialist_weight_lower_half",
            ]
        )
        gate_consistency_audit = pd.DataFrame()
        if not novelty_specialist_predictions.empty:
            adaptive_frames: list[pd.DataFrame] = []
            adaptive_metrics_frames: list[pd.DataFrame] = []
            specialist_lookup = novelty_specialist_predictions[
                [
                    "backbone_id",
                    "novelty_specialist_prediction",
                ]
            ].copy()
            for base_model_name, adaptive_model_name, specialist_weight, gating_rule in [
                (
                    "natural_auc_priority",
                    "adaptive_natural_priority",
                    1.0,
                    "lower_half_specialist_switch",
                ),
                (
                    "knownness_robust_priority",
                    "adaptive_knownness_robust_priority",
                    1.0,
                    "lower_half_specialist_switch",
                ),
                (
                    "knownness_robust_priority",
                    "adaptive_knownness_blend_priority",
                    0.5,
                    "lower_half_specialist_blend_0p50",
                ),
                (
                    "support_calibrated_priority",
                    "adaptive_support_calibrated_blend_priority",
                    0.5,
                    "lower_half_specialist_blend_0p50",
                ),
                (
                    "support_synergy_priority",
                    "adaptive_support_synergy_blend_priority",
                    0.5,
                    "lower_half_specialist_blend_0p50",
                ),
                (
                    "host_transfer_synergy_priority",
                    "adaptive_host_transfer_synergy_blend_priority",
                    0.7,
                    "lower_half_specialist_blend_0p70",
                ),
                (
                    "threat_architecture_priority",
                    "adaptive_threat_architecture_blend_priority",
                    0.5,
                    "lower_half_specialist_blend_0p50",
                ),
            ]:
                if base_model_name not in predictions["model_name"].astype(str).unique():
                    continue
                base_predictions = predictions.loc[
                    predictions["model_name"] == base_model_name,
                    ["backbone_id", "oof_prediction", "spread_label"],
                ].rename(columns={"oof_prediction": "base_oof_prediction"})
                frame = base_predictions.merge(
                    knownness_meta[
                        [
                            "backbone_id",
                            "knownness_score",
                            "knownness_half",
                            "knownness_quartile",
                        ]
                    ],
                    on="backbone_id",
                    how="left",
                    validate="1:1",
                )
                frame = frame.merge(
                    specialist_lookup,
                    on="backbone_id",
                    how="left",
                )
                lower_half_mask = frame["knownness_half"].astype(str).eq("lower_half")
                frame["upper_half_route_prediction"] = frame["base_oof_prediction"].astype(float)
                frame["lower_half_route_prediction"] = specialist_weight * frame[
                    "novelty_specialist_prediction"
                ].fillna(frame["base_oof_prediction"]).astype(float) + (
                    1.0 - specialist_weight
                ) * frame["base_oof_prediction"].astype(float)
                frame["adaptive_prediction"] = np.where(
                    lower_half_mask,
                    frame["lower_half_route_prediction"],
                    frame["upper_half_route_prediction"],
                )
                frame["model_name"] = adaptive_model_name
                frame["base_model_name"] = base_model_name
                frame["specialist_model_name"] = "novelty_specialist_priority"
                frame["gating_rule"] = gating_rule
                frame["specialist_weight_lower_half"] = float(specialist_weight)
                frame["prediction_source"] = np.where(
                    lower_half_mask & (specialist_weight >= 0.999),
                    "novelty_specialist_priority",
                    np.where(
                        lower_half_mask & (specialist_weight > 0.0),
                        f"blend_{specialist_weight:.2f}_{base_model_name}_novelty_specialist_priority",
                        base_model_name,
                    ),
                )
                frame["prediction_source"] = np.where(
                    lower_half_mask & (specialist_weight <= 0.0),
                    base_model_name,
                    frame["prediction_source"],
                )
                adaptive_frames.append(frame.copy())
                adaptive_metrics_frames.append(
                    _summarize_prediction_frame(
                        frame,
                        prediction_column="adaptive_prediction",
                        model_name=adaptive_model_name,
                        extra={
                            "base_model_name": base_model_name,
                            "specialist_model_name": "novelty_specialist_priority",
                            "gating_rule": gating_rule,
                            "specialist_weight_lower_half": float(specialist_weight),
                        },
                    )
                )
            if adaptive_frames:
                adaptive_gated_predictions = pd.concat(adaptive_frames, ignore_index=True)
            if adaptive_metrics_frames:
                adaptive_gated_metrics = pd.concat(adaptive_metrics_frames, ignore_index=True)
            if not adaptive_gated_predictions.empty:
                gate_consistency_audit = build_gate_consistency_audit(adaptive_gated_predictions)
        adaptive_gated_predictions.to_csv(adaptive_gated_predictions_output, sep="\t", index=False)
        adaptive_gated_metrics.to_csv(adaptive_gated_metrics_output, sep="\t", index=False)
        gate_consistency_audit.to_csv(gate_consistency_output, sep="\t", index=False)

        primary = predictions.loc[predictions["model_name"] == primary_model_name].copy()
        primary["prediction_bin"] = pd.qcut(primary["oof_prediction"], q=8, duplicates="drop")
        calibration = (
            primary.groupby("prediction_bin")
            .agg(
                mean_prediction=("oof_prediction", "mean"),
                observed_rate=("spread_label", "mean"),
                n_backbones=("backbone_id", "nunique"),
            )
            .reset_index()
        )
        calibration = calibration.sort_values("mean_prediction").reset_index(drop=True)
        calibration["prediction_bin_index"] = calibration.index + 1
        z = 1.96
        n = calibration["n_backbones"].clip(lower=1).astype(float)
        p = calibration["observed_rate"].clip(lower=0.0, upper=1.0).astype(float)
        denominator = 1.0 + (z**2 / n)
        center = (p + (z**2 / (2.0 * n))) / denominator
        margin = z * (((p * (1.0 - p) / n) + (z**2 / (4.0 * (n**2)))) ** 0.5) / denominator
        calibration["observed_rate_se"] = (p * (1.0 - p) / n) ** 0.5
        calibration["observed_rate_ci_lower"] = (center - margin).clip(lower=0.0)
        calibration["observed_rate_ci_upper"] = (center + margin).clip(upper=1.0)
        calibration["absolute_calibration_gap"] = (
            calibration["observed_rate"] - calibration["mean_prediction"]
        ).abs()
        calibration.to_csv(calibration_output, sep="\t", index=False)
        family_summary = build_model_family_summary(model_metrics)
        family_summary.to_csv(family_summary_output, sep="\t", index=False)

        subgroup_models = []
        for model_name in [
            primary_model_name,
            conservative_model_name,
            governance_model_name,
            "knownness_robust_priority",
            "host_transfer_synergy_priority",
            "ecology_clinical_priority",
            "full_priority",
            "baseline_both",
            "baseline_country_count",
            "source_only",
        ]:
            if model_name not in subgroup_models:
                subgroup_models.append(model_name)
        subgroup_performance = build_model_subgroup_performance(
            predictions, scored, model_names=subgroup_models
        )
        subgroup_performance.to_csv(subgroup_output, sep="\t", index=False)

        calibration_metrics = build_calibration_metric_table(
            predictions, model_names=subgroup_models
        )
        calibration_metrics.to_csv(calibration_metrics_output, sep="\t", index=False)

        comparison_models = [
            conservative_model_name,
            "host_transfer_synergy_priority",
            "full_priority",
            "baseline_both",
            "baseline_country_count",
            "source_only",
        ]
        comparison_table = build_model_comparison_table(
            predictions,
            primary_model_name=primary_model_name,
            comparison_model_names=comparison_models,
        )
        comparison_table.to_csv(comparison_output, sep="\t", index=False)
        incremental_targets = [
            model_name
            for model_name in [
                "baseline_both",
                conservative_model_name,
                "natural_auc_priority",
                "support_synergy_priority",
                "host_transfer_synergy_priority",
            ]
            if model_name in predictions["model_name"].astype(str).unique()
            and model_name != primary_model_name
        ]
        if incremental_targets:
            incremental_value = build_model_comparison_table(
                predictions,
                primary_model_name=primary_model_name,
                comparison_model_names=incremental_targets,
            )
        else:
            incremental_value = pd.DataFrame()
        incremental_value.to_csv(incremental_value_output, sep="\t", index=False)

        primary_columns = MODULE_A_FEATURE_SETS[primary_model_name]
        coefficient_table = build_standardized_coefficient_table(
            scored,
            model_name=primary_model_name,
            columns=primary_columns,
        )
        coefficient_table.to_csv(coefficients_output, sep="\t", index=False)

        coefficient_stability = build_coefficient_stability_table(
            scored,
            model_name=primary_model_name,
            columns=primary_columns,
        )
        coefficient_stability.to_csv(coefficient_stability_output, sep="\t", index=False)

        dropout_table = build_feature_dropout_audit(
            scored,
            model_name=primary_model_name,
            columns=primary_columns,
            n_jobs=min(8, os.cpu_count() or 1),
        )
        dropout_table.to_csv(dropout_output, sep="\t", index=False)

        source_balance_resampling = build_source_balance_resampling_table(
            scored,
            model_name=primary_model_name,
            n_jobs=min(8, os.cpu_count() or 1),
        )
        source_balance_resampling.to_csv(source_balance_resampling_output, sep="\t", index=False)

        holdout_models = [
            model_name
            for model_name in model_metrics["model_name"].astype(str).tolist()
            if model_name in MODULE_A_FEATURE_SETS
        ]
        group_holdout = build_group_holdout_performance(
            scored,
            model_names=holdout_models,
            group_columns=[
                "dominant_source",
                "dominant_region_train",
                "dominant_genus_train",
                "dominant_host_family_train",
                "dominant_host_order_train",
                "dominant_mobility_train",
                "dominant_primary_replicon_train",
                "dominant_primary_replicon_family_train",
                "dominant_amr_class_train",
            ],
            min_group_size=25,
            max_groups_per_column=8,
            n_jobs=min(8, os.cpu_count() or 1),
        )
        group_holdout.to_csv(group_holdout_output, sep="\t", index=False)

        permutation_models = []
        for model_name in [
            primary_model_name,
            conservative_model_name,
            "full_priority",
            "baseline_both",
            "baseline_country_count",
            "source_only",
        ]:
            if model_name not in permutation_models:
                permutation_models.append(model_name)
        permutation_detail, permutation_summary = build_permutation_null_tables(
            predictions,
            model_names=permutation_models,
            n_permutations=2000,
        )
        permutation_detail.to_csv(permutation_detail_output, sep="\t", index=False)
        permutation_summary.to_csv(permutation_summary_output, sep="\t", index=False)

        official_model_names = get_official_model_names(
            [
                primary_model_name,
                get_governance_model_name(model_metrics["model_name"].astype(str).tolist()),
                "baseline_both",
            ]
        )
        selection_adjusted_detail, selection_adjusted_summary = (
            build_selection_adjusted_permutation_null(
                scored,
                model_names=list(official_model_names),
                primary_model_name=primary_model_name,
                n_permutations=200,
                n_splits=5,
                n_repeats=5,
                seed=42,
            )
        )
        selection_adjusted_detail.to_csv(
            selection_adjusted_permutation_detail_output, sep="\t", index=False
        )
        selection_adjusted_summary.to_csv(
            selection_adjusted_permutation_summary_output, sep="\t", index=False
        )

        negative_control = build_negative_control_audit(
            scored,
            primary_model_name=primary_model_name,
            n_repeats=5,
        )
        negative_control.to_csv(negative_control_output, sep="\t", index=False)
        future_sentinel = build_future_sentinel_audit(
            scored,
            predictions=predictions,
            primary_model_name=primary_model_name,
            model_names=list(official_model_names),
        )
        future_sentinel.to_csv(future_sentinel_output, sep="\t", index=False)

        logistic_impl = build_logistic_implementation_audit(
            scored,
            model_name=primary_model_name,
            columns=primary_columns,
        )
        logistic_impl.to_csv(logistic_impl_output, sep="\t", index=False)
        convergence_models = []
        for model_name in [
            primary_model_name,
            conservative_model_name,
            "baseline_both",
            "source_only",
        ]:
            if model_name not in convergence_models:
                convergence_models.append(model_name)
        logistic_convergence = build_logistic_convergence_audit(
            scored,
            model_names=convergence_models,
        )
        logistic_convergence.to_csv(logistic_convergence_output, sep="\t", index=False)

        simplicity_summary = build_model_simplicity_summary(
            model_metrics,
            predictions,
            primary_model_name=primary_model_name,
            conservative_model_name=conservative_model_name,
        )
        simplicity_summary.to_csv(simplicity_output, sep="\t", index=False)

        knownness_summary, knownness_strata = build_knownness_audit_tables(
            predictions,
            scored,
            primary_model_name=primary_model_name,
            baseline_model_name="baseline_both",
        )
        knownness_summary.to_csv(knownness_summary_output, sep="\t", index=False)
        knownness_strata.to_csv(knownness_strata_output, sep="\t", index=False)
        run.set_rows_out("source_validation_rows", int(len(source_table)))
        run.set_rows_out("calibration_rows", int(len(calibration)))
        run.set_rows_out("subgroup_validation_rows", int(len(subgroup_performance)))
        run.set_rows_out("dropout_rows", int(len(dropout_table)))
        run.set_rows_out("source_balance_resampling_rows", int(len(source_balance_resampling)))
        run.set_rows_out("group_holdout_rows", int(len(group_holdout)))
        run.set_rows_out("permutation_summary_rows", int(len(permutation_summary)))
        run.set_rows_out("negative_control_rows", int(len(negative_control)))
        run.set_rows_out("future_sentinel_rows", int(len(future_sentinel)))
        run.set_rows_out("logistic_impl_rows", int(len(logistic_impl)))
        run.set_rows_out("logistic_convergence_rows", int(len(logistic_convergence)))
        run.set_rows_out("backbone_purity_atlas_rows", int(len(purity_atlas)))
        run.set_rows_out("assignment_confidence_summary_rows", int(len(assignment_summary)))
        run.set_rows_out("incremental_value_rows", int(len(incremental_value)))
        run.set_rows_out("novelty_specialist_metrics_rows", int(len(novelty_specialist_metrics)))
        run.set_rows_out(
            "novelty_specialist_prediction_rows", int(len(novelty_specialist_predictions))
        )
        run.set_rows_out("adaptive_gated_metrics_rows", int(len(adaptive_gated_metrics)))
        run.set_rows_out("adaptive_gated_prediction_rows", int(len(adaptive_gated_predictions)))
        run.set_rows_out("knownness_strata_rows", int(len(knownness_strata)))
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
