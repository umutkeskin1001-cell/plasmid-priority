#!/usr/bin/env python3
"""Run lightweight sensitivity analyses on the scored backbone table."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import build_amr_consensus
from plasmid_priority.backbone import (
    assign_backbone_ids,
    assign_backbone_ids_training_only,
    compute_backbone_coherence,
)
from plasmid_priority.config import (
    DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
    build_context,
    context_config_paths,
)
from plasmid_priority.features import (
    build_backbone_table,
    build_training_canonical_table,
    compute_feature_a,
    compute_feature_h,
    compute_feature_t,
)
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    evaluate_model_name,
    fit_full_model_predictions,
    get_primary_model_name,
)
from plasmid_priority.modeling.module_a import _model_fit_kwargs
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_priority_bootstrap_stability_table,
    build_variant_rank_consistency_table,
)
from plasmid_priority.scoring import (
    DEFAULT_NORMALIZATION_METHOD,
    build_scored_backbone_table,
    recompute_priority_from_reference,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    load_signature_manifest,
    materialize_recorded_paths,
    project_python_source_paths,
    write_signature_manifest,
)
from plasmid_priority.utils.parallel import limit_native_threads

DEFAULT_PRIMARY_MODEL = get_primary_model_name(MODULE_A_FEATURE_SETS.keys())


def _resolve_parallel_jobs(requested_jobs: int | None, *, max_tasks: int, cap: int = 4) -> int:
    if max_tasks <= 1:
        return 1
    env_cap = os.getenv("PLASMID_PRIORITY_MAX_JOBS")
    if env_cap:
        try:
            cap = max(1, min(cap, int(env_cap)))
        except ValueError:
            pass
    if requested_jobs is None:
        requested = min(cap, os.cpu_count() or 1)
    else:
        requested = int(requested_jobs)
    return max(1, min(requested, max_tasks, cap))


def _model_metrics(
    scored: pd.DataFrame,
    *,
    model_name: str | None = None,
    n_repeats: int = 5,
    include_ci: bool = True,
) -> dict[str, float]:
    return _model_metrics_with_config(
        scored,
        model_name=model_name,
        fit_config=None,
        n_repeats=n_repeats,
        include_ci=include_ci,
    )


def _model_metrics_with_config(
    scored: pd.DataFrame,
    *,
    model_name: str | None = None,
    fit_config: dict[str, object] | None = None,
    n_repeats: int = 5,
    include_ci: bool = True,
) -> dict[str, float]:
    chosen_model = model_name or DEFAULT_PRIMARY_MODEL
    resolved_fit_config = _model_fit_kwargs(chosen_model, fit_config)
    result = evaluate_model_name(
        scored,
        model_name=chosen_model,
        n_repeats=n_repeats,
        fit_config=fit_config,
        include_ci=include_ci,
    )
    payload = dict(result.metrics)
    payload["evaluated_model_name"] = chosen_model
    payload["n_eligible_backbones"] = int(scored["spread_label"].notna().sum())
    payload.update({str(key): value for key, value in resolved_fit_config.items()})
    return payload


def _rebuild_scored(
    records: pd.DataFrame,
    amr_hits: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
    amr_identity_min: float | None = None,
    amr_coverage_min: float | None = None,
    force_fallback_backbones: bool = False,
    training_only_backbones: bool = False,
    assigned_records_cache: dict[tuple[int, bool, bool], pd.DataFrame] | None = None,
    amr_consensus_cache: dict[tuple[float | None, float | None], pd.DataFrame] | None = None,
    component_cache: dict[tuple[object, ...], dict[str, object]] | None = None,
    scored_cache: dict[tuple[object, ...], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    cache_key = (
        int(split_year),
        int(test_year_end),
        normalization_method,
        amr_identity_min,
        amr_coverage_min,
        bool(force_fallback_backbones),
        bool(training_only_backbones),
    )
    if scored_cache is not None and cache_key in scored_cache:
        return scored_cache[cache_key].copy()

    components = _prepare_scored_components(
        records,
        amr_hits,
        split_year=split_year,
        amr_identity_min=amr_identity_min,
        amr_coverage_min=amr_coverage_min,
        force_fallback_backbones=force_fallback_backbones,
        training_only_backbones=training_only_backbones,
        assigned_records_cache=assigned_records_cache,
        amr_consensus_cache=amr_consensus_cache,
        component_cache=component_cache,
    )
    scored = _score_components_for_horizon(
        components,
        test_year_end=test_year_end,
        normalization_method=normalization_method,
    )
    if scored_cache is not None:
        scored_cache[cache_key] = scored.copy()
    return scored


def _prepare_scored_components(
    records: pd.DataFrame,
    amr_hits: pd.DataFrame,
    *,
    split_year: int,
    amr_identity_min: float | None = None,
    amr_coverage_min: float | None = None,
    force_fallback_backbones: bool = False,
    training_only_backbones: bool = False,
    assigned_records_cache: dict[tuple[int, bool, bool], pd.DataFrame] | None = None,
    amr_consensus_cache: dict[tuple[float | None, float | None], pd.DataFrame] | None = None,
    component_cache: dict[tuple[object, ...], dict[str, object]] | None = None,
) -> dict[str, object]:
    component_key = (
        int(split_year),
        amr_identity_min,
        amr_coverage_min,
        bool(force_fallback_backbones),
        bool(training_only_backbones),
    )
    if component_cache is not None and component_key in component_cache:
        return component_cache[component_key]

    assignment_key = (
        int(split_year),
        bool(force_fallback_backbones),
        bool(training_only_backbones),
    )
    if assigned_records_cache is not None and assignment_key in assigned_records_cache:
        records = assigned_records_cache[assignment_key].copy()
    else:
        records = records.copy()
        if training_only_backbones:
            records = assign_backbone_ids_training_only(records, split_year=split_year)
        if force_fallback_backbones:
            records = assign_backbone_ids(
                records.assign(primary_cluster_id=""), backbone_assignment_mode="all_records"
            )
        if assigned_records_cache is not None:
            assigned_records_cache[assignment_key] = records.copy()

    consensus_key = (amr_identity_min, amr_coverage_min)
    if amr_consensus_cache is not None and consensus_key in amr_consensus_cache:
        amr_consensus = amr_consensus_cache[consensus_key].copy()
    else:
        filtered_hits = amr_hits.copy()
        if amr_identity_min is not None:
            filtered_hits = filtered_hits.loc[
                pd.to_numeric(filtered_hits["sequence_identity"], errors="coerce")
                >= amr_identity_min
            ].copy()
        if amr_coverage_min is not None:
            filtered_hits = filtered_hits.loc[
                pd.to_numeric(filtered_hits["coverage_percentage"], errors="coerce")
                >= amr_coverage_min
            ].copy()
        amr_consensus = build_amr_consensus(filtered_hits)
        if amr_consensus_cache is not None:
            amr_consensus_cache[consensus_key] = amr_consensus.copy()
    training_canonical = build_training_canonical_table(
        records, amr_consensus, split_year=split_year
    )
    feature_t = compute_feature_t(training_canonical)
    feature_h = compute_feature_h(records, split_year=split_year)
    feature_a = compute_feature_a(training_canonical)
    coherence = compute_backbone_coherence(records, split_year=split_year)
    assignment_mode_series = (
        records.get("backbone_assignment_mode", pd.Series(dtype=str))
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    assignment_mode = (
        str(assignment_mode_series.iloc[0]).strip()
        if not assignment_mode_series.empty
        else "training_only"
    )
    backbone_base = build_backbone_table(
        records,
        coherence,
        split_year=split_year,
        test_year_end=split_year,
        backbone_assignment_mode=assignment_mode,
    ).copy()
    working_records = records.copy()
    years = pd.to_numeric(working_records["resolved_year"], errors="coerce").fillna(0).astype(int)
    working_records["resolved_year_int"] = years
    working_records["country_clean"] = working_records["country"].fillna("").astype(str).str.strip()
    if "backbone_seen_in_training" in working_records.columns:
        working_records["_seen_float"] = (
            working_records["backbone_seen_in_training"].fillna(False).astype(bool).astype(float)
        )
    training_pairs = working_records.loc[
        (years <= split_year) & working_records["country_clean"].ne(""),
        ["backbone_id", "country_clean"],
    ].drop_duplicates()
    future_country_first_year = (
        working_records.loc[
            (years > split_year) & working_records["country_clean"].ne(""),
            ["backbone_id", "country_clean", "resolved_year_int"],
        ]
        .sort_values("resolved_year_int")
        .drop_duplicates(["backbone_id", "country_clean"], keep="first")
        .reset_index(drop=True)
    )
    if not future_country_first_year.empty:
        if training_pairs.empty:
            future_country_first_year["is_new_country"] = True
        else:
            training_index = pd.MultiIndex.from_frame(
                training_pairs[["backbone_id", "country_clean"]]
            )
            future_index = pd.MultiIndex.from_frame(
                future_country_first_year[["backbone_id", "country_clean"]]
            )
            future_country_first_year["is_new_country"] = ~future_index.isin(training_index)
    else:
        future_country_first_year["is_new_country"] = pd.Series(dtype=bool)
    components = {
        "cache_key": component_key,
        "split_year": int(split_year),
        "records": working_records,
        "feature_t": feature_t,
        "feature_h": feature_h,
        "feature_a": feature_a,
        "backbone_base": backbone_base,
        "future_country_first_year": future_country_first_year,
    }
    if component_cache is not None:
        component_cache[component_key] = components
    return components


def _score_components_for_horizon(
    components: dict[str, object],
    *,
    test_year_end: int,
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
) -> pd.DataFrame:
    split_year = int(components["split_year"])
    backbone_table = pd.DataFrame(components["backbone_base"]).copy()
    future_country_first_year = pd.DataFrame(components["future_country_first_year"])
    if not future_country_first_year.empty:
        visible_future = future_country_first_year.loc[
            future_country_first_year["resolved_year_int"].astype(int) <= int(test_year_end)
        ].copy()
        n_countries_test = visible_future.groupby("backbone_id", sort=False).size()
        n_new_countries = (
            visible_future.loc[visible_future["is_new_country"]]
            .groupby("backbone_id", sort=False)
            .size()
        )
    else:
        n_countries_test = pd.Series(dtype=int)
        n_new_countries = pd.Series(dtype=int)
    backbone_table["n_countries_test"] = (
        backbone_table["backbone_id"].map(n_countries_test).fillna(0).astype(int)
    )
    backbone_table["n_new_countries"] = (
        backbone_table["backbone_id"].map(n_new_countries).fillna(0).astype(int)
    )
    backbone_table["split_year"] = int(split_year)
    backbone_table["test_year_end"] = int(test_year_end)
    eligible = backbone_table["n_countries_train"].between(1, 3, inclusive="both")
    spread_label = pd.Series(np.nan, index=backbone_table.index, dtype=float)
    spread_label.loc[eligible] = (
        backbone_table.loc[eligible, "n_new_countries"]
        .ge(DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD)
        .astype(int)
    )
    backbone_table["spread_label"] = spread_label
    backbone_table["visibility_expansion_label"] = spread_label
    records = pd.DataFrame(components["records"])
    if "_seen_float" in records.columns:
        future_rows = records.loc[
            (records["resolved_year_int"].astype(int) > split_year)
            & (records["resolved_year_int"].astype(int) <= int(test_year_end))
        ].copy()
        if not future_rows.empty:
            seen_stats = future_rows.groupby("backbone_id", sort=False)["_seen_float"].agg(
                ["sum", "mean"]
            )
            backbone_table["n_test_records_seen_in_training"] = (
                backbone_table["backbone_id"].map(seen_stats["sum"]).fillna(0).astype(int)
            )
            backbone_table["test_seen_in_training_fraction"] = (
                backbone_table["backbone_id"].map(seen_stats["mean"]).fillna(0.0).astype(float)
            )
        else:
            backbone_table["n_test_records_seen_in_training"] = 0
            backbone_table["test_seen_in_training_fraction"] = 0.0
    return build_scored_backbone_table(
        backbone_table,
        pd.DataFrame(components["feature_t"]),
        pd.DataFrame(components["feature_h"]),
        pd.DataFrame(components["feature_a"]),
        normalization_method=normalization_method,
    )


def _source_balanced_subset(scored: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    eligible["dominant_source"] = np.where(
        eligible["refseq_share_train"].fillna(0.0) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    counts = eligible["dominant_source"].value_counts()
    if counts.empty or len(counts) < 2:
        return eligible
    n_per_group = int(counts.min())
    sampled = (
        eligible.groupby("dominant_source", group_keys=False, sort=False)
        .sample(n=n_per_group, random_state=seed)
        .drop(columns=["dominant_source"])
        .reset_index(drop=True)
    )
    return sampled


def _rescore_existing_table(scored: pd.DataFrame, *, normalization_method: str) -> pd.DataFrame:
    training_reference = scored.loc[scored["member_count_train"].fillna(0).astype(int) > 0].copy()
    rescored = recompute_priority_from_reference(
        scored,
        training_reference,
        normalization_method=normalization_method,
    )
    return rescored


def _apply_outcome_rule(
    scored: pd.DataFrame,
    *,
    n_new_column: str,
    threshold: int,
    min_train_countries: int = 1,
    max_train_countries: int | None = 3,
) -> pd.DataFrame:
    frame = scored.copy()
    train_countries = frame["n_countries_train"].fillna(0).astype(int)
    n_new = frame[n_new_column].fillna(0).astype(float)
    eligible = train_countries >= min_train_countries
    if max_train_countries is not None:
        eligible = eligible & (train_countries <= max_train_countries)
    label = pd.Series(np.nan, index=frame.index, dtype=float)
    label.loc[eligible] = (n_new.loc[eligible] >= threshold).astype(int)
    frame["spread_label"] = label
    return frame


def _stable_country_new_counts(
    records: pd.DataFrame,
    *,
    split_year: int,
    test_year_end: int,
    min_global_records_per_period: int = 0,
) -> pd.Series:
    working = records.copy()
    years = pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    working["country_clean"] = working["country"].fillna("").astype(str).str.strip()
    training = working.loc[(years <= split_year) & working["country_clean"].ne("")].copy()
    testing = working.loc[
        (years > split_year) & (years <= test_year_end) & working["country_clean"].ne("")
    ].copy()
    if training.empty or testing.empty:
        return pd.Series(dtype=int)

    train_country_counts = training.groupby("country_clean", sort=False).size()
    test_country_counts = testing.groupby("country_clean", sort=False).size()
    stable_countries = set(train_country_counts.index) & set(test_country_counts.index)
    if min_global_records_per_period > 0:
        stable_countries = {
            country
            for country in stable_countries
            if int(train_country_counts.get(country, 0)) >= min_global_records_per_period
            and int(test_country_counts.get(country, 0)) >= min_global_records_per_period
        }
    if not stable_countries:
        return pd.Series(dtype=int)

    training_pairs = training.loc[
        training["country_clean"].isin(stable_countries), ["backbone_id", "country_clean"]
    ].drop_duplicates()
    testing_pairs = testing.loc[
        testing["country_clean"].isin(stable_countries), ["backbone_id", "country_clean"]
    ].drop_duplicates()
    if training_pairs.empty or testing_pairs.empty:
        return pd.Series(dtype=int)

    new_pairs = testing_pairs.merge(
        training_pairs,
        on=["backbone_id", "country_clean"],
        how="left",
        indicator=True,
    )
    new_pairs = new_pairs.loc[new_pairs["_merge"] == "left_only"]
    return new_pairs.groupby("backbone_id", sort=False).size()


def _evaluate_variant_payload(
    task: tuple[str, pd.DataFrame, str | None, dict[str, object] | None],
) -> tuple[str, dict[str, object]]:
    name, frame, model_name, fit_config = task
    eligible = frame.loc[frame["spread_label"].notna()]
    if len(eligible) < 20 or eligible["spread_label"].nunique() < 2:
        return name, {"skipped": True}
    return name, _model_metrics_with_config(frame, model_name=model_name, fit_config=fit_config)


def _evaluate_rolling_mode_task(
    task: tuple[int, int, int, str, pd.DataFrame, str],
) -> dict[str, object]:
    split_year, window_end, horizon_years, assignment_mode, variant_scored, default_primary = task
    eligible = variant_scored.loc[variant_scored["spread_label"].notna()].copy()
    if len(eligible) < 20 or eligible["spread_label"].nunique() < 2:
        return {
            "split_year": int(split_year),
            "test_year_end": int(window_end),
            "horizon_years": int(horizon_years),
            "backbone_assignment_mode": assignment_mode,
            "eligible_ids": set(eligible["backbone_id"].astype(str)),
            "status": "skipped_insufficient_label_variation",
            "rolling_row": {
                "split_year": int(split_year),
                "test_year_end": int(window_end),
                "horizon_years": int(horizon_years),
                "backbone_assignment_mode": assignment_mode,
                "model_name": default_primary,
                "n_backbones": int(len(variant_scored)),
                "n_eligible_backbones": int(len(eligible)),
                "status": "skipped_insufficient_label_variation",
            },
            "annual_freeze_row": None,
        }

    metrics = _model_metrics(
        variant_scored,
        model_name=default_primary,
        n_repeats=5,
        include_ci=False,
    )
    rolling_row = {
        "split_year": int(split_year),
        "test_year_end": int(window_end),
        "horizon_years": int(horizon_years),
        "backbone_assignment_mode": assignment_mode,
        "model_name": default_primary,
        "n_backbones": int(len(variant_scored)),
        "n_eligible_backbones": int(metrics["n_eligible_backbones"]),
        "n_positive": int(metrics["n_positive"]),
        "positive_prevalence": float(metrics["positive_prevalence"]),
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
        "average_precision_lift": float(metrics["average_precision_lift"]),
        "average_precision_enrichment": float(metrics["average_precision_enrichment"]),
        "brier_score": float(metrics["brier_score"]),
        "status": "ok",
    }
    annual_freeze_row = None
    if assignment_mode == "training_only":
        freeze_predictions = fit_full_model_predictions(
            variant_scored, model_name=default_primary
        ).rename(columns={"prediction": "freeze_candidate_score"})
        freeze_candidates = (
            variant_scored.loc[variant_scored["member_count_train"].fillna(0).astype(int) > 0]
            .merge(freeze_predictions, on="backbone_id", how="left", validate="1:1")
            .sort_values(["freeze_candidate_score", "priority_index"], ascending=[False, False])
            .head(25)
            .copy()
        )
        annual_freeze_row = {
            "split_year": int(split_year),
            "test_year_end": int(window_end),
            "horizon_years": int(horizon_years),
            "backbone_assignment_mode": assignment_mode,
            "n_candidates": int(len(freeze_candidates)),
            "n_positive_candidates": int(
                freeze_candidates["spread_label"].fillna(0).astype(int).sum()
            )
            if not freeze_candidates.empty
            else 0,
            "precision_at_25": float(
                freeze_candidates["spread_label"].fillna(0).astype(float).mean()
            )
            if not freeze_candidates.empty
            else np.nan,
            "mean_n_new_countries": float(freeze_candidates["n_new_countries"].fillna(0).mean())
            if not freeze_candidates.empty
            else np.nan,
            "top_backbone_id": str(freeze_candidates.iloc[0]["backbone_id"])
            if not freeze_candidates.empty
            else "",
        }
    return {
        "split_year": int(split_year),
        "test_year_end": int(window_end),
        "horizon_years": int(horizon_years),
        "backbone_assignment_mode": assignment_mode,
        "eligible_ids": set(eligible["backbone_id"].astype(str)),
        "status": "ok",
        "metrics": metrics,
        "rolling_row": rolling_row,
        "annual_freeze_row": annual_freeze_row,
    }


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.data_dir / "scores/backbone_scored.tsv"
    backbones_path = context.data_dir / "silver/plasmid_backbones.tsv"
    amr_hits_path = context.data_dir / "silver/plasmid_amr_hits.tsv"
    config_paths = context_config_paths(context)
    manifest_path = context.data_dir / "analysis/22_run_sensitivity.manifest.json"
    output_path = context.data_dir / "analysis/sensitivity_summary.json"
    rolling_output_path = context.data_dir / "analysis/rolling_temporal_validation.tsv"
    rolling_diagnostic_output_path = (
        context.data_dir / "analysis/rolling_assignment_diagnostics.tsv"
    )
    rank_stability_output_path = context.data_dir / "analysis/candidate_rank_stability.tsv"
    variant_consistency_output_path = (
        context.data_dir / "analysis/candidate_variant_consistency.tsv"
    )
    freeze_output_path = context.data_dir / "analysis/prospective_candidate_freeze.tsv"
    annual_freeze_summary_output_path = (
        context.data_dir / "analysis/annual_candidate_freeze_summary.tsv"
    )
    ensure_directory(output_path.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/22_run_sensitivity.py",
    )
    input_paths = [scored_path, backbones_path, amr_hits_path, *config_paths]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
    }

    with ManagedScriptRun(context, "22_run_sensitivity") as run:
        for path in (scored_path, backbones_path, amr_hits_path, *config_paths):
            run.record_input(path)
        run.record_output(output_path)
        run.record_output(rolling_output_path)
        run.record_output(rolling_diagnostic_output_path)
        run.record_output(rank_stability_output_path)
        run.record_output(variant_consistency_output_path)
        run.record_output(freeze_output_path)
        run.record_output(annual_freeze_summary_output_path)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached sensitivity outputs.")
            run.set_metric("cache_hit", True)
            return 0
        scored = read_tsv(scored_path)
        records = read_tsv(backbones_path)
        amr_hits = read_tsv(amr_hits_path)
        pipeline = context.pipeline_settings
        assigned_records_cache: dict[tuple[int, bool, bool], pd.DataFrame] = {}
        amr_consensus_cache: dict[tuple[float | None, float | None], pd.DataFrame] = {}
        component_cache: dict[tuple[object, ...], dict[str, object]] = {}
        scored_cache: dict[tuple[object, ...], pd.DataFrame] = {}

        default_primary = DEFAULT_PRIMARY_MODEL
        main_threshold = int(pipeline.min_new_countries_for_spread)

        variants: dict[str, object] = {
            "default": (scored, default_primary, None),
            "low_coherence_excluded": scored.loc[
                scored["coherence_score"].fillna(0.0) >= 0.5
            ].copy(),
            "member_count_train_ge_3": scored.loc[
                scored["member_count_train"].fillna(0).astype(int) >= 3
            ].copy(),
            "alternate_normalization_rank_percentile": _rescore_existing_table(
                scored,
                normalization_method="rank_percentile",
            ),
            "alternate_normalization_yeo_johnson": _rescore_existing_table(
                scored,
                normalization_method="yeo_johnson_sigmoid",
            ),
            "strict_geometric_priority_as_main": (
                scored.assign(priority_index=scored["strict_priority_index"]),
                "full_priority",
                None,
            ),
            "source_balanced_rerun": _source_balanced_subset(scored),
            "class_plus_knownness_balanced_primary": (
                scored,
                default_primary,
                {"sample_weight_mode": "class_balanced+knownness_balanced"},
            ),
            "knownness_balanced_primary": (
                scored,
                default_primary,
                {"sample_weight_mode": "knownness_balanced"},
            ),
            "source_plus_class_balanced_primary": (
                scored,
                default_primary,
                {"sample_weight_mode": "source_balanced+class_balanced"},
            ),
            "alternate_outcome_threshold_1": _apply_outcome_rule(
                scored,
                n_new_column="n_new_countries",
                threshold=1,
                min_train_countries=1,
                max_train_countries=3,
            ),
            "alternate_outcome_threshold_2": _apply_outcome_rule(
                scored,
                n_new_column="n_new_countries",
                threshold=2,
                min_train_countries=1,
                max_train_countries=3,
            ),
            "expanded_eligibility_ge_1": _apply_outcome_rule(
                scored,
                n_new_column="n_new_countries",
                threshold=main_threshold,
                min_train_countries=1,
                max_train_countries=None,
            ),
        }
        for l2_penalty in (0.1, 1.0, 2.5, 5.0, 10.0):
            variant_name = f"primary_l2_{str(l2_penalty).replace('.', 'p')}"
            variants[variant_name] = (
                scored,
                default_primary,
                {"l2": float(l2_penalty)},
            )
        stable_new = _stable_country_new_counts(
            records,
            split_year=pipeline.split_year,
            test_year_end=2023,
            min_global_records_per_period=0,
        )
        stable_dense_new = _stable_country_new_counts(
            records,
            split_year=pipeline.split_year,
            test_year_end=2023,
            min_global_records_per_period=10,
        )
        stable_country_frame = scored.copy()
        stable_country_frame["n_new_stable_countries"] = (
            scored["backbone_id"].map(stable_new).fillna(0).astype(int)
        )
        variants["stable_country_outcome"] = _apply_outcome_rule(
            stable_country_frame,
            n_new_column="n_new_stable_countries",
            threshold=main_threshold,
            min_train_countries=1,
            max_train_countries=3,
        )
        stable_dense_country_frame = scored.copy()
        stable_dense_country_frame["n_new_stable_dense_countries"] = (
            scored["backbone_id"].map(stable_dense_new).fillna(0).astype(int)
        )
        variants["stable_dense_country_outcome"] = _apply_outcome_rule(
            stable_dense_country_frame,
            n_new_column="n_new_stable_dense_countries",
            threshold=main_threshold,
            min_train_countries=1,
            max_train_countries=3,
        )
        # Rebuild-heavy variants share caches so the same assignment/consensus work is not repeated.
        variants["alternate_split_2014"] = _rebuild_scored(
            records,
            amr_hits,
            split_year=2014,
            assigned_records_cache=assigned_records_cache,
            amr_consensus_cache=amr_consensus_cache,
            component_cache=component_cache,
            scored_cache=scored_cache,
        )
        variants["alternate_split_2016"] = _rebuild_scored(
            records,
            amr_hits,
            split_year=2016,
            assigned_records_cache=assigned_records_cache,
            amr_consensus_cache=amr_consensus_cache,
            component_cache=component_cache,
            scored_cache=scored_cache,
        )
        variants["strict_amr_identity99_coverage95"] = _rebuild_scored(
            records,
            amr_hits,
            amr_identity_min=99.0,
            amr_coverage_min=95.0,
            assigned_records_cache=assigned_records_cache,
            amr_consensus_cache=amr_consensus_cache,
            component_cache=component_cache,
            scored_cache=scored_cache,
        )
        variants["training_only_backbone_rerun"] = _rebuild_scored(
            records,
            amr_hits,
            training_only_backbones=True,
            assigned_records_cache=assigned_records_cache,
            amr_consensus_cache=amr_consensus_cache,
            component_cache=component_cache,
            scored_cache=scored_cache,
        )
        variants["fallback_backbone_rerun"] = _rebuild_scored(
            records,
            amr_hits,
            force_fallback_backbones=True,
            assigned_records_cache=assigned_records_cache,
            amr_consensus_cache=amr_consensus_cache,
            component_cache=component_cache,
            scored_cache=scored_cache,
        )

        # T4: Alternate outcome thresholds — compare looser and stricter labels
        # against the published main threshold without silently redefining it.
        if main_threshold != 3:
            variants["alternate_outcome_threshold_3"] = _apply_outcome_rule(
                scored,
                n_new_column="n_new_countries",
                threshold=3,
                min_train_countries=1,
                max_train_countries=3,
            )
        variants["alternate_outcome_threshold_4"] = _apply_outcome_rule(
            scored,
            n_new_column="n_new_countries",
            threshold=4,
            min_train_countries=1,
            max_train_countries=3,
        )
        variants["alternate_outcome_threshold_5"] = _apply_outcome_rule(
            scored,
            n_new_column="n_new_countries",
            threshold=5,
            min_train_countries=1,
            max_train_countries=3,
        )

        # T6: Parsimonious model (without H_support_norm_residual)
        variants["parsimonious_model"] = (scored, "parsimonious_priority", None)

        variant_frames: dict[str, pd.DataFrame] = {}
        variant_tasks: list[tuple[str, pd.DataFrame, str | None, dict[str, object] | None]] = []
        for name, variant in variants.items():
            fit_config = None
            if isinstance(variant, tuple):
                if len(variant) == 3:
                    frame, model_name, fit_config = variant
                else:
                    frame, model_name = variant
            else:
                frame, model_name = variant, None
            variant_frames[name] = frame
            variant_tasks.append((name, frame, model_name, fit_config))
        variant_jobs = _resolve_parallel_jobs(None, max_tasks=len(variant_tasks))
        if variant_jobs > 1:
            with limit_native_threads(1):
                with ThreadPoolExecutor(max_workers=variant_jobs) as executor:
                    payload = dict(executor.map(_evaluate_variant_payload, variant_tasks))
        else:
            payload = dict(_evaluate_variant_payload(task) for task in variant_tasks)

        rolling_rows: list[dict[str, object]] = []
        rolling_diagnostic_rows: list[dict[str, object]] = []
        annual_freeze_rows: list[dict[str, object]] = []
        rolling_tasks: list[tuple[int, int, int, str, pd.DataFrame, str]] = []
        rolling_keys: list[tuple[int, int, int]] = []
        for split_year in range(2012, 2019):
            for horizon_years in (1, 3, 5, 8):
                window_end = min(split_year + horizon_years, 2023)
                if window_end <= split_year:
                    continue
                rolling_keys.append((int(split_year), int(window_end), int(horizon_years)))
                for assignment_mode, training_only in (
                    ("all_records", False),
                    ("training_only", True),
                ):
                    variant_scored = _rebuild_scored(
                        records,
                        amr_hits,
                        split_year=split_year,
                        test_year_end=window_end,
                        training_only_backbones=training_only,
                        assigned_records_cache=assigned_records_cache,
                        amr_consensus_cache=amr_consensus_cache,
                        component_cache=component_cache,
                        scored_cache=scored_cache,
                    )
                    rolling_tasks.append(
                        (
                            int(split_year),
                            int(window_end),
                            int(horizon_years),
                            assignment_mode,
                            variant_scored,
                            default_primary,
                        )
                    )
        rolling_jobs = _resolve_parallel_jobs(None, max_tasks=len(rolling_tasks))
        if rolling_jobs > 1:
            with limit_native_threads(1):
                with ThreadPoolExecutor(max_workers=rolling_jobs) as executor:
                    rolling_results = list(executor.map(_evaluate_rolling_mode_task, rolling_tasks))
        else:
            rolling_results = [_evaluate_rolling_mode_task(task) for task in rolling_tasks]

        mode_results_by_key = {
            (
                result["split_year"],
                result["test_year_end"],
                result["backbone_assignment_mode"],
            ): result
            for result in rolling_results
        }
        for result in rolling_results:
            rolling_rows.append(result["rolling_row"])
            if result.get("annual_freeze_row") is not None:
                annual_freeze_rows.append(result["annual_freeze_row"])

        for split_year, window_end, horizon_years in rolling_keys:
            all_records_result = mode_results_by_key.get(
                (split_year, window_end, "all_records"), {}
            )
            training_only_result = mode_results_by_key.get(
                (split_year, window_end, "training_only"), {}
            )
            all_ids = set(all_records_result.get("eligible_ids", set()))
            training_ids = set(training_only_result.get("eligible_ids", set()))
            overlap = all_ids & training_ids
            union = all_ids | training_ids
            all_metrics = all_records_result.get("metrics", {})
            training_metrics = training_only_result.get("metrics", {})
            assignment_key = (int(split_year), False, True)
            assigned_training_only = assigned_records_cache.get(assignment_key)
            future_unseen_row_fraction = np.nan
            future_unseen_backbone_fraction = np.nan
            if assigned_training_only is not None:
                assigned_years = (
                    pd.to_numeric(assigned_training_only["resolved_year"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                future_rows = assigned_training_only.loc[
                    (assigned_years > split_year) & (assigned_years <= window_end)
                ].copy()
                if not future_rows.empty:
                    unseen_mask = (
                        future_rows["backbone_assignment_rule"]
                        .astype(str)
                        .eq("unseen_after_training")
                    )
                    future_unseen_row_fraction = float(unseen_mask.mean())
                    future_unseen_backbone_fraction = float(
                        future_rows.loc[unseen_mask, "backbone_id"].astype(str).nunique()
                        / max(future_rows["backbone_id"].astype(str).nunique(), 1)
                    )
            rolling_diagnostic_rows.append(
                {
                    "split_year": split_year,
                    "test_year_end": window_end,
                    "horizon_years": int(horizon_years),
                    "all_records_status": all_records_result.get("status", "missing"),
                    "training_only_status": training_only_result.get("status", "missing"),
                    "all_records_n_eligible": int(len(all_ids)),
                    "training_only_n_eligible": int(len(training_ids)),
                    "eligible_overlap_count": int(len(overlap)),
                    "eligible_union_count": int(len(union)),
                    "eligible_overlap_fraction": float(len(overlap) / len(union)) if union else 0.0,
                    "eligible_identical": bool(all_ids == training_ids),
                    "training_only_future_unseen_row_fraction": future_unseen_row_fraction,
                    "training_only_future_unseen_backbone_fraction": future_unseen_backbone_fraction,
                    "roc_auc_delta_training_only_minus_all_records": (
                        float(training_metrics["roc_auc"] - all_metrics["roc_auc"])
                        if all_records_result.get("status") == "ok"
                        and training_only_result.get("status") == "ok"
                        else np.nan
                    ),
                    "average_precision_delta_training_only_minus_all_records": (
                        float(
                            training_metrics["average_precision"] - all_metrics["average_precision"]
                        )
                        if all_records_result.get("status") == "ok"
                        and training_only_result.get("status") == "ok"
                        else np.nan
                    ),
                }
            )

        rolling_table = pd.DataFrame(rolling_rows)
        rolling_table.to_csv(rolling_output_path, sep="\t", index=False)
        rolling_diagnostic_table = pd.DataFrame(rolling_diagnostic_rows)
        rolling_diagnostic_table.to_csv(rolling_diagnostic_output_path, sep="\t", index=False)
        annual_freeze_summary = pd.DataFrame(annual_freeze_rows)
        annual_freeze_summary.to_csv(annual_freeze_summary_output_path, sep="\t", index=False)

        consistency_variants = {
            name: frame
            for name, frame in variant_frames.items()
            if name
            in {
                "default",
                "low_coherence_excluded",
                "member_count_train_ge_3",
                "alternate_normalization_rank_percentile",
                "strict_geometric_priority_as_main",
                "strict_amr_identity99_coverage95",
                "training_only_backbone_rerun",
                "fallback_backbone_rerun",
            }
        }
        variant_consistency = build_variant_rank_consistency_table(
            scored,
            consistency_variants,
            candidate_n=50,
            top_k=25,
            model_name=default_primary,
            n_jobs=4,
        )
        variant_consistency.to_csv(variant_consistency_output_path, sep="\t", index=False)

        rank_stability = build_priority_bootstrap_stability_table(
            scored,
            candidate_n=50,
            top_k=25,
            n_bootstrap=100,
            model_name=default_primary,
            n_jobs=4,
        )
        rank_stability.to_csv(rank_stability_output_path, sep="\t", index=False)

        freeze_scored = variant_frames.get("training_only_backbone_rerun", pd.DataFrame()).copy()
        if freeze_scored.empty:
            freeze_scored = _rebuild_scored(
                records,
                amr_hits,
                split_year=pipeline.split_year,
                test_year_end=2023,
                training_only_backbones=True,
                assigned_records_cache=assigned_records_cache,
                amr_consensus_cache=amr_consensus_cache,
                component_cache=component_cache,
                scored_cache=scored_cache,
            )
        freeze_predictions = fit_full_model_predictions(
            freeze_scored, model_name=default_primary
        ).rename(columns={"prediction": "freeze_candidate_score"})
        freeze_candidates = (
            freeze_scored.loc[freeze_scored["member_count_train"].fillna(0).astype(int) > 0]
            .merge(freeze_predictions, on="backbone_id", how="left", validate="1:1")
            .sort_values(["freeze_candidate_score", "priority_index"], ascending=[False, False])
            .head(50)
            .copy()
            .reset_index(drop=True)
        )
        freeze_candidates["freeze_rank"] = np.arange(1, len(freeze_candidates) + 1)
        freeze_candidates["freeze_split_year"] = int(pipeline.split_year)
        freeze_candidates["evaluation_year_start"] = int(pipeline.split_year) + 1
        freeze_candidates["evaluation_year_end"] = 2023
        freeze_candidates["backbone_assignment_mode"] = "training_only"
        leading_columns = [
            "freeze_rank",
            "freeze_split_year",
            "evaluation_year_start",
            "evaluation_year_end",
            "backbone_assignment_mode",
            "backbone_id",
            "freeze_candidate_score",
            "priority_index",
            "spread_label",
            "n_new_countries",
            "n_countries_train",
            "member_count_train",
            "coherence_score",
            "T_eff_norm",
            "H_eff_norm",
            "A_eff_norm",
        ]
        ordered_columns = leading_columns + [
            column for column in freeze_candidates.columns if column not in leading_columns
        ]
        freeze_candidates = freeze_candidates[ordered_columns]
        freeze_candidates.to_csv(freeze_output_path, sep="\t", index=False)

        atomic_write_json(output_path, payload)
        run.set_metric("variants_evaluated", len(payload))
        run.set_rows_out("rolling_temporal_rows", int(len(rolling_table)))
        run.set_rows_out("rolling_assignment_diagnostic_rows", int(len(rolling_diagnostic_table)))
        run.set_rows_out("candidate_rank_stability_rows", int(len(rank_stability)))
        run.set_rows_out("candidate_variant_consistency_rows", int(len(variant_consistency)))
        run.set_rows_out("prospective_candidate_freeze_rows", int(len(freeze_candidates)))
        run.set_rows_out("annual_candidate_freeze_rows", int(len(annual_freeze_summary)))
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
