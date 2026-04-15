"""Evaluation and reporting helpers for the geo spread branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from plasmid_priority.geo_spread.dataset import (
    prepare_geo_spread_scored_table,
    resolve_geo_spread_dataset_model_names,
)
from plasmid_priority.geo_spread.select import (
    build_geo_spread_adaptive_result,
    build_geo_spread_blended_result,
    build_geo_spread_meta_result,
)
from plasmid_priority.geo_spread.train import fit_geo_spread_branch
from plasmid_priority.modeling.module_a import ModelResult
from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def _knownness_metrics_for_predictions(merged: pd.DataFrame, score_column: str) -> dict[str, float]:
    knownness = pd.to_numeric(merged.get("knownness_score"), errors="coerce")
    labels = pd.to_numeric(merged.get("spread_label"), errors="coerce")
    score = pd.to_numeric(merged.get(score_column), errors="coerce")
    valid = knownness.notna() & labels.notna() & score.notna()
    if not valid.any():
        return {}
    knownness = knownness.loc[valid].astype(float)
    labels = labels.loc[valid].astype(int)
    score = score.loc[valid].astype(float)
    threshold = float(np.nanquantile(knownness, 0.5))
    low_mask = knownness <= threshold
    high_mask = knownness > threshold
    metrics: dict[str, float] = {
        "knownness_threshold": threshold,
        "low_knownness_share": float(np.mean(low_mask.astype(float))),
        "high_knownness_share": float(np.mean(high_mask.astype(float))),
    }
    if low_mask.any() and labels.loc[low_mask].nunique() >= 2:
        metrics["low_knownness_roc_auc"] = float(
            roc_auc_score(labels.loc[low_mask], score.loc[low_mask])
        )
        metrics["low_knownness_average_precision"] = float(
            average_precision(labels.loc[low_mask], score.loc[low_mask])
        )
    if high_mask.any() and labels.loc[high_mask].nunique() >= 2:
        metrics["high_knownness_roc_auc"] = float(
            roc_auc_score(labels.loc[high_mask], score.loc[high_mask])
        )
        metrics["high_knownness_average_precision"] = float(
            average_precision(labels.loc[high_mask], score.loc[high_mask])
        )
    if knownness.nunique() >= 4:
        quantiles = pd.qcut(
            knownness.rank(method="average"),
            q=min(4, int(knownness.nunique())),
            labels=False,
            duplicates="drop",
        )
        aucs: list[float] = []
        aps: list[float] = []
        for code in sorted(pd.Series(quantiles).dropna().astype(int).unique().tolist()):
            mask = quantiles == code
            if labels.loc[mask].nunique() < 2:
                continue
            aucs.append(float(roc_auc_score(labels.loc[mask], score.loc[mask])))
            aps.append(float(average_precision(labels.loc[mask], score.loc[mask])))
        if aucs:
            metrics["worst_knownness_quartile_roc_auc"] = float(min(aucs))
            metrics["mean_knownness_quartile_roc_auc"] = float(np.mean(aucs))
        if aps:
            metrics["worst_knownness_quartile_average_precision"] = float(min(aps))
            metrics["mean_knownness_quartile_average_precision"] = float(np.mean(aps))
    return metrics


def _auxiliary_target_metrics(merged: pd.DataFrame, score_column: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    score = pd.to_numeric(merged.get(score_column), errors="coerce")
    for target_name in (
        "event_within_1y_label",
        "event_within_3y_label",
        "event_within_5y_label",
        "macro_region_jump_label",
    ):
        if target_name not in merged.columns:
            continue
        labels = pd.to_numeric(merged[target_name], errors="coerce")
        valid = labels.notna() & score.notna()
        if not valid.any():
            continue
        labels = labels.loc[valid].astype(int)
        score_valid = score.loc[valid].astype(float)
        if labels.nunique() < 2:
            continue
        prefix = target_name.replace("_label", "")
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(labels, score_valid))
        metrics[f"{prefix}_average_precision"] = float(average_precision(labels, score_valid))
    return metrics


def _augment_geo_result_metrics(
    result: ModelResult,
    *,
    prepared_scored: pd.DataFrame,
) -> None:
    if result.status != "ok" or result.predictions.empty:
        return
    score_column = "oof_prediction" if "oof_prediction" in result.predictions.columns else "prediction"
    if score_column not in result.predictions.columns:
        return
    merge_columns = [
        column
        for column in (
            "backbone_id",
            "spread_label",
            "knownness_score",
            "event_within_1y_label",
            "event_within_3y_label",
            "event_within_5y_label",
            "macro_region_jump_label",
        )
        if column in prepared_scored.columns
    ]
    if "backbone_id" not in merge_columns:
        return
    merged = result.predictions.merge(
        prepared_scored.loc[:, merge_columns].drop_duplicates(subset=["backbone_id"]),
        on="backbone_id",
        how="left",
        suffixes=("", "_prepared"),
    )
    if "spread_label_prepared" in merged.columns and "spread_label" in merged.columns:
        merged["spread_label"] = pd.to_numeric(merged["spread_label"], errors="coerce").fillna(
            pd.to_numeric(merged["spread_label_prepared"], errors="coerce")
        )
        merged = merged.drop(columns=["spread_label_prepared"])
    result.metrics.update(_knownness_metrics_for_predictions(merged, score_column))
    result.metrics.update(_auxiliary_target_metrics(merged, score_column))


def build_geo_spread_model_summary(results: Mapping[str, ModelResult]) -> pd.DataFrame:
    """Summarize geo spread model metrics in a compact table."""
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        row = {
            "model_name": str(model_name),
            "status": result.status,
            "error_message": result.error_message,
        }
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_geo_spread_prediction_table(results: Mapping[str, ModelResult]) -> pd.DataFrame:
    """Combine geo spread predictions into a single branch table."""
    frames: list[pd.DataFrame] = []
    for model_name, result in results.items():
        frame = result.predictions.copy()
        if frame.empty:
            continue
        frame["model_name"] = str(model_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_geo_spread_branch(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    include_research: bool = False,
    include_ablation: bool = False,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    n_jobs: int | None = 1,
    config: Mapping[str, Any] | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
    include_blend: bool = True,
    include_adaptive: bool = True,
    include_meta: bool = True,
) -> dict[str, ModelResult]:
    """Evaluate the geo spread branch model surface."""
    if model_names is None:
        model_names = resolve_geo_spread_dataset_model_names(
            config,
            include_research=include_research,
            include_ablation=include_ablation,
        )
    prepared_scored = prepare_geo_spread_scored_table(scored, config=config, records=records)
    results = fit_geo_spread_branch(
        scored,
        model_names=model_names,
        include_research=include_research,
        include_ablation=include_ablation,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        n_jobs=n_jobs,
        config=config,
        records=records,
        include_ci=include_ci,
        prepared_scored=prepared_scored,
    )
    if include_blend:
        blend = build_geo_spread_blended_result(results, include_ci=include_ci)
        results[blend.name] = blend
    if include_adaptive:
        adaptive = build_geo_spread_adaptive_result(
            results,
            scored=prepared_scored,
            include_ci=include_ci,
        )
        results[adaptive.name] = adaptive
    if include_meta:
        meta = build_geo_spread_meta_result(
            results,
            scored=prepared_scored,
            include_ci=include_ci,
        )
        results[meta.name] = meta
    for result in results.values():
        _augment_geo_result_metrics(result, prepared_scored=prepared_scored)
    return results
