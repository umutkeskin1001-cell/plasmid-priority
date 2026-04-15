"""Evaluation and reporting helpers for the bio transfer branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from plasmid_priority.bio_transfer.dataset import (
    prepare_bio_transfer_scored_table,
    resolve_bio_transfer_dataset_model_names,
)
from plasmid_priority.bio_transfer.features import build_bio_transfer_features
from plasmid_priority.bio_transfer.train import fit_bio_transfer_branch
from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def _augment_bio_result_metrics(result: Any, *, prepared_scored: pd.DataFrame) -> None:
    if getattr(result, "status", "") != "ok" or getattr(result, "predictions", pd.DataFrame()).empty:
        return
    score_column = "oof_prediction" if "oof_prediction" in result.predictions.columns else "prediction"
    if score_column not in result.predictions.columns:
        return
    merge_columns = [
        column
        for column in (
            "backbone_id",
            "bio_transfer_label",
            "knownness_score",
            "n_new_host_genera_future",
            "n_new_host_families_future",
            "host_phylo_dispersion_gain_future",
        )
        if column in prepared_scored.columns
    ]
    merged = result.predictions.merge(
        prepared_scored.loc[:, merge_columns].drop_duplicates(subset=["backbone_id"]),
        on="backbone_id",
        how="left",
        suffixes=("", "_prepared"),
    )
    if "bio_transfer_label_prepared" in merged.columns and "bio_transfer_label" in merged.columns:
        merged["bio_transfer_label"] = pd.to_numeric(merged["bio_transfer_label"], errors="coerce").fillna(
            pd.to_numeric(merged["bio_transfer_label_prepared"], errors="coerce")
        )
        merged = merged.drop(columns=["bio_transfer_label_prepared"])
    labels = pd.to_numeric(merged.get("bio_transfer_label"), errors="coerce")
    score = pd.to_numeric(merged.get(score_column), errors="coerce")
    valid = labels.notna() & score.notna()
    if valid.any() and labels.loc[valid].nunique() >= 2:
        result.metrics["roc_auc"] = float(roc_auc_score(labels.loc[valid].astype(int), score.loc[valid]))
        result.metrics["average_precision"] = float(average_precision(labels.loc[valid].astype(int), score.loc[valid]))
    result.metrics["n_backbones"] = int(len(merged))
    result.metrics["n_positive"] = int(labels.fillna(0).astype(int).sum())
    if "knownness_score" in merged.columns:
        knownness = pd.to_numeric(merged["knownness_score"], errors="coerce")
        result.metrics["mean_knownness_score"] = float(np.nanmean(knownness.to_numpy(dtype=float)))


def build_bio_transfer_model_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        row = {
            "model_name": str(model_name),
            "status": getattr(result, "status", "ok"),
            "error_message": getattr(result, "error_message", None),
        }
        row.update(getattr(result, "metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def build_bio_transfer_prediction_table(results: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name, result in results.items():
        frame = getattr(result, "predictions", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["model_name"] = str(model_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_bio_transfer_branch(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    include_research: bool = False,
    include_ablation: bool = False,
    n_splits: int = 3,
    n_repeats: int = 2,
    seed: int = 42,
    n_jobs: int | None = 1,
    config: Mapping[str, Any] | None = None,
    records: pd.DataFrame | None = None,
    include_ci: bool = True,
) -> dict[str, Any]:
    if model_names is None:
        model_names = resolve_bio_transfer_dataset_model_names(
            config,
            include_research=include_research,
            include_ablation=include_ablation,
        )
    prepared_scored = prepare_bio_transfer_scored_table(scored, config=config, records=records)
    results = fit_bio_transfer_branch(
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
    for result in results.values():
        _augment_bio_result_metrics(result, prepared_scored=prepared_scored)
    return results
