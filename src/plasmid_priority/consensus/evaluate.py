"""Evaluation helpers for the consensus branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from plasmid_priority.consensus.fuse import (
    build_operational_consensus_frame,
    build_research_consensus_frame,
)
from plasmid_priority.consensus.specs import resolve_consensus_model_names
from plasmid_priority.modeling.module_a import _evaluate_prediction_set
from plasmid_priority.modeling.module_a_support import ModelResult
from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def _resolve_consensus_weights(config: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(config, Mapping):
        return None
    branch_block = config.get("consensus", {})
    if isinstance(branch_block, Mapping):
        weights = branch_block.get("fusion_weights")
        if isinstance(weights, Mapping):
            return {str(key): float(value) for key, value in weights.items()}
    pipeline_block = config.get("pipeline", {})
    if isinstance(pipeline_block, Mapping):
        weights = pipeline_block.get("consensus_weights")
        if isinstance(weights, Mapping):
            return {str(key): float(value) for key, value in weights.items()}
    return None


def _model_result_from_frame(name: str, frame: pd.DataFrame) -> Any:
    if frame.empty or "consensus_score" not in frame.columns:
        return ModelResult(
            name=name, metrics={}, predictions=frame.copy(), status="ok", error_message=None
        )
    working = frame.loc[:, ~frame.columns.duplicated()].copy()
    label_series = (
        working["spread_label"]
        if "spread_label" in working.columns
        else pd.Series(np.nan, index=working.index)
    )
    y = pd.to_numeric(label_series, errors="coerce").fillna(-1).astype(int)
    preds = pd.to_numeric(working["consensus_score"], errors="coerce").fillna(0.0)
    valid = y >= 0
    extra_metrics = {
        "branch_agreement_score": float(
            pd.to_numeric(working["branch_agreement_score"], errors="coerce").mean()
        ),
        "review_fraction": float(
            pd.to_numeric(working["consensus_review_flag"], errors="coerce")
            .fillna(False)
            .astype(bool)
            .mean()
        ),
        "ood_rate": float(
            pd.concat(
                [
                    working.get("ood_geo", pd.Series(False, index=working.index)),
                    working.get("ood_bio_transfer", pd.Series(False, index=working.index)),
                    working.get("ood_clinical_hazard", pd.Series(False, index=working.index)),
                ],
                axis=1,
            )
            .fillna(False)
            .astype(bool)
            .any(axis=1)
            .mean()
        ),
    }
    if valid.any() and y.loc[valid].nunique() >= 2:
        y_valid = y.loc[valid].to_numpy(dtype=int)
        preds_valid = preds.loc[valid].to_numpy(dtype=float)
        extra_metrics["roc_auc"] = float(roc_auc_score(y_valid, preds_valid))
        extra_metrics["average_precision"] = float(
            average_precision(y_valid, preds_valid)
        )
    if not valid.any():
        return ModelResult(
            name=name,
            metrics=extra_metrics,
            predictions=working.loc[:, ~working.columns.duplicated()].copy(),
            status="ok",
            error_message=None,
        )
    result = _evaluate_prediction_set(
        name,
        y.loc[valid].to_numpy(dtype=int),
        preds.loc[valid].to_numpy(dtype=float),
        working.loc[valid, "backbone_id"].astype(str),
        include_ci=True,
        extra_metrics=extra_metrics,
        prediction_detail=working.loc[valid].copy(),
    )
    if isinstance(getattr(result, "predictions", None), pd.DataFrame):
        result.predictions = result.predictions.loc[
            :, ~result.predictions.columns.duplicated()
        ].copy()
    return result


def build_consensus_model_summary(results: Mapping[str, Any]) -> pd.DataFrame:
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


def build_consensus_prediction_table(results: Mapping[str, Any]) -> pd.DataFrame:
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


def evaluate_consensus_branch(
    merged: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_names = tuple(model_names or resolve_consensus_model_names(config))
    if merged.empty:
        return {name: _model_result_from_frame(name, merged) for name in model_names}
    weights = _resolve_consensus_weights(config)
    operational = build_operational_consensus_frame(merged, weights=weights)
    research = build_research_consensus_frame(merged)
    return {
        "operational_consensus": _model_result_from_frame("operational_consensus", operational),
        "research_consensus": _model_result_from_frame("research_consensus", research),
    }
