"""Helper functions extracted from scripts/24_build_reports.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.utils.dataframe import read_tsv


def metrics_to_frame(metrics_path: Path) -> pd.DataFrame:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for model_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        row = {
            "model_name": model_name,
            "status": str(metrics.get("status", "ok") or "ok"),
            "error_message": metrics.get("error_message"),
        }
        row.update(
            {key: value for key, value in metrics.items() if key not in {"status", "error_message"}}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_tsv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def official_report_model_names(
    *,
    primary_model_name: str | None,
    governance_model_name: str | None,
    baseline_model_name: str = "baseline_both",
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                str(primary_model_name or "").strip(),
                str(governance_model_name or "").strip(),
                str(baseline_model_name or "").strip(),
            ]
        )
    )


def format_scorecard_rank_text(
    scorecard_row: pd.Series,
    *,
    total_models: int,
    track_label_hint: str | None = None,
) -> str:
    selection_rank = pd.to_numeric(
        pd.Series([scorecard_row.get("selection_rank", np.nan)]), errors="coerce"
    ).iloc[0]
    if pd.isna(selection_rank):
        return "`NA`"
    rank_text = f"`{int(selection_rank)}/{int(total_models)}` overall"
    track_label = str(track_label_hint or scorecard_row.get("model_track", "") or "").strip()
    track_rank_column = {
        "discovery": "discovery_track_rank",
        "governance": "governance_track_rank",
        "baseline": "baseline_track_rank",
    }.get(track_label, "track_rank")
    track_rank = pd.to_numeric(
        pd.Series([scorecard_row.get(track_rank_column, scorecard_row.get("track_rank", np.nan))]),
        errors="coerce",
    ).iloc[0]
    if track_label and pd.notna(track_rank):
        rank_text += f"; `{int(track_rank)}` within the {track_label} track"
    return rank_text


def brier_skill_score(brier_score_value: object, prevalence_value: object) -> float:
    brier_score_numeric = pd.to_numeric(pd.Series([brier_score_value]), errors="coerce").iloc[0]
    prevalence_numeric = pd.to_numeric(pd.Series([prevalence_value]), errors="coerce").iloc[0]
    if pd.isna(brier_score_numeric) or pd.isna(prevalence_numeric):
        return float("nan")
    prevalence = float(prevalence_numeric)
    baseline_brier = prevalence * (1.0 - prevalence)
    if baseline_brier <= 0.0:
        return float("nan")
    return float(1.0 - (float(brier_score_numeric) / baseline_brier))


def build_spatial_holdout_summary(group_holdout: pd.DataFrame) -> pd.DataFrame:
    if group_holdout.empty:
        return pd.DataFrame()
    working = group_holdout.loc[
        group_holdout.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
        & group_holdout.get("group_column", pd.Series(dtype=str))
        .astype(str)
        .eq("dominant_region_train")
    ].copy()
    if working.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model_name, frame in working.groupby("model_name", sort=False):
        weights = pd.to_numeric(
            frame.get("n_test_backbones", pd.Series(0.0, index=frame.index)), errors="coerce"
        ).fillna(0.0)
        roc_auc = pd.to_numeric(
            frame.get("roc_auc", pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
        if weights.gt(0).any() and roc_auc.notna().any():
            weighted_auc = float(
                np.average(
                    roc_auc.fillna(0.0), weights=np.clip(weights.to_numpy(dtype=float), 1.0, None)
                )
            )
        else:
            weighted_auc = float(roc_auc.mean()) if roc_auc.notna().any() else np.nan
        best_region_row = frame.sort_values("roc_auc", ascending=False).head(1)
        worst_region_row = frame.sort_values("roc_auc", ascending=True).head(1)
        rows.append(
            {
                "model_name": str(model_name),
                "spatial_holdout_roc_auc": weighted_auc,
                "spatial_holdout_regions": int(frame["group_value"].astype(str).nunique()),
                "spatial_holdout_n_backbones": int(weights.sum()),
                "best_spatial_holdout_region": str(best_region_row.iloc[0]["group_value"])
                if not best_region_row.empty
                else "",
                "best_spatial_holdout_region_roc_auc": float(best_region_row.iloc[0]["roc_auc"])
                if not best_region_row.empty
                else np.nan,
                "worst_spatial_holdout_region": str(worst_region_row.iloc[0]["group_value"])
                if not worst_region_row.empty
                else "",
                "worst_spatial_holdout_region_roc_auc": float(worst_region_row.iloc[0]["roc_auc"])
                if not worst_region_row.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
