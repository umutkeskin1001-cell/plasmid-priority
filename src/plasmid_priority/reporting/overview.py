"""Compact overview panels for report assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def build_report_overview_table(
    *,
    model_selection_summary: pd.DataFrame,
    decision_yield: pd.DataFrame,
    threshold_utility_summary: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    candidate_case_studies: pd.DataFrame,
    false_negative_audit: pd.DataFrame,
) -> pd.DataFrame:
    if model_selection_summary.empty:
        return pd.DataFrame()
    selection_row = model_selection_summary.iloc[0]
    primary_model = str(selection_row.get("published_primary_model", ""))
    governance_model = str(selection_row.get("governance_primary_model", ""))
    conservative_model = str(selection_row.get("conservative_model_name", ""))

    def _series_lookup(frame: pd.DataFrame, model_name: str, column: str) -> float:
        if frame.empty or model_name == "" or "model_name" not in frame.columns:
            return float("nan")
        match = frame.loc[frame["model_name"].astype(str) == str(model_name)]
        if match.empty or column not in match.columns:
            return float("nan")
        return _safe_float(match.iloc[0].get(column, np.nan))

    def _decision_lookup(model_name: str, top_k: int, column: str) -> float:
        if decision_yield.empty:
            return float("nan")
        top_k_series = pd.to_numeric(
            decision_yield.get("top_k", pd.Series(dtype=float)),
            errors="coerce",
        )
        match = decision_yield.loc[
            (decision_yield.get("model_name", pd.Series(dtype=str)).astype(str) == model_name)
            & (top_k_series == top_k)
        ]
        if match.empty or column not in match.columns:
            return float("nan")
        return float(pd.to_numeric(match.iloc[0].get(column, np.nan), errors="coerce"))

    def _utility_lookup(model_name: str, column: str) -> float:
        if threshold_utility_summary.empty:
            return float("nan")
        match = threshold_utility_summary.loc[
            threshold_utility_summary.get("model_name", pd.Series(dtype=str)).astype(str)
            == model_name
        ]
        if match.empty or column not in match.columns:
            return float("nan")
        return float(pd.to_numeric(match.iloc[0].get(column, np.nan), errors="coerce"))

    rows: list[dict[str, object]] = [
        {
            "panel_item": "published_primary_model",
            "metric_name": "model_name",
            "metric_value": primary_model,
        },
        {
            "panel_item": "published_primary_roc_auc",
            "metric_name": "roc_auc",
            "metric_value": _safe_float(selection_row.get("published_primary_roc_auc", np.nan)),
        },
        {
            "panel_item": "published_primary_ap",
            "metric_name": "average_precision",
            "metric_value": _safe_float(
                selection_row.get("published_primary_average_precision", np.nan)
            ),
        },
        {
            "panel_item": "published_primary_top_10_precision",
            "metric_name": "precision_at_k",
            "metric_value": _decision_lookup(primary_model, 10, "precision_at_k"),
        },
        {
            "panel_item": "published_primary_optimal_threshold",
            "metric_name": "optimal_threshold",
            "metric_value": _utility_lookup(primary_model, "optimal_threshold"),
        },
        {
            "panel_item": "governance_model",
            "metric_name": "model_name",
            "metric_value": governance_model,
        },
        {
            "panel_item": "governance_roc_auc",
            "metric_name": "roc_auc",
            "metric_value": _safe_float(selection_row.get("governance_primary_roc_auc", np.nan)),
        },
        {
            "panel_item": "governance_top_10_precision",
            "metric_name": "precision_at_k",
            "metric_value": _decision_lookup(governance_model, 10, "precision_at_k"),
        },
        {
            "panel_item": "governance_optimal_threshold",
            "metric_name": "optimal_threshold",
            "metric_value": _utility_lookup(governance_model, "optimal_threshold"),
        },
        {
            "panel_item": "conservative_model",
            "metric_name": "model_name",
            "metric_value": conservative_model,
        },
        {
            "panel_item": "candidate_portfolio_size",
            "metric_name": "n_rows",
            "metric_value": int(len(candidate_portfolio)),
        },
        {
            "panel_item": "candidate_case_studies_size",
            "metric_name": "n_rows",
            "metric_value": int(len(candidate_case_studies)),
        },
        {
            "panel_item": "false_negative_audit_size",
            "metric_name": "n_rows",
            "metric_value": int(len(false_negative_audit)),
        },
    ]
    if not threshold_utility_summary.empty:
        utility_rows = threshold_utility_summary.loc[
            threshold_utility_summary.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .isin({primary_model, governance_model, conservative_model})
        ].copy()
        if not utility_rows.empty:
            utility_rows["panel_item"] = utility_rows["model_name"].astype(str) + "_utility"
            utility_rows["metric_name"] = "optimal_threshold_utility_per_sample"
            utility_rows["metric_value"] = pd.to_numeric(
                utility_rows.get(
                    "optimal_threshold_utility_per_sample",
                    pd.Series(np.nan, index=utility_rows.index),
                ),
                errors="coerce",
            )
            rows.extend(
                utility_rows[["panel_item", "metric_name", "metric_value"]].to_dict(
                    orient="records"
                )
            )
    return pd.DataFrame(rows)
