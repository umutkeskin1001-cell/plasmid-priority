"""Geo spread branch reporting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from plasmid_priority.modeling.module_a import ModelResult


from plasmid_priority.shared.report_utils import safe_series_value as _safe_series_value
from plasmid_priority.shared.report_utils import metric_value as _metric_value


def build_geo_spread_report_card(
    results: Mapping[str, ModelResult],
    *,
    calibration_summary: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
    selection_scorecard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact report card for the geo spread branch."""
    rows: list[dict[str, Any]] = []
    calibration_frame = calibration_summary if calibration_summary is not None else pd.DataFrame()
    ordered_results = sorted(
        results.items(),
        key=lambda item: (
            -float(item[1].metrics.get("roc_auc", float("-inf"))),
            str(item[0]),
        ),
    )
    for rank, (model_name, result) in enumerate(ordered_results, start=1):
        row: dict[str, Any] = {
            "rank": int(rank),
            "model_name": str(model_name),
            "status": result.status,
            "roc_auc": result.metrics.get("roc_auc", float("nan")),
            "average_precision": result.metrics.get("average_precision", float("nan")),
            "brier_score": result.metrics.get("brier_score", float("nan")),
            "expected_calibration_error": result.metrics.get(
                "expected_calibration_error", float("nan")
            ),
            "top_k_precision": result.metrics.get("precision_at_top_10", float("nan")),
            "top_k_recall": result.metrics.get("recall_at_top_10", float("nan")),
            "decision_utility_score": result.metrics.get("decision_utility_score", float("nan")),
            "novelty_adjusted_average_precision": result.metrics.get(
                "novelty_adjusted_average_precision", float("nan")
            ),
            "low_knownness_roc_auc": result.metrics.get("low_knownness_roc_auc", float("nan")),
            "low_knownness_average_precision": result.metrics.get(
                "low_knownness_average_precision", float("nan")
            ),
            "worst_knownness_quartile_roc_auc": result.metrics.get(
                "worst_knownness_quartile_roc_auc", float("nan")
            ),
            "event_within_3y_roc_auc": result.metrics.get("event_within_3y_roc_auc", float("nan")),
            "n_backbones": result.metrics.get("n_backbones", len(result.predictions)),
            "n_positive": result.metrics.get("n_positive", float("nan")),
            "error_message": result.error_message,
        }
        if not calibration_frame.empty:
            match = calibration_frame.loc[
                calibration_frame.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            ]
            if not match.empty:
                row["calibration_method"] = _safe_series_value(match, "calibration_method", "none")
                row["abstain_rate"] = _safe_series_value(match, "abstain_rate")
                row["mean_confidence"] = _safe_series_value(match, "mean_confidence")
                row["ood_rate"] = _safe_series_value(match, "ood_rate")
                row["calibrated_expected_calibration_error"] = _safe_series_value(
                    match, "calibrated_expected_calibration_error"
                )
                row["calibrated_brier_score"] = _safe_series_value(match, "calibrated_brier_score")
        rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["is_primary_candidate"] = (
        report["model_name"]
        .astype(str)
        .eq(str(provenance.get("primary_model_name", "")) if provenance else "")
    )
    report["is_headline_candidate"] = report["rank"].eq(1)
    if provenance:
        for key in (
            "benchmark_name",
            "split_year",
            "run_signature",
            "config_hash",
            "input_hash",
            "feature_surface_hash",
        ):
            report[key] = provenance.get(key)
    if selection_scorecard is not None and not selection_scorecard.empty:
        report = report.merge(
            selection_scorecard.loc[
                :,
                [
                    column
                    for column in selection_scorecard.columns
                    if column in {"model_name", "selection_score", "selection_rank"}
                ],
            ],
            on="model_name",
            how="left",
        )
        recommended_model = str(selection_scorecard.iloc[0]["model_name"])
        report["is_recommended_primary"] = report["model_name"].astype(str).eq(recommended_model)
    return report


def format_geo_spread_report_markdown(
    report_card: pd.DataFrame,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Render the geo spread report card as compact markdown."""
    if report_card.empty:
        return "# Geo spread report\n\nNo eligible models were evaluated.\n"

    top_row = report_card.iloc[0]
    recommended_row = (
        report_card.loc[
            report_card.get(
                "is_recommended_primary", pd.Series(False, index=report_card.index)
            ).fillna(False)
        ]
        if "is_recommended_primary" in report_card.columns
        else pd.DataFrame()
    )
    provenance_lines = []
    if provenance:
        provenance_lines = [
            f"- benchmark: `{provenance.get('benchmark_name', 'geo_spread_v1')}`",
            f"- split_year: `{provenance.get('split_year', '')}`",
            f"- run_signature: `{provenance.get('run_signature', '')}`",
            f"- primary_model_name: `{provenance.get('primary_model_name', '')}`",
        ]
        recommended = str(provenance.get("recommended_primary_model_name", "") or "").strip()
        if recommended:
            provenance_lines.append(f"- recommended_primary_model_name: `{recommended}`")

    lines = [
        "# Geo spread report",
        "",
        "## Branch summary",
        f"- evaluated_models: `{int(len(report_card))}`",
        f"- best_predictive_model: `{top_row['model_name']}`",
        f"- best_predictive_roc_auc: `{float(top_row['roc_auc']):.3f}`",
        f"- best_predictive_average_precision: `{float(top_row['average_precision']):.3f}`",
    ]
    best_low_knownness_auc = pd.to_numeric(
        pd.Series([top_row.get("low_knownness_roc_auc", float("nan"))]), errors="coerce"
    ).iloc[0]
    worst_low_knownness_auc = pd.to_numeric(
        pd.Series([top_row.get("worst_knownness_quartile_roc_auc", float("nan"))]), errors="coerce"
    ).iloc[0]
    within_3y_auc = pd.to_numeric(
        pd.Series([top_row.get("event_within_3y_roc_auc", float("nan"))]), errors="coerce"
    ).iloc[0]
    if pd.notna(best_low_knownness_auc):
        lines.append(
            f"- best_predictive_low_knownness_roc_auc: `{float(best_low_knownness_auc):.3f}`"
        )
    if pd.notna(worst_low_knownness_auc):
        worst_auc_value = float(worst_low_knownness_auc)
        lines.append(f"- best_predictive_worst_knownness_quartile_roc_auc: `{worst_auc_value:.3f}`")
    if pd.notna(within_3y_auc):
        lines.append(f"- best_predictive_event_within_3y_roc_auc: `{float(within_3y_auc):.3f}`")
    if not recommended_row.empty:
        recommended = recommended_row.iloc[0]
        lines.append(f"- recommended_primary_model: `{recommended['model_name']}`")
        if "selection_score" in recommended.index:
            recommended_score = _metric_value(recommended, "selection_score")
            lines.append(f"- recommended_primary_selection_score: `{recommended_score:.3f}`")
    if "abstain_rate" in report_card.columns:
        reference_row = recommended_row.iloc[0] if not recommended_row.empty else top_row
        best_abstain = pd.to_numeric(
            pd.Series([reference_row.get("abstain_rate", float("nan"))]), errors="coerce"
        ).iloc[0]
        if pd.notna(best_abstain):
            lines.append(f"- recommended_primary_abstain_rate: `{float(best_abstain):.3f}`")
    if provenance_lines:
        lines.extend(["", "## Provenance", *provenance_lines])
    lines.extend(
        [
            "",
            "## Ranked models",
        ]
    )
    for _, row in report_card.iterrows():
        low_knownness_auc = _metric_value(row, "low_knownness_roc_auc")
        low_knownness_suffix = (
            f", low-knownness AUC `{float(low_knownness_auc):.3f}`"
            if pd.notna(low_knownness_auc)
            else ""
        )
        worst_low_knownness_auc = _metric_value(row, "worst_knownness_quartile_roc_auc")
        worst_low_knownness_suffix = (
            f", worst-knownness quartile AUC `{float(worst_low_knownness_auc):.3f}`"
            if pd.notna(worst_low_knownness_auc)
            else ""
        )
        ece_value = float(
            pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "calibrated_expected_calibration_error",
                            row.get("expected_calibration_error", float("nan")),
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
        )
        roc_auc = _metric_value(row, "roc_auc")
        average_precision = _metric_value(row, "average_precision")
        lines.append(
            f"- `{row['model_name']}`: ROC AUC `{roc_auc:.3f}`, AP "
            f"`{average_precision:.3f}`, calibrated ECE `{ece_value:.3f}`"
            f"{low_knownness_suffix}{worst_low_knownness_suffix}"
        )
    lines.append("")
    return "\n".join(lines)
