"""Clinical hazard branch reporting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


from plasmid_priority.shared.report_utils import safe_series_value as _safe_series_value
from plasmid_priority.shared.report_utils import metric_value as _metric_value


def build_clinical_hazard_report_card(
    results: Mapping[str, Any],
    *,
    calibration_summary: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
    selection_scorecard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration_frame = calibration_summary if calibration_summary is not None else pd.DataFrame()
    ordered_results = sorted(
        results.items(),
        key=lambda item: (-float(item[1].metrics.get("roc_auc", float("-inf"))), str(item[0])),
    )
    for rank, (model_name, result) in enumerate(ordered_results, start=1):
        row = {
            "rank": int(rank),
            "model_name": str(model_name),
            "status": getattr(result, "status", "ok"),
            "roc_auc": result.metrics.get("roc_auc", float("nan")),
            "average_precision": result.metrics.get("average_precision", float("nan")),
            "brier_score": result.metrics.get("brier_score", float("nan")),
            "expected_calibration_error": result.metrics.get(
                "expected_calibration_error", float("nan")
            ),
            "n_backbones": result.metrics.get(
                "n_backbones", len(getattr(result, "predictions", []))
            ),
            "n_positive": result.metrics.get("n_positive", float("nan")),
            "error_message": getattr(result, "error_message", None),
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
    return report


def format_clinical_hazard_report_markdown(
    report_card: pd.DataFrame,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    if report_card.empty:
        return "# Clinical hazard report\n\nNo eligible models were evaluated.\n"
    top_row = report_card.iloc[0]
    lines = [
        "# Clinical hazard report",
        "",
        "## Branch summary",
        f"- evaluated_models: `{int(len(report_card))}`",
        f"- best_predictive_model: `{top_row['model_name']}`",
        f"- best_predictive_roc_auc: `{float(top_row['roc_auc']):.3f}`",
        f"- best_predictive_average_precision: `{float(top_row['average_precision']):.3f}`",
    ]
    if provenance:
        lines.extend(
            [
                "",
                "## Provenance",
                f"- benchmark: `{provenance.get('benchmark_name', 'clinical_hazard_v1')}`",
                f"- split_year: `{provenance.get('split_year', '')}`",
                f"- run_signature: `{provenance.get('run_signature', '')}`",
                f"- primary_model_name: `{provenance.get('primary_model_name', '')}`",
            ]
        )
    lines.extend(["", "## Ranked models"])
    for _, row in report_card.iterrows():
        roc_auc = _metric_value(row, "roc_auc")
        average_precision = _metric_value(row, "average_precision")
        lines.append(
            f"- `{row['model_name']}`: ROC AUC `{roc_auc:.3f}`, AP `{average_precision:.3f}`"
        )
    lines.append("")
    return "\n".join(lines)
