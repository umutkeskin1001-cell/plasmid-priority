"""Surface assembly helpers extracted from scripts/24_build_reports.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.reporting.model_audit import build_official_benchmark_panel


def lookup_decision_yield(
    decision_yield: pd.DataFrame, model_name: str, top_k: int
) -> pd.Series | None:
    if decision_yield.empty:
        return None
    match = decision_yield.loc[
        (decision_yield["model_name"] == model_name) & (decision_yield["top_k"] == int(top_k))
    ]
    if match.empty:
        return None
    return match.iloc[0]


def build_official_benchmark_context(
    model_selection_summary: pd.DataFrame,
    decision_yield: pd.DataFrame,
    benchmark_protocol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if model_selection_summary.empty and decision_yield.empty:
        return pd.DataFrame()
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )
    primary_model_name = str(selection_row.get("published_primary_model", "") or "").strip()
    conservative_model_name = str(selection_row.get("conservative_model_name", "") or "").strip()
    governance_model_name = str(selection_row.get("governance_primary_model", "") or "").strip()

    def _yield_metrics(model_name: str, top_k: int) -> tuple[float, float]:
        if not model_name:
            return (np.nan, np.nan)
        row = lookup_decision_yield(decision_yield, model_name, top_k)
        if row is None:
            return (np.nan, np.nan)
        return (
            float(row.get("precision_at_k", np.nan)),
            float(row.get("recall_at_k", np.nan)),
        )

    primary_top10_precision, primary_top10_recall = _yield_metrics(primary_model_name, 10)
    primary_top25_precision, primary_top25_recall = _yield_metrics(primary_model_name, 25)
    governance_top10_precision, governance_top10_recall = _yield_metrics(governance_model_name, 10)
    governance_top25_precision, governance_top25_recall = _yield_metrics(governance_model_name, 25)
    conservative_top10_precision, conservative_top10_recall = _yield_metrics(
        conservative_model_name, 10
    )

    def _selection_value(column: str) -> float:
        value = selection_row.get(column, np.nan)
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])

    panel_size = (
        int(len(build_official_benchmark_panel(benchmark_protocol)))
        if benchmark_protocol is not None and not benchmark_protocol.empty
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "official_primary_model": primary_model_name,
                "official_conservative_model": conservative_model_name,
                "official_governance_model": governance_model_name,
                "official_primary_top_10_precision": primary_top10_precision,
                "official_primary_top_10_recall": primary_top10_recall,
                "official_primary_top_25_precision": primary_top25_precision,
                "official_primary_top_25_recall": primary_top25_recall,
                "official_primary_decision_utility_score": _selection_value(
                    "published_primary_decision_utility_score"
                ),
                "official_primary_optimal_decision_threshold": _selection_value(
                    "published_primary_optimal_decision_threshold"
                ),
                "official_governance_top_10_precision": governance_top10_precision,
                "official_governance_top_10_recall": governance_top10_recall,
                "official_governance_top_25_precision": governance_top25_precision,
                "official_governance_top_25_recall": governance_top25_recall,
                "official_governance_decision_utility_score": _selection_value(
                    "governance_primary_decision_utility_score"
                ),
                "official_governance_optimal_decision_threshold": _selection_value(
                    "governance_primary_optimal_decision_threshold"
                ),
                "official_conservative_top_10_precision": conservative_top10_precision,
                "official_conservative_top_10_recall": conservative_top10_recall,
                "official_conservative_decision_utility_score": _selection_value(
                    "conservative_decision_utility_score"
                ),
                "official_conservative_optimal_decision_threshold": _selection_value(
                    "conservative_optimal_decision_threshold"
                ),
                "official_benchmark_panel_size": panel_size,
            }
        ]
    )


def attach_official_benchmark_context(
    frame: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty or context.empty:
        return frame
    working = frame.copy()
    context_row = context.iloc[0]
    for column, value in context_row.items():
        working[column] = value
    return working


def prune_duplicate_table_artifacts(
    core_dir: Path, diag_dir: Path, core_file_names: set[str]
) -> None:
    core_names = {path.name for path in core_dir.glob("*.tsv")}
    diag_names = {path.name for path in diag_dir.glob("*.tsv")}
    for name in sorted(core_names & diag_names):
        stale_path = diag_dir / name if name in core_file_names else core_dir / name
        if stale_path.exists():
            stale_path.unlink()


def prune_shadowed_report_tables(
    core_dir: Path,
    diag_dir: Path,
    analysis_dir: Path,
    *,
    preserve_file_names: set[str] | None = None,
) -> None:
    preserve_file_names = preserve_file_names or set()
    analysis_names = {path.name for path in analysis_dir.glob("*.tsv")}
    for directory in (core_dir, diag_dir):
        for path in directory.glob("*.tsv"):
            if path.name in preserve_file_names:
                continue
            if path.name in analysis_names and path.exists():
                path.unlink()


def select_summary_candidate_briefs(
    candidate_briefs: pd.DataFrame, *, per_track: int = 5
) -> pd.DataFrame:
    if candidate_briefs.empty or "portfolio_track" not in candidate_briefs.columns:
        return candidate_briefs.head(per_track * 2).copy()
    tracks = ["established_high_risk", "novel_signal"]
    frames: list[pd.DataFrame] = []
    for track_order, track in enumerate(tracks):
        frame = (
            candidate_briefs.loc[candidate_briefs["portfolio_track"] == track]
            .head(per_track)
            .copy()
        )
        if frame.empty:
            continue
        frame["_summary_track_order"] = track_order
        frame["_summary_row_order"] = range(len(frame))
        frames.append(frame)
    if not frames:
        return candidate_briefs.head(per_track * 2).copy()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.sort_values(["_summary_row_order", "_summary_track_order"])
        .drop(columns=["_summary_track_order", "_summary_row_order"])
        .reset_index(drop=True)
    )


def primary_baseline_delta_text(model_metrics: pd.DataFrame, primary_model_name: str) -> str:
    matching_rows = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(primary_model_name)
    ].head(1)
    if matching_rows.empty:
        return "NA"
    row = matching_rows.iloc[0]
    delta = pd.to_numeric(pd.Series([row.get("delta_vs_baseline_roc_auc")]), errors="coerce").iloc[
        0
    ]
    ci_low = pd.to_numeric(
        pd.Series([row.get("delta_vs_baseline_ci_lower")]), errors="coerce"
    ).iloc[0]
    ci_high = pd.to_numeric(
        pd.Series([row.get("delta_vs_baseline_ci_upper")]), errors="coerce"
    ).iloc[0]
    if pd.isna(delta):
        return "NA"
    if pd.notna(ci_low) and pd.notna(ci_high):
        return f"{float(delta):+.3f} [{float(ci_low):+.3f}, {float(ci_high):+.3f}]"
    return f"{float(delta):+.3f}"
