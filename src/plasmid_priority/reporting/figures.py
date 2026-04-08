"""Final report figure generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plasmid_priority_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score as skl_roc_auc_score

from plasmid_priority.reporting.figure_style import (
    PALETTE,
)
from plasmid_priority.reporting.figure_style import (
    annotate_bar_values as _annotate_bar_values,
)
from plasmid_priority.reporting.figure_style import (
    apply_axis_style as _apply_axis_style,
)
from plasmid_priority.reporting.figure_style import (
    candidate_tick_label as _candidate_tick_label,
)
from plasmid_priority.reporting.figure_style import (
    format_pvalue as _format_pvalue,
)
from plasmid_priority.reporting.figure_style import (
    palette_sequence as _palette_sequence,
)
from plasmid_priority.reporting.figure_style import (
    pretty_feature_label as _pretty_feature_label,
)
from plasmid_priority.reporting.figure_style import (
    pretty_model_label as _pretty_model_label,
)
from plasmid_priority.reporting.figure_style import (
    pretty_sensitivity_label as _pretty_sensitivity_label,
)
from plasmid_priority.reporting.figure_style import (
    stable_jitter as _stable_jitter,
)
from plasmid_priority.reporting.figure_style import (
    style as _style,
)
from plasmid_priority.utils.files import ensure_directory
from plasmid_priority.validation.metrics import average_precision


def plot_score_distribution(scored: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = scored.copy()
    working["operational_priority_index"] = working.get(
        "operational_priority_index", working.get("priority_index", 0.0)
    ).fillna(working.get("priority_index", 0.0))
    working["training_support_group"] = np.where(
        working["member_count_train"].fillna(0).astype(int) > 0,
        "training-supported",
        "no training support",
    )
    low_threshold = 0.25
    eligible = working.loc[working["spread_label"].notna()].copy()
    eligible_low_cluster = eligible.loc[
        eligible["operational_priority_index"].fillna(0.0) < low_threshold
    ].copy()
    bio_threshold = (
        float(eligible["bio_priority_index"].quantile(0.25)) if not eligible.empty else 0.25
    )
    evidence_threshold = (
        float(eligible["evidence_support_index"].quantile(0.25)) if not eligible.empty else 0.25
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.4))
    bins = np.linspace(0.0, 1.0, 36)
    support_order = ["no training support", "training-supported"]
    support_summary = []
    for group in support_order:
        frame = working.loc[working["training_support_group"] == group].copy()
        total = int(len(frame))
        low_n = (
            int(frame["operational_priority_index"].fillna(0.0).lt(low_threshold).sum())
            if total
            else 0
        )
        support_summary.append(
            {
                "group": group,
                "total": total,
                "low_n": low_n,
                "remaining": max(total - low_n, 0),
                "low_share": float(low_n / total) if total else 0.0,
            }
        )
    positions = np.arange(len(support_summary))
    axes[0].bar(
        positions,
        [row["remaining"] for row in support_summary],
        color=[PALETTE["muted"], PALETTE["primary"]],
        edgecolor="#23313A",
        linewidth=0.4,
        label=f"score >= {low_threshold:.2f}",
    )
    axes[0].bar(
        positions,
        [row["low_n"] for row in support_summary],
        bottom=[row["remaining"] for row in support_summary],
        color=PALETTE["accent"],
        edgecolor="#23313A",
        linewidth=0.4,
        label=f"score < {low_threshold:.2f}",
    )
    for idx, row in enumerate(support_summary):
        total = int(cast(int, row["total"]))
        axes[0].text(
            idx,
            total + max(18, total * 0.01),
            f"n={row['total']}\nlow={row['low_share']:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("Support Status vs Low-Score Backlog")
    axes[0].set_ylabel("Backbone count")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(["no training\nsupport", "training-\nsupported"])
    axes[0].legend(frameon=False, loc="upper right")
    _apply_axis_style(axes[0], grid_axis="y")

    eligible_bio = eligible["bio_priority_index"].fillna(0.0)
    eligible_evidence = eligible["evidence_support_index"].fillna(0.0)
    axes[1].hist(
        [eligible_bio, eligible_evidence],
        bins=bins,
        color=[PALETTE["support"], PALETTE["accent"]],
        edgecolor="white",
        alpha=0.82,
        density=True,
        label=["bio priority index", "evidence support index"],
    )
    axes[1].axvline(bio_threshold, color=PALETTE["support"], linestyle=":", linewidth=1.5)
    axes[1].axvline(evidence_threshold, color=PALETTE["accent"], linestyle=":", linewidth=1.5)
    axes[1].set_title("Eligible Cohort: Biology vs Evidence")
    axes[1].set_xlabel("index value")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].text(
        0.98,
        0.96,
        (
            f"eligible cohort: n={len(eligible)}\n"
            f"bio q1: {bio_threshold:.2f}\n"
            f"evidence q1: {evidence_threshold:.2f}"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[1], grid_axis="y")

    component_labels = {
        "T_eff_norm": "T most\nlimiting",
        "H_eff_norm": "H most\nlimiting",
        "A_eff_norm": "A most\nlimiting",
        "tie": "tied\nlimiters",
    }
    decomposition_counts = {label: 0 for label in component_labels.values()}
    for row in eligible_low_cluster[["T_eff_norm", "H_eff_norm", "A_eff_norm"]].to_dict(
        orient="records"
    ):
        values = {
            key: float(row.get(key, np.nan)) for key in ("T_eff_norm", "H_eff_norm", "A_eff_norm")
        }
        finite = {key: value for key, value in values.items() if np.isfinite(value)}
        if not finite:
            continue
        min_value = min(finite.values())
        tied = [key for key, value in finite.items() if abs(value - min_value) <= 0.03]
        if len(tied) > 1:
            decomposition_counts[component_labels["tie"]] += 1
        else:
            decomposition_counts[component_labels[tied[0]]] += 1
    positions = np.arange(len(decomposition_counts))
    counts = list(decomposition_counts.values())
    axes[2].bar(
        positions,
        counts,
        color=[PALETTE["accent"], PALETTE["primary"], PALETTE["support"], PALETTE["muted"]],
        edgecolor="#23313A",
        linewidth=0.4,
    )
    for idx, total in enumerate(counts):
        if total == 0:
            continue
        axes[2].text(idx, total + 8, str(int(total)), ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Most Limiting Effective Component in the Low-Score Cohort")
    axes[2].set_xlabel("Effective component pattern")
    axes[2].set_ylabel("Backbone count")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(list(decomposition_counts.keys()))
    axes[2].text(
        0.98,
        0.96,
        (
            f"eligible low-operational cohort: n={len(eligible_low_cluster)}\n"
            f"low-score cutoff: < {low_threshold:.2f}\n"
            "classification uses the smallest effective\n"
            "T / H / A component per backbone"
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[2], grid_axis="y")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.24, wspace=0.34)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc_curve(predictions: pd.DataFrame, output_path: Path, model_names: list[str]) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = _palette_sequence(len(model_names))
    for color, model_name in zip(colors, model_names):
        frame = predictions.loc[predictions["model_name"] == model_name]
        fpr, tpr, _ = roc_curve(frame["spread_label"], frame["oof_prediction"])
        auc = float(skl_roc_auc_score(frame["spread_label"], frame["oof_prediction"]))
        ax.plot(
            fpr,
            tpr,
            label=f"{_pretty_model_label(model_name)} ({auc:.3f})",
            color=color,
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    _apply_axis_style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pr_curve(predictions: pd.DataFrame, output_path: Path, model_names: list[str]) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = _palette_sequence(len(model_names))
    prevalence = (
        float(predictions.loc[predictions["model_name"] == model_names[0], "spread_label"].mean())
        if model_names
        else 0.0
    )
    for color, model_name in zip(colors, model_names):
        frame = predictions.loc[predictions["model_name"] == model_name]
        precision, recall, _ = precision_recall_curve(
            frame["spread_label"], frame["oof_prediction"]
        )
        ap = float(average_precision(
            frame["spread_label"].to_numpy(),
            frame["oof_prediction"].to_numpy()
        ))
        lift = ap - prevalence
        ax.plot(
            recall,
            precision,
            label=f"{_pretty_model_label(model_name)} (AP {ap:.3f}, lift {lift:+.3f})",
            color=color,
            linewidth=2,
        )
    ax.axhline(
        prevalence,
        linestyle="--",
        color=PALETTE["muted"],
        linewidth=1,
        label=f"class prevalence ({prevalence:.3f})",
    )
    ax.set_title("Precision-Recall Curve With Prevalence Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    _apply_axis_style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _calibration_bin_frame(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return pd.DataFrame(
            columns=["mean_prediction", "observed_rate", "n_backbones", "bin_index"]
        )
    order = np.argsort(y_score)
    bins = [
        indices
        for indices in np.array_split(order, min(int(n_bins), len(y_true)))
        if len(indices) > 0
    ]
    rows = []
    for bin_index, indices in enumerate(bins, start=1):
        rows.append(
            {
                "bin_index": int(bin_index),
                "mean_prediction": float(y_score[indices].mean()),
                "observed_rate": float(y_true[indices].mean()),
                "n_backbones": int(len(indices)),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_diagram(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    _style()
    calibration = _calibration_bin_frame(y_true, y_score, n_bins=n_bins)
    if ax is None:
        fig, (top_ax, bottom_ax) = plt.subplots(
            2,
            1,
            figsize=(7.6, 8.6),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.2]},
        )
    else:
        top_ax, bottom_ax = ax
        fig = cast(plt.Figure, top_ax.figure)
    if calibration.empty:
        return fig

    top_ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
    top_ax.plot(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        color=PALETTE["primary"],
        linewidth=2,
        marker="o",
    )
    point_sizes = 28 + 1.4 * calibration["n_backbones"].to_numpy(dtype=float)
    top_ax.scatter(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        s=point_sizes,
        color=PALETTE["accent"],
        edgecolor="#23313A",
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
    )
    for row in calibration.itertuples(index=False):
        top_ax.annotate(
            f"n={int(row.n_backbones)}",
            (row.mean_prediction, row.observed_rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    top_ax.set_ylabel("Observed positive fraction")
    top_ax.set_xlim(0.0, 1.0)
    top_ax.set_ylim(0.0, 1.0)
    _apply_axis_style(top_ax, grid_axis="both")

    bottom_ax.bar(
        calibration["mean_prediction"],
        calibration["n_backbones"],
        width=max(0.9 / max(len(calibration), 1), 0.05),
        color=PALETTE["support"],
        edgecolor="#23313A",
        linewidth=0.4,
        alpha=0.85,
    )
    bottom_ax.set_xlabel("Mean predicted probability")
    bottom_ax.set_ylabel("Bin count")
    bottom_ax.set_xlim(0.0, 1.0)
    _apply_axis_style(bottom_ax, grid_axis="y")
    return fig


def plot_calibration(calibration: pd.DataFrame, output_path: Path, model_name: str) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        figsize=(7.6, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )
    top_ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
    top_ax.plot(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        color=PALETTE["primary"],
        linewidth=2,
        marker="o",
    )
    lower = calibration.get("observed_rate_ci_lower", calibration["observed_rate"]).fillna(
        calibration["observed_rate"]
    )
    upper = calibration.get("observed_rate_ci_upper", calibration["observed_rate"]).fillna(
        calibration["observed_rate"]
    )
    yerr = np.vstack(
        [
            (calibration["observed_rate"] - lower).clip(lower=0.0).to_numpy(dtype=float),
            (upper - calibration["observed_rate"]).clip(lower=0.0).to_numpy(dtype=float),
        ]
    )
    top_ax.errorbar(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        yerr=yerr,
        fmt="none",
        ecolor="#23313A",
        capsize=3,
        linewidth=1,
        zorder=2,
    )
    point_sizes = 28 + 1.4 * calibration["n_backbones"].clip(lower=0).to_numpy(dtype=float)
    top_ax.scatter(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        s=point_sizes,
        color=PALETTE["accent"],
        edgecolor="#23313A",
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
    )
    for row in calibration.itertuples(index=False):
        top_ax.annotate(
            f"n={int(row.n_backbones)}",
            (row.mean_prediction, row.observed_rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    top_ax.set_title(f"Calibration Diagram ({_pretty_model_label(model_name)})")
    top_ax.set_ylabel("Observed visibility-expansion rate")
    top_ax.set_xlim(0.0, 1.0)
    top_ax.set_ylim(0.0, 1.0)
    _apply_axis_style(top_ax, grid_axis="both")
    bottom_ax.bar(
        calibration["mean_prediction"],
        calibration["n_backbones"],
        width=max(0.9 / max(len(calibration), 1), 0.05),
        color=PALETTE["support"],
        edgecolor="#23313A",
        linewidth=0.4,
        alpha=0.85,
    )
    bottom_ax.set_xlabel("Mean predicted probability")
    bottom_ax.set_ylabel("Bin count")
    bottom_ax.set_xlim(0.0, 1.0)
    _apply_axis_style(bottom_ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration_threshold_summary(
    calibration: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    if calibration.empty and threshold_sensitivity.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))

    calibration_frame = calibration.copy()
    if not calibration_frame.empty:
        calibration_frame["mean_prediction"] = pd.to_numeric(
            calibration_frame.get(
                "mean_prediction",
                pd.Series(np.nan, index=calibration_frame.index),
            ),
            errors="coerce",
        )
        calibration_frame["observed_rate"] = pd.to_numeric(
            calibration_frame.get(
                "observed_rate",
                pd.Series(np.nan, index=calibration_frame.index),
            ),
            errors="coerce",
        )
        calibration_frame["n_backbones"] = pd.to_numeric(
            calibration_frame.get("n_backbones", pd.Series(0.0, index=calibration_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        calibration_frame = calibration_frame.loc[
            calibration_frame["mean_prediction"].notna()
            & calibration_frame["observed_rate"].notna()
        ].copy()

    if calibration_frame.empty:
        axes[0].text(
            0.5,
            0.5,
            "Calibration data unavailable",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_axis_off()
    else:
        axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
        axes[0].plot(
            calibration_frame["mean_prediction"],
            calibration_frame["observed_rate"],
            color=PALETTE["primary"],
            linewidth=2,
            marker="o",
        )
        point_sizes = 30 + 1.6 * calibration_frame["n_backbones"].clip(lower=0).to_numpy(
            dtype=float
        )
        axes[0].scatter(
            calibration_frame["mean_prediction"],
            calibration_frame["observed_rate"],
            s=point_sizes,
            color=PALETTE["accent"],
            edgecolor="#23313A",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        for row in calibration_frame.itertuples(index=False):
            axes[0].annotate(
                f"n={int(row.n_backbones)}",
                (row.mean_prediction, row.observed_rate),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        calibration_error = np.abs(
            calibration_frame["observed_rate"].to_numpy(dtype=float)
            - calibration_frame["mean_prediction"].to_numpy(dtype=float)
        )
        finite_mask = np.isfinite(calibration_error)
        finite_error = calibration_error[finite_mask]
        weights = calibration_frame["n_backbones"].to_numpy(dtype=float)
        if finite_error.size:
            if np.isfinite(weights).any() and float(np.nansum(weights)) > 0.0:
                ece = float(
                    np.average(
                        finite_error,
                        weights=np.clip(weights[finite_mask], 0.0, None),
                    )
                )
            else:
                ece = float(finite_error.mean())
            max_ce = float(finite_error.max())
        else:
            ece = float("nan")
            max_ce = float("nan")
        axes[0].text(
            0.98,
            0.04,
            f"bins={len(calibration_frame)}\nECE={ece:.3f}\nMax CE={max_ce:.3f}",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
        )
        axes[0].set_title(f"Calibration ({_pretty_model_label(model_name)})")
        axes[0].set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Observed visibility-expansion rate")
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        _apply_axis_style(axes[0], grid_axis="both")

    threshold_frame = threshold_sensitivity.copy()
    threshold_column = "new_country_threshold"
    if "new_country_threshold" not in threshold_frame.columns:
        threshold_column = "threshold"
    if not threshold_frame.empty:
        threshold_frame[threshold_column] = pd.to_numeric(
            threshold_frame.get(threshold_column, pd.Series(np.nan, index=threshold_frame.index)),
            errors="coerce",
        )
        threshold_frame["roc_auc"] = pd.to_numeric(
            threshold_frame.get("roc_auc", pd.Series(np.nan, index=threshold_frame.index)),
            errors="coerce",
        )
        threshold_frame["roc_auc_ci_lower"] = pd.to_numeric(
            threshold_frame.get(
                "roc_auc_ci_lower",
                threshold_frame.get(
                    "roc_auc",
                    pd.Series(np.nan, index=threshold_frame.index),
                ),
            ),
            errors="coerce",
        )
        threshold_frame["roc_auc_ci_upper"] = pd.to_numeric(
            threshold_frame.get(
                "roc_auc_ci_upper",
                threshold_frame.get(
                    "roc_auc",
                    pd.Series(np.nan, index=threshold_frame.index),
                ),
            ),
            errors="coerce",
        )
        threshold_frame["average_precision"] = pd.to_numeric(
            threshold_frame.get(
                "average_precision", pd.Series(np.nan, index=threshold_frame.index)
            ),
            errors="coerce",
        )
        threshold_frame["n_eligible_backbones"] = pd.to_numeric(
            threshold_frame.get(
                "n_eligible_backbones", pd.Series(0.0, index=threshold_frame.index)
            ),
            errors="coerce",
        ).fillna(0.0)
        threshold_frame = threshold_frame.loc[
            threshold_frame[threshold_column].notna() & threshold_frame["roc_auc"].notna()
        ].copy()
        threshold_frame = threshold_frame.sort_values(threshold_column)
        default_threshold = threshold_frame.loc[
            threshold_frame.get("variant", pd.Series("", index=threshold_frame.index))
            .astype(str)
            .eq("default"),
            threshold_column,
        ].head(1)
    else:
        default_threshold = pd.Series(dtype=float)

    if threshold_frame.empty:
        axes[1].text(
            0.5,
            0.5,
            "Threshold sensitivity data unavailable",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[1].set_axis_off()
    else:
        x_values = threshold_frame[threshold_column].to_numpy(dtype=float)
        roc_auc_values = threshold_frame["roc_auc"].to_numpy(dtype=float)
        ap_values = threshold_frame["average_precision"].to_numpy(dtype=float)
        lower = threshold_frame["roc_auc_ci_lower"].to_numpy(dtype=float)
        upper = threshold_frame["roc_auc_ci_upper"].to_numpy(dtype=float)
        ax = axes[1]
        ax.plot(
            x_values,
            roc_auc_values,
            color=PALETTE["primary"],
            marker="o",
            linewidth=2.2,
            label="ROC AUC",
        )
        ci_mask = np.isfinite(lower) & np.isfinite(upper)
        if ci_mask.any():
            ax.fill_between(
                x_values[ci_mask],
                lower[ci_mask],
                upper[ci_mask],
                color=PALETTE["support"],
                alpha=0.22,
                label="ROC AUC CI",
            )
        ax.plot(
            x_values,
            ap_values,
            color=PALETTE["accent"],
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label="Average precision",
        )
        point_sizes = 34 + 1.2 * threshold_frame["n_eligible_backbones"].clip(lower=0).to_numpy(
            dtype=float
        )
        ax.scatter(
            x_values,
            roc_auc_values,
            s=point_sizes,
            color=PALETTE["primary"],
            edgecolor="#23313A",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            x_values,
            ap_values,
            s=point_sizes * 0.75,
            color=PALETTE["accent"],
            edgecolor="#23313A",
            linewidth=0.5,
            alpha=0.8,
            zorder=3,
        )
        for row in threshold_frame.itertuples(index=False):
            threshold_value = int(getattr(row, threshold_column))
            auc_value = float(row.roc_auc)
            ap_value = float(row.average_precision)
            ax.annotate(
                f"T={threshold_value}\nAUC {auc_value:.3f}\nAP {ap_value:.3f}",
                (threshold_value, auc_value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
            )
        if not default_threshold.empty and pd.notna(default_threshold.iloc[0]):
            ax.axvline(
                float(default_threshold.iloc[0]),
                color=PALETTE["muted"],
                linestyle=":",
                linewidth=1,
            )
        ax.set_title("Outcome-threshold sensitivity")
        ax.set_xlabel("Later new-country threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([int(value) for value in x_values.tolist()])
        ax.legend(frameon=False, loc="lower left")
        _apply_axis_style(ax, grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics_bar(
    model_metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    metric: str,
    *,
    highlight_models: set[str] | None = None,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    highlight_models = highlight_models or set()
    ordered = model_metrics.sort_values(metric, ascending=True).reset_index(drop=True)
    base_colors = _palette_sequence(len(ordered))
    colors = [
        PALETTE["primary"] if name in highlight_models else base_colors[index]
        for index, name in enumerate(ordered["model_name"])
    ]
    labels = [_pretty_model_label(name) for name in ordered["model_name"].astype(str)]
    ax.barh(labels, ordered[metric], color=colors, edgecolor="#23313A", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_xlim(0.0, max(1.0, float(ordered[metric].max()) + 0.08))
    for idx, value in enumerate(ordered[metric].tolist()):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ablation_summary(model_metrics: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    order = ["T_only", "H_only", "A_only", "T_plus_H", "T_plus_A", "H_plus_A", "T_plus_H_plus_A"]
    working = (
        model_metrics.set_index("model_name")
        .reindex(order)
        .dropna(subset=["roc_auc"])
        .reset_index()
    )
    if working.empty:
        return
    color_map = {
        "T_only": PALETTE["primary"],
        "H_only": PALETTE["support"],
        "A_only": PALETTE["accent"],
        "T_plus_H": "#5B8AA7",
        "T_plus_A": "#B86B2F",
        "H_plus_A": "#6A8F7A",
        "T_plus_H_plus_A": PALETTE["high"],
    }
    colors = [color_map.get(name, PALETTE["muted"]) for name in working["model_name"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    positions = np.arange(len(working))
    ax.bar(positions, working["roc_auc"], color=colors, edgecolor="#23313A", linewidth=0.4)
    ax.set_title("Ablation Summary")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_pretty_model_label(name) for name in working["model_name"]], rotation=30, ha="right"
    )
    ax.set_ylim(0.0, 1.0)
    _annotate_bar_values(ax, working["roc_auc"].tolist(), positions.tolist())
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensitivity_summary(sensitivity: dict[str, dict[str, float]], output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    rows = []
    for name, metrics in sensitivity.items():
        if metrics.get("skipped"):
            continue
        rows.append(
            {
                "variant": name,
                "roc_auc": metrics["roc_auc"],
                "roc_auc_ci_lower": metrics.get("roc_auc_ci_lower", metrics["roc_auc"]),
                "roc_auc_ci_upper": metrics.get("roc_auc_ci_upper", metrics["roc_auc"]),
                "average_precision_lift": metrics.get("average_precision_lift", np.nan),
                "average_precision_ci_lower": metrics.get("average_precision_ci_lower", np.nan),
                "average_precision_ci_upper": metrics.get("average_precision_ci_upper", np.nan),
                "positive_prevalence": metrics.get("positive_prevalence", np.nan),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame["variant_label"] = frame["variant"].astype(str).map(_pretty_sensitivity_label)
    frame["ap_lift_ci_lower"] = frame["average_precision_ci_lower"].fillna(
        frame["average_precision_lift"]
    ) - frame["positive_prevalence"].fillna(0.0)
    frame["ap_lift_ci_upper"] = frame["average_precision_ci_upper"].fillna(
        frame["average_precision_lift"]
    ) - frame["positive_prevalence"].fillna(0.0)
    default_row = frame.loc[frame["variant"].isin(["default", "parsimonious_model"])].head(1)
    use_delta = not default_row.empty
    if use_delta:
        default_roc = float(default_row.iloc[0]["roc_auc"])
        default_lift = float(default_row.iloc[0]["average_precision_lift"])
        frame["roc_center"] = frame["roc_auc"] - default_roc
        frame["roc_low"] = frame["roc_auc_ci_lower"] - default_roc
        frame["roc_high"] = frame["roc_auc_ci_upper"] - default_roc
        frame["lift_center"] = frame["average_precision_lift"] - default_lift
        frame["lift_low"] = frame["ap_lift_ci_lower"] - default_lift
        frame["lift_high"] = frame["ap_lift_ci_upper"] - default_lift
        frame = frame.sort_values("roc_center", ascending=True)
    else:
        frame["roc_center"] = frame["roc_auc"]
        frame["roc_low"] = frame["roc_auc_ci_lower"]
        frame["roc_high"] = frame["roc_auc_ci_upper"]
        frame["lift_center"] = frame["average_precision_lift"]
        frame["lift_low"] = frame["ap_lift_ci_lower"]
        frame["lift_high"] = frame["ap_lift_ci_upper"]
        frame = frame.sort_values("roc_center", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    positions = np.arange(len(frame))
    axes[0].hlines(
        positions, frame["roc_low"], frame["roc_high"], color=PALETTE["muted"], linewidth=2
    )
    axes[0].scatter(frame["roc_center"], positions, color=PALETTE["accent"], s=55, zorder=3)
    axes[0].axvline(0.0 if use_delta else 0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title(
        "Sensitivity ROC AUC" if not use_delta else "Sensitivity ROC AUC Delta vs Default"
    )
    axes[0].set_xlabel("ROC AUC" if not use_delta else "delta ROC AUC vs default")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(frame["variant_label"].tolist())
    for idx, value in enumerate(frame["roc_auc"].tolist()):
        axes[0].text(
            frame.iloc[idx]["roc_high"] + 0.004, idx, f"{value:.3f}", va="center", fontsize=8
        )
    _apply_axis_style(axes[0], grid_axis="x")

    axes[1].hlines(
        positions, frame["lift_low"], frame["lift_high"], color=PALETTE["muted"], linewidth=2
    )
    axes[1].scatter(frame["lift_center"], positions, color=PALETTE["primary"], s=55, zorder=3)
    axes[1].axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_title(
        "Sensitivity AP Lift Above Prevalence"
        if not use_delta
        else "Sensitivity AP-Lift Delta vs Default"
    )
    axes[1].set_xlabel("AP lift" if not use_delta else "delta AP lift vs default")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    for idx, value in enumerate(frame["average_precision_lift"].fillna(0.0).tolist()):
        axes[1].text(
            frame.iloc[idx]["lift_high"] + 0.004, idx, f"{value:+.3f}", va="center", fontsize=8
        )
    _apply_axis_style(axes[1], grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_sensitivity_curve(
    sensitivity: dict[str, dict[str, float]], output_path: Path
) -> None:
    _style()
    ensure_directory(output_path.parent)
    rows = []
    default_threshold = 3
    default_metrics = sensitivity.get("default") or sensitivity.get("parsimonious_model") or {}
    if default_metrics and not default_metrics.get("skipped"):
        rows.append(
            {
                "threshold": default_threshold,
                "roc_auc": float(default_metrics["roc_auc"]),
                "roc_auc_ci_lower": float(
                    default_metrics.get("roc_auc_ci_lower", default_metrics["roc_auc"])
                ),
                "roc_auc_ci_upper": float(
                    default_metrics.get("roc_auc_ci_upper", default_metrics["roc_auc"])
                ),
                "average_precision": float(default_metrics["average_precision"]),
            }
        )
    for name, metrics in sensitivity.items():
        if not str(name).startswith("alternate_outcome_threshold_") or metrics.get("skipped"):
            continue
        threshold = int(str(name).rsplit("_", 1)[-1])
        rows.append(
            {
                "threshold": threshold,
                "roc_auc": float(metrics["roc_auc"]),
                "roc_auc_ci_lower": float(metrics.get("roc_auc_ci_lower", metrics["roc_auc"])),
                "roc_auc_ci_upper": float(metrics.get("roc_auc_ci_upper", metrics["roc_auc"])),
                "average_precision": float(metrics["average_precision"]),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(
        frame["threshold"], frame["roc_auc"], color=PALETTE["primary"], marker="o", linewidth=2.2
    )
    ax.fill_between(
        frame["threshold"],
        frame["roc_auc_ci_lower"],
        frame["roc_auc_ci_upper"],
        color=PALETTE["support"],
        alpha=0.22,
    )
    for row in frame.itertuples(index=False):
        ax.text(row.threshold, row.roc_auc + 0.012, f"{row.roc_auc:.3f}", ha="center", fontsize=9)
    ax.set_title("Outcome Threshold Sensitivity")
    ax.set_xlabel("Later new-country threshold")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(frame["threshold"].tolist())
    ax.set_ylim(
        max(0.0, frame["roc_auc_ci_lower"].min() - 0.04),
        min(1.0, frame["roc_auc_ci_upper"].max() + 0.06),
    )
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_roc_pr_curves(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str,
    thresholds: tuple[int, ...] = (2, 3, 4),
) -> None:
    _style()
    ensure_directory(output_path.parent)
    prediction_frame = predictions.loc[
        predictions["model_name"].astype(str) == str(model_name), ["backbone_id", "oof_prediction"]
    ].copy()
    if prediction_frame.empty:
        return
    outcome_column = (
        "n_new_countries_recomputed"
        if "n_new_countries_recomputed" in scored.columns
        else "n_new_countries"
    )
    if outcome_column not in scored.columns:
        return
    merged = (
        scored[["backbone_id", outcome_column]]
        .copy()
        .merge(prediction_frame, on="backbone_id", how="inner")
    )
    merged[outcome_column] = pd.to_numeric(merged[outcome_column], errors="coerce")
    merged["oof_prediction"] = pd.to_numeric(merged["oof_prediction"], errors="coerce")
    merged = merged.loc[merged[outcome_column].notna() & merged["oof_prediction"].notna()].copy()
    if merged.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    colors = [PALETTE["support"], PALETTE["primary"], PALETTE["accent"]]
    for color, threshold in zip(colors, thresholds):
        labels = merged[outcome_column].ge(int(threshold)).astype(int)
        if labels.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, merged["oof_prediction"])
        precision, recall, _ = precision_recall_curve(labels, merged["oof_prediction"])
        auc = float(skl_roc_auc_score(labels, merged["oof_prediction"]))
        ap = float(average_precision(
            labels.to_numpy(),
            merged["oof_prediction"].to_numpy()
        ))
        prevalence = float(labels.mean())
        axes[0].plot(
            fpr, tpr, color=color, linewidth=2.2, label=f"threshold >= {threshold} (AUC {auc:.3f})"
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=2.2,
            label=f"threshold >= {threshold} (AP {ap:.3f})",
        )
        axes[1].axhline(prevalence, color=color, linestyle=":", linewidth=1, alpha=0.7)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
    axes[0].set_title("Threshold-Specific ROC Curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right")
    _apply_axis_style(axes[0], grid_axis="both")

    axes[1].set_title("Threshold-Specific PR Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, loc="lower left")
    _apply_axis_style(axes[1], grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_l2_sensitivity(sensitivity: dict[str, dict[str, float]], output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    rows = []
    for name, metrics in sensitivity.items():
        if not str(name).startswith("primary_l2_") or metrics.get("skipped"):
            continue
        rows.append(
            {
                "l2": float(
                    metrics.get("l2", str(name).replace("primary_l2_", "").replace("p", "."))
                ),
                "roc_auc": float(metrics["roc_auc"]),
                "roc_auc_ci_lower": float(metrics.get("roc_auc_ci_lower", metrics["roc_auc"])),
                "roc_auc_ci_upper": float(metrics.get("roc_auc_ci_upper", metrics["roc_auc"])),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("l2")
    if frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes[0].plot(frame["l2"], frame["roc_auc"], color=PALETTE["primary"], marker="o", linewidth=2.2)
    axes[0].fill_between(
        frame["l2"],
        frame["roc_auc_ci_lower"],
        frame["roc_auc_ci_upper"],
        color=PALETTE["support"],
        alpha=0.22,
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Primary Model L2 Sensitivity")
    axes[0].set_xlabel("L2 penalty (log scale)")
    axes[0].set_ylabel("ROC AUC")
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].plot(
        frame["l2"],
        frame["average_precision_lift"],
        color=PALETTE["accent"],
        marker="o",
        linewidth=2.2,
    )
    axes[1].axhline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_title("Primary Model AP Lift vs L2")
    axes[1].set_xlabel("L2 penalty (log scale)")
    axes[1].set_ylabel("AP lift above prevalence")
    _apply_axis_style(axes[1], grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pathogen_detection_support(comparison: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = comparison.copy()
    if working.empty:
        return
    combined = working.loc[working["pathogen_dataset"] == "combined"].copy()
    if combined.empty:
        combined = working.head(1).copy()
    row = combined.iloc[0]
    positions = np.arange(2)
    labels = ["high", "low"]
    colors = [PALETTE["high"], PALETTE["low"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    matching_values = [
        float(row["mean_matching_fraction_high"]),
        float(row["mean_matching_fraction_low"]),
    ]
    support_values = [float(row["support_fraction_high"]), float(row["support_fraction_low"])]

    axes[0].bar(positions, matching_values, color=colors, edgecolor="#23313A", linewidth=0.4)
    axes[0].set_title(
        "Combined PD Matching Fraction "
        f"({_format_pvalue(float(row['permutation_p_mean_matching_fraction']))})"
    )
    axes[0].set_ylabel("Mean matching fraction")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, max(0.2, max(matching_values) + 0.05))
    _annotate_bar_values(axes[0], matching_values, positions.tolist())
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].bar(positions, support_values, color=colors, edgecolor="#23313A", linewidth=0.4)
    axes[1].set_title(
        "Combined PD Any-Support Fraction "
        f"({_format_pvalue(float(row['permutation_p_support_fraction']))})"
    )
    axes[1].set_ylabel("Backbone share with any support")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    _annotate_bar_values(axes[1], support_values, positions.tolist())
    _apply_axis_style(axes[1], grid_axis="y")

    fig.suptitle(
        (
            "Pathogen Detection Descriptive Support: dominant-species exact match plus "
            "at least one shared top AMR gene"
        ),
        y=1.04,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pathogen_detection_strata_summary(comparison: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = comparison.copy()
    working = working.loc[
        working["pathogen_dataset"].isin(["combined", "clinical", "environmental"])
    ]
    if working.empty:
        return
    working = (
        working.set_index("pathogen_dataset")
        .reindex(["combined", "clinical", "environmental"])
        .dropna(how="all")
        .reset_index()
    )
    positions = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    low_values = working["mean_matching_fraction_low"].fillna(0.0).to_numpy(dtype=float)
    high_values = working["mean_matching_fraction_high"].fillna(0.0).to_numpy(dtype=float)
    for idx, row in enumerate(working.itertuples(index=False)):
        ax.plot(
            [positions[idx], positions[idx]],
            [row.mean_matching_fraction_low, row.mean_matching_fraction_high],
            color=PALETTE["muted"],
            linewidth=2,
            zorder=1,
        )
        ax.text(
            positions[idx],
            max(row.mean_matching_fraction_low, row.mean_matching_fraction_high) + 0.016,
            f"n={int(row.n_high)}/{int(row.n_low)}\n{_format_pvalue(float(row.permutation_p_mean_matching_fraction))}",
            ha="center",
            fontsize=8,
        )
    ax.scatter(positions, low_values, color=PALETTE["low"], s=80, label="low", zorder=2)
    ax.scatter(positions, high_values, color=PALETTE["high"], s=80, label="high", zorder=2)
    ax.set_title("Pathogen Detection Strata Effect Sizes")
    ax.set_ylabel("Mean matching fraction")
    ax.set_xticks(positions)
    ax.set_xticklabels(working["pathogen_dataset"].tolist())
    upper = max(
        high_values.max() if len(high_values) else 0.0, low_values.max() if len(low_values) else 0.0
    )
    ax.set_ylim(0.0, upper + 0.06)
    ax.legend(frameon=False)
    matching_rule = (
        str(working["matching_rule"].dropna().iloc[0])
        if "matching_rule" in working.columns and working["matching_rule"].notna().any()
        else ""
    )
    if matching_rule:
        ax.text(
            0.98,
            0.04,
            matching_rule,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
        )
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_knownness_vs_oof_scatter(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    prediction_frame = predictions.loc[
        predictions["model_name"].astype(str) == str(model_name), ["backbone_id", "oof_prediction"]
    ].copy()
    if prediction_frame.empty:
        return
    working = scored.merge(prediction_frame, on="backbone_id", how="inner")
    if working.empty or "knownness_score" not in working.columns:
        return
    working = working.loc[
        working["spread_label"].notna() & working["knownness_score"].notna()
    ].copy()
    if working.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    label_colors = (
        working["spread_label"]
        .astype(int)
        .map({1: PALETTE["low"], 0: PALETTE["primary"]})
        .fillna(PALETTE["muted"])
    )
    member_sizes = np.clip(
        np.sqrt(working["member_count_train"].fillna(0.0).to_numpy(dtype=float) + 1.0) * 18.0,
        20.0,
        220.0,
    )
    ax.scatter(
        working["knownness_score"],
        working["oof_prediction"],
        s=member_sizes,
        c=label_colors,
        alpha=0.70,
        edgecolor="#23313A",
        linewidth=0.35,
    )
    lower_half_boundary = (
        float(
            working.loc[
                working["knownness_half"].astype(str) == "lower_half", "knownness_score"
            ].max()
        )
        if (working["knownness_half"].astype(str) == "lower_half").any()
        else np.nan
    )
    if np.isfinite(lower_half_boundary):
        ax.axvline(lower_half_boundary, color=PALETTE["accent"], linestyle="--", linewidth=1.4)
        ax.text(
            lower_half_boundary + 0.01,
            0.04,
            "lower-half boundary",
            fontsize=9,
            color=PALETTE["accent"],
        )
    ax.set_title(f"Knownness vs OOF Score ({_pretty_model_label(model_name)})")
    ax.set_xlabel("Knownness score")
    ax.set_ylabel("OOF prediction")
    ax.text(
        0.98,
        0.04,
        "point size = training members\nred = later spread-positive",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tha_radar(
    portfolio_df: pd.DataFrame,
    scored: pd.DataFrame,
    output_path: Path,
    *,
    features: list[str] | None = None,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    features = features or ["T_eff_norm", "H_specialization_norm", "A_eff_norm"]
    if portfolio_df.empty or any(feature not in scored.columns for feature in features):
        return
    selected = []
    for track in ("established_high_risk", "novel_signal"):
        frame = (
            portfolio_df.loc[portfolio_df["portfolio_track"].astype(str) == track]
            .sort_values("track_rank")
            .head(10)
        )
        if not frame.empty:
            selected.append(frame[["backbone_id", "portfolio_track"]])
    if not selected:
        return
    working = pd.concat(selected, ignore_index=True).merge(
        scored[["backbone_id", *features]].drop_duplicates("backbone_id"),
        on="backbone_id",
        how="left",
    )
    if working.empty:
        return
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})
    palette = {
        "established_high_risk": "#0F6A8B",
        "novel_signal": "#C84C09",
    }
    for row in working.itertuples(index=False):
        values = [float(getattr(row, feature, 0.0) or 0.0) for feature in features]
        values += values[:1]
        color = palette.get(str(row.portfolio_track), PALETTE["muted"])
        alpha = 0.55 if str(row.portfolio_track) == "established_high_risk" else 0.35
        ax.plot(angles, values, color=color, linewidth=1.5, alpha=0.85)
        ax.fill(angles, values, color=color, alpha=alpha * 0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_pretty_feature_label(feature) for feature in features])
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("T / H / A Candidate Geometry", pad=24)
    legend_handles = [
        plt.Line2D(
            [0], [0], color=palette["established_high_risk"], lw=2, label="established shortlist"
        ),
        plt.Line2D([0], [0], color=palette["novel_signal"], lw=2, label="novel-signal watchlist"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_temporal_design(
    output_path: Path,
    *,
    split_year: int = 2015,
    year_start: int = 2005,
    year_end: int = 2023,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.hlines(0.0, year_start, year_end, color="#23313A", linewidth=2.0)
    ax.axvspan(year_start, split_year, color=PALETTE["support"], alpha=0.18)
    ax.axvspan(split_year, year_end, color=PALETTE["accent"], alpha=0.16)
    ax.axvline(split_year, color=PALETTE["low"], linestyle="--", linewidth=2)
    for year in range(year_start, year_end + 1, 2):
        ax.vlines(year, -0.08, 0.08, color="#23313A", linewidth=1)
        ax.text(year, -0.18, str(year), ha="center", va="top", fontsize=9)
    ax.text(
        (year_start + split_year) / 2,
        0.42,
        "Training era\nT / H / A extraction",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.text(
        (split_year + year_end) / 2,
        0.42,
        "Outcome era\nnew-country visibility only",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.annotate(
        "",
        xy=(split_year - 0.2, 0.24),
        xytext=(year_start + 0.8, 0.24),
        arrowprops={"arrowstyle": "<->", "color": PALETTE["support"], "linewidth": 2},
    )
    ax.annotate(
        "",
        xy=(year_end - 0.8, 0.24),
        xytext=(split_year + 0.2, 0.24),
        arrowprops={"arrowstyle": "<->", "color": PALETTE["accent"], "linewidth": 2},
    )
    ax.text(
        split_year,
        0.78,
        f"split at {split_year}",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["low"],
    )
    ax.text(
        year_start + 0.3, -0.55, "No post-split rows contribute to feature computation.", fontsize=9
    )
    ax.text(
        year_start + 0.3,
        -0.73,
        "Only the later window contributes to the spread label.",
        fontsize=9,
    )
    ax.set_title("Retrospective Temporal Design")
    ax.set_xlim(year_start - 0.4, year_end + 0.4)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_benchmark_comparison(
    model_metrics: pd.DataFrame,
    output_path: Path,
    *,
    primary_model_name: str,
    governance_model_name: str | None = None,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    focus_order = [
        "baseline_member_count",
        "baseline_country_count",
        "baseline_both",
        "bio_clean_priority",
        governance_model_name or "",
        primary_model_name,
    ]
    working = model_metrics.loc[
        model_metrics["model_name"].astype(str).isin([name for name in focus_order if name])
    ].copy()
    if working.empty:
        return
    working["plot_order"] = (
        working["model_name"]
        .astype(str)
        .map({name: idx for idx, name in enumerate(focus_order) if name})
    )
    working = working.sort_values("plot_order")
    baseline_auc = (
        float(working.loc[working["model_name"].astype(str) == "baseline_both", "roc_auc"].iloc[0])
        if (working["model_name"].astype(str) == "baseline_both").any()
        else np.nan
    )
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    positions = np.arange(len(working))
    colors = []
    for model_name in working["model_name"].astype(str):
        if model_name == primary_model_name:
            colors.append(PALETTE["primary"])
        elif governance_model_name and model_name == governance_model_name:
            colors.append(PALETTE["accent"])
        elif model_name == "baseline_both":
            colors.append(PALETTE["muted"])
        else:
            colors.append("#B8B2A6")
    ax.bar(positions, working["roc_auc"], color=colors, edgecolor="#23313A", linewidth=0.4)
    if {"roc_auc_ci_lower", "roc_auc_ci_upper"}.issubset(working.columns):
        lower = (
            (working["roc_auc"] - working["roc_auc_ci_lower"]).clip(lower=0.0).to_numpy(dtype=float)
        )
        upper = (
            (working["roc_auc_ci_upper"] - working["roc_auc"]).clip(lower=0.0).to_numpy(dtype=float)
        )
        ax.errorbar(
            positions,
            working["roc_auc"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#23313A",
            capsize=3,
            linewidth=1,
        )
    if np.isfinite(baseline_auc):
        ax.axhline(baseline_auc, color=PALETTE["muted"], linestyle="--", linewidth=1.2)
    for idx, row in enumerate(working.itertuples(index=False)):
        label = f"{float(row.roc_auc):.3f}"
        if np.isfinite(baseline_auc):
            label += f"\nΔ {float(row.roc_auc) - baseline_auc:+.3f}"
        ax.text(idx, float(row.roc_auc) + 0.018, label, ha="center", va="bottom", fontsize=8)
    ax.set_title("Baseline vs Discovery/Governance Benchmarks")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_pretty_model_label(name) for name in working["model_name"].astype(str)],
        rotation=18,
        ha="right",
    )
    ax.set_ylim(0.0, min(1.0, float(working["roc_auc"].max()) + 0.12))
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_false_negative_heatmap(
    false_negative_audit: pd.DataFrame,
    scored: pd.DataFrame,
    output_path: Path,
    *,
    max_rows: int = 30,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    if false_negative_audit.empty:
        return
    feature_columns = [
        "T_eff_norm",
        "H_specialization_norm",
        "A_eff_norm",
        "knownness_score",
        "member_count_train",
        "coherence_score",
    ]
    available = [column for column in feature_columns if column in scored.columns]
    if not available:
        return
    working = false_negative_audit.head(max_rows).copy()
    missing_columns = [column for column in available if column not in working.columns]
    if missing_columns:
        working = working.merge(
            scored[["backbone_id", *missing_columns]].drop_duplicates("backbone_id"),
            on="backbone_id",
            how="left",
        )
    available = [column for column in available if column in working.columns]
    if working.empty:
        return
    matrix = working[available].copy().fillna(0.0)
    for column in matrix.columns:
        values = matrix[column].to_numpy(dtype=float)
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if np.isfinite(min_value) and np.isfinite(max_value) and max_value > min_value:
            matrix[column] = (values - min_value) / (max_value - min_value)
        else:
            matrix[column] = 0.0
    fig, ax = plt.subplots(figsize=(10.8, max(5.6, 0.22 * len(working))))
    image = ax.imshow(
        matrix.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0
    )
    ax.set_title("False-Negative Feature Profile Heatmap")
    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels(
        [_pretty_feature_label(column) for column in available], rotation=25, ha="right"
    )
    ytick_labels = [
        f"{backbone_id} | {str(flags or 'none')}"
        for backbone_id, flags in zip(
            working["backbone_id"].astype(str).tolist(),
            working.get("miss_driver_flags", pd.Series("none", index=working.index))
            .astype(str)
            .tolist(),
        )
    ]
    ax.set_yticks(np.arange(len(working)))
    ax.set_yticklabels(ytick_labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="within-column normalized value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_core_model_coefficient_heatmap(
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    if coefficients.empty:
        return
    working = coefficients.copy()
    if (
        "coefficient" not in working.columns
        or "model_name" not in working.columns
        or "feature_name" not in working.columns
    ):
        return
    working["normalized_coefficient"] = working.groupby("model_name")["coefficient"].transform(
        lambda column: column / max(float(np.abs(column).max()), 1e-9)
    )
    model_order = list(dict.fromkeys(working["model_name"].astype(str).tolist()))
    pivot = working.pivot_table(
        index="feature_name",
        columns="model_name",
        values="normalized_coefficient",
        aggfunc="first",
    ).reindex(columns=model_order)
    feature_strength = pivot.abs().max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[feature_strength.index]
    fig, ax = plt.subplots(figsize=(9.8, max(6.0, 0.28 * len(pivot))))
    image = ax.imshow(
        pivot.fillna(0.0).to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    ax.set_title("Core Model Coefficient Heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [_pretty_model_label(column) for column in pivot.columns], rotation=20, ha="right"
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_pretty_feature_label(index) for index in pivot.index], fontsize=8)
    fig.colorbar(
        image, ax=ax, fraction=0.025, pad=0.02, label="within-model normalized coefficient"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_primary_model_coefficients(
    coefficients: pd.DataFrame,
    output_path: Path,
    model_name: str,
    *,
    coefficient_stability: pd.DataFrame | None = None,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    ordered = coefficients.sort_values("coefficient")
    colors = (
        ordered["direction"].map({"positive": PALETTE["high"], "negative": PALETTE["low"]}).tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["coefficient"], color=colors)
    if coefficient_stability is not None and not coefficient_stability.empty:
        stability = coefficient_stability.set_index("feature_name")
        means = (
            ordered["feature_name"]
            .map(stability["mean_coefficient"])
            .fillna(ordered["coefficient"])
            .to_numpy(dtype=float)
        )
        sds = (
            ordered["feature_name"]
            .map(stability["std_coefficient"])
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ax.errorbar(
            means, positions, xerr=sds, fmt="none", ecolor="#23313A", capsize=3, linewidth=1
        )
    ax.axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    max_extent = float(
        max(
            np.max(np.abs(ordered["coefficient"].to_numpy(dtype=float))) if len(ordered) else 0.0,
            np.max(np.abs(means) + sds)
            if coefficient_stability is not None and not coefficient_stability.empty
            else 0.0,
        )
    )
    span = max(max_extent * 1.18, 0.15)
    ax.set_xlim(min(-0.12, -span * 0.06), span)
    ax.set_title(f"Standardized Coefficients ({_pretty_model_label(model_name)})")
    ax.set_xlabel("Coefficient")
    ax.set_yticks(positions)
    ax.set_yticklabels([_pretty_feature_label(name) for name in ordered["feature_name"].tolist()])
    for idx, value in enumerate(ordered["coefficient"].tolist()):
        x_position = value + (0.03 * span if value >= 0 else -0.03 * span)
        ax.text(
            x_position,
            idx,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_dropout_importance(
    dropout: pd.DataFrame, output_path: Path, model_name: str
) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = dropout.loc[dropout["feature_name"] != "__full_model__"].copy()
    working = working.sort_values("roc_auc_drop_vs_full", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [_pretty_feature_label(name) for name in working["feature_name"].tolist()]
    colors = [PALETTE["accent"]] * len(working)
    if colors:
        colors[-1] = PALETTE["primary"]
    ax.barh(labels, working["roc_auc_drop_vs_full"], color=colors)
    ax.set_title(f"Feature Dropout Impact ({_pretty_model_label(model_name)})")
    ax.set_xlabel("ROC AUC drop after removing feature")
    x_max = float(working["roc_auc_drop_vs_full"].max()) if len(working) else 0.0
    ax.set_xlim(0.0, max(0.05, x_max + 0.012))
    for idx, value in enumerate(working["roc_auc_drop_vs_full"].tolist()):
        ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=9)
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_family_summary(family_summary: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    palette = {
        "easy_proxy": PALETTE["muted"],
        "handcrafted_score": PALETTE["primary"],
        "legacy_biological_core": "#7AA5BD",
        "biological_core": PALETTE["support"],
        "evidence_aware": PALETTE["high"],
        "legacy_integrated": PALETTE["accent"],
    }
    ordered = family_summary.sort_values("roc_auc", ascending=False)
    colors = ordered["evidence_role"].map(palette).fillna(PALETTE["support"]).tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [_pretty_model_label(name) for name in ordered["model_name"].astype(str)]
    ax.barh(labels, ordered["roc_auc"], color=colors)
    ax.set_title("Model Family Summary")
    ax.set_xlabel("ROC AUC")
    for idx, value in enumerate(ordered["roc_auc"].tolist()):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_comparison_deltas(
    comparison: pd.DataFrame, output_path: Path, model_name: str
) -> None:
    _style()
    ensure_directory(output_path.parent)
    if comparison.empty:
        return
    preferred_order = {
        "baseline_both": 0,
        "baseline_country_count": 1,
        "full_priority": 2,
        "bio_clean_priority": 3,
        "support_calibrated_priority": 11,
        "support_synergy_priority": 12,
        "T_plus_H_plus_A": 4,
        "source_only": 99,
    }
    ordered = comparison.copy()
    ordered["plot_order"] = ordered["comparison_model_name"].map(preferred_order).fillna(99)
    ordered = ordered.sort_values(["plot_order", "delta_roc_auc"])
    core = ordered.loc[ordered["comparison_model_name"] != "source_only"].copy()
    weak_control = ordered.loc[ordered["comparison_model_name"] == "source_only"].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.5),
        gridspec_kw={"width_ratios": [4.2, 1.4]},
    )
    centers = np.arange(len(core))
    xerr = (
        np.vstack(
            [
                (core["delta_roc_auc"] - core["delta_roc_auc_ci_lower"])
                .clip(lower=0.0)
                .to_numpy(dtype=float),
                (core["delta_roc_auc_ci_upper"] - core["delta_roc_auc"])
                .clip(lower=0.0)
                .to_numpy(dtype=float),
            ]
        )
        if not core.empty
        else np.zeros((2, 0), dtype=float)
    )
    colors = (
        core["comparison_model_name"]
        .map(
            {
                "baseline_both": PALETTE["primary"],
                "full_priority": PALETTE["support"],
                "T_plus_H_plus_A": PALETTE["accent"],
            }
        )
        .fillna(PALETTE["support"])
    )
    axes[0].barh(centers, core["delta_roc_auc"], color=colors, edgecolor="#23313A", linewidth=0.4)
    if len(core) > 0:
        axes[0].errorbar(
            core["delta_roc_auc"],
            centers,
            xerr=xerr,
            fmt="none",
            ecolor="#23313A",
            capsize=3,
            linewidth=1,
        )
    axes[0].axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title(f"ROC AUC Gain vs Main Comparators ({_pretty_model_label(model_name)})")
    axes[0].set_xlabel("Primary minus comparator ROC AUC")
    axes[0].set_yticks(centers)
    axes[0].set_yticklabels(
        [_pretty_model_label(name) for name in core["comparison_model_name"].tolist()]
    )
    axes[0].invert_yaxis()
    for idx, row in enumerate(core.itertuples(index=False)):
        pvalue = getattr(row, "delta_roc_auc_delong_pvalue", np.nan)
        label = f"{float(row.delta_roc_auc):+.3f}"
        if np.isfinite(pvalue):
            label += f"\nDeLong {_format_pvalue(float(pvalue))}"
        axes[0].text(
            float(row.delta_roc_auc_ci_upper) + 0.006,
            idx,
            label,
            va="center",
            fontsize=8,
        )
    _apply_axis_style(axes[0], grid_axis="x")

    axes[1].axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    if not weak_control.empty:
        row = weak_control.iloc[0]
        delta = float(row["delta_roc_auc"])
        lower = max(delta - float(row["delta_roc_auc_ci_lower"]), 0.0)
        upper = max(float(row["delta_roc_auc_ci_upper"]) - delta, 0.0)
        axes[1].barh([0], [delta], color=PALETTE["muted"], edgecolor="#23313A", linewidth=0.4)
        axes[1].errorbar(
            [delta],
            [0],
            xerr=np.array([[lower], [upper]]),
            fmt="none",
            ecolor="#23313A",
            capsize=3,
            linewidth=1,
        )
        axes[1].set_yticks([0])
        axes[1].set_yticklabels([_pretty_model_label("source_only")])
        axes[1].set_xlabel("Delta ROC AUC")
        axes[1].set_title("Weak Control")
        axes[1].text(
            0.05,
            0.06,
            "weak comparator",
            transform=axes[1].transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
        )
    else:
        axes[1].set_axis_off()
    _apply_axis_style(axes[1], grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_knownness_audit(summary: pd.DataFrame, output_path: Path, model_name: str) -> None:
    _style()
    ensure_directory(output_path.parent)
    if summary.empty:
        return
    row = summary.iloc[0]
    q1_supported = bool(row.get("lowest_knownness_quartile_supported", False))
    pretty_labels = ["overall", "lower-half\nknownness"]
    primary_values = [
        float(row.get("overall_primary_roc_auc", np.nan)),
        float(row.get("lower_half_knownness_primary_roc_auc", np.nan)),
    ]
    baseline_values = [
        float(row.get("overall_baseline_roc_auc", np.nan)),
        float(row.get("lower_half_knownness_baseline_roc_auc", np.nan)),
    ]
    if q1_supported:
        pretty_labels.append("lowest-knownness\nquartile")
        primary_values.append(float(row.get("lowest_knownness_quartile_primary_roc_auc", np.nan)))
        baseline_values.append(float(row.get("lowest_knownness_quartile_baseline_roc_auc", np.nan)))
    pretty_labels.append("matched\nvisibility strata")
    primary_values.append(float(row.get("matched_strata_primary_weighted_roc_auc", np.nan)))
    baseline_values.append(float(row.get("matched_strata_baseline_weighted_roc_auc", np.nan)))
    correlation_rows = pd.DataFrame(
        {
            "label": [
                "primary prediction",
                "counts-only baseline",
                "bio priority index",
                "evidence support index",
            ],
            "value": [
                float(row.get("primary_prediction_vs_knownness_spearman", np.nan)),
                float(row.get("baseline_prediction_vs_knownness_spearman", np.nan)),
                float(row.get("bio_priority_index_vs_knownness_spearman", np.nan)),
                float(
                    row.get(
                        "evidence_support_index_vs_knownness_spearman",
                        row.get("operational_priority_index_vs_knownness_spearman", np.nan),
                    )
                ),
            ],
        }
    )

    top_k = int(row.get("top_k", 25))
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    positions = np.arange(len(pretty_labels))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        primary_values,
        width=width,
        color=PALETTE["primary"],
        label=_pretty_model_label(model_name),
    )
    axes[0].bar(
        positions + width / 2,
        baseline_values,
        width=width,
        color=PALETTE["muted"],
        label=_pretty_model_label("baseline_both"),
    )
    axes[0].set_title("Discrimination After Controlling for Knownness")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(pretty_labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    _apply_axis_style(axes[0], grid_axis="y")

    corr_positions = np.arange(len(correlation_rows))
    axes[1].bar(
        corr_positions,
        correlation_rows["value"],
        color=[PALETTE["primary"], PALETTE["muted"], PALETTE["support"], PALETTE["accent"]],
    )
    axes[1].axhline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_title("Knownness Coupling Across Score Axes")
    axes[1].set_ylabel("Spearman correlation")
    axes[1].set_xticks(corr_positions)
    axes[1].set_xticklabels(correlation_rows["label"].tolist(), rotation=18, ha="right")
    for idx, value in enumerate(correlation_rows["value"].tolist()):
        axes[1].text(
            idx, value + (0.02 if value >= 0 else -0.05), f"{value:.2f}", ha="center", fontsize=9
        )
    _apply_axis_style(axes[1], grid_axis="y")

    composition_labels = [f"top-{top_k}\nlower-half"]
    composition_values = [float(row.get("top_k_lower_half_knownness_fraction", np.nan))]
    reference_values = [0.5]
    if q1_supported:
        composition_labels.append(f"top-{top_k}\nlowest quartile")
        composition_values.append(
            float(row.get("top_k_lowest_quartile_knownness_fraction", np.nan))
        )
        reference_values.append(0.25)
    composition_labels.extend(
        [
            "cohort\nmean knownness",
            f"top-{top_k}\nmean knownness",
        ]
    )
    composition_values.extend(
        [
            float(row.get("eligible_mean_knownness_score", np.nan)),
            float(row.get("top_k_mean_knownness_score", np.nan)),
        ]
    )
    reference_values.extend([np.nan, np.nan])
    composition_positions = np.arange(len(composition_values))
    axes[2].bar(
        composition_positions,
        composition_values,
        color=[PALETTE["support"], PALETTE["support"], PALETTE["muted"], PALETTE["primary"]],
        edgecolor="#23313A",
        linewidth=0.4,
    )
    for reference in (0.5, 0.25):
        axes[2].axhline(reference, color="#A89E8A", linestyle=":", linewidth=1)
        axes[2].text(
            len(composition_values) - 0.1,
            reference + 0.015,
            f"cohort expectation {reference:.2f}",
            ha="right",
            fontsize=8,
            color="#6E6558",
        )
    axes[2].set_title("How Novel the Main Top-Ranked Set Is")
    axes[2].set_ylabel("Fraction or mean knownness")
    axes[2].set_xticks(composition_positions)
    axes[2].set_xticklabels(composition_labels)
    axes[2].set_ylim(0.0, 1.0)
    for idx, value in enumerate(composition_values):
        if np.isfinite(value):
            axes[2].text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    axes[2].text(
        0.98,
        0.04,
        (
            "Main ranking is not novelty-heavy by itself.\n"
            "Use the companion novelty frontier/watchlist for new-signal candidates."
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[2], grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_source_balance_resampling(
    resampling: pd.DataFrame, output_path: Path, model_name: str
) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        resampling["resample_index"],
        resampling["roc_auc"],
        marker="o",
        color=PALETTE["primary"],
        linewidth=1.8,
    )
    mean_auc = float(resampling["roc_auc"].mean())
    sd_auc = float(resampling["roc_auc"].std()) if len(resampling) > 1 else 0.0
    ax.axhline(mean_auc, color=PALETTE["accent"], linestyle="--", linewidth=1.5)
    ax.fill_between(
        resampling["resample_index"],
        mean_auc - sd_auc,
        mean_auc + sd_auc,
        color=PALETTE["accent"],
        alpha=0.15,
        linewidth=0.0,
    )
    ax.set_title(f"Source-Balanced Resampling ROC AUC ({_pretty_model_label(model_name)})")
    ax.set_xlabel("Resample index")
    ax.set_ylabel("ROC AUC")
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_card_mechanism_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = comparison.copy()
    working["abs_delta"] = working["prevalence_delta_high_minus_low"].abs()
    working = working.sort_values("abs_delta", ascending=False).head(12).sort_values("high")
    positions = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.hlines(positions, working["low"], working["high"], color=PALETTE["muted"], linewidth=2)
    ax.scatter(
        working["low"], positions, color=PALETTE["low"], s=55, label="low prevalence", zorder=3
    )
    ax.scatter(
        working["high"], positions, color=PALETTE["high"], s=55, label="high prevalence", zorder=3
    )
    ax.set_title("CARD Resistance Mechanism Prevalence")
    ax.set_xlabel("Backbone prevalence")
    ax.set_yticks(positions)
    ax.set_yticklabels(working["card_resistance_mechanism"].tolist())
    ax.legend(frameon=False, loc="lower right")
    for y, row in enumerate(working.itertuples(index=False)):
        ax.text(
            row.high + 0.015, y, f"{row.high:.2f}", va="center", fontsize=8, color=PALETTE["high"]
        )
        ax.text(
            max(row.low - 0.015, 0.01),
            y,
            f"{row.low:.2f}",
            va="center",
            ha="right",
            fontsize=8,
            color=PALETTE["low"],
        )
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mobsuite_support_summary(summary: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = summary.copy()
    coverage = working["n_with_literature_support"] / working["n_backbones"].replace(0, pd.NA)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    colors = (
        working["priority_group"].map({"high": PALETTE["high"], "low": PALETTE["low"]}).tolist()
    )

    axes[0].bar(working["priority_group"], coverage.fillna(0.0), color=colors)
    axes[0].set_title("MOB-suite Literature Coverage")
    axes[0].set_ylabel("Share of selected backbones")
    axes[0].set_ylim(0.0, 1.0)
    _annotate_bar_values(axes[0], coverage.fillna(0.0).tolist(), list(range(len(working))))
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].bar(
        working["priority_group"], working["mean_reported_host_range_taxid_count"], color=colors
    )
    axes[1].set_title("Mean Reported Host-Range Breadth")
    axes[1].set_ylabel("Unique reported host-range taxids")
    _annotate_bar_values(
        axes[1],
        working["mean_reported_host_range_taxid_count"].fillna(0.0).tolist(),
        list(range(len(working))),
    )
    _apply_axis_style(axes[1], grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_who_mia_category_support(comparison: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    ordered_categories = ["HPCIA", "CIA", "HIA", "IA"]
    working = comparison.copy()
    working = (
        working.set_index("who_mia_category").reindex(ordered_categories).fillna(0.0).reset_index()
    )
    positions = list(range(len(working)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        [position - width / 2 for position in positions],
        working.get("high", pd.Series([0.0] * len(working))).tolist(),
        width=width,
        color=PALETTE["high"],
        label="high",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        working.get("low", pd.Series([0.0] * len(working))).tolist(),
        width=width,
        color=PALETTE["low"],
        label="low",
    )
    ax.set_title("WHO MIA Category Coverage")
    ax.set_ylabel("Backbone prevalence")
    ax.set_xticks(positions)
    ax.set_xticklabels(working["who_mia_category"].tolist())
    ax.legend(frameon=False)
    for position, high, low in zip(
        positions,
        working.get("high", pd.Series([0.0] * len(working))).tolist(),
        working.get("low", pd.Series([0.0] * len(working))).tolist(),
    ):
        ax.text(
            position - width / 2, high + 0.01, f"{high:.2f}", ha="center", va="bottom", fontsize=8
        )
        ax.text(
            position + width / 2, low + 0.01, f"{low:.2f}", ha="center", va="bottom", fontsize=8
        )
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_amrfinder_concordance(summary: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = summary.loc[summary["priority_group"] != "overall"].copy()
    if working.empty:
        return
    working["amr_evidence_fraction"] = working["n_with_any_amr_evidence"] / working[
        "n_sequences"
    ].replace(0, pd.NA)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    colors = (
        working["priority_group"].map({"high": PALETTE["high"], "low": PALETTE["low"]}).tolist()
    )
    gene_metric = working.get("mean_gene_jaccard_nonempty", working["mean_gene_jaccard"]).fillna(
        0.0
    )
    class_metric = working.get("mean_class_jaccard_nonempty", working["mean_class_jaccard"]).fillna(
        0.0
    )
    positions = np.arange(len(working))
    axes[0].bar(positions, working["amr_evidence_fraction"].fillna(0.0), color=colors)
    axes[0].set_title("AMR Evidence Coverage (probe panel)")
    axes[0].set_ylabel("Share with any AMR evidence")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(working["priority_group"].tolist())
    working_reset = working.reset_index(drop=True)
    for idx, row in enumerate(working_reset.itertuples(index=False)):
        axes[0].text(
            idx,
            min(float(row.amr_evidence_fraction) + 0.04, 0.96),
            f"n={int(row.n_sequences)}\nAMR={int(row.n_with_any_amr_evidence)}",
            ha="center",
            fontsize=8,
        )
    _apply_axis_style(axes[0], grid_axis="y")

    gene_draw = gene_metric.copy()
    gene_draw.loc[working["n_with_any_amr_evidence"].fillna(0).astype(int) < 3] = np.nan
    axes[1].bar(positions, gene_draw.fillna(0.0), color=colors)
    axes[1].set_title("Gene Concordance on Evaluable Panel")
    axes[1].set_ylabel("Mean gene-set Jaccard")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(working["priority_group"].tolist())
    _apply_axis_style(axes[1], grid_axis="y")

    class_draw = class_metric.copy()
    class_draw.loc[working["n_with_any_amr_evidence"].fillna(0).astype(int) < 3] = np.nan
    axes[2].bar(positions, class_draw.fillna(0.0), color=colors)
    axes[2].set_title("Class Concordance on Evaluable Panel")
    axes[2].set_ylabel("Mean class-set Jaccard")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(working["priority_group"].tolist())
    _apply_axis_style(axes[2], grid_axis="y")

    for axis, metric_column in (
        (axes[1], "n_with_any_amr_evidence"),
        (axes[2], "n_with_any_amr_evidence"),
    ):
        for idx, row in enumerate(working_reset.itertuples(index=False)):
            evaluable_n = int(getattr(row, metric_column))
            metric_value = gene_metric.iloc[idx] if axis is axes[1] else class_metric.iloc[idx]
            if evaluable_n >= 3:
                label = f"n={evaluable_n}"
            elif evaluable_n > 0:
                label = f"n={evaluable_n}\ninsufficient"
            else:
                label = "not evaluable"
            axis.text(
                idx,
                max(metric_value if pd.notna(metric_value) else 0.0, 0.02) + 0.04,
                label,
                ha="center",
                fontsize=8,
            )
    axes[0].text(
        0.98,
        0.04,
        "Small probe panel; low group is underpowered for concordance.",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rolling_temporal_validation(
    rolling: pd.DataFrame,
    output_path: Path,
    model_name: str,
    diagnostics: pd.DataFrame | None = None,
) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = rolling.loc[
        (rolling["status"] == "ok") & (rolling["model_name"] == model_name)
    ].copy()
    if working.empty:
        return
    if "horizon_years" in working.columns and working["horizon_years"].nunique() > 1:
        preferred_horizon = (
            3
            if (working["horizon_years"] == 3).any()
            else int(sorted(working["horizon_years"].dropna().astype(int).unique())[0])
        )
        working = working.loc[
            working["horizon_years"].fillna(preferred_horizon).astype(int) == preferred_horizon
        ].copy()
        if (
            diagnostics is not None
            and not diagnostics.empty
            and "horizon_years" in diagnostics.columns
        ):
            diagnostics = diagnostics.loc[
                diagnostics["horizon_years"].fillna(preferred_horizon).astype(int)
                == preferred_horizon
            ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1))
    palette = {"all_records": PALETTE["primary"], "training_only": PALETTE["accent"]}
    mode_labels = {
        "all_records": "all records assignment",
        "training_only": "training-only assignment",
    }
    identical_curves = False
    if diagnostics is not None and not diagnostics.empty:
        identical_curves = bool(
            diagnostics["eligible_identical"].fillna(False).all()
            and diagnostics["roc_auc_delta_training_only_minus_all_records"].fillna(0.0).abs().max()
            < 1e-10
        )
    group_iterable = (
        [("all_records", working.loc[working["backbone_assignment_mode"] == "all_records"])]
        if identical_curves
        else list(working.groupby("backbone_assignment_mode", sort=False))
    )
    for mode, frame in group_iterable:
        ordered = frame.sort_values("split_year")
        axes[0].plot(
            ordered["split_year"],
            ordered["roc_auc"],
            marker="o",
            linewidth=2,
            label=mode_labels.get(str(mode), str(mode)),
            color=palette.get(str(mode), PALETTE["support"]),
        )
    axes[0].set_title(f"Rolling Temporal Validation ({_pretty_model_label(model_name)})")
    axes[0].set_xlabel("Training cutoff year")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    _apply_axis_style(axes[0], grid_axis="y")
    if identical_curves:
        axes[0].text(
            0.02,
            0.05,
            (
                "eligible cohort stayed identical in every window;\n"
                "assignment changes mainly affected rows outside the evaluated set."
            ),
            transform=axes[0].transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
        )
    for mode, frame in group_iterable:
        ordered = frame.sort_values("split_year")
        ap_values = ordered.get("average_precision_lift", ordered["average_precision"]).astype(
            float
        )
        axes[1].plot(
            ordered["split_year"],
            ap_values,
            marker="o",
            linewidth=2,
            label=mode_labels.get(str(mode), str(mode)),
            color=palette.get(str(mode), PALETTE["support"]),
        )
        if "positive_prevalence" in ordered.columns:
            prevalence_values = ordered["positive_prevalence"].astype(float)
            axes[1].scatter(
                ordered["split_year"],
                prevalence_values,
                marker="x",
                color=palette.get(str(mode), PALETTE["support"]),
                alpha=0.55,
            )
    axes[1].axhline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_title("Average Precision Lift Above Prevalence")
    axes[1].set_xlabel("Training cutoff year")
    axes[1].set_ylabel("AP - prevalence")
    axes[1].legend(frameon=False)
    _apply_axis_style(axes[1], grid_axis="y")
    axes[1].text(
        0.98,
        0.05,
        "x markers show raw positive prevalence in each window",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    if diagnostics is not None and not diagnostics.empty:
        ordered_diag = diagnostics.sort_values("split_year")
        unseen_share = ordered_diag["training_only_future_unseen_row_fraction"].fillna(0.0)
        unseen_backbone_share = ordered_diag[
            "training_only_future_unseen_backbone_fraction"
        ].fillna(0.0)
        axes[2].plot(
            ordered_diag["split_year"],
            unseen_share,
            marker="o",
            linewidth=2,
            color=PALETTE["accent"],
            label="future rows reassigned to unseen",
        )
        axes[2].plot(
            ordered_diag["split_year"],
            unseen_backbone_share,
            marker="s",
            linewidth=2,
            color=PALETTE["support"],
            label="future backbones unseen in training",
        )
        axes[2].set_title("Why Assignment-Invariance Can Still Be Real")
        axes[2].set_xlabel("Training cutoff year")
        axes[2].set_ylabel("Training-only reassignment share")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].legend(frameon=False)
        _apply_axis_style(axes[2], grid_axis="y")
    else:
        axes[2].set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_permutation_null_summary(summary: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = summary.copy()
    if working.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    positions = list(range(len(working)))
    observed = working["observed_roc_auc"].fillna(0.0).tolist()
    null_cutoff = working["null_roc_auc_q975"].fillna(0.0).tolist()
    ax.bar(positions, observed, color=PALETTE["primary"], alpha=0.85, label="Observed ROC AUC")
    ax.scatter(positions, null_cutoff, color=PALETTE["accent"], s=80, label="97.5% null ROC AUC")
    ax.set_title("Permutation Null Audit")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(working["model_name"].tolist(), rotation=35)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    _apply_axis_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_candidate_stability(candidates: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    if candidates.empty:
        return
    working = candidates.copy()
    if "base_rank" not in working.columns:
        working = working.sort_values("priority_index", ascending=False).reset_index(drop=True)
        working["base_rank"] = range(1, len(working) + 1)
    working = working.sort_values("base_rank").head(12).copy()
    if "bootstrap_top_k_frequency" not in working.columns:
        working["bootstrap_top_k_frequency"] = 0.0
    if "variant_top_k_frequency" not in working.columns:
        working["variant_top_k_frequency"] = 0.0
    if "bootstrap_top_25_frequency" not in working.columns:
        working["bootstrap_top_25_frequency"] = working["bootstrap_top_k_frequency"]
    if "variant_top_25_frequency" not in working.columns:
        working["variant_top_25_frequency"] = working["variant_top_k_frequency"]
    positions = np.arange(len(working))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
    core_mask = working["base_rank"].astype(int).le(10).to_numpy(dtype=bool)
    bootstrap_retention = np.where(
        core_mask,
        working.get("bootstrap_top_10_frequency", working["bootstrap_top_k_frequency"])
        .fillna(0.0)
        .to_numpy(dtype=float),
        working.get("bootstrap_top_25_frequency", working["bootstrap_top_k_frequency"])
        .fillna(0.0)
        .to_numpy(dtype=float),
    )
    variant_retention = np.where(
        core_mask,
        working.get("variant_top_10_frequency", working["variant_top_k_frequency"])
        .fillna(0.0)
        .to_numpy(dtype=float),
        working.get("variant_top_25_frequency", working["variant_top_k_frequency"])
        .fillna(0.0)
        .to_numpy(dtype=float),
    )
    axes[0].bar(
        positions - width / 2,
        bootstrap_retention.tolist(),
        width=width,
        color=PALETTE["primary"],
        label="bootstrap retention in intended slice",
    )
    axes[0].bar(
        positions + width / 2,
        variant_retention.tolist(),
        width=width,
        color=PALETTE["accent"],
        label="variant retention in intended slice",
    )
    axes[0].axvline(9.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title("Retention in Core Top-10 or Watchlist Top-25")
    axes[0].set_ylabel("Retention frequency")
    axes[0].set_xticks(positions)
    tick_labels = [
        f"#{int(row.base_rank)} {_candidate_tick_label(row)}"
        for row in working.itertuples(index=False)
    ]
    axes[0].set_xticklabels(tick_labels, rotation=30, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    axes[0].text(4.5, 1.02, "core top-10", ha="center", va="bottom", fontsize=9)
    axes[0].text(10.5, 1.02, "watchlist top-25", ha="center", va="bottom", fontsize=9)
    _apply_axis_style(axes[0], grid_axis="y")

    rank_std = (
        working.get("bootstrap_rank_std", pd.Series([0.0] * len(working)))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    axes[1].errorbar(
        positions,
        working.get("bootstrap_mean_rank", pd.Series(range(1, len(working) + 1))).to_numpy(
            dtype=float
        ),
        yerr=rank_std,
        fmt="o",
        color=PALETTE["primary"],
        ecolor=PALETTE["muted"],
        capsize=3,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Bootstrap Mean Rank +/- SD")
    axes[1].set_ylabel("Rank (lower is better)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(tick_labels, rotation=30, ha="right")
    axes[1].axhline(10, color=PALETTE["accent"], linestyle=":", linewidth=1)
    axes[1].axhline(25, color=PALETTE["muted"], linestyle=":", linewidth=1)
    axes[1].text(
        0.98,
        0.04,
        (
            "Ranks 1-10 are judged by top-10 retention;\n"
            "ranks 11+ are judged as watchlist top-25 entries."
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[1], grid_axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_novelty_frontier(frontier: pd.DataFrame, output_path: Path, model_name: str) -> None:
    _style()
    ensure_directory(output_path.parent)
    if frontier.empty:
        return
    required = {
        "baseline_both_oof_prediction",
        "primary_model_oof_prediction",
        "novelty_margin_vs_baseline",
    }
    if not required.issubset(set(frontier.columns)):
        return
    working = frontier.dropna(
        subset=[
            "baseline_both_oof_prediction",
            "primary_model_oof_prediction",
            "novelty_margin_vs_baseline",
        ]
    ).copy()
    if working.empty:
        return

    working["knownness_half"] = (
        working.get("knownness_half", pd.Series("unknown", index=working.index))
        .fillna("unknown")
        .astype(str)
    )
    watchlist = working.loc[working["knownness_half"] == "lower_half"].copy()
    watchlist = watchlist.sort_values(
        ["novelty_margin_vs_baseline", "primary_model_oof_prediction"], ascending=[False, False]
    ).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    for knownness_half, color, label in (
        ("upper_half", PALETTE["muted"], "upper-half knownness"),
        ("lower_half", PALETTE["accent"], "lower-half knownness"),
    ):
        frame = working.loc[working["knownness_half"] == knownness_half].copy()
        if frame.empty:
            continue
        frame["x_plot"] = (
            frame["baseline_both_oof_prediction"].astype(float)
            + frame["backbone_id"]
            .astype(str)
            .map(lambda value: _stable_jitter(value, salt="novelty_frontier_x"))
        ).clip(lower=0.0, upper=1.0)
        frame["y_plot"] = (
            frame["primary_model_oof_prediction"].astype(float)
            + frame["backbone_id"]
            .astype(str)
            .map(lambda value: _stable_jitter(value, salt="novelty_frontier_y"))
        ).clip(lower=0.0, upper=1.0)
        axes[0].scatter(
            frame["x_plot"],
            frame["y_plot"],
            s=28,
            color=color,
            alpha=0.45 if knownness_half == "upper_half" else 0.70,
            label=label,
            edgecolor="none",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6E6558", linewidth=1)
    label_offsets = [
        (5, 5),
        (5, -12),
        (-34, 6),
        (-30, -12),
        (10, 14),
        (10, -18),
        (18, 8),
        (-24, 14),
    ]
    for row, offset in zip(watchlist.head(6).itertuples(index=False), label_offsets):
        x_plot = float(
            np.clip(
                row.baseline_both_oof_prediction
                + _stable_jitter(str(row.backbone_id), salt="novelty_frontier_x"),
                0.0,
                1.0,
            )
        )
        y_plot = float(
            np.clip(
                row.primary_model_oof_prediction
                + _stable_jitter(str(row.backbone_id), salt="novelty_frontier_y"),
                0.0,
                1.0,
            )
        )
        axes[0].scatter(
            x_plot, y_plot, s=80, facecolor="none", edgecolor="#23313A", linewidth=1.1, zorder=4
        )
        axes[0].annotate(
            str(row.backbone_id),
            (x_plot, y_plot),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_title("Primary vs Counts-Only Baseline")
    axes[0].set_xlabel(f"{_pretty_model_label('baseline_both')} OOF prediction")
    axes[0].set_ylabel(f"{_pretty_model_label(model_name)} OOF prediction")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].text(
        0.03,
        0.97,
        "Above diagonal = published primary assigns\nhigher risk than the counts-only baseline.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[0], grid_axis="both")

    if "knownness_score" in working.columns and working["knownness_score"].notna().any():
        for knownness_half, color, label in (
            ("upper_half", PALETTE["muted"], "upper-half knownness"),
            ("lower_half", PALETTE["accent"], "lower-half knownness"),
        ):
            frame = working.loc[working["knownness_half"] == knownness_half].copy()
            if frame.empty:
                continue
            axes[1].scatter(
                frame["knownness_score"],
                frame["novelty_margin_vs_baseline"],
                s=30,
                color=color,
                alpha=0.55 if knownness_half == "upper_half" else 0.75,
                edgecolor="none",
                label=label,
            )
        axes[1].axhline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
        median_knownness = float(working["knownness_score"].median())
        axes[1].axvline(median_knownness, color=PALETTE["muted"], linestyle=":", linewidth=1)
        for row, offset in zip(watchlist.head(8).itertuples(index=False), label_offsets):
            axes[1].scatter(
                row.knownness_score,
                row.novelty_margin_vs_baseline,
                s=80,
                facecolor="none",
                edgecolor="#23313A",
                linewidth=1.1,
                zorder=4,
            )
            axes[1].annotate(
                str(row.backbone_id),
                (row.knownness_score, row.novelty_margin_vs_baseline),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].set_title("Knownness vs Novelty Margin")
        axes[1].set_xlabel("Knownness score (lower = less established)")
        axes[1].set_ylabel("Primary minus counts-only prediction")
        axes[1].legend(frameon=False, loc="upper right")
        _apply_axis_style(axes[1], grid_axis="x")
    else:
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_negative_control_audit(audit: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    if audit.empty:
        return
    working = audit.copy().sort_values("roc_auc", ascending=True)
    palette = []
    for audit_name in working["audit_name"].astype(str):
        if audit_name == "primary_model":
            palette.append(PALETTE["primary"])
        elif audit_name.startswith("primary_plus"):
            palette.append(PALETTE["accent"])
        else:
            palette.append(PALETTE["muted"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(working["audit_name"], working["roc_auc"], color=palette)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_title("Negative-Control Feature Audit")
    ax.set_xlabel("ROC AUC")
    for idx, value in enumerate(working["roc_auc"].tolist()):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_temporal_drift_summary(summary: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    if summary.empty:
        return
    working = summary.sort_values("resolved_year").copy()
    dense = (
        working.loc[~working["low_density_year"].fillna(False)].copy()
        if "low_density_year" in working.columns
        else working.loc[working["n_records"].fillna(0).astype(int) >= 20].copy()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    if "low_density_year" in working.columns and working["low_density_year"].any():
        low_density = working.loc[working["low_density_year"]].copy()
        axes[0].axvspan(
            low_density["resolved_year"].min() - 0.5,
            low_density["resolved_year"].max() + 0.5,
            color="#E6E0D3",
            alpha=0.35,
        )
        axes[1].axvspan(
            low_density["resolved_year"].min() - 0.5,
            low_density["resolved_year"].max() + 0.5,
            color="#E6E0D3",
            alpha=0.35,
        )

    axes[0].scatter(
        working["resolved_year"], working["n_records"], color=PALETTE["primary"], alpha=0.25, s=18
    )
    axes[0].scatter(
        working["resolved_year"], working["n_backbones"], color=PALETTE["accent"], alpha=0.25, s=18
    )
    axes[0].plot(
        dense["resolved_year"],
        dense["n_records_rolling3"],
        color=PALETTE["primary"],
        linewidth=2,
        label="records (3-year mean)",
    )
    axes[0].plot(
        dense["resolved_year"],
        dense["n_backbones_rolling3"],
        color=PALETTE["accent"],
        linewidth=2,
        label="backbones (3-year mean)",
    )
    axes[0].set_title("Temporal Record Density")
    axes[0].set_xlabel("Resolved year")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.03,
        0.94,
        "Shaded years had sparse records and are down-weighted visually.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
    )
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].scatter(
        working["resolved_year"],
        working["refseq_record_fraction"].fillna(0.0),
        color=PALETTE["support"],
        alpha=0.25,
        s=18,
    )
    axes[1].scatter(
        working["resolved_year"],
        working["mobilizable_fraction"].fillna(0.0),
        color=PALETTE["high"],
        alpha=0.25,
        s=18,
    )
    axes[1].plot(
        dense["resolved_year"],
        dense["refseq_record_fraction_rolling3"].fillna(0.0),
        color=PALETTE["support"],
        linewidth=2,
        label="RefSeq fraction (3-year mean)",
    )
    axes[1].plot(
        dense["resolved_year"],
        dense["mobilizable_fraction_rolling3"].fillna(0.0),
        color=PALETTE["high"],
        linewidth=2,
        label="Mobilizable fraction (3-year mean)",
    )
    axes[1].set_title("Temporal Composition Drift")
    axes[1].set_xlabel("Resolved year")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    _apply_axis_style(axes[1], grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_module_f_enrichment(top_hits: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = top_hits.copy()
    if working.empty:
        return
    working["log2_odds_ratio"] = np.log2(
        working["odds_ratio"].replace(0.0, np.nan).replace(np.inf, 1e9)
    )
    working = (
        working.sort_values(["q_value", "log2_odds_ratio"], ascending=[True, False]).head(12).copy()
    )
    working["display_label"] = (
        working["feature_group"].astype(str) + ": " + working["feature_value"].astype(str)
    )
    positions = np.arange(len(working))
    colors = [
        PALETTE["primary"]
        if group in {"primary_replicon", "dominant_mpf_type"}
        else PALETTE["accent"]
        if group in {"amr_class", "dominant_amr_gene_family"}
        else PALETTE["support"]
        for group in working["feature_group"].astype(str)
    ]
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    ax.barh(positions, working["log2_odds_ratio"], color=colors, edgecolor="#23313A", linewidth=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(working["display_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("log2 odds ratio for visibility-positive enrichment")
    ax.set_title("Module F: Independent Genomic Signature Enrichment")
    for idx, row in enumerate(working.itertuples(index=False)):
        ax.text(
            float(row.log2_odds_ratio) + 0.04,
            idx,
            f"q={float(row.q_value):.3f}\nΔprev={float(row.prevalence_delta):+.2f}",
            va="center",
            fontsize=8,
        )
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_all_figures(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    model_metrics: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    dropout_table: pd.DataFrame,
    candidate_stability: pd.DataFrame,
    *,
    threshold_sensitivity: pd.DataFrame | None = None,
    candidate_portfolio: pd.DataFrame | None = None,
    false_negative_audit: pd.DataFrame | None = None,
    core_model_coefficients: pd.DataFrame | None = None,
    governance_model_name: str | None = None,
    figures_dir: Path,
    primary_model_name: str,
) -> list[str]:
    core_dir = figures_dir.parent / "core_figures"
    ensure_directory(core_dir)
    candidate_portfolio = candidate_portfolio if candidate_portfolio is not None else pd.DataFrame()
    false_negative_audit = (
        false_negative_audit if false_negative_audit is not None else pd.DataFrame()
    )
    threshold_sensitivity = (
        threshold_sensitivity if threshold_sensitivity is not None else pd.DataFrame()
    )
    core_model_coefficients = (
        core_model_coefficients if core_model_coefficients is not None else pd.DataFrame()
    )
    outputs = []
    output = core_dir / "score_distribution.png"
    plot_score_distribution(scored, output)
    outputs.append(str(output))

    available_prediction_models = set(predictions["model_name"].astype(str))
    curve_models = []
    for model_name in [
        primary_model_name,
        "full_priority",
        "baseline_both",
        "source_only",
        "random_score_control",
    ]:
        if model_name in available_prediction_models and model_name not in curve_models:
            curve_models.append(model_name)

    output = core_dir / "roc_curve.png"
    plot_roc_curve(predictions, output, curve_models)
    outputs.append(str(output))

    output = core_dir / "pr_curve.png"
    plot_pr_curve(predictions, output, curve_models)
    outputs.append(str(output))

    output = core_dir / "calibration_plot.png"
    plot_calibration(calibration, output, primary_model_name)
    outputs.append(str(output))

    if not threshold_sensitivity.empty:
        output = core_dir / "calibration_threshold_summary.png"
        plot_calibration_threshold_summary(
            calibration,
            threshold_sensitivity,
            output,
            primary_model_name,
        )
        outputs.append(str(output))

    output = core_dir / "threshold_roc_pr_curves.png"
    plot_threshold_roc_pr_curves(scored, predictions, output, model_name=primary_model_name)
    outputs.append(str(output))

    if not candidate_portfolio.empty:
        output = core_dir / "tha_candidate_radar.png"
        plot_tha_radar(candidate_portfolio, scored, output)
        outputs.append(str(output))

    output = core_dir / "knownness_vs_oof_score_scatter.png"
    plot_knownness_vs_oof_scatter(scored, predictions, output, model_name=primary_model_name)
    outputs.append(str(output))

    output = core_dir / "temporal_design.png"
    plot_temporal_design(output)
    outputs.append(str(output))

    output = core_dir / "baseline_vs_full_model_comparison.png"
    plot_benchmark_comparison(
        model_metrics,
        output,
        primary_model_name=primary_model_name,
        governance_model_name=governance_model_name,
    )
    outputs.append(str(output))

    if not coefficient_table.empty:
        output = core_dir / "primary_model_coefficients.png"
        plot_primary_model_coefficients(
            coefficient_table,
            output,
            primary_model_name,
            coefficient_stability=coefficient_stability,
        )
        outputs.append(str(output))

    if not dropout_table.empty:
        output = core_dir / "feature_dropout_importance.png"
        plot_feature_dropout_importance(dropout_table, output, primary_model_name)
        outputs.append(str(output))

    if not candidate_stability.empty:
        output = core_dir / "candidate_stability.png"
        plot_candidate_stability(candidate_stability, output)
        outputs.append(str(output))

    if not false_negative_audit.empty:
        output = core_dir / "false_negative_heatmap.png"
        plot_false_negative_heatmap(false_negative_audit, scored, output)
        outputs.append(str(output))

    if not core_model_coefficients.empty:
        output = core_dir / "core_model_coefficient_heatmap.png"
        plot_core_model_coefficient_heatmap(core_model_coefficients, output)
        outputs.append(str(output))

    return outputs
