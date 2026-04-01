"""Final report figure generation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plasmid_priority_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score as skl_roc_auc_score, roc_curve

from plasmid_priority.utils.files import ensure_directory


PALETTE = {
    "primary": "#0072B2",
    "accent": "#E69F00",
    "support": "#56B4E9",
    "muted": "#7A7A7A",
    "low": "#D55E00",
    "high": "#009E73",
    "background": "#FAF8F3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["background"],
            "axes.edgecolor": "#23313A",
            "axes.labelcolor": "#23313A",
            "xtick.color": "#23313A",
            "ytick.color": "#23313A",
            "text.color": "#23313A",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "lines.markersize": 6,
            "axes.grid": False,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "savefig.facecolor": PALETTE["background"],
        }
    )


def _palette_sequence(n: int) -> list[str]:
    base = [
        PALETTE["primary"],
        PALETTE["accent"],
        PALETTE["support"],
        PALETTE["muted"],
        PALETTE["high"],
        PALETTE["low"],
    ]
    return [base[index % len(base)] for index in range(n)]


def _annotate_bar_values(ax: plt.Axes, values: list[float], positions: list[float], *, fmt: str = "{:.2f}", fontsize: int = 9) -> None:
    for position, value in zip(positions, values):
        ax.text(
            position,
            value + 0.015,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _apply_axis_style(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D8D1C3", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def _format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def _stable_jitter(text: str, *, salt: str, scale: float = 0.012) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return (unit - 0.5) * scale


def _candidate_tick_label(row: pd.Series) -> str:
    if hasattr(row, "get"):
        return str(row.get("backbone_id", "")).strip()
    return str(getattr(row, "backbone_id", "")).strip()


def _pretty_model_label(model_name: str) -> str:
    labels = {
        "parsimonious_priority": "legacy published primary model",
        "natural_auc_priority": "augmented biological model",
        "phylogeny_aware_priority": "phylogeny-aware biological model",
        "structured_signal_priority": "structure-aware biological model",
        "ecology_clinical_priority": "ecology-clinical biological model",
        "knownness_robust_priority": "knownness-robust biological model",
        "support_calibrated_priority": "support-calibrated biological model",
        "support_synergy_priority": "support-synergy biological model",
        "host_transfer_synergy_priority": "host-transfer synergy biological model",
        "threat_architecture_priority": "threat-architecture biological model",
        "adaptive_natural_priority": "knownness-gated natural audit",
        "adaptive_knownness_robust_priority": "knownness-gated switch audit",
        "adaptive_knownness_blend_priority": "knownness-gated blended audit",
        "adaptive_support_calibrated_blend_priority": "support-calibrated gated audit",
        "adaptive_support_synergy_blend_priority": "support-synergy gated audit",
        "adaptive_host_transfer_synergy_blend_priority": "host-transfer synergy gated audit",
        "adaptive_threat_architecture_blend_priority": "threat-architecture gated audit",
        "visibility_adjusted_priority": "visibility-adjusted model",
        "balanced_evidence_priority": "balanced evidence model",
        "evidence_aware_priority": "primary evidence-aware model",
        "bio_clean_priority": "bio-clean model",
        "proxy_light_priority": "legacy support-adjusted model",
        "enhanced_priority": "enhanced integrated model",
        "baseline_both": "counts-only baseline",
        "baseline_member_count": "member-count baseline",
        "baseline_country_count": "country-count baseline",
        "full_priority": "counts + conservative index",
        "T_plus_H_plus_A": "biological core (T+H+A)",
        "source_only": "source-only weak control",
        "random_score_control": "random-score control",
        "label_permutation": "label-permutation control",
        "T_only": "T only",
        "H_only": "H only",
        "A_only": "A only",
        "T_plus_H": "T + H",
        "T_plus_A": "T + A",
        "H_plus_A": "H + A",
    }
    return labels.get(str(model_name), str(model_name))


def _pretty_feature_label(feature_name: str) -> str:
    labels = {
        "T_eff_norm": "T: mobility (support-adjusted)",
        "H_eff_norm": "H: host breadth (support-adjusted)",
        "T_raw_norm": "T: raw mobility breadth",
        "H_breadth_norm": "H breadth only",
        "H_specialization_norm": "H specialization only",
        "H_phylogenetic_specialization_norm": "H specialization (taxonomy-aware)",
        "H_phylogenetic_norm": "H breadth (taxonomy-aware)",
        "host_phylogenetic_dispersion_norm": "host phylogenetic dispersion",
        "host_taxon_evenness_norm": "host taxon evenness",
        "H_support_norm": "H support only",
        "H_support_norm_residual": "H: residual support beyond visibility",
        "host_support_factor": "host support factor",
        "H_raw": "H raw breadth",
        "A_eff_norm": "A: AMR burden (support-adjusted)",
        "A_raw_norm": "A: raw AMR burden",
        "A_recurrence_norm": "A: recurrent AMR structure",
        "amr_support_norm_residual": "A: residual evidence beyond visibility",
        "support_shrinkage_norm": "evidence depth",
        "amr_support_norm": "AMR evidence support",
        "bio_priority_index": "bio priority index",
        "evidence_support_index": "evidence support index",
        "operational_priority_index": "operational priority index",
        "coherence_score": "backbone coherence",
        "orit_support": "oriT support",
        "log1p_member_count_train": "training member count",
        "log1p_n_countries_train": "training country count",
        "priority_index": "conservative priority index",
        "refseq_share_train": "RefSeq fraction",
    }
    return labels.get(str(feature_name), str(feature_name))


def _pretty_sensitivity_label(variant: str) -> str:
    if str(variant).startswith("primary_l2_"):
        label = str(variant).replace("primary_l2_", "").replace("p", ".")
        return f"primary model (L2={label})"
    labels = {
        "default": "default",
        "parsimonious_model": "published primary model",
        "alternate_normalization_robust_sigmoid": "robust sigmoid normalization",
        "alternate_normalization_yeo_johnson": "Yeo-Johnson normalization",
        "alternate_outcome_threshold_1": "outcome threshold >=1",
        "alternate_outcome_threshold_2": "outcome threshold >=2",
        "alternate_outcome_threshold_4": "outcome threshold >=4",
        "alternate_outcome_threshold_5": "outcome threshold >=5",
        "alternate_split_2014": "train/test split at 2014",
        "alternate_split_2016": "train/test split at 2016",
        "class_balanced_primary": "class-balanced primary model",
        "expanded_eligibility_ge_1": "expanded eligibility (>=1 train country)",
        "fallback_backbone_rerun": "fallback backbone reassignment",
        "low_coherence_excluded": "exclude low-coherence backbones",
        "member_count_train_ge_3": "training member count >=3",
        "source_balanced_rerun": "source-balanced rerun",
        "source_plus_class_balanced_primary": "source + class-balanced primary",
        "stable_country_outcome": "stable-country outcome",
        "stable_dense_country_outcome": "stable dense-country outcome",
        "strict_amr_identity99_coverage95": "strict AMR identity/coverage",
        "strict_geometric_priority_as_main": "strict geometric priority headline",
        "training_only_backbone_rerun": "training-only backbone reassignment",
    }
    token = str(variant)
    return labels.get(token, token.replace("_", " "))


def plot_score_distribution(scored: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    working = scored.copy()
    working["operational_priority_index"] = working.get("operational_priority_index", working.get("priority_index", 0.0)).fillna(
        working.get("priority_index", 0.0)
    )
    working["training_support_group"] = np.where(
        working["member_count_train"].fillna(0).astype(int) > 0,
        "training-supported",
        "no training support",
    )
    low_threshold = 0.25
    low_cluster = working.loc[working["operational_priority_index"].fillna(0.0) < low_threshold].copy()
    eligible = working.loc[working["spread_label"].notna()].copy()
    eligible_low_cluster = eligible.loc[eligible["operational_priority_index"].fillna(0.0) < low_threshold].copy()
    bio_threshold = float(eligible["bio_priority_index"].quantile(0.25)) if not eligible.empty else 0.25
    evidence_threshold = float(eligible["evidence_support_index"].quantile(0.25)) if not eligible.empty else 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.4))
    bins = np.linspace(0.0, 1.0, 36)
    support_order = ["no training support", "training-supported"]
    support_summary = []
    for group in support_order:
        frame = working.loc[working["training_support_group"] == group].copy()
        total = int(len(frame))
        low_n = int(frame["operational_priority_index"].fillna(0.0).lt(low_threshold).sum()) if total else 0
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
        axes[0].text(
            idx,
            row["total"] + max(18, row["total"] * 0.01),
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
    for row in eligible_low_cluster[["T_eff_norm", "H_eff_norm", "A_eff_norm"]].to_dict(orient="records"):
        values = {key: float(row.get(key, np.nan)) for key in ("T_eff_norm", "H_eff_norm", "A_eff_norm")}
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
        ax.plot(fpr, tpr, label=f"{_pretty_model_label(model_name)} ({auc:.3f})", color=color, linewidth=2)
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
    prevalence = float(predictions.loc[predictions["model_name"] == model_names[0], "spread_label"].mean()) if model_names else 0.0
    for color, model_name in zip(colors, model_names):
        frame = predictions.loc[predictions["model_name"] == model_name]
        precision, recall, _ = precision_recall_curve(frame["spread_label"], frame["oof_prediction"])
        ap = float(average_precision_score(frame["spread_label"], frame["oof_prediction"]))
        lift = ap - prevalence
        ax.plot(
            recall,
            precision,
            label=f"{_pretty_model_label(model_name)} (AP {ap:.3f}, lift {lift:+.3f})",
            color=color,
            linewidth=2,
        )
    ax.axhline(prevalence, linestyle="--", color=PALETTE["muted"], linewidth=1, label=f"class prevalence ({prevalence:.3f})")
    ax.set_title("Precision-Recall Curve With Prevalence Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    _apply_axis_style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame, output_path: Path, model_name: str) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    lower = calibration.get("observed_rate_ci_lower", calibration["observed_rate"]).fillna(calibration["observed_rate"])
    upper = calibration.get("observed_rate_ci_upper", calibration["observed_rate"]).fillna(calibration["observed_rate"])
    yerr = np.vstack(
        [
            (calibration["observed_rate"] - lower).clip(lower=0.0).to_numpy(dtype=float),
            (upper - calibration["observed_rate"]).clip(lower=0.0).to_numpy(dtype=float),
        ]
    )
    ax.errorbar(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        yerr=yerr,
        marker="o",
        color=PALETTE["primary"],
        linewidth=2,
        capsize=3,
    )
    point_sizes = 25 + 1.5 * calibration["n_backbones"].clip(lower=0).to_numpy(dtype=float)
    ax.scatter(
        calibration["mean_prediction"],
        calibration["observed_rate"],
        s=point_sizes,
        color=PALETTE["accent"],
        edgecolor="#23313A",
        linewidth=0.5,
        zorder=3,
        alpha=0.9,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
    for row in calibration.itertuples(index=False):
        ax.annotate(
            f"n={int(row.n_backbones)}",
            (row.mean_prediction, row.observed_rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_title(f"Calibration Plot ({_pretty_model_label(model_name)})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed visibility-expansion rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _apply_axis_style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics_bar(model_metrics: pd.DataFrame, output_path: Path, title: str, metric: str, *, highlight_models: set[str] | None = None) -> None:
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
    ax.set_xticklabels([_pretty_model_label(name) for name in working["model_name"]], rotation=30, ha="right")
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
    frame["ap_lift_ci_lower"] = frame["average_precision_ci_lower"].fillna(frame["average_precision_lift"]) - frame["positive_prevalence"].fillna(0.0)
    frame["ap_lift_ci_upper"] = frame["average_precision_ci_upper"].fillna(frame["average_precision_lift"]) - frame["positive_prevalence"].fillna(0.0)
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
    axes[0].hlines(positions, frame["roc_low"], frame["roc_high"], color=PALETTE["muted"], linewidth=2)
    axes[0].scatter(frame["roc_center"], positions, color=PALETTE["accent"], s=55, zorder=3)
    axes[0].axvline(0.0 if use_delta else 0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title("Sensitivity ROC AUC" if not use_delta else "Sensitivity ROC AUC Delta vs Default")
    axes[0].set_xlabel("ROC AUC" if not use_delta else "delta ROC AUC vs default")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(frame["variant_label"].tolist())
    for idx, value in enumerate(frame["roc_auc"].tolist()):
        axes[0].text(frame.iloc[idx]["roc_high"] + 0.004, idx, f"{value:.3f}", va="center", fontsize=8)
    _apply_axis_style(axes[0], grid_axis="x")

    axes[1].hlines(positions, frame["lift_low"], frame["lift_high"], color=PALETTE["muted"], linewidth=2)
    axes[1].scatter(frame["lift_center"], positions, color=PALETTE["primary"], s=55, zorder=3)
    axes[1].axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[1].set_title("Sensitivity AP Lift Above Prevalence" if not use_delta else "Sensitivity AP-Lift Delta vs Default")
    axes[1].set_xlabel("AP lift" if not use_delta else "delta AP lift vs default")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    for idx, value in enumerate(frame["average_precision_lift"].fillna(0.0).tolist()):
        axes[1].text(frame.iloc[idx]["lift_high"] + 0.004, idx, f"{value:+.3f}", va="center", fontsize=8)
    _apply_axis_style(axes[1], grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_sensitivity_curve(sensitivity: dict[str, dict[str, float]], output_path: Path) -> None:
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
                "roc_auc_ci_lower": float(default_metrics.get("roc_auc_ci_lower", default_metrics["roc_auc"])),
                "roc_auc_ci_upper": float(default_metrics.get("roc_auc_ci_upper", default_metrics["roc_auc"])),
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
    ax.plot(frame["threshold"], frame["roc_auc"], color=PALETTE["primary"], marker="o", linewidth=2.2)
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
    ax.set_ylim(max(0.0, frame["roc_auc_ci_lower"].min() - 0.04), min(1.0, frame["roc_auc_ci_upper"].max() + 0.06))
    _apply_axis_style(ax, grid_axis="y")
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
                "l2": float(metrics.get("l2", str(name).replace("primary_l2_", "").replace("p", "."))),
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
    axes[0].fill_between(frame["l2"], frame["roc_auc_ci_lower"], frame["roc_auc_ci_upper"], color=PALETTE["support"], alpha=0.22)
    axes[0].set_xscale("log")
    axes[0].set_title("Primary Model L2 Sensitivity")
    axes[0].set_xlabel("L2 penalty (log scale)")
    axes[0].set_ylabel("ROC AUC")
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].plot(frame["l2"], frame["average_precision_lift"], color=PALETTE["accent"], marker="o", linewidth=2.2)
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
    matching_values = [float(row["mean_matching_fraction_high"]), float(row["mean_matching_fraction_low"])]
    support_values = [float(row["support_fraction_high"]), float(row["support_fraction_low"])]

    axes[0].bar(positions, matching_values, color=colors, edgecolor="#23313A", linewidth=0.4)
    axes[0].set_title(f"Combined PD Matching Fraction ({_format_pvalue(float(row['permutation_p_mean_matching_fraction']))})")
    axes[0].set_ylabel("Mean matching fraction")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, max(0.2, max(matching_values) + 0.05))
    _annotate_bar_values(axes[0], matching_values, positions.tolist())
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].bar(positions, support_values, color=colors, edgecolor="#23313A", linewidth=0.4)
    axes[1].set_title(f"Combined PD Any-Support Fraction ({_format_pvalue(float(row['permutation_p_support_fraction']))})")
    axes[1].set_ylabel("Backbone share with any support")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    _annotate_bar_values(axes[1], support_values, positions.tolist())
    _apply_axis_style(axes[1], grid_axis="y")

    fig.suptitle(
        "Pathogen Detection Descriptive Support: dominant-species exact match plus at least one shared top AMR gene",
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
    working = working.loc[working["pathogen_dataset"].isin(["combined", "clinical", "environmental"])]
    if working.empty:
        return
    working = working.set_index("pathogen_dataset").reindex(["combined", "clinical", "environmental"]).dropna(how="all").reset_index()
    positions = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    low_values = working["mean_matching_fraction_low"].fillna(0.0).to_numpy(dtype=float)
    high_values = working["mean_matching_fraction_high"].fillna(0.0).to_numpy(dtype=float)
    for idx, row in working.iterrows():
        ax.plot(
            [positions[idx], positions[idx]],
            [row["mean_matching_fraction_low"], row["mean_matching_fraction_high"]],
            color=PALETTE["muted"],
            linewidth=2,
            zorder=1,
        )
        ax.text(
            positions[idx],
            max(row["mean_matching_fraction_low"], row["mean_matching_fraction_high"]) + 0.016,
            f"n={int(row['n_high'])}/{int(row['n_low'])}\n{_format_pvalue(float(row['permutation_p_mean_matching_fraction']))}",
            ha="center",
            fontsize=8,
        )
    ax.scatter(positions, low_values, color=PALETTE["low"], s=80, label="low", zorder=2)
    ax.scatter(positions, high_values, color=PALETTE["high"], s=80, label="high", zorder=2)
    ax.set_title("Pathogen Detection Strata Effect Sizes")
    ax.set_ylabel("Mean matching fraction")
    ax.set_xticks(positions)
    ax.set_xticklabels(working["pathogen_dataset"].tolist())
    upper = max(high_values.max() if len(high_values) else 0.0, low_values.max() if len(low_values) else 0.0)
    ax.set_ylim(0.0, upper + 0.06)
    ax.legend(frameon=False)
    matching_rule = str(working["matching_rule"].dropna().iloc[0]) if "matching_rule" in working.columns and working["matching_rule"].notna().any() else ""
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
    colors = ordered["direction"].map({"positive": PALETTE["high"], "negative": PALETTE["low"]}).tolist()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["coefficient"], color=colors)
    if coefficient_stability is not None and not coefficient_stability.empty:
        stability = coefficient_stability.set_index("feature_name")
        means = ordered["feature_name"].map(stability["mean_coefficient"]).fillna(ordered["coefficient"]).to_numpy(dtype=float)
        sds = ordered["feature_name"].map(stability["std_coefficient"]).fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(means, positions, xerr=sds, fmt="none", ecolor="#23313A", capsize=3, linewidth=1)
    ax.axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    max_extent = float(
        max(
            np.max(np.abs(ordered["coefficient"].to_numpy(dtype=float))) if len(ordered) else 0.0,
            np.max(np.abs(means) + sds) if coefficient_stability is not None and not coefficient_stability.empty else 0.0,
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
        ax.text(x_position, idx, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    _apply_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_dropout_importance(dropout: pd.DataFrame, output_path: Path, model_name: str) -> None:
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


def plot_model_comparison_deltas(comparison: pd.DataFrame, output_path: Path, model_name: str) -> None:
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
    xerr = np.vstack(
        [
            (core["delta_roc_auc"] - core["delta_roc_auc_ci_lower"]).clip(lower=0.0).to_numpy(dtype=float),
            (core["delta_roc_auc_ci_upper"] - core["delta_roc_auc"]).clip(lower=0.0).to_numpy(dtype=float),
        ]
    ) if not core.empty else np.zeros((2, 0), dtype=float)
    colors = core["comparison_model_name"].map(
        {
            "baseline_both": PALETTE["primary"],
            "full_priority": PALETTE["support"],
            "T_plus_H_plus_A": PALETTE["accent"],
        }
    ).fillna(PALETTE["support"])
    axes[0].barh(centers, core["delta_roc_auc"], color=colors, edgecolor="#23313A", linewidth=0.4)
    if len(core) > 0:
        axes[0].errorbar(core["delta_roc_auc"], centers, xerr=xerr, fmt="none", ecolor="#23313A", capsize=3, linewidth=1)
    axes[0].axvline(0.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title(f"ROC AUC Gain vs Main Comparators ({_pretty_model_label(model_name)})")
    axes[0].set_xlabel("Primary minus comparator ROC AUC")
    axes[0].set_yticks(centers)
    axes[0].set_yticklabels([_pretty_model_label(name) for name in core["comparison_model_name"].tolist()])
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
        axes[1].errorbar([delta], [0], xerr=np.array([[lower], [upper]]), fmt="none", ecolor="#23313A", capsize=3, linewidth=1)
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
    axes[0].bar(positions - width / 2, primary_values, width=width, color=PALETTE["primary"], label=_pretty_model_label(model_name))
    axes[0].bar(positions + width / 2, baseline_values, width=width, color=PALETTE["muted"], label=_pretty_model_label("baseline_both"))
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
        axes[1].text(idx, value + (0.02 if value >= 0 else -0.05), f"{value:.2f}", ha="center", fontsize=9)
    _apply_axis_style(axes[1], grid_axis="y")

    composition_labels = [f"top-{top_k}\nlower-half"]
    composition_values = [float(row.get("top_k_lower_half_knownness_fraction", np.nan))]
    reference_values = [0.5]
    if q1_supported:
        composition_labels.append(f"top-{top_k}\nlowest quartile")
        composition_values.append(float(row.get("top_k_lowest_quartile_knownness_fraction", np.nan)))
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
        axes[2].text(len(composition_values) - 0.1, reference + 0.015, f"cohort expectation {reference:.2f}", ha="right", fontsize=8, color="#6E6558")
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
        "Main ranking is not novelty-heavy by itself.\nUse the companion novelty frontier/watchlist for new-signal candidates.",
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


def plot_source_balance_resampling(resampling: pd.DataFrame, output_path: Path, model_name: str) -> None:
    _style()
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(resampling["resample_index"], resampling["roc_auc"], marker="o", color=PALETTE["primary"], linewidth=1.8)
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
    ax.scatter(working["low"], positions, color=PALETTE["low"], s=55, label="low prevalence", zorder=3)
    ax.scatter(working["high"], positions, color=PALETTE["high"], s=55, label="high prevalence", zorder=3)
    ax.set_title("CARD Resistance Mechanism Prevalence")
    ax.set_xlabel("Backbone prevalence")
    ax.set_yticks(positions)
    ax.set_yticklabels(working["card_resistance_mechanism"].tolist())
    ax.legend(frameon=False, loc="lower right")
    for y, row in enumerate(working.itertuples(index=False)):
        ax.text(row.high + 0.015, y, f"{row.high:.2f}", va="center", fontsize=8, color=PALETTE["high"])
        ax.text(max(row.low - 0.015, 0.01), y, f"{row.low:.2f}", va="center", ha="right", fontsize=8, color=PALETTE["low"])
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
    colors = working["priority_group"].map({"high": PALETTE["high"], "low": PALETTE["low"]}).tolist()

    axes[0].bar(working["priority_group"], coverage.fillna(0.0), color=colors)
    axes[0].set_title("MOB-suite Literature Coverage")
    axes[0].set_ylabel("Share of selected backbones")
    axes[0].set_ylim(0.0, 1.0)
    _annotate_bar_values(axes[0], coverage.fillna(0.0).tolist(), list(range(len(working))))
    _apply_axis_style(axes[0], grid_axis="y")

    axes[1].bar(working["priority_group"], working["mean_reported_host_range_taxid_count"], color=colors)
    axes[1].set_title("Mean Reported Host-Range Breadth")
    axes[1].set_ylabel("Unique reported host-range taxids")
    _annotate_bar_values(axes[1], working["mean_reported_host_range_taxid_count"].fillna(0.0).tolist(), list(range(len(working))))
    _apply_axis_style(axes[1], grid_axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_who_mia_category_support(comparison: pd.DataFrame, output_path: Path) -> None:
    _style()
    ensure_directory(output_path.parent)
    ordered_categories = ["HPCIA", "CIA", "HIA", "IA"]
    working = comparison.copy()
    working = working.set_index("who_mia_category").reindex(ordered_categories).fillna(0.0).reset_index()
    positions = list(range(len(working)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar([position - width / 2 for position in positions], working.get("high", pd.Series([0.0] * len(working))).tolist(), width=width, color=PALETTE["high"], label="high")
    ax.bar([position + width / 2 for position in positions], working.get("low", pd.Series([0.0] * len(working))).tolist(), width=width, color=PALETTE["low"], label="low")
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
        ax.text(position - width / 2, high + 0.01, f"{high:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(position + width / 2, low + 0.01, f"{low:.2f}", ha="center", va="bottom", fontsize=8)
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
    working["amr_evidence_fraction"] = working["n_with_any_amr_evidence"] / working["n_sequences"].replace(0, pd.NA)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    colors = working["priority_group"].map({"high": PALETTE["high"], "low": PALETTE["low"]}).tolist()
    gene_metric = working.get("mean_gene_jaccard_nonempty", working["mean_gene_jaccard"]).fillna(0.0)
    class_metric = working.get("mean_class_jaccard_nonempty", working["mean_class_jaccard"]).fillna(0.0)
    positions = np.arange(len(working))
    axes[0].bar(positions, working["amr_evidence_fraction"].fillna(0.0), color=colors)
    axes[0].set_title("AMR Evidence Coverage (probe panel)")
    axes[0].set_ylabel("Share with any AMR evidence")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(working["priority_group"].tolist())
    for idx, row in working.reset_index(drop=True).iterrows():
        axes[0].text(
            idx,
            min(float(row["amr_evidence_fraction"]) + 0.04, 0.96),
            f"n={int(row['n_sequences'])}\nAMR={int(row['n_with_any_amr_evidence'])}",
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
        for idx, row in working.reset_index(drop=True).iterrows():
            evaluable_n = int(row[metric_column])
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
    working = rolling.loc[(rolling["status"] == "ok") & (rolling["model_name"] == model_name)].copy()
    if working.empty:
        return
    if "horizon_years" in working.columns and working["horizon_years"].nunique() > 1:
        preferred_horizon = 3 if (working["horizon_years"] == 3).any() else int(sorted(working["horizon_years"].dropna().astype(int).unique())[0])
        working = working.loc[working["horizon_years"].fillna(preferred_horizon).astype(int) == preferred_horizon].copy()
        if diagnostics is not None and not diagnostics.empty and "horizon_years" in diagnostics.columns:
            diagnostics = diagnostics.loc[
                diagnostics["horizon_years"].fillna(preferred_horizon).astype(int) == preferred_horizon
            ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1))
    palette = {"all_records": PALETTE["primary"], "training_only": PALETTE["accent"]}
    mode_labels = {"all_records": "all records assignment", "training_only": "training-only assignment"}
    identical_curves = False
    if diagnostics is not None and not diagnostics.empty:
        identical_curves = bool(
            diagnostics["eligible_identical"].fillna(False).all()
            and diagnostics["roc_auc_delta_training_only_minus_all_records"].fillna(0.0).abs().max() < 1e-10
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
            "eligible cohort stayed identical in every window;\nassignment changes mainly affected rows outside the evaluated set.",
            transform=axes[0].transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F7F1", "edgecolor": "#C8C0B4"},
        )
    for mode, frame in group_iterable:
        ordered = frame.sort_values("split_year")
        ap_values = ordered.get("average_precision_lift", ordered["average_precision"]).astype(float)
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
        unseen_backbone_share = ordered_diag["training_only_future_unseen_backbone_fraction"].fillna(0.0)
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
        working.get("bootstrap_top_10_frequency", working["bootstrap_top_k_frequency"]).fillna(0.0).to_numpy(dtype=float),
        working.get("bootstrap_top_25_frequency", working["bootstrap_top_k_frequency"]).fillna(0.0).to_numpy(dtype=float),
    )
    variant_retention = np.where(
        core_mask,
        working.get("variant_top_10_frequency", working["variant_top_k_frequency"]).fillna(0.0).to_numpy(dtype=float),
        working.get("variant_top_25_frequency", working["variant_top_k_frequency"]).fillna(0.0).to_numpy(dtype=float),
    )
    axes[0].bar(positions - width / 2, bootstrap_retention.tolist(), width=width, color=PALETTE["primary"], label="bootstrap retention in intended slice")
    axes[0].bar(positions + width / 2, variant_retention.tolist(), width=width, color=PALETTE["accent"], label="variant retention in intended slice")
    axes[0].axvline(9.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
    axes[0].set_title("Retention in Core Top-10 or Watchlist Top-25")
    axes[0].set_ylabel("Retention frequency")
    axes[0].set_xticks(positions)
    tick_labels = [f"#{int(row.base_rank)} {_candidate_tick_label(row)}" for row in working.itertuples(index=False)]
    axes[0].set_xticklabels(tick_labels, rotation=30, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    axes[0].text(4.5, 1.02, "core top-10", ha="center", va="bottom", fontsize=9)
    axes[0].text(10.5, 1.02, "watchlist top-25", ha="center", va="bottom", fontsize=9)
    _apply_axis_style(axes[0], grid_axis="y")

    rank_std = working.get("bootstrap_rank_std", pd.Series([0.0] * len(working))).fillna(0.0).to_numpy(dtype=float)
    axes[1].errorbar(
        positions,
        working.get("bootstrap_mean_rank", pd.Series(range(1, len(working) + 1))).to_numpy(dtype=float),
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
        "Ranks 1-10 are judged by top-10 retention;\nranks 11+ are judged as watchlist top-25 entries.",
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
    required = {"baseline_both_oof_prediction", "primary_model_oof_prediction", "novelty_margin_vs_baseline"}
    if not required.issubset(set(frontier.columns)):
        return
    working = frontier.dropna(
        subset=["baseline_both_oof_prediction", "primary_model_oof_prediction", "novelty_margin_vs_baseline"]
    ).copy()
    if working.empty:
        return

    working["knownness_half"] = working.get("knownness_half", pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
    watchlist = working.loc[working["knownness_half"] == "lower_half"].copy()
    watchlist = watchlist.sort_values(["novelty_margin_vs_baseline", "primary_model_oof_prediction"], ascending=[False, False]).head(12)

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
            + frame["backbone_id"].astype(str).map(lambda value: _stable_jitter(value, salt="novelty_frontier_x"))
        ).clip(lower=0.0, upper=1.0)
        frame["y_plot"] = (
            frame["primary_model_oof_prediction"].astype(float)
            + frame["backbone_id"].astype(str).map(lambda value: _stable_jitter(value, salt="novelty_frontier_y"))
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
    label_offsets = [(5, 5), (5, -12), (-34, 6), (-30, -12), (10, 14), (10, -18), (18, 8), (-24, 14)]
    for row, offset in zip(watchlist.head(6).itertuples(index=False), label_offsets):
        x_plot = float(np.clip(row.baseline_both_oof_prediction + _stable_jitter(str(row.backbone_id), salt="novelty_frontier_x"), 0.0, 1.0))
        y_plot = float(np.clip(row.primary_model_oof_prediction + _stable_jitter(str(row.backbone_id), salt="novelty_frontier_y"), 0.0, 1.0))
        axes[0].scatter(x_plot, y_plot, s=80, facecolor="none", edgecolor="#23313A", linewidth=1.1, zorder=4)
        axes[0].annotate(str(row.backbone_id), (x_plot, y_plot), xytext=offset, textcoords="offset points", fontsize=8)
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
            axes[1].scatter(row.knownness_score, row.novelty_margin_vs_baseline, s=80, facecolor="none", edgecolor="#23313A", linewidth=1.1, zorder=4)
            axes[1].annotate(str(row.backbone_id), (row.knownness_score, row.novelty_margin_vs_baseline), xytext=offset, textcoords="offset points", fontsize=8)
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
    dense = working.loc[~working["low_density_year"].fillna(False)].copy() if "low_density_year" in working.columns else working.loc[working["n_records"].fillna(0).astype(int) >= 20].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    if "low_density_year" in working.columns and working["low_density_year"].any():
        low_density = working.loc[working["low_density_year"]].copy()
        axes[0].axvspan(low_density["resolved_year"].min() - 0.5, low_density["resolved_year"].max() + 0.5, color="#E6E0D3", alpha=0.35)
        axes[1].axvspan(low_density["resolved_year"].min() - 0.5, low_density["resolved_year"].max() + 0.5, color="#E6E0D3", alpha=0.35)

    axes[0].scatter(working["resolved_year"], working["n_records"], color=PALETTE["primary"], alpha=0.25, s=18)
    axes[0].scatter(working["resolved_year"], working["n_backbones"], color=PALETTE["accent"], alpha=0.25, s=18)
    axes[0].plot(dense["resolved_year"], dense["n_records_rolling3"], color=PALETTE["primary"], linewidth=2, label="records (3-year mean)")
    axes[0].plot(dense["resolved_year"], dense["n_backbones_rolling3"], color=PALETTE["accent"], linewidth=2, label="backbones (3-year mean)")
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

    axes[1].scatter(working["resolved_year"], working["refseq_record_fraction"].fillna(0.0), color=PALETTE["support"], alpha=0.25, s=18)
    axes[1].scatter(working["resolved_year"], working["mobilizable_fraction"].fillna(0.0), color=PALETTE["high"], alpha=0.25, s=18)
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
    working["log2_odds_ratio"] = np.log2(working["odds_ratio"].replace(0.0, np.nan).replace(np.inf, 1e9))
    working = working.sort_values(["q_value", "log2_odds_ratio"], ascending=[True, False]).head(12).copy()
    working["display_label"] = working["feature_group"].astype(str) + ": " + working["feature_value"].astype(str)
    positions = np.arange(len(working))
    colors = [
        PALETTE["primary"] if group in {"primary_replicon", "dominant_mpf_type"} else
        PALETTE["accent"] if group in {"amr_class", "dominant_amr_gene_family"} else
        PALETTE["support"]
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
    figures_dir: Path,
    primary_model_name: str,
) -> list[str]:
    core_dir = figures_dir.parent / "core_figures"
    ensure_directory(core_dir)
    outputs = []
    output = core_dir / "score_distribution.png"
    plot_score_distribution(scored, output)
    outputs.append(str(output))

    available_prediction_models = set(predictions["model_name"].astype(str))
    curve_models = []
    for model_name in [primary_model_name, "full_priority", "baseline_both", "source_only", "random_score_control"]:
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

    baseline_models = model_metrics.loc[
        model_metrics["model_name"].isin(
            [
                "baseline_member_count",
                "baseline_country_count",
                "baseline_both",
                "full_priority",
                primary_model_name,
                "source_only",
            ]
        )
    ]
    output = core_dir / "baseline_vs_full_model_comparison.png"
    plot_metrics_bar(
        baseline_models,
        output,
        "Baseline vs Priority Model Comparison (point estimates)",
        "roc_auc",
        highlight_models={primary_model_name, "full_priority"},
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

    return outputs
