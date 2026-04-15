"""Shared figure styling and labeling helpers."""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = {
    "primary": "#0072B2",
    "accent": "#E69F00",
    "support": "#56B4E9",
    "muted": "#7A7A7A",
    "low": "#D55E00",
    "high": "#009E73",
    "background": "#FAF8F3",
}


def style() -> None:
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


def palette_sequence(n: int) -> list[str]:
    base = [
        PALETTE["primary"],
        PALETTE["accent"],
        PALETTE["support"],
        PALETTE["muted"],
        PALETTE["high"],
        PALETTE["low"],
    ]
    return [base[index % len(base)] for index in range(n)]


def annotate_bar_values(
    ax: plt.Axes,
    values: Sequence[float | int],
    positions: Sequence[float | int],
    *,
    fmt: str = "{:.2f}",
    fontsize: int = 9,
) -> None:
    for position, value in zip(positions, values):
        ax.text(
            position,
            value + 0.015,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def apply_axis_style(ax: plt.Axes, *, grid_axis: Literal["both", "x", "y"] | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#D8D1C3", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def stable_jitter(text: str, *, salt: str, scale: float = 0.012) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return (unit - 0.5) * scale


def candidate_tick_label(row: pd.Series) -> str:
    if hasattr(row, "get"):
        return str(row.get("backbone_id", "")).strip()
    return str(getattr(row, "backbone_id", "")).strip()


def pretty_model_label(model_name: str) -> str:
    labels = {
        "parsimonious_priority": "legacy published primary model",
        "natural_auc_priority": "augmented biological model",
        "phylogeny_aware_priority": "phylogeny-aware biological model",
        "structured_signal_priority": "structure-aware biological model",
        "ecology_clinical_priority": "ecology-clinical biological model",
        "knownness_robust_priority": "knownness-robust biological model",
        "support_calibrated_priority": "support-calibrated biological model",
        "support_synergy_priority": "support-synergy biological model",
        "phylo_support_fusion_priority": "phylo-support fusion model",
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


def pretty_feature_label(feature_name: str) -> str:
    labels = {
        "T_eff_norm": "T: mobility (support-adjusted)",
        "H_obs_norm": "H observed breadth",
        "H_obs_specialization_norm": "H observed specialization",
        "H_eff_norm": "H: host breadth (support-adjusted)",
        "T_raw_norm": "T: raw mobility breadth",
        "H_breadth_norm": "H breadth only",
        "H_specialization_norm": "H specialization only",
        "H_phylogenetic_specialization_norm": "H specialization (taxonomy-aware)",
        "H_phylogenetic_norm": "H breadth (taxonomy-aware)",
        "host_phylogenetic_dispersion_norm": "host phylogenetic dispersion",
        "host_taxon_evenness_norm": "host taxon evenness",
        "H_support_norm": "H support only",
        "H_support": "H support composite",
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


def pretty_sensitivity_label(variant: str) -> str:
    if str(variant).startswith("primary_l2_"):
        label = str(variant).replace("primary_l2_", "").replace("p", ".")
        return f"primary model (L2={label})"
    labels = {
        "default": "default",
        "parsimonious_model": "published primary model",
        "alternate_normalization_rank_percentile": "rank-percentile normalization",
        "alternate_normalization_yeo_johnson": "Yeo-Johnson normalization",
        "alternate_outcome_threshold_1": "outcome threshold >=1",
        "alternate_outcome_threshold_2": "outcome threshold >=2",
        "alternate_outcome_threshold_4": "outcome threshold >=4",
        "alternate_outcome_threshold_5": "outcome threshold >=5",
        "alternate_split_2014": "train/test split at 2014",
        "alternate_split_2016": "train/test split at 2016",
        "class_plus_knownness_balanced_primary": "class + knownness-balanced primary model",
        "expanded_eligibility_ge_1": "expanded eligibility (>=1 train country)",
        "fallback_backbone_rerun": "fallback backbone reassignment",
        "knownness_balanced_primary": "knownness-balanced primary model",
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
