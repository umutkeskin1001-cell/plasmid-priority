"""Model audit tables for jury-facing validation and explainability."""

from __future__ import annotations

import ast
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    annotate_knownness_metadata,
    build_single_model_pareto_screen,
    evaluate_model_name,
    fit_full_model_predictions,
    fit_predict_model_holdout,
    get_feature_track,
    get_model_track,
)
from plasmid_priority.modeling.module_a import (
    _compute_sample_weight,
    _eligible_xy,
    _ensure_feature_columns,
    _oof_predictions_from_eligible,
)
from plasmid_priority.modeling.module_a_support import (
    _fit_kwarg_float,
    _fit_kwarg_int,
    _fit_kwarg_mode,
    _model_fit_kwargs,
    _oof_predictions,
    _standardize_apply,
    _standardize_fit,
    _stratified_folds,
)
from plasmid_priority.modeling.single_model_pareto import (
    add_failure_severity,
    build_pareto_shortlist,
)
from plasmid_priority.reporting.candidate_tables import (
    annotate_candidate_explanation_fields as _annotate_candidate_explanation_fields,
)
from plasmid_priority.reporting.candidate_tables import (
    build_candidate_dossier_table as _build_candidate_dossier_table,
)
from plasmid_priority.reporting.candidate_tables import (
    build_candidate_portfolio_table as _build_candidate_portfolio_table,
)
from plasmid_priority.reporting.candidate_tables import (
    build_candidate_risk_table as _build_candidate_risk_table,
)
from plasmid_priority.reporting.candidate_tables import (
    build_decision_yield_table as _build_decision_yield_table,
)
from plasmid_priority.reporting.candidate_tables import (
    build_threshold_flip_table as _build_threshold_flip_table,
)
from plasmid_priority.reporting.candidate_tables import (
    build_threshold_utility_table as _build_threshold_utility_table,
)
from plasmid_priority.scoring import DEFAULT_NORMALIZATION_METHOD, recompute_priority_from_reference
from plasmid_priority.utils.dataframe import coalescing_left_merge
from plasmid_priority.utils.parallel import limit_native_threads
from plasmid_priority.validation import (
    average_precision,
    average_precision_enrichment,
    average_precision_lift,
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    paired_auc_delong,
    paired_bootstrap_deltas,
    positive_prevalence,
    roc_auc_score,
)


class _BootstrapCandidateStats(TypedDict):
    top_hits: int
    top_hits_top10: int
    top_hits_top25: int
    ranks: list[int]


FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "matched_knownness_gap_min": -0.005,
    "source_holdout_gap_min": -0.005,
    "spatial_holdout_gap_min": -0.03,
    "ece_max": 0.05,
    "selection_adjusted_p_max": 0.01,
}

_CALIBRATION_METHODS: tuple[str, ...] = ("raw", "platt", "isotonic", "beta")
ADAPTIVE_GATED_PUBLIC_COLUMNS: tuple[str, ...] = (
    "backbone_id",
    "adaptive_prediction",
    "spread_label",
    "knownness_score",
    "knownness_half",
    "knownness_quartile",
    "model_name",
    "base_model_name",
    "specialist_model_name",
    "gating_rule",
    "prediction_source",
    "specialist_weight_lower_half",
    "base_oof_prediction",
    "novelty_specialist_prediction",
    "upper_half_route_prediction",
    "lower_half_route_prediction",
)


annotate_candidate_explanation_fields = _annotate_candidate_explanation_fields
build_candidate_dossier_table = _build_candidate_dossier_table
build_candidate_portfolio_table = _build_candidate_portfolio_table
build_candidate_risk_table = _build_candidate_risk_table
build_decision_yield_table = _build_decision_yield_table
build_threshold_flip_table = _build_threshold_flip_table
build_threshold_utility_table = _build_threshold_utility_table


def sanitize_adaptive_gated_predictions(adaptive_predictions: pd.DataFrame) -> pd.DataFrame:
    """Persist only the public adaptive-gating schema expected by reports and audits."""
    if adaptive_predictions.empty:
        return pd.DataFrame(columns=list(ADAPTIVE_GATED_PUBLIC_COLUMNS))

    working = adaptive_predictions.copy()
    for column in ADAPTIVE_GATED_PUBLIC_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    return working.loc[:, list(ADAPTIVE_GATED_PUBLIC_COLUMNS)].copy()


def _stable_unit_interval(text: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    return float(int.from_bytes(digest[:8], "big") / 2**64)


def _resolve_parallel_jobs(requested_jobs: int | None, *, max_tasks: int, cap: int = 8) -> int:
    if max_tasks <= 1:
        return 1
    env_cap = os.getenv("PLASMID_PRIORITY_MAX_JOBS")
    if env_cap:
        try:
            cap = max(1, min(cap, int(env_cap)))
        except ValueError:
            pass
    if requested_jobs is None:
        requested = min(cap, os.cpu_count() or 1)
    else:
        requested = int(requested_jobs)
    return max(1, min(requested, max_tasks, cap))


def _active_model_metrics(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty or "status" not in model_metrics.columns:
        return model_metrics.copy()
    active = model_metrics.loc[model_metrics["status"].fillna("ok").astype(str).eq("ok")].copy()
    return active if not active.empty else model_metrics.copy()


def build_model_family_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the most decision-relevant models and label their evidence role."""
    model_metrics = _active_model_metrics(model_metrics)
    selected = [
        ("source_only", "easy_proxy", "source composition only"),
        ("baseline_both", "easy_proxy", "training visibility counts only"),
        (
            "full_priority",
            "handcrafted_score",
            "counts plus arithmetic conservative priority index",
        ),
        ("T_plus_H_plus_A", "legacy_biological_core", "legacy support-adjusted T/H/A components"),
        (
            "bio_clean_priority",
            "biological_core",
            "raw biological T/A plus host specialization and oriT support",
        ),
        (
            "bio_residual_synergy_priority",
            "discovery_synergy_research",
            "discovery-safe biological core with predeclared mobility, observed-host-range, coherence, and AMR synergy terms",
        ),
        (
            "hybrid_agreement_priority",
            "hybrid_agreement_research",
            "stacked discovery logistic plus nonlinear gradient-boosted surrogate with isotonic blending and explicit agreement review flags",
        ),
        (
            "firth_parsimonious_priority",
            "rare_event_bias_reduced",
            "parsimonious discovery surface refit with Firth bias reduction for separation- and rare-event-robust coefficients",
        ),
        (
            "parsimonious_priority",
            "legacy_published_primary",
            "support-adjusted T/H/A plus coherence; retained as the legacy interpretable benchmark",
        ),
        (
            "natural_auc_priority",
            "augmented_biological_core",
            "biological core plus external host-range, backbone purity, assignment confidence, mash-based novelty, and replicon architecture",
        ),
        (
            "phylogeny_aware_priority",
            "phylogeny_augmented_core",
            "augmented biological core with host specialization replaced by taxonomy-aware phylogenetic host breadth",
        ),
        (
            "structured_signal_priority",
            "structure_augmented_core",
            "phylogeny-aware biological core plus phylogenetically augmented host specialization, host dispersion, host evenness, recurrent AMR structure, replicon multiplicity, and orthogonal PlasmidFinder complexity",
        ),
        (
            "ecology_clinical_priority",
            "eco_clinical_augmented_core",
            "augmented biological core plus clinical-context prevalence and ecological-context diversity",
        ),
        (
            "knownness_robust_priority",
            "knownness_robust_core",
            "augmented biological core plus clinical-context prevalence, ecological-context diversity, recurrent AMR structure, and pMLST coherence under class+knownness balancing",
        ),
        (
            "support_calibrated_priority",
            "support_calibrated_core",
            "knownness-robust biological core plus explicit host-range support, pMLST presence, and AMR support depth for sparse-annotation error recovery",
        ),
        (
            "support_synergy_priority",
            "support_synergy_core",
            "support-calibrated biological core plus normalized pMLST prevalence, orthogonal PlasmidFinder replicon support, residualized AMR support, guarded context support, metadata support depth, external host-range magnitude, and host-range x transfer synergy; retained as the lighter predecessor to the current fusion benchmark",
        ),
        (
            "monotonic_latent_priority",
            "monotonic_latent_core",
            "support-calibrated biological core plus saturating AMR burden, replicon multiplicity, host-range, and eco-clinical latent axes for nonlinear biological documentation",
        ),
        (
            "regime_stability_priority",
            "governance_regime_stability",
            "knownness- and source-residualized biological core with monotonic saturation and guardrail-first weighting for the governance track",
        ),
        (
            "phylo_support_fusion_priority",
            "published_primary",
            "support-synergy biological core plus phylogenetically augmented host specialization, host dispersion, explicit replicon multiplicity, and orthogonal PlasmidFinder structure/support; chosen as the current primary benchmark",
        ),
        (
            "host_transfer_synergy_priority",
            "error_focused_augmented_core",
            "knownness-robust biological core plus explicit host-transfer synergy and external host-range support for sparse-backbone error recovery",
        ),
        (
            "threat_architecture_priority",
            "threat_architecture_audit",
            "host-transfer augmented biological core plus decomposed AMR richness and burden, AMR clinical-threat burden, and replicon multiplicity for sparse-backbone error recovery",
        ),
        (
            "contextual_bio_priority",
            "contextual_biological_core",
            "augmented biological core plus PMLST coherence and eco-clinical context diversity",
        ),
        (
            "visibility_adjusted_priority",
            "deconfounded_evidence",
            "raw biological T/A plus host specialization and visibility-adjusted support residuals",
        ),
        (
            "balanced_evidence_priority",
            "evidence_aware",
            "raw biological axes plus axis-specific evidence depth",
        ),
        (
            "evidence_aware_priority",
            "support_heavy_evidence",
            "raw mobility plus global evidence-depth support",
        ),
        ("proxy_light_priority", "legacy_integrated", "legacy support-adjusted integrated model"),
        (
            "enhanced_priority",
            "proxy_integrated",
            "support-adjusted model plus explicit count proxies",
        ),
    ]
    available = model_metrics.set_index("model_name", drop=False)
    rows = []
    reference_model = None
    if "visibility_adjusted_priority" in available.index:
        reference_model = "visibility_adjusted_priority"
    elif "balanced_evidence_priority" in available.index:
        reference_model = "balanced_evidence_priority"
    elif "evidence_aware_priority" in available.index:
        reference_model = "evidence_aware_priority"
    elif "enhanced_priority" in available.index:
        reference_model = "enhanced_priority"
    elif not available.empty:
        reference_model = str(available["roc_auc"].astype(float).idxmax())
    reference_auc = (
        float(available.loc[reference_model, "roc_auc"])
        if reference_model in available.index
        else np.nan
    )
    enhanced_auc = (
        float(available.loc["enhanced_priority", "roc_auc"])
        if "enhanced_priority" in available.index
        else np.nan
    )
    for model_name, evidence_role, evidence_summary in selected:
        if model_name not in available.index:
            continue
        row = available.loc[model_name].to_dict()
        row["model_track"] = _safe_model_track(model_name)
        row["track_summary"] = _model_track_summary(str(row["model_track"]))
        row["evidence_role"] = evidence_role
        row["evidence_summary"] = evidence_summary
        row["primary_reference_model"] = reference_model
        row["delta_auc_vs_primary_reference"] = (
            float(row["roc_auc"]) - reference_auc if np.isfinite(reference_auc) else np.nan
        )
        row["delta_auc_vs_enhanced_priority"] = (
            float(row["roc_auc"]) - enhanced_auc if np.isfinite(enhanced_auc) else np.nan
        )
        row["leakage_review_required"] = bool(
            np.isfinite(float(row["roc_auc"])) and float(row["roc_auc"]) >= 0.90
        )
        row["leakage_review_reason"] = (
            "roc_auc_ge_0p90_on_current_feature_universe" if row["leakage_review_required"] else ""
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _guardrail_loss_from_scorecard_row(scorecard_row: pd.Series) -> float:
    knownness_gap = pd.to_numeric(
        pd.Series([scorecard_row.get("knownness_matched_gap", np.nan)]), errors="coerce"
    ).iloc[0]
    source_gap = pd.to_numeric(
        pd.Series([scorecard_row.get("source_holdout_gap", np.nan)]), errors="coerce"
    ).iloc[0]
    gaps: list[float] = []
    if pd.notna(knownness_gap):
        gaps.append(abs(float(knownness_gap)))
    if pd.notna(source_gap):
        gaps.append(abs(float(source_gap)))
    if not gaps:
        return float("nan")
    loss = float(sum(gaps))
    if bool(scorecard_row.get("leakage_review_required", False)):
        loss += 0.25
    return loss


def _guardrail_loss_series(scorecard: pd.DataFrame) -> pd.Series:
    knownness_gap = pd.to_numeric(
        scorecard.get("knownness_matched_gap", pd.Series(np.nan, index=scorecard.index)),
        errors="coerce",
    ).abs()
    source_gap = pd.to_numeric(
        scorecard.get("source_holdout_gap", pd.Series(np.nan, index=scorecard.index)),
        errors="coerce",
    ).abs()
    loss = knownness_gap.fillna(0.0) + source_gap.fillna(0.0)
    has_any_gap = knownness_gap.notna() | source_gap.notna()
    loss = loss.where(has_any_gap, np.nan)
    leakage_penalty = (
        pd.to_numeric(
            scorecard.get("leakage_review_required", pd.Series(False, index=scorecard.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )
    return loss + 0.25 * leakage_penalty


def _safe_model_track(model_name: object) -> str:
    try:
        return str(get_model_track(str(model_name)))
    except (KeyError, TypeError, ValueError):
        return "unclassified"


def _model_track_summary(track: str) -> str:
    summaries = {
        "baseline": "count- and source-proxy control surface",
        "discovery": "pre-event biological discovery surface",
        "governance": "support-augmented governance/watch-only surface",
        "unclassified": "unclassified experimental surface",
    }
    return summaries.get(str(track), "unclassified experimental surface")


def _select_governance_scorecard_row(scorecard: pd.DataFrame) -> pd.Series:
    if scorecard.empty or "model_name" not in scorecard.columns:
        return pd.Series(dtype=object)
    working = scorecard.copy()
    if "model_track" not in working.columns:
        working["model_track"] = working["model_name"].map(_safe_model_track)
    governance_mask = working["model_track"].astype(str).eq("governance")
    if governance_mask.any():
        working = working.loc[governance_mask].copy()
    if "roc_auc" not in working.columns:
        working["roc_auc"] = np.nan
    if "average_precision" not in working.columns:
        working["average_precision"] = np.nan
    working["guardrail_loss"] = _guardrail_loss_series(working)
    working["governance_priority_score"] = (
        pd.to_numeric(
            working.get("roc_auc", pd.Series(np.nan, index=working.index)), errors="coerce"
        ).fillna(0.0)
        - working["guardrail_loss"].fillna(1.0)
        - 0.25
        * pd.to_numeric(
            working.get("leakage_review_required", pd.Series(False, index=working.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )
    strict_mask = (
        working.get("strict_knownness_acceptance_flag", pd.Series(False, index=working.index))
        .fillna(False)
        .astype(bool)
    )
    if strict_mask.any():
        working = working.loc[strict_mask].copy()
    working = working.sort_values(
        ["governance_priority_score", "guardrail_loss", "roc_auc", "average_precision"],
        ascending=[False, True, False, False],
        kind="mergesort",
    )
    return working.iloc[0] if not working.empty else pd.Series(dtype=object)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].rank(method="average").corr(frame["right"].rank(method="average")))


def _rank_percentile_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    valid = values.notna()
    result = pd.Series(0.0, index=series.index, dtype=float)
    if not valid.any():
        return result
    ranked = values.loc[valid].rank(method="average", pct=True)
    result.loc[valid] = ranked.to_numpy(dtype=float)
    return result


def build_h_feature_diagnostics(
    scored: pd.DataFrame,
    *,
    model_metrics: pd.DataFrame | None = None,
    coefficient_table: pd.DataFrame | None = None,
    dropout_table: pd.DataFrame | None = None,
    mobsuite_detail: pd.DataFrame | None = None,
    score_column: str = "priority_index",
) -> pd.DataFrame:
    """Summarize whether H behaves like a plausible host-range signal."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()

    ranking_column = score_column if score_column in scored.columns else "priority_index"
    h_feature_names = {
        "H_eff_norm",
        "H_breadth_norm",
        "H_specialization_norm",
        "H_support_norm",
        "H_support_norm_residual",
    }
    row: dict[str, object] = {
        "n_scored_backbones": int(len(scored)),
        "n_eligible_backbones": int(len(eligible)),
        "nonzero_h_fraction_all": float(scored["H_eff_norm"].fillna(0.0).gt(0.0).mean()),
        "nonzero_h_fraction_eligible": float(eligible["H_eff_norm"].fillna(0.0).gt(0.0).mean()),
        "h_eff_norm_mean_positive": float(
            eligible.loc[eligible["spread_label"] == 1, "H_eff_norm"].fillna(0.0).mean()
        ),
        "h_eff_norm_mean_negative": float(
            eligible.loc[eligible["spread_label"] == 0, "H_eff_norm"].fillna(0.0).mean()
        ),
        "h_eff_norm_vs_spread_label_spearman": _safe_spearman(
            eligible["H_eff_norm"], eligible["spread_label"]
        ),
        "h_eff_norm_vs_member_count_train_spearman": _safe_spearman(
            eligible["H_eff_norm"],
            eligible["log1p_member_count_train"],
        ),
        "h_eff_norm_vs_n_countries_train_spearman": _safe_spearman(
            eligible["H_eff_norm"],
            eligible["log1p_n_countries_train"],
        ),
        "h_eff_norm_top100_mean": float(
            scored.sort_values(ranking_column, ascending=False)
            .head(100)["H_eff_norm"]
            .fillna(0.0)
            .mean()
        ),
        "h_eff_norm_bottom100_mean": float(
            scored.sort_values(ranking_column, ascending=True)
            .head(100)["H_eff_norm"]
            .fillna(0.0)
            .mean()
        ),
    }
    if "H_breadth_norm" in eligible.columns:
        row["h_breadth_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_breadth_norm"],
            eligible["spread_label"],
        )
        row["h_breadth_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_breadth_norm"],
            eligible["log1p_member_count_train"],
        )
        row["h_breadth_norm_top100_mean"] = float(
            scored.sort_values(ranking_column, ascending=False)
            .head(100)["H_breadth_norm"]
            .fillna(0.0)
            .mean()
        )
        row["h_breadth_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True)
            .head(100)["H_breadth_norm"]
            .fillna(0.0)
            .mean()
        )
    if "H_specialization_norm" in eligible.columns:
        row["h_specialization_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_specialization_norm"],
            eligible["spread_label"],
        )
        row["h_specialization_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_specialization_norm"],
            eligible["log1p_member_count_train"],
        )
        row["h_specialization_norm_top100_mean"] = float(
            scored.sort_values(ranking_column, ascending=False)
            .head(100)["H_specialization_norm"]
            .fillna(0.0)
            .mean()
        )
        row["h_specialization_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True)
            .head(100)["H_specialization_norm"]
            .fillna(0.0)
            .mean()
        )
    if "H_genus_richness_norm" in eligible.columns:
        row["h_genus_richness_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_genus_richness_norm"],
            eligible["spread_label"],
        )
        row["h_genus_richness_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_genus_richness_norm"],
            eligible["log1p_member_count_train"],
        )
    if "H_genus_norm" in eligible.columns:
        row["h_genus_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_genus_norm"],
            eligible["spread_label"],
        )
        row["h_genus_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_genus_norm"],
            eligible["log1p_member_count_train"],
        )
    if "H_support_norm" in eligible.columns:
        row["h_support_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_support_norm"],
            eligible["spread_label"],
        )
        row["h_support_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_support_norm"],
            eligible["log1p_member_count_train"],
        )
    if "H_raw" in eligible.columns:
        row["h_raw_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_raw"], eligible["spread_label"]
        )
        row["h_raw_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_raw"],
            eligible["log1p_member_count_train"],
        )
    if "H_phylogenetic_norm" in eligible.columns:
        row["h_phylogenetic_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["H_phylogenetic_norm"],
            eligible["spread_label"],
        )
        row["h_phylogenetic_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["H_phylogenetic_norm"],
            eligible["log1p_member_count_train"],
        )
        row["h_phylogenetic_norm_top100_mean"] = float(
            scored.sort_values(ranking_column, ascending=False)
            .head(100)["H_phylogenetic_norm"]
            .fillna(0.0)
            .mean()
        )
        row["h_phylogenetic_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True)
            .head(100)["H_phylogenetic_norm"]
            .fillna(0.0)
            .mean()
        )
    if "host_phylogenetic_dispersion_norm" in eligible.columns:
        row["host_phylogenetic_dispersion_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["host_phylogenetic_dispersion_norm"],
            eligible["spread_label"],
        )
        row["host_phylogenetic_dispersion_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["host_phylogenetic_dispersion_norm"],
            eligible["log1p_member_count_train"],
        )
    if "host_taxon_evenness_norm" in eligible.columns:
        row["host_taxon_evenness_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["host_taxon_evenness_norm"],
            eligible["spread_label"],
        )
        row["host_taxon_evenness_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["host_taxon_evenness_norm"],
            eligible["log1p_member_count_train"],
        )
    if "A_recurrence_norm" in eligible.columns:
        row["a_recurrence_norm_vs_spread_label_spearman"] = _safe_spearman(
            eligible["A_recurrence_norm"],
            eligible["spread_label"],
        )
        row["a_recurrence_norm_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["A_recurrence_norm"],
            eligible["log1p_member_count_train"],
        )
    if "phylo_pairwise_dispersion_score" in eligible.columns:
        row["phylo_pairwise_dispersion_score_vs_spread_label_spearman"] = _safe_spearman(
            eligible["phylo_pairwise_dispersion_score"],
            eligible["spread_label"],
        )
    if "host_support_factor" in eligible.columns:
        row["host_support_factor_vs_spread_label_spearman"] = _safe_spearman(
            eligible["host_support_factor"],
            eligible["spread_label"],
        )
        row["host_support_factor_vs_member_count_train_spearman"] = _safe_spearman(
            eligible["host_support_factor"],
            eligible["log1p_member_count_train"],
        )

    if coefficient_table is not None and not coefficient_table.empty:
        matches = coefficient_table.loc[
            coefficient_table["feature_name"].isin(h_feature_names)
        ].copy()
        if "abs_coefficient" not in matches.columns and "coefficient" in matches.columns:
            matches["abs_coefficient"] = matches["coefficient"].astype(float).abs()
        row["primary_model_uses_h"] = bool(len(matches))
        row["primary_model_h_features"] = (
            ",".join(matches["feature_name"].astype(str).tolist()) if not matches.empty else ""
        )
        row["primary_model_h_total_abs_coefficient"] = (
            float(matches["abs_coefficient"].sum()) if not matches.empty else np.nan
        )
        if not matches.empty:
            dominant = matches.sort_values("abs_coefficient", ascending=False).iloc[0]
            row["primary_model_h_primary_feature"] = str(dominant["feature_name"])
            row["primary_model_h_coefficient"] = float(dominant["coefficient"])
    else:
        row["primary_model_uses_h"] = False
    if dropout_table is not None and not dropout_table.empty:
        matches = dropout_table.loc[
            dropout_table["feature_name"].isin(h_feature_names), "roc_auc_drop_vs_full"
        ]
        row["h_feature_dropout_auc_drop"] = float(matches.sum()) if not matches.empty else np.nan
    if model_metrics is not None and not model_metrics.empty:
        available = model_metrics.set_index("model_name", drop=False)
        for model_name, output_name in (
            ("H_only", "h_only_roc_auc"),
            ("T_plus_H", "t_plus_h_roc_auc"),
            ("T_plus_H_plus_A", "t_plus_h_plus_a_roc_auc"),
            ("proxy_light_priority", "proxy_light_priority_roc_auc"),
            ("enhanced_priority", "enhanced_priority_roc_auc"),
        ):
            if model_name in available.index:
                row[output_name] = float(available.loc[model_name, "roc_auc"])
    if mobsuite_detail is not None and not mobsuite_detail.empty:
        supported = mobsuite_detail.loc[
            mobsuite_detail["mobsuite_any_literature_support"].fillna(False).astype(bool)
            & mobsuite_detail["mobsuite_reported_host_range_taxid_count"]
            .fillna(0)
            .astype(float)
            .gt(0.0)
        ].copy()
        row["mobsuite_supported_backbones"] = int(len(supported))
        if not supported.empty and "priority_group" in supported.columns:
            group_summary = supported.groupby("priority_group", as_index=True)[
                "mobsuite_reported_host_range_taxid_count"
            ].agg(["count", "mean"])
            row["mobsuite_high_literature_supported_n"] = (
                int(group_summary.loc["high", "count"]) if "high" in group_summary.index else 0
            )
            row["mobsuite_low_literature_supported_n"] = (
                int(group_summary.loc["low", "count"]) if "low" in group_summary.index else 0
            )
            row["mobsuite_high_mean_reported_host_range_taxid_count"] = (
                float(group_summary.loc["high", "mean"])
                if "high" in group_summary.index
                else np.nan
            )
            row["mobsuite_low_mean_reported_host_range_taxid_count"] = (
                float(group_summary.loc["low", "mean"]) if "low" in group_summary.index else np.nan
            )
            if "high" in group_summary.index and "low" in group_summary.index:
                row["mobsuite_high_minus_low_mean_reported_host_range_taxid_count"] = float(
                    group_summary.loc["high", "mean"] - group_summary.loc["low", "mean"]
                )
    return pd.DataFrame([row])


def build_knownness_audit_tables(
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    primary_model_name: str,
    baseline_model_name: str = "baseline_both",
    top_k: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify whether the primary model adds discrimination beyond visibility/knownness proxies."""
    primary = predictions.loc[
        predictions["model_name"] == primary_model_name,
        ["backbone_id", "oof_prediction", "spread_label"],
    ].rename(columns={"oof_prediction": "primary_prediction"})
    baseline = predictions.loc[
        predictions["model_name"] == baseline_model_name,
        ["backbone_id", "oof_prediction"],
    ].rename(columns={"oof_prediction": "baseline_prediction"})
    meta_columns = [
        "backbone_id",
        "priority_index",
        "operational_priority_index",
        "bio_priority_index",
        "evidence_support_index",
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "refseq_share_train",
    ]
    merged = primary.merge(baseline, on="backbone_id", how="inner", validate="1:1")
    available_meta_columns = [column for column in meta_columns if column in scored.columns]
    merged = merged.merge(
        scored[available_meta_columns], on="backbone_id", how="left", validate="1:1"
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    for column in (
        "operational_priority_index",
        "bio_priority_index",
        "evidence_support_index",
        "priority_index",
    ):
        if column not in merged.columns:
            merged[column] = np.nan

    merged["operational_priority_index"] = merged["operational_priority_index"].fillna(
        merged["priority_index"]
    )
    merged = annotate_knownness_metadata(merged)

    summary_row: dict[str, object] = {
        "primary_model_name": primary_model_name,
        "baseline_model_name": baseline_model_name,
        "n_backbones": int(len(merged)),
        "n_positive": int(merged["spread_label"].sum()),
        "overall_primary_roc_auc": roc_auc_score(
            merged["spread_label"], merged["primary_prediction"]
        ),
        "overall_baseline_roc_auc": roc_auc_score(
            merged["spread_label"], merged["baseline_prediction"]
        ),
        "overall_delta_roc_auc": roc_auc_score(merged["spread_label"], merged["primary_prediction"])
        - roc_auc_score(merged["spread_label"], merged["baseline_prediction"]),
        "primary_prediction_vs_knownness_spearman": _safe_spearman(
            merged["primary_prediction"],
            merged["knownness_score"],
        ),
        "baseline_prediction_vs_knownness_spearman": _safe_spearman(
            merged["baseline_prediction"],
            merged["knownness_score"],
        ),
        "operational_priority_index_vs_knownness_spearman": _safe_spearman(
            merged["operational_priority_index"],
            merged["knownness_score"],
        ),
        "bio_priority_index_vs_knownness_spearman": _safe_spearman(
            merged["bio_priority_index"],
            merged["knownness_score"],
        ),
        "evidence_support_index_vs_knownness_spearman": _safe_spearman(
            merged["evidence_support_index"],
            merged["knownness_score"],
        ),
        "matched_strata_count": 0,
        "matched_strata_n_backbones": 0,
        "matched_strata_primary_weighted_roc_auc": np.nan,
        "matched_strata_baseline_weighted_roc_auc": np.nan,
        "matched_strata_weighted_delta_roc_auc": np.nan,
        "lower_half_knownness_n_backbones": 0,
        "lower_half_knownness_n_positive": 0,
        "lower_half_knownness_primary_roc_auc": np.nan,
        "lower_half_knownness_baseline_roc_auc": np.nan,
        "lower_half_knownness_delta_roc_auc": np.nan,
        "lowest_knownness_quartile_supported": bool(
            (merged["knownness_quartile"] == "q1_lowest").any()
        ),
        "lowest_knownness_quartile_n_backbones": 0,
        "lowest_knownness_quartile_n_positive": 0,
        "lowest_knownness_quartile_primary_roc_auc": np.nan,
        "lowest_knownness_quartile_baseline_roc_auc": np.nan,
        "lowest_knownness_quartile_delta_roc_auc": np.nan,
    }
    summary_row["priority_index_vs_knownness_spearman"] = summary_row[
        "operational_priority_index_vs_knownness_spearman"
    ]

    cohort_specs = [
        ("lower_half_knownness", merged["knownness_half"] == "lower_half"),
        ("lowest_knownness_quartile", merged["knownness_quartile"] == "q1_lowest"),
    ]
    for label, mask in cohort_specs:
        cohort = merged.loc[mask].copy()
        if cohort.empty or cohort["spread_label"].nunique() < 2:
            continue
        summary_row[f"{label}_n_backbones"] = int(len(cohort))
        summary_row[f"{label}_n_positive"] = int(cohort["spread_label"].sum())
        summary_row[f"{label}_primary_roc_auc"] = roc_auc_score(
            cohort["spread_label"],
            cohort["primary_prediction"],
        )
        summary_row[f"{label}_baseline_roc_auc"] = roc_auc_score(
            cohort["spread_label"],
            cohort["baseline_prediction"],
        )
        summary_row[f"{label}_delta_roc_auc"] = float(
            cast(float, summary_row[f"{label}_primary_roc_auc"])
            - cast(float, summary_row[f"{label}_baseline_roc_auc"])
        )

    top_candidates = merged.sort_values("primary_prediction", ascending=False).head(top_k).copy()
    summary_row["top_k"] = int(top_k)
    summary_row["top_k_mean_knownness_score"] = float(top_candidates["knownness_score"].mean())
    summary_row["eligible_mean_knownness_score"] = float(merged["knownness_score"].mean())
    summary_row["top_k_lower_half_knownness_count"] = int(
        (top_candidates["knownness_half"] == "lower_half").sum()
    )
    summary_row["top_k_lower_half_knownness_fraction"] = float(
        (top_candidates["knownness_half"] == "lower_half").mean()
    )
    summary_row["top_k_lowest_quartile_knownness_count"] = int(
        (top_candidates["knownness_quartile"] == "q1_lowest").sum()
    )
    summary_row["top_k_lowest_quartile_knownness_fraction"] = float(
        (top_candidates["knownness_quartile"] == "q1_lowest").mean()
    )

    stratum_rows: list[dict[str, object]] = []
    for keys, frame in merged.groupby(
        ["member_count_band", "country_count_band", "source_band"],
        sort=False,
        observed=True,
    ):
        if len(frame) < 20 or frame["spread_label"].nunique() < 2:
            continue
        primary_auc = roc_auc_score(frame["spread_label"], frame["primary_prediction"])
        baseline_auc = roc_auc_score(frame["spread_label"], frame["baseline_prediction"])
        stratum_rows.append(
            {
                "member_count_band": str(keys[0]),
                "country_count_band": str(keys[1]),
                "source_band": str(keys[2]),
                "n_backbones": int(len(frame)),
                "n_positive": int(frame["spread_label"].sum()),
                "mean_knownness_score": float(frame["knownness_score"].mean()),
                "primary_roc_auc": primary_auc,
                "baseline_roc_auc": baseline_auc,
                "delta_roc_auc": primary_auc - baseline_auc,
            }
        )
    strata = pd.DataFrame(stratum_rows)
    if not strata.empty:
        weighted_primary = float(
            np.average(strata["primary_roc_auc"], weights=strata["n_backbones"])
        )
        weighted_baseline = float(
            np.average(strata["baseline_roc_auc"], weights=strata["n_backbones"])
        )
        summary_row["matched_strata_count"] = int(len(strata))
        summary_row["matched_strata_n_backbones"] = int(strata["n_backbones"].sum())
        summary_row["matched_strata_primary_weighted_roc_auc"] = weighted_primary
        summary_row["matched_strata_baseline_weighted_roc_auc"] = weighted_baseline
        summary_row["matched_strata_weighted_delta_roc_auc"] = weighted_primary - weighted_baseline

    if strata.empty:
        return pd.DataFrame([summary_row]), strata

    return pd.DataFrame([summary_row]), strata.sort_values(
        ["delta_roc_auc", "n_backbones"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_novelty_margin_summary(
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    primary_model_name: str,
    baseline_model_name: str = "baseline_both",
    top_k: int = 25,
) -> pd.DataFrame:
    """Summarize where the primary model outperforms a counts-only baseline at candidate level."""
    primary = predictions.loc[
        predictions["model_name"] == primary_model_name,
        ["backbone_id", "oof_prediction", "spread_label"],
    ].rename(columns={"oof_prediction": "primary_prediction"})
    baseline = predictions.loc[
        predictions["model_name"] == baseline_model_name,
        ["backbone_id", "oof_prediction"],
    ].rename(columns={"oof_prediction": "baseline_prediction"})
    meta_columns = [
        "backbone_id",
        "priority_index",
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "refseq_share_train",
    ]
    merged = primary.merge(baseline, on="backbone_id", how="inner", validate="1:1")
    merged = merged.merge(scored[meta_columns], on="backbone_id", how="left", validate="1:1")
    if merged.empty:
        return pd.DataFrame()

    merged = annotate_knownness_metadata(merged)
    merged["novelty_margin"] = merged["primary_prediction"] - merged["baseline_prediction"]

    row: dict[str, object] = {
        "primary_model_name": primary_model_name,
        "baseline_model_name": baseline_model_name,
        "n_backbones": int(len(merged)),
        "n_positive": int(merged["spread_label"].sum()),
        "novelty_margin_overall_roc_auc": roc_auc_score(
            merged["spread_label"], merged["novelty_margin"]
        ),
        "novelty_margin_overall_average_precision": average_precision(
            merged["spread_label"],
            merged["novelty_margin"],
        ),
        "novelty_margin_vs_knownness_spearman": _safe_spearman(
            merged["novelty_margin"],
            merged["knownness_score"],
        ),
        "novelty_margin_vs_priority_index_spearman": _safe_spearman(
            merged["novelty_margin"],
            merged["priority_index"],
        ),
    }

    for label, mask in (
        ("lower_half_knownness", merged["knownness_half"] == "lower_half"),
        ("lowest_knownness_quartile", merged["knownness_quartile"] == "q1_lowest"),
    ):
        cohort = merged.loc[mask].copy()
        row[f"{label}_n_backbones"] = int(len(cohort))
        row[f"{label}_n_positive"] = int(cohort["spread_label"].sum()) if not cohort.empty else 0
        if not cohort.empty and cohort["spread_label"].nunique() >= 2:
            row[f"{label}_novelty_margin_roc_auc"] = roc_auc_score(
                cohort["spread_label"],
                cohort["novelty_margin"],
            )
            row[f"{label}_novelty_margin_average_precision"] = average_precision(
                cohort["spread_label"],
                cohort["novelty_margin"],
            )

    watchlist = merged.loc[merged["knownness_half"] == "lower_half"].copy()
    watchlist = watchlist.sort_values(
        ["novelty_margin", "primary_prediction", "priority_index"],
        ascending=[False, False, False],
    ).head(top_k)
    row["top_k"] = int(top_k)
    row["watchlist_positive_count"] = (
        int(watchlist["spread_label"].sum()) if not watchlist.empty else 0
    )
    row["watchlist_positive_fraction"] = (
        float(watchlist["spread_label"].mean()) if not watchlist.empty else np.nan
    )
    row["watchlist_mean_novelty_margin"] = (
        float(watchlist["novelty_margin"].mean()) if not watchlist.empty else np.nan
    )
    row["watchlist_mean_primary_prediction"] = (
        float(watchlist["primary_prediction"].mean()) if not watchlist.empty else np.nan
    )
    row["watchlist_mean_baseline_prediction"] = (
        float(watchlist["baseline_prediction"].mean()) if not watchlist.empty else np.nan
    )
    row["watchlist_mean_knownness_score"] = (
        float(watchlist["knownness_score"].mean()) if not watchlist.empty else np.nan
    )
    return pd.DataFrame([row])


def _build_gate_consistency_row(
    frame: pd.DataFrame,
    *,
    model_name: str,
    gate_name: str,
    lower_label: str,
    upper_label: str,
    lower_route_column: str,
    upper_route_column: str,
    lower_mask: pd.Series,
    upper_mask: pd.Series,
    near_fraction: float,
    min_n: int,
) -> dict[str, object] | None:
    working = frame.loc[
        lower_mask | upper_mask,
        ["backbone_id", "knownness_score", lower_route_column, upper_route_column],
    ].copy()
    working["knownness_score"] = pd.to_numeric(working["knownness_score"], errors="coerce")
    working[lower_route_column] = pd.to_numeric(working[lower_route_column], errors="coerce")
    working[upper_route_column] = pd.to_numeric(working[upper_route_column], errors="coerce")
    working = working.dropna(subset=["knownness_score", lower_route_column, upper_route_column])
    if working.empty:
        return None

    lower_scores = frame.loc[lower_mask, "knownness_score"]
    upper_scores = frame.loc[upper_mask, "knownness_score"]
    if lower_scores.empty or upper_scores.empty:
        return None
    lower_boundary = float(pd.to_numeric(lower_scores, errors="coerce").dropna().max())
    upper_boundary = float(pd.to_numeric(upper_scores, errors="coerce").dropna().min())
    if not np.isfinite(lower_boundary) or not np.isfinite(upper_boundary):
        return None

    working["distance_to_gate"] = np.minimum(
        (working["knownness_score"] - lower_boundary).abs(),
        (working["knownness_score"] - upper_boundary).abs(),
    )
    near_n = min(len(working), max(int(min_n), int(np.ceil(len(working) * near_fraction))))
    near = (
        working.sort_values(
            ["distance_to_gate", "knownness_score"], ascending=[True, True], kind="mergesort"
        )
        .head(near_n)
        .copy()
    )
    if near.empty:
        return None

    delta_all = (working[lower_route_column] - working[upper_route_column]).abs()
    delta_near = (near[lower_route_column] - near[upper_route_column]).abs()
    mean_delta_near = float(delta_near.mean())
    p90_delta_near = float(delta_near.quantile(0.90))
    route_spearman_near = _safe_spearman(near[lower_route_column], near[upper_route_column])
    if (
        mean_delta_near <= 0.08
        and p90_delta_near <= 0.12
        and np.isfinite(route_spearman_near)
        and route_spearman_near >= 0.95
    ):
        gate_tier = "stable"
    elif (
        mean_delta_near <= 0.10
        and p90_delta_near <= 0.15
        and np.isfinite(route_spearman_near)
        and route_spearman_near >= 0.90
    ):
        gate_tier = "moderate"
    else:
        gate_tier = "unstable"
    return {
        "model_name": model_name,
        "gate_name": gate_name,
        "lower_label": lower_label,
        "upper_label": upper_label,
        "lower_route_column": lower_route_column,
        "upper_route_column": upper_route_column,
        "n_gate_candidates": int(len(working)),
        "n_near_gate": int(len(near)),
        "lower_boundary_max_knownness": lower_boundary,
        "upper_boundary_min_knownness": upper_boundary,
        "boundary_gap": float(upper_boundary - lower_boundary),
        "mean_abs_route_delta_all": float(delta_all.mean()),
        "mean_abs_route_delta_near_gate": mean_delta_near,
        "median_abs_route_delta_near_gate": float(delta_near.median()),
        "p90_abs_route_delta_near_gate": p90_delta_near,
        "max_abs_route_delta_near_gate": float(delta_near.max()),
        "fraction_abs_route_delta_ge_0p05_near_gate": float(delta_near.ge(0.05).mean()),
        "fraction_abs_route_delta_ge_0p10_near_gate": float(delta_near.ge(0.10).mean()),
        "route_spearman_near_gate": route_spearman_near,
        "gate_consistency_tier": gate_tier,
    }


def build_gate_consistency_audit(
    adaptive_predictions: pd.DataFrame,
    *,
    near_fraction: float = 0.10,
    min_n: int = 40,
) -> pd.DataFrame:
    """Quantify how similar route scores remain for backbones close to adaptive gates."""
    if adaptive_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model_name, frame in adaptive_predictions.groupby("model_name", sort=False):
        if frame.empty or "knownness_score" not in frame.columns:
            continue
        lower_mask = (
            frame.get("knownness_half", pd.Series("", index=frame.index))
            .astype(str)
            .eq("lower_half")
        )
        upper_mask = (
            frame.get("knownness_half", pd.Series("", index=frame.index))
            .astype(str)
            .eq("upper_half")
        )
        if {
            "lower_half_route_prediction",
            "upper_half_route_prediction",
        } <= set(frame.columns):
            row = _build_gate_consistency_row(
                frame,
                model_name=str(model_name),
                gate_name="half_boundary",
                lower_label="lower_half",
                upper_label="upper_half",
                lower_route_column="lower_half_route_prediction",
                upper_route_column="upper_half_route_prediction",
                lower_mask=lower_mask,
                upper_mask=upper_mask,
                near_fraction=near_fraction,
                min_n=min_n,
            )
            if row is not None:
                rows.append(row)

        quartiles = frame.get("knownness_quartile", pd.Series("", index=frame.index)).astype(str)
        if {
            "q1_route_prediction",
            "q2_route_prediction",
        } <= set(frame.columns):
            row = _build_gate_consistency_row(
                frame,
                model_name=str(model_name),
                gate_name="q1_q2_boundary",
                lower_label="q1_lowest",
                upper_label="q2",
                lower_route_column="q1_route_prediction",
                upper_route_column="q2_route_prediction",
                lower_mask=quartiles.eq("q1_lowest"),
                upper_mask=quartiles.eq("q2"),
                near_fraction=near_fraction,
                min_n=min_n,
            )
            if row is not None:
                rows.append(row)
        if {
            "q2_route_prediction",
            "q3_route_prediction",
        } <= set(frame.columns):
            row = _build_gate_consistency_row(
                frame,
                model_name=str(model_name),
                gate_name="q2_q3_boundary",
                lower_label="q2",
                upper_label="q3",
                lower_route_column="q2_route_prediction",
                upper_route_column="q3_route_prediction",
                lower_mask=quartiles.eq("q2"),
                upper_mask=quartiles.eq("q3"),
                near_fraction=near_fraction,
                min_n=min_n,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_score_distribution_diagnostics(
    scored: pd.DataFrame,
    *,
    low_score_threshold: float = 0.25,
    high_score_threshold: float = 0.70,
) -> pd.DataFrame:
    """Explain priority-score clustering by support status and limiting component."""
    if scored.empty:
        return pd.DataFrame()

    working = scored.copy()
    component_columns = {
        "T": "T_eff_norm",
        "H": "H_eff_norm",
        "A": "A_eff_norm",
    }
    component_frame = working[list(component_columns.values())].fillna(0.0)
    dominant_floor = component_frame.idxmin(axis=1).map(
        {value: key for key, value in component_columns.items()}
    )
    working["dominant_floor_component"] = dominant_floor.fillna("unknown")

    segments = {
        "all_backbones": pd.Series(True, index=working.index),
        "training_supported": working["member_count_train"].fillna(0).astype(int) > 0,
        "eligible_candidate_cohort": working["spread_label"].notna()
        if "spread_label" in working.columns
        else pd.Series(False, index=working.index),
        "no_training_support": working["member_count_train"].fillna(0).astype(int) == 0,
        "low_score_cluster": working["priority_index"].fillna(0.0) < low_score_threshold,
        "low_score_supported": (working["priority_index"].fillna(0.0) < low_score_threshold)
        & working["member_count_train"].fillna(0).astype(int).gt(0),
        "low_score_supported_eligible": (
            (working["priority_index"].fillna(0.0) < low_score_threshold)
            & working["member_count_train"].fillna(0).astype(int).gt(0)
            & (working["spread_label"].notna() if "spread_label" in working.columns else False)
        ),
        "low_score_no_training_support": (
            working["priority_index"].fillna(0.0) < low_score_threshold
        )
        & working["member_count_train"].fillna(0).astype(int).eq(0),
        "high_score_cluster": working["priority_index"].fillna(0.0) >= high_score_threshold,
    }

    rows: list[dict[str, object]] = []
    total = max(len(working), 1)
    for segment_name, mask in segments.items():
        frame = working.loc[mask].copy()
        if frame.empty:
            continue
        floor_share = frame["dominant_floor_component"].value_counts(normalize=True)
        rows.append(
            {
                "segment": segment_name,
                "low_score_threshold": float(low_score_threshold),
                "high_score_threshold": float(high_score_threshold),
                "n_backbones": int(len(frame)),
                "share_of_all_backbones": float(len(frame) / total),
                "mean_priority_index": float(frame["priority_index"].fillna(0.0).mean()),
                "mean_operational_priority_index": float(
                    frame.get("operational_priority_index", frame["priority_index"])
                    .fillna(0.0)
                    .mean()
                ),
                "mean_bio_priority_index": float(
                    frame.get("bio_priority_index", pd.Series(0.0, index=frame.index))
                    .fillna(0.0)
                    .mean()
                ),
                "mean_evidence_support_index": float(
                    frame.get("evidence_support_index", pd.Series(0.0, index=frame.index))
                    .fillna(0.0)
                    .mean()
                ),
                "median_priority_index": float(frame["priority_index"].fillna(0.0).median()),
                "mean_member_count_train": float(frame["member_count_train"].fillna(0.0).mean()),
                "median_member_count_train": float(
                    frame["member_count_train"].fillna(0.0).median()
                ),
                "zero_training_support_fraction": float(
                    frame["member_count_train"].fillna(0).astype(int).eq(0).mean()
                ),
                "mean_T_eff_norm": float(frame["T_eff_norm"].fillna(0.0).mean()),
                "mean_H_eff_norm": float(frame["H_eff_norm"].fillna(0.0).mean()),
                "mean_A_eff_norm": float(frame["A_eff_norm"].fillna(0.0).mean()),
                "dominant_floor_T_fraction": float(floor_share.get("T", 0.0)),
                "dominant_floor_H_fraction": float(floor_share.get("H", 0.0)),
                "dominant_floor_A_fraction": float(floor_share.get("A", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_score_axis_summary(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model_name: str,
    baseline_model_name: str = "baseline_both",
) -> pd.DataFrame:
    """Summarize biological, evidence, and operational axes on a common knownness scale."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["operational_priority_index"] = eligible.get(
        "operational_priority_index", eligible.get("priority_index", 0.0)
    ).fillna(eligible.get("priority_index", 0.0))
    if "bio_priority_index" not in eligible.columns:
        eligible["bio_priority_index"] = np.nan
    if "evidence_support_index" not in eligible.columns:
        eligible["evidence_support_index"] = np.nan
    if "H_specialization_norm" not in eligible.columns and "H_breadth_norm" in eligible.columns:
        eligible["H_specialization_norm"] = (1.0 - eligible["H_breadth_norm"].fillna(0.0)).clip(
            lower=0.0, upper=1.0
        )

    for column in ("log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"):
        if column not in eligible.columns:
            eligible[column] = 0.0
        eligible[f"{column}_rank"] = _rank_percentile_series(eligible[column].fillna(0.0))
    eligible["knownness_score"] = (
        eligible["log1p_member_count_train_rank"]
        + eligible["log1p_n_countries_train_rank"]
        + eligible["refseq_share_train_rank"]
    ) / 3.0
    prediction_frames = {}
    for model_name, output_column in (
        (primary_model_name, "primary_prediction"),
        (baseline_model_name, "baseline_prediction"),
    ):
        if model_name not in set(predictions["model_name"].astype(str)):
            continue
        prediction_frames[output_column] = predictions.loc[
            predictions["model_name"] == model_name,
            ["backbone_id", "oof_prediction"],
        ].rename(columns={"oof_prediction": output_column})
    for frame in prediction_frames.values():
        eligible = eligible.merge(frame, on="backbone_id", how="left", validate="1:1")

    rows: list[dict[str, object]] = []
    axis_specs = [
        ("primary_prediction", f"{primary_model_name}_prediction", "primary_prediction"),
        ("baseline_prediction", f"{baseline_model_name}_prediction", "baseline_prediction"),
        ("operational_priority_index", "operational_priority_index", "operational_priority_index"),
        ("bio_priority_index", "bio_priority_index", "bio_priority_index"),
        ("evidence_support_index", "evidence_support_index", "evidence_support_index"),
        ("H_specialization_norm", "H_specialization_norm", "H_specialization_norm"),
    ]
    for axis_key, axis_name, column in axis_specs:
        if column not in eligible.columns:
            continue
        frame = eligible[["spread_label", "knownness_score", column]].dropna()
        if frame.empty or frame[column].nunique() < 2:
            continue
        rows.append(
            {
                "axis_key": axis_key,
                "axis_name": axis_name,
                "n_backbones": int(len(frame)),
                "roc_auc": roc_auc_score(frame["spread_label"], frame[column]),
                "average_precision": average_precision(frame["spread_label"], frame[column]),
                "positive_prevalence": positive_prevalence(frame["spread_label"]),
                "average_precision_lift": average_precision_lift(
                    frame["spread_label"], frame[column]
                ),
                "average_precision_enrichment": average_precision_enrichment(
                    frame["spread_label"], frame[column]
                ),
                "knownness_spearman": _safe_spearman(frame[column], frame["knownness_score"]),
                "mean_value": float(frame[column].mean()),
                "median_value": float(frame[column].median()),
                "zero_fraction": float(frame[column].eq(0.0).mean()),
                "lower_quartile_fraction": float(
                    frame[column].le(frame[column].quantile(0.25)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_component_floor_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    """Expose how often each component is structurally zero and what score that implies."""
    if scored.empty:
        return pd.DataFrame()

    training_supported = scored["member_count_train"].fillna(0).astype(int) > 0
    eligible = scored["spread_label"].notna()
    rows: list[dict[str, object]] = []
    component_specs = [
        ("T", "T_eff", "T_eff_norm"),
        ("H", "H_eff", "H_eff_norm"),
        ("A", "A_eff", "A_eff_norm"),
    ]
    for component_name, raw_column, norm_column in component_specs:
        raw = scored[raw_column].fillna(0.0).astype(float)
        norm = scored[norm_column].fillna(0.0).astype(float)
        zero_mask = raw <= 0.0
        training_zero = zero_mask & training_supported
        eligible_zero = zero_mask & eligible
        rows.append(
            {
                "component": component_name,
                "n_backbones": int(len(scored)),
                "n_training_supported": int(training_supported.sum()),
                "n_eligible": int(eligible.sum()),
                "zero_fraction_all": float(zero_mask.mean()),
                "zero_fraction_training_reference": float(
                    training_zero.sum() / training_supported.sum()
                )
                if training_supported.any()
                else np.nan,
                "zero_fraction_eligible": float(eligible_zero.sum() / eligible.sum())
                if eligible.any()
                else np.nan,
                "normalized_value_when_raw_zero_min": float(norm.loc[zero_mask].min())
                if zero_mask.any()
                else np.nan,
                "normalized_value_when_raw_zero_median": float(norm.loc[zero_mask].median())
                if zero_mask.any()
                else np.nan,
                "normalized_value_when_raw_zero_max": float(norm.loc[zero_mask].max())
                if zero_mask.any()
                else np.nan,
                "normalized_value_when_raw_positive_min": float(norm.loc[raw > 0.0].min())
                if (raw > 0.0).any()
                else np.nan,
                "normalized_value_when_raw_positive_median": float(norm.loc[raw > 0.0].median())
                if (raw > 0.0).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_amrfinder_coverage_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Make AMRFinder support coverage explicit before interpreting concordance values."""
    if summary.empty:
        return pd.DataFrame()
    working = summary.loc[summary["priority_group"] != "overall"].copy()
    if working.empty:
        return pd.DataFrame()
    working["amrfinder_hit_fraction"] = working["n_with_amrfinder_hits"] / working[
        "n_sequences"
    ].replace(0, np.nan)
    working["amr_evidence_fraction"] = working["n_with_any_amr_evidence"] / working[
        "n_sequences"
    ].replace(0, np.nan)
    working["nonempty_concordance_evaluable_fraction"] = working["amr_evidence_fraction"]
    return working


def build_model_subgroup_performance(
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    model_names: list[str],
) -> pd.DataFrame:
    """Compute subgroup-specific discrimination metrics for selected models."""
    backbone_meta = scored[
        [
            "backbone_id",
            "refseq_share_train",
            "member_count_train",
            "n_countries_train",
        ]
    ].copy()
    backbone_meta["dominant_source"] = np.where(
        backbone_meta["refseq_share_train"].fillna(0.0) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    backbone_meta["member_count_band"] = pd.cut(
        backbone_meta["member_count_train"].fillna(0).astype(float),
        bins=[-np.inf, 1, 2, np.inf],
        labels=["1", "2", "3_plus"],
    ).astype(str)
    backbone_meta["country_count_band"] = (
        backbone_meta["n_countries_train"].fillna(0).astype(int).astype(str)
    )

    merged = predictions.merge(backbone_meta, on="backbone_id", how="left", validate="m:1")
    merged = merged.loc[merged["model_name"].isin(model_names)].copy()

    subgroup_specs = [
        ("overall", None),
        ("dominant_source", "dominant_source"),
        ("member_count_band", "member_count_band"),
        ("country_count_band", "country_count_band"),
    ]
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = merged.loc[merged["model_name"] == model_name].copy()
        if frame.empty:
            continue
        for subgroup_name, column in subgroup_specs:
            if column is None:
                subsets = [("all", frame)]
            else:
                subsets = list(frame.groupby(column, dropna=False))
            for subgroup_value, subset in subsets:
                y = subset["spread_label"].to_numpy(dtype=int)
                preds = subset["oof_prediction"].to_numpy(dtype=float)
                if len(subset) < 20 or len(np.unique(y)) < 2:
                    rows.append(
                        {
                            "model_name": model_name,
                            "subgroup_name": subgroup_name,
                            "subgroup_value": str(subgroup_value),
                            "n_backbones": int(len(subset)),
                            "n_positive": int(y.sum()),
                            "roc_auc": np.nan,
                            "average_precision": np.nan,
                            "brier_score": np.nan,
                            "status": "skipped_insufficient_label_variation",
                        }
                    )
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "subgroup_name": subgroup_name,
                        "subgroup_value": str(subgroup_value),
                        "n_backbones": int(len(subset)),
                        "n_positive": int(y.sum()),
                        "roc_auc": roc_auc_score(y, preds),
                        "average_precision": average_precision(y, preds),
                        "brier_score": brier_score(y, preds),
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def build_model_comparison_table(
    predictions: pd.DataFrame,
    *,
    primary_model_name: str,
    comparison_model_names: list[str],
) -> pd.DataFrame:
    """Build paired bootstrap comparisons between the primary model and comparators."""
    required_models = [primary_model_name] + [
        name for name in comparison_model_names if name != primary_model_name
    ]
    wide = (
        predictions.loc[
            predictions["model_name"].isin(required_models),
            ["backbone_id", "model_name", "oof_prediction", "spread_label"],
        ]
        .pivot_table(
            index="backbone_id", columns="model_name", values="oof_prediction", aggfunc="first"
        )
        .reset_index()
    )
    primary_labels = (
        predictions.loc[
            predictions["model_name"] == primary_model_name, ["backbone_id", "spread_label"]
        ]
        .drop_duplicates("backbone_id")
        .rename(columns={"spread_label": "primary_spread_label"})
    )
    wide = wide.merge(primary_labels, on="backbone_id", how="inner", validate="1:1")
    if primary_model_name not in wide.columns:
        return pd.DataFrame()
    rows = []
    valid_primary = wide[primary_model_name].notna() & wide["primary_spread_label"].notna()
    base = wide.loc[valid_primary].copy()
    if base.empty:
        return pd.DataFrame()
    for comparator in comparison_model_names:
        if comparator == primary_model_name:
            continue
        if comparator not in base.columns:
            continue
        merged = base.loc[base[comparator].notna()].copy()
        if merged.empty:
            continue
        y = merged["primary_spread_label"].to_numpy(dtype=int)
        primary_scores = merged[primary_model_name].to_numpy(dtype=float)
        other_scores = merged[comparator].to_numpy(dtype=float)
        discrimination_deltas = paired_bootstrap_deltas(
            y,
            primary_scores,
            other_scores,
            {
                "roc_auc": roc_auc_score,
                "average_precision": average_precision,
            },
        )
        brier_delta = paired_bootstrap_deltas(
            y,
            other_scores,
            primary_scores,
            {"brier_score": brier_score},
        )["brier_score"]
        delong = paired_auc_delong(y, primary_scores, other_scores)
        primary_roc_auc = roc_auc_score(y, primary_scores)
        comparison_roc_auc = roc_auc_score(y, other_scores)
        primary_average_precision = average_precision(y, primary_scores)
        comparison_average_precision = average_precision(y, other_scores)
        rows.append(
            {
                "primary_model_name": primary_model_name,
                "comparison_model_name": comparator,
                "n_backbones": int(len(merged)),
                "positive_prevalence": positive_prevalence(y),
                "primary_roc_auc": primary_roc_auc,
                "comparison_roc_auc": comparison_roc_auc,
                "delta_roc_auc": discrimination_deltas["roc_auc"]["delta"],
                "delta_roc_auc_ci_lower": discrimination_deltas["roc_auc"]["lower"],
                "delta_roc_auc_ci_upper": discrimination_deltas["roc_auc"]["upper"],
                "delta_roc_auc_delong": delong["delta_auc"],
                "delta_roc_auc_delong_variance": delong["var_delta"],
                "delta_roc_auc_delong_z": delong["z_score"],
                "delta_roc_auc_delong_pvalue": delong["p_value"],
                "primary_average_precision": primary_average_precision,
                "comparison_average_precision": comparison_average_precision,
                "primary_average_precision_lift": average_precision_lift(y, primary_scores),
                "comparison_average_precision_lift": average_precision_lift(y, other_scores),
                "delta_average_precision": discrimination_deltas["average_precision"]["delta"],
                "delta_average_precision_ci_lower": discrimination_deltas["average_precision"][
                    "lower"
                ],
                "delta_average_precision_ci_upper": discrimination_deltas["average_precision"][
                    "upper"
                ],
                "delta_brier_improvement": brier_delta["delta"],
                "delta_brier_ci_lower": brier_delta["lower"],
                "delta_brier_ci_upper": brier_delta["upper"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False).reset_index(drop=True)


def _clip_probability_array(values: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    return cast(np.ndarray, np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps))


def _calibration_feature_matrix(preds: np.ndarray, *, method: str) -> np.ndarray:
    clipped = _clip_probability_array(preds)
    if method == "platt":
        logit = np.log(clipped / (1.0 - clipped))
        return logit.reshape(-1, 1)
    if method == "beta":
        return np.column_stack([np.log(clipped), np.log1p(-clipped)])
    raise ValueError(f"unsupported calibration method: {method}")


def _fit_calibration_transform(
    y: np.ndarray,
    preds: np.ndarray,
    *,
    method: str,
) -> LogisticRegression | IsotonicRegression | None:
    y = np.asarray(y, dtype=int)
    preds = _clip_probability_array(preds)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    if method == "platt":
        calibrator = LogisticRegression(
            C=1_000_000.0,
            max_iter=1_000,
            solver="lbfgs",
        )
        calibrator.fit(_calibration_feature_matrix(preds, method=method), y)
        return cast(LogisticRegression, calibrator)
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(preds, y)
        return cast(IsotonicRegression, calibrator)
    if method == "beta":
        calibrator = LogisticRegression(
            C=1_000_000.0,
            max_iter=1_000,
            solver="lbfgs",
        )
        calibrator.fit(_calibration_feature_matrix(preds, method=method), y)
        return cast(LogisticRegression, calibrator)
    raise ValueError(f"unsupported calibration method: {method}")


def _apply_calibration_transform(
    preds: np.ndarray,
    *,
    method: str,
    calibrator: LogisticRegression | IsotonicRegression | None,
) -> np.ndarray:
    clipped = _clip_probability_array(preds)
    if method == "raw" or calibrator is None:
        return clipped
    if method == "platt" or method == "beta":
        proba = cast(
            LogisticRegression,
            calibrator,
        ).predict_proba(_calibration_feature_matrix(clipped, method=method))[:, 1]
        return _clip_probability_array(cast(np.ndarray, proba))
    if method == "isotonic":
        proba = cast(IsotonicRegression, calibrator).predict(clipped)
        return _clip_probability_array(np.asarray(proba, dtype=float))
    raise ValueError(f"unsupported calibration method: {method}")


def _calibration_metrics_from_arrays(
    y: np.ndarray,
    preds: np.ndarray,
    *,
    method: str,
) -> dict[str, object]:
    y = np.asarray(y, dtype=int)
    preds = _clip_probability_array(preds)

    # Compute calibration slope and intercept via linear regression
    calibration_slope = float("nan")
    calibration_intercept = float("nan")
    if len(y) > 1 and np.unique(y).size > 1:
        try:
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression()
            reg.fit(preds.reshape(-1, 1), y)
            calibration_slope = float(reg.coef_[0])
            calibration_intercept = float(reg.intercept_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            import warnings

            warnings.warn(f"Calibration slope/intercept calculation failed: {e}")
            calibration_slope = float("nan")
            calibration_intercept = float("nan")

    return {
        "calibration_method": method,
        "calibration_metric_family": "fixed_bin_probability_calibration",
        "calibration_metric_semantics": "fixed_bin_diagnostics",
        "n_backbones": int(len(y)),
        "mean_prediction": float(preds.mean()) if len(y) else float("nan"),
        "observed_rate": float(y.mean()) if len(y) else float("nan"),
        "brier_score": brier_score(y, preds) if len(y) else float("nan"),
        "ece": expected_calibration_error(y, preds) if len(y) else float("nan"),
        "expected_calibration_error": expected_calibration_error(y, preds)
        if len(y)
        else float("nan"),
        "max_calibration_error": max_calibration_error(y, preds) if len(y) else float("nan"),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def _nested_calibrated_predictions(
    y: np.ndarray,
    preds: np.ndarray,
    *,
    method: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    preds = _clip_probability_array(preds)
    if method == "raw" or len(y) == 0 or np.unique(y).size < 2:
        return preds
    calibrated = np.zeros(len(preds), dtype=float)
    counts = np.zeros(len(preds), dtype=float)
    for fold_indices in _stratified_folds(
        np.asarray(y, dtype=int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    ):
        for test_idx in fold_indices:
            train_mask = np.ones(len(preds), dtype=bool)
            train_mask[test_idx] = False
            calibrator = _fit_calibration_transform(y[train_mask], preds[train_mask], method=method)
            fold_preds = _apply_calibration_transform(
                preds[test_idx],
                method=method,
                calibrator=calibrator,
            )
            calibrated[test_idx] += fold_preds
            counts[test_idx] += 1.0
    counts[counts == 0] = 1.0
    return _clip_probability_array(calibrated / counts)


def build_calibration_metric_table(
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    calibration_methods: tuple[str, ...] = _CALIBRATION_METHODS,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute nested calibration-quality summaries for selected models."""
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = predictions.loc[predictions["model_name"] == model_name].copy()
        if frame.empty:
            continue
        y = frame["spread_label"].to_numpy(dtype=int)
        raw_preds = frame["oof_prediction"].to_numpy(dtype=float)
        raw_metrics: dict[str, object] = _calibration_metrics_from_arrays(
            y, raw_preds, method="raw"
        )
        raw_metrics.update(
            {
                "model_name": model_name,
                "evaluation_split": "oof",
                "calibration_strategy": "identity",
                "calibration_gain_vs_raw_brier": 0.0,
                "calibration_gain_vs_raw_ece": 0.0,
            }
        )
        rows.append(raw_metrics)
        for method in calibration_methods:
            if method == "raw":
                continue
            calibrated = _nested_calibrated_predictions(
                y,
                raw_preds,
                method=method,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
            )
            calibrated_metrics: dict[str, object] = _calibration_metrics_from_arrays(
                y, calibrated, method=method
            )
            raw_brier_score = cast(float, raw_metrics["brier_score"])
            raw_ece = cast(float, raw_metrics["ece"])
            calibrated_brier_score = cast(float, calibrated_metrics["brier_score"])
            calibrated_ece = cast(float, calibrated_metrics["ece"])
            calibrated_metrics.update(
                {
                    "model_name": model_name,
                    "evaluation_split": "oof",
                    "calibration_strategy": "nested_oof",
                    "calibration_gain_vs_raw_brier": raw_brier_score - calibrated_brier_score,
                    "calibration_gain_vs_raw_ece": raw_ece - calibrated_ece,
                }
            )
            rows.append(calibrated_metrics)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    method_order = {method: index for index, method in enumerate(_CALIBRATION_METHODS)}
    result["_calibration_method_order"] = (
        result["calibration_method"].map(method_order).fillna(len(method_order))
    )
    return (
        result.sort_values(
            ["model_name", "evaluation_split", "_calibration_method_order"],
            kind="mergesort",
        )
        .drop(columns=["_calibration_method_order"])
        .reset_index(drop=True)
    )


def build_blocked_holdout_calibration_table(
    scored: pd.DataFrame,
    *,
    model_names: list[str],
    group_columns: list[str],
    calibration_methods: tuple[str, ...] = _CALIBRATION_METHODS,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Evaluate nested calibration transfer on strict blocked holdout splits."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["spread_label"] = eligible["spread_label"].astype(int)

    rows: list[dict[str, object]] = []
    tasks: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame, str]] = []
    for group_column in group_columns:
        if group_column not in eligible.columns:
            continue
        working = eligible.copy()
        working[group_column] = working[group_column].fillna("unknown").astype(str)
        counts = working[group_column].value_counts()
        if group_column == "dominant_source":
            selected_groups = counts.index.tolist()
        else:
            selected_groups = counts.loc[counts >= 25].head(8).index.tolist()
        for group_value in selected_groups:
            test = working.loc[working[group_column] == group_value].copy()
            train = working.loc[working[group_column] != group_value].copy()
            base_row = {
                "group_column": group_column,
                "group_value": str(group_value),
                "n_test_backbones": int(len(test)),
                "n_train_backbones": int(len(train)),
            }
            for model_name in model_names:
                if (
                    len(test) < 25
                    or test["spread_label"].nunique() < 2
                    or train["spread_label"].nunique() < 2
                ):
                    rows.append(
                        {
                            **base_row,
                            "model_name": model_name,
                            "evaluation_split": "blocked_holdout",
                            "calibration_method": "raw",
                            "calibration_strategy": "skipped_insufficient_label_variation",
                            "n_backbones": int(len(test)),
                            "mean_prediction": np.nan,
                            "observed_rate": np.nan,
                            "brier_score": np.nan,
                            "ece": np.nan,
                            "expected_calibration_error": np.nan,
                            "max_calibration_error": np.nan,
                            "calibration_gain_vs_raw_brier": np.nan,
                            "calibration_gain_vs_raw_ece": np.nan,
                            "status": "skipped_insufficient_label_variation",
                        }
                    )
                    continue
                tasks.append((base_row, train, test, model_name))

    def _evaluate_holdout_task(
        task: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, str],
    ) -> list[dict[str, object]]:
        base_row, train, test, model_name = task
        train_result = evaluate_model_name(
            train,
            model_name=model_name,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            include_ci=False,
        )
        train_frame = train_result.predictions.copy()
        if "oof_prediction" not in train_frame.columns or train_frame.empty:
            return [
                {
                    **base_row,
                    "model_name": model_name,
                    "evaluation_split": "blocked_holdout",
                    "calibration_method": method,
                    "calibration_strategy": "train_fit_holdout_apply",
                    "n_backbones": int(len(test)),
                    "mean_prediction": np.nan,
                    "observed_rate": np.nan,
                    "brier_score": np.nan,
                    "ece": np.nan,
                    "expected_calibration_error": np.nan,
                    "max_calibration_error": np.nan,
                    "calibration_gain_vs_raw_brier": np.nan,
                    "calibration_gain_vs_raw_ece": np.nan,
                    "status": "skipped_fit_failure",
                }
                for method in calibration_methods
            ]

        y_train = train_frame["spread_label"].to_numpy(dtype=int)
        source_preds = train_frame["oof_prediction"].to_numpy(dtype=float)
        prediction_table = fit_predict_model_holdout(train, test, model_name=model_name)
        if prediction_table.empty or prediction_table["spread_label"].nunique() < 2:
            return [
                {
                    **base_row,
                    "model_name": model_name,
                    "evaluation_split": "blocked_holdout",
                    "calibration_method": method,
                    "calibration_strategy": "train_fit_holdout_apply",
                    "n_backbones": int(len(test)),
                    "mean_prediction": np.nan,
                    "observed_rate": np.nan,
                    "brier_score": np.nan,
                    "ece": np.nan,
                    "expected_calibration_error": np.nan,
                    "max_calibration_error": np.nan,
                    "calibration_gain_vs_raw_brier": np.nan,
                    "calibration_gain_vs_raw_ece": np.nan,
                    "status": "skipped_fit_failure",
                }
                for method in calibration_methods
            ]
        y_test = prediction_table["spread_label"].to_numpy(dtype=int)
        target_preds = prediction_table["prediction"].to_numpy(dtype=float)
        raw_metrics = _calibration_metrics_from_arrays(y_test, target_preds, method="raw")
        raw_brier_score = cast(float, raw_metrics["brier_score"])
        raw_ece = cast(float, raw_metrics["ece"])
        holdout_rows: list[dict[str, object]] = []
        for method in calibration_methods:
            if method == "raw":
                metrics = dict(raw_metrics)
                metrics.update(
                    {
                        **base_row,
                        "model_name": model_name,
                        "evaluation_split": "blocked_holdout",
                        "calibration_strategy": "identity",
                        "calibration_gain_vs_raw_brier": 0.0,
                        "calibration_gain_vs_raw_ece": 0.0,
                        "status": "ok",
                    }
                )
                holdout_rows.append(metrics)
                continue
            calibrator = _fit_calibration_transform(y_train, source_preds, method=method)
            calibrated = _apply_calibration_transform(
                target_preds,
                method=method,
                calibrator=calibrator,
            )
            metrics = _calibration_metrics_from_arrays(y_test, calibrated, method=method)
            calibrated_brier_score = cast(float, metrics["brier_score"])
            calibrated_ece = cast(float, metrics["ece"])
            metrics.update(
                {
                    **base_row,
                    "model_name": model_name,
                    "evaluation_split": "blocked_holdout",
                    "calibration_strategy": "train_fit_holdout_apply",
                    "calibration_gain_vs_raw_brier": raw_brier_score - calibrated_brier_score,
                    "calibration_gain_vs_raw_ece": raw_ece - calibrated_ece,
                    "status": "ok",
                }
            )
            holdout_rows.append(metrics)
        return holdout_rows

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(tasks))
    if jobs > 1 and tasks:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                for result in executor.map(_evaluate_holdout_task, tasks):
                    rows.extend(result)
    else:
        for task in tasks:
            rows.extend(_evaluate_holdout_task(task))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["model_name", "group_column", "group_value", "calibration_method"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_blocked_holdout_calibration_summary(
    blocked_holdout_calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate blocked-holdout calibration metrics to a model-level summary."""
    if blocked_holdout_calibration.empty:
        return pd.DataFrame()
    working = blocked_holdout_calibration.loc[
        blocked_holdout_calibration.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
    ].copy()
    if working.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (model_name, calibration_method), frame in working.groupby(
        ["model_name", "calibration_method"], sort=False
    ):
        weights = pd.to_numeric(
            frame.get("n_test_backbones", pd.Series(np.nan, index=frame.index)), errors="coerce"
        ).fillna(0.0)
        valid = frame["brier_score"].notna() & frame["ece"].notna()
        if not valid.any():
            continue
        valid_weights = np.clip(weights.loc[valid].to_numpy(dtype=float), 1.0, None)
        if valid_weights.size and np.isfinite(valid_weights).any():
            mean_prediction = float(
                np.average(
                    pd.to_numeric(frame.loc[valid, "mean_prediction"], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    weights=valid_weights,
                )
            )
            observed_rate = float(
                np.average(
                    pd.to_numeric(frame.loc[valid, "observed_rate"], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    weights=valid_weights,
                )
            )
            brier_score_value = float(
                np.average(
                    pd.to_numeric(frame.loc[valid, "brier_score"], errors="coerce").to_numpy(
                        dtype=float
                    ),
                    weights=valid_weights,
                )
            )
            ece_value = float(
                np.average(
                    pd.to_numeric(frame.loc[valid, "ece"], errors="coerce").to_numpy(dtype=float),
                    weights=valid_weights,
                )
            )
            max_calibration_error_value = float(
                np.average(
                    pd.to_numeric(
                        frame.loc[valid, "max_calibration_error"], errors="coerce"
                    ).to_numpy(dtype=float),
                    weights=valid_weights,
                )
            )
        else:
            mean_prediction = float(pd.to_numeric(frame["mean_prediction"], errors="coerce").mean())
            observed_rate = float(pd.to_numeric(frame["observed_rate"], errors="coerce").mean())
            brier_score_value = float(pd.to_numeric(frame["brier_score"], errors="coerce").mean())
            ece_value = float(pd.to_numeric(frame["ece"], errors="coerce").mean())
            max_calibration_error_value = float(
                pd.to_numeric(frame["max_calibration_error"], errors="coerce").mean()
            )
        rows.append(
            {
                "model_name": str(model_name),
                "evaluation_split": "blocked_holdout",
                "calibration_method": str(calibration_method),
                "calibration_strategy": "train_fit_holdout_apply"
                if calibration_method != "raw"
                else "identity",
                "n_backbones": int(weights.sum()),
                "n_groups": int(frame[["group_column", "group_value"]].drop_duplicates().shape[0]),
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "brier_score": brier_score_value,
                "ece": ece_value,
                "expected_calibration_error": ece_value,
                "max_calibration_error": max_calibration_error_value,
                "calibration_gain_vs_raw_brier": float(
                    pd.to_numeric(frame["calibration_gain_vs_raw_brier"], errors="coerce").mean()
                ),
                "calibration_gain_vs_raw_ece": float(
                    pd.to_numeric(frame["calibration_gain_vs_raw_ece"], errors="coerce").mean()
                ),
                "status": "ok",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    method_order = {method: index for index, method in enumerate(_CALIBRATION_METHODS)}
    result["_calibration_method_order"] = (
        result["calibration_method"].map(method_order).fillna(len(method_order))
    )
    return (
        result.sort_values(["model_name", "_calibration_method_order"], kind="mergesort")
        .drop(columns=["_calibration_method_order"])
        .reset_index(drop=True)
    )


def build_source_balance_resampling_table(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_resamples: int = 20,
    seed: int = 42,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Repeatedly resample a source-balanced cohort and measure model stability."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    eligible["dominant_source"] = np.where(
        eligible["refseq_share_train"].fillna(0.0) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    counts = eligible["dominant_source"].value_counts()
    if len(counts) < 2:
        return pd.DataFrame()
    n_per_group = int(counts.min())
    rng = np.random.default_rng(seed)
    grouped_frames = [frame.copy() for _, frame in eligible.groupby("dominant_source", sort=False)]
    sample_plan = [
        (resample_index, int(rng.integers(0, 1_000_000_000)))
        for resample_index in range(1, n_resamples + 1)
    ]

    def _evaluate_resample(task: tuple[int, int]) -> dict[str, object]:
        resample_index, sample_seed = task
        sampled_frames = [
            frame.sample(n=n_per_group, random_state=sample_seed) for frame in grouped_frames
        ]
        sampled = pd.concat(sampled_frames, ignore_index=True).drop(columns=["dominant_source"])
        result = evaluate_model_name(
            sampled, model_name=model_name, n_repeats=2, seed=sample_seed, include_ci=False
        )
        return {
            "model_name": model_name,
            "resample_index": resample_index,
            "sample_seed": sample_seed,
            "n_backbones": int(len(sampled)),
            "n_per_source_group": n_per_group,
            "roc_auc": result.metrics["roc_auc"],
            "average_precision": result.metrics["average_precision"],
            "brier_score": result.metrics["brier_score"],
        }

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(sample_plan))
    if jobs > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                rows = list(executor.map(_evaluate_resample, sample_plan))
    else:
        rows = [_evaluate_resample(task) for task in sample_plan]
    return pd.DataFrame(rows)


def build_negative_control_audit(
    scored: pd.DataFrame,
    *,
    primary_model_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Check that deterministic nuisance features do not create meaningful discrimination.

    All feature-set specs are evaluated using the same L2 regularization and
    sample-weight configuration as the primary model so that delta comparisons
    are methodologically valid.
    """
    if primary_model_name not in MODULE_A_FEATURE_SETS:
        return pd.DataFrame()

    working = scored.copy()
    backbone_ids = working["backbone_id"].fillna("").astype(str)
    working["negative_control_noise_a"] = backbone_ids.map(
        lambda value: _stable_unit_interval(value, salt="noise_a")
    )
    working["negative_control_noise_b"] = backbone_ids.map(
        lambda value: _stable_unit_interval(value, salt="noise_b")
    )
    working["negative_control_length"] = backbone_ids.str.len().astype(float)

    # Use the same fit parameters as the primary model for all specs so that
    # the "primary_model" baseline row matches the official evaluation and the
    # delta columns are computed against a consistent reference.
    fit_kwargs = _model_fit_kwargs(primary_model_name)

    specs = [
        ("primary_model", MODULE_A_FEATURE_SETS[primary_model_name]),
        ("negative_control_noise_a_only", ["negative_control_noise_a"]),
        (
            "negative_control_noise_ab_only",
            ["negative_control_noise_a", "negative_control_noise_b"],
        ),
        ("negative_control_length_only", ["negative_control_length"]),
        (
            "primary_plus_negative_control_a",
            MODULE_A_FEATURE_SETS[primary_model_name] + ["negative_control_noise_a"],
        ),
        (
            "primary_plus_negative_control_ab",
            MODULE_A_FEATURE_SETS[primary_model_name]
            + ["negative_control_noise_a", "negative_control_noise_b"],
        ),
    ]

    rows: list[dict[str, object]] = []
    primary_metrics: dict[str, float] | None = None
    for audit_name, columns in specs:
        eligible, X, y = _eligible_xy(working, columns)
        sample_weight = _compute_sample_weight(eligible, mode=_fit_kwarg_mode(fit_kwargs))
        preds = _oof_predictions(
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            sample_weight=sample_weight,
            l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
        )
        metrics = {
            "roc_auc": roc_auc_score(y, preds),
            "average_precision": average_precision(y, preds),
            "brier_score": brier_score(y, preds),
        }
        if audit_name == "primary_model":
            primary_metrics = dict(metrics)
        rows.append(
            {
                "audit_name": audit_name,
                "feature_columns": ",".join(columns),
                "n_features": int(len(columns)),
                "roc_auc": float(metrics["roc_auc"]),
                "average_precision": float(metrics["average_precision"]),
                "brier_score": float(metrics["brier_score"]),
            }
        )

    audit = pd.DataFrame(rows)
    if primary_metrics is not None:
        audit["delta_roc_auc_vs_primary"] = audit["roc_auc"] - float(primary_metrics["roc_auc"])
        audit["delta_average_precision_vs_primary"] = audit["average_precision"] - float(
            primary_metrics["average_precision"]
        )
        audit["delta_brier_vs_primary"] = audit["brier_score"] - float(
            primary_metrics["brier_score"]
        )
    return audit


def build_future_sentinel_audit(
    scored: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    primary_model_name: str | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
    sentinel_feature_name: str = "future_label_sentinel",
) -> pd.DataFrame:
    """Document that an obviously leaky future-only sentinel is excluded from discovery models."""
    eligible = scored.loc[scored.get("spread_label", pd.Series(dtype=float)).notna()].copy()
    if eligible.empty or eligible["spread_label"].astype(int).nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "audit_name": sentinel_feature_name,
                    "audit_status": "skipped_insufficient_label_variation",
                    "sentinel_feature_name": sentinel_feature_name,
                    "sentinel_source": "future_outcome_label",
                }
            ]
        )
    eligible["backbone_id"] = eligible["backbone_id"].astype(str)
    y = eligible["spread_label"].astype(int).to_numpy()
    sentinel_scores = eligible["spread_label"].astype(float).to_numpy()
    discovery_models = [
        str(model_name)
        for model_name in (model_names or list(MODULE_A_FEATURE_SETS))
        if str(get_model_track(str(model_name))) == "discovery"
    ]
    official_discovery_models_use_sentinel = any(
        sentinel_feature_name in MODULE_A_FEATURE_SETS.get(model_name, [])
        for model_name in discovery_models
    )
    try:
        sentinel_track = str(get_feature_track(sentinel_feature_name))
        discovery_contract_forbidden = sentinel_track != "discovery"
    except (KeyError, TypeError, ValueError):
        sentinel_track = "unregistered"
        discovery_contract_forbidden = True

    primary_roc_auc = np.nan
    primary_average_precision = np.nan
    if (
        predictions is not None
        and not predictions.empty
        and primary_model_name is not None
        and str(primary_model_name).strip()
    ):
        primary_frame = predictions.loc[
            predictions.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(str(primary_model_name))
        ].copy()
        if not primary_frame.empty:
            primary_frame["backbone_id"] = primary_frame["backbone_id"].astype(str)
            primary_frame = primary_frame.merge(
                eligible[["backbone_id", "spread_label"]],
                on="backbone_id",
                how="inner",
                validate="1:1",
                suffixes=("", "_eligible"),
            )
            if (
                not primary_frame.empty
                and primary_frame["spread_label_eligible"].astype(int).nunique() >= 2
            ):
                y_primary = primary_frame["spread_label_eligible"].astype(int).to_numpy()
                preds_primary = primary_frame["oof_prediction"].astype(float).to_numpy()
                primary_roc_auc = roc_auc_score(y_primary, preds_primary)
                primary_average_precision = average_precision(y_primary, preds_primary)

    sentinel_roc_auc = roc_auc_score(y, sentinel_scores)
    sentinel_average_precision = average_precision(y, sentinel_scores)
    sentinel_brier = brier_score(y, sentinel_scores)
    audit_status = (
        "pass"
        if discovery_contract_forbidden and not official_discovery_models_use_sentinel
        else "fail"
    )
    rationale = (
        "Synthetic future-only sentinel is structurally excluded from official discovery models."
        if audit_status == "pass"
        else "Official discovery surface would admit an explicitly future-derived sentinel."
    )
    return pd.DataFrame(
        [
            {
                "audit_name": sentinel_feature_name,
                "audit_status": audit_status,
                "audit_rationale": rationale,
                "sentinel_feature_name": sentinel_feature_name,
                "sentinel_source": "future_outcome_label",
                "sentinel_track": sentinel_track,
                "discovery_contract_forbidden": bool(discovery_contract_forbidden),
                "official_discovery_models_checked": ",".join(discovery_models),
                "n_discovery_models_checked": int(len(discovery_models)),
                "official_discovery_models_use_sentinel": bool(
                    official_discovery_models_use_sentinel
                ),
                "n_backbones": int(len(eligible)),
                "n_positive": int((y == 1).sum()),
                "sentinel_only_roc_auc": float(sentinel_roc_auc),
                "sentinel_only_average_precision": float(sentinel_average_precision),
                "sentinel_only_brier_score": float(sentinel_brier),
                "primary_model_name": str(primary_model_name or ""),
                "primary_roc_auc": float(primary_roc_auc) if pd.notna(primary_roc_auc) else np.nan,
                "primary_average_precision": float(primary_average_precision)
                if pd.notna(primary_average_precision)
                else np.nan,
                "delta_roc_auc_vs_primary": float(sentinel_roc_auc - primary_roc_auc)
                if pd.notna(primary_roc_auc)
                else np.nan,
                "delta_average_precision_vs_primary": float(
                    sentinel_average_precision - primary_average_precision
                )
                if pd.notna(primary_average_precision)
                else np.nan,
            }
        ]
    )


def build_permutation_null_tables(
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    n_permutations: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate empirical null distributions by permuting labels against fixed model scores."""
    rng = np.random.default_rng(seed)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = predictions.loc[predictions["model_name"] == model_name].copy()
        if frame.empty:
            continue
        y = frame["spread_label"].to_numpy(dtype=int)
        preds = frame["oof_prediction"].to_numpy(dtype=float)
        observed_auc = roc_auc_score(y, preds)
        observed_ap = average_precision(y, preds)
        n = len(y)
        positives = int((y == 1).sum())
        negatives = int(n - positives)
        if positives == 0 or negatives == 0:
            continue
        order_desc = np.argsort(-preds, kind="mergesort")
        order_asc = np.argsort(preds, kind="mergesort")
        sorted_scores = preds[order_asc]
        _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
        average_ranks = first_idx + (counts + 1.0) / 2.0
        score_ranks = np.empty(n, dtype=float)
        score_ranks[order_asc] = np.repeat(average_ranks, counts)
        denominator = np.arange(1, n + 1, dtype=float)
        null_aucs: list[float] = []
        null_aps: list[float] = []
        permutation_index = 1
        batch_size = min(256, n_permutations)
        while permutation_index <= n_permutations:
            current_batch = min(batch_size, n_permutations - permutation_index + 1)
            permuted_batch = np.vstack([rng.permutation(y) for _ in range(current_batch)]).astype(
                int, copy=False
            )
            rank_sums = permuted_batch @ score_ranks
            auc_values = (rank_sums - positives * (positives + 1) / 2.0) / (positives * negatives)
            permuted_sorted = permuted_batch[:, order_desc]
            tp = np.cumsum(permuted_sorted == 1, axis=1, dtype=float)
            precision = tp / denominator
            ap_values = (precision * (permuted_sorted == 1)).sum(axis=1) / max(positives, 1)
            for null_auc, null_ap in zip(auc_values.tolist(), ap_values.tolist(), strict=False):
                null_aucs.append(float(null_auc))
                null_aps.append(float(null_ap))
                detail_rows.append(
                    {
                        "model_name": model_name,
                        "permutation_index": permutation_index,
                        "null_roc_auc": float(null_auc),
                        "null_average_precision": float(null_ap),
                    }
                )
                permutation_index += 1
        summary_rows.append(
            {
                "model_name": model_name,
                "n_backbones": int(len(frame)),
                "n_permutations": int(n_permutations),
                "positive_prevalence": positive_prevalence(y),
                "observed_roc_auc": observed_auc,
                "null_roc_auc_mean": float(np.mean(null_aucs)),
                "null_roc_auc_std": float(np.std(null_aucs)),
                "null_roc_auc_q975": float(np.quantile(null_aucs, 0.975)),
                "empirical_p_roc_auc": float(
                    (1 + sum(value >= observed_auc for value in null_aucs)) / (n_permutations + 1)
                ),
                "observed_average_precision": observed_ap,
                "observed_average_precision_lift": average_precision_lift(y, preds),
                "observed_average_precision_enrichment": average_precision_enrichment(y, preds),
                "null_average_precision_mean": float(np.mean(null_aps)),
                "null_average_precision_std": float(np.std(null_aps)),
                "null_average_precision_q975": float(np.quantile(null_aps, 0.975)),
                "empirical_p_average_precision": float(
                    (1 + sum(value >= observed_ap for value in null_aps)) / (n_permutations + 1)
                ),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_selection_adjusted_permutation_null(
    scored: pd.DataFrame,
    *,
    model_names: list[str],
    primary_model_name: str,
    n_permutations: int = 200,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a model-selection-adjusted null by refitting the official model surface.

    For each label permutation, every candidate model in `model_names` is refit with the
    permuted labels under the same OOF protocol. The null distribution is then defined by the
    best ROC AUC achieved within that permutation, which accounts for post-hoc model choice
    across the official surface.
    """
    selection_scope = [name for name in dict.fromkeys(model_names) if name in MODULE_A_FEATURE_SETS]
    if primary_model_name not in selection_scope:
        selection_scope = [*selection_scope, primary_model_name]
        selection_scope = [name for name in selection_scope if name in MODULE_A_FEATURE_SETS]
    if not selection_scope or primary_model_name not in set(selection_scope):
        return pd.DataFrame(), pd.DataFrame()

    eligible_mask = scored.get("spread_label", pd.Series(index=scored.index)).notna()
    if not bool(eligible_mask.any()):
        return pd.DataFrame(), pd.DataFrame()
    y_true = scored.loc[eligible_mask, "spread_label"].astype(int).to_numpy(dtype=int)
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Prepare the per-model design matrix once; permutations only change labels.
    prepared_inputs: dict[str, tuple[pd.DataFrame, list[str], dict[str, object]]] = {}
    for model_name in selection_scope:
        columns = MODULE_A_FEATURE_SETS[model_name]
        eligible = _ensure_feature_columns(scored, columns).loc[eligible_mask].copy()
        eligible["spread_label"] = eligible["spread_label"].astype(int)
        prepared_inputs[model_name] = (eligible, columns, _model_fit_kwargs(model_name))

    observed_metrics: dict[str, tuple[float, float]] = {}
    for model_name in selection_scope:
        eligible, columns, fit_kwargs = prepared_inputs[model_name]
        preds, y = _oof_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
        )
        observed_metrics[model_name] = (
            float(roc_auc_score(y, preds)),
            float(average_precision(y, preds)),
        )

    rng = np.random.default_rng(seed)
    detail_rows: list[dict[str, object]] = []
    selected_null_aucs: list[float] = []
    selected_null_aps: list[float] = []
    selected_model_names: list[str] = []

    for permutation_index in range(1, max(int(n_permutations), 0) + 1):
        permuted_y = rng.permutation(y_true)
        fold_groups = _stratified_folds(
            permuted_y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        permutation_rows: list[dict[str, object]] = []
        for model_name in selection_scope:
            eligible, columns, fit_kwargs = prepared_inputs[model_name]
            preds, _ = _oof_predictions_from_eligible(
                eligible,
                columns=columns,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                fit_kwargs=fit_kwargs,
                y_override=permuted_y,
                folds_per_repeat=fold_groups,
            )
            permutation_rows.append(
                {
                    "model_name": model_name,
                    "null_roc_auc": float(roc_auc_score(permuted_y, preds)),
                    "null_average_precision": float(average_precision(permuted_y, preds)),
                }
            )

        permutation_frame = pd.DataFrame(permutation_rows).sort_values(
            ["null_roc_auc", "null_average_precision", "model_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected_row = permutation_frame.iloc[0]
        selected_null_auc = float(selected_row["null_roc_auc"])
        selected_null_ap = float(selected_row["null_average_precision"])
        selected_model_name = str(selected_row["model_name"])
        selected_null_aucs.append(selected_null_auc)
        selected_null_aps.append(selected_null_ap)
        selected_model_names.append(selected_model_name)
        detail_rows.append(
            {
                "permutation_index": permutation_index,
                "selection_scope": "official_model_surface",
                "n_models_in_scope": int(len(selection_scope)),
                "selected_model_name": selected_model_name,
                "selected_null_roc_auc": selected_null_auc,
                "selected_null_average_precision": selected_null_ap,
            }
        )

    if not selected_null_aucs:
        return pd.DataFrame(detail_rows), pd.DataFrame()

    winner_counts = pd.Series(selected_model_names, dtype=object).value_counts()
    modal_selected_model = str(winner_counts.index[0]) if not winner_counts.empty else ""
    modal_selected_share = (
        float(winner_counts.iloc[0] / max(len(selected_model_names), 1))
        if not winner_counts.empty
        else 0.0
    )

    summary_rows: list[dict[str, object]] = []
    for model_name in selection_scope:
        observed_auc, observed_ap = observed_metrics[model_name]
        summary_rows.append(
            {
                "model_name": model_name,
                "null_protocol": "selection_adjusted_official_model_refit",
                "selection_scope": "official_model_surface",
                "selection_reference_model": primary_model_name,
                "n_models_in_scope": int(len(selection_scope)),
                "n_permutations": int(n_permutations),
                "observed_roc_auc": observed_auc,
                "observed_average_precision": observed_ap,
                "null_roc_auc_mean": float(np.mean(selected_null_aucs)),
                "null_roc_auc_std": float(np.std(selected_null_aucs)),
                "null_roc_auc_q975": float(np.quantile(selected_null_aucs, 0.975)),
                "selection_adjusted_empirical_p_roc_auc": float(
                    (1 + sum(value >= observed_auc for value in selected_null_aucs))
                    / (len(selected_null_aucs) + 1)
                ),
                "null_average_precision_mean": float(np.mean(selected_null_aps)),
                "null_average_precision_std": float(np.std(selected_null_aps)),
                "null_average_precision_q975": float(np.quantile(selected_null_aps, 0.975)),
                "selection_adjusted_empirical_p_average_precision": float(
                    (1 + sum(value >= observed_ap for value in selected_null_aps))
                    / (len(selected_null_aps) + 1)
                ),
                "modal_selected_model_name": modal_selected_model,
                "modal_selected_model_share": modal_selected_share,
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_consensus_candidate_ranking(
    candidate_context: pd.DataFrame,
    *,
    primary_score_column: str,
    conservative_score_column: str,
    top_k: int = 50,
) -> pd.DataFrame:
    """Rank candidates by agreement across evidence-aware and conservative views."""
    if candidate_context.empty:
        return pd.DataFrame()

    working = candidate_context.copy()
    working = working.loc[working["member_count_train"].fillna(0).astype(int) > 0].copy()
    if "spread_label" in working.columns:
        working = working.loc[working["spread_label"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    for column in (
        primary_score_column,
        conservative_score_column,
        "bio_priority_index",
        "operational_priority_index",
        "evidence_support_index",
        "coherence_score",
    ):
        if column not in working.columns:
            working[column] = np.nan

    working = working.loc[
        working[primary_score_column].notna() & working[conservative_score_column].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame()
    working["primary_candidate_score"] = working[primary_score_column].astype(float)
    working["conservative_candidate_score"] = working[conservative_score_column].astype(float)
    working["primary_rank"] = working["primary_candidate_score"].rank(
        method="average", ascending=False
    )
    working["conservative_rank"] = working["conservative_candidate_score"].rank(
        method="average", ascending=False
    )
    working["bio_rank"] = (
        working["bio_priority_index"].fillna(0.0).rank(method="average", ascending=False)
    )
    working["consensus_rank_mean"] = (working["primary_rank"] + working["conservative_rank"]) / 2.0
    working["rank_disagreement_primary_vs_conservative"] = (
        working["primary_rank"] - working["conservative_rank"]
    ).abs()
    working["top25_primary"] = working["primary_rank"] <= 25
    working["top25_conservative"] = working["conservative_rank"] <= 25
    working["top25_bio"] = working["bio_rank"] <= 25
    working["consensus_support_count"] = (
        working["top25_primary"].astype(int)
        + working["top25_conservative"].astype(int)
        + working["top25_bio"].astype(int)
    )
    working["consensus_candidate_score"] = 1.0 - (
        working["consensus_rank_mean"] / max(float(len(working)), 1.0)
    )
    working = working.sort_values(
        [
            "consensus_support_count",
            "consensus_rank_mean",
            "rank_disagreement_primary_vs_conservative",
            "coherence_score",
            "bio_priority_index",
            primary_score_column,
        ],
        ascending=[False, True, True, False, False, False],
    ).reset_index(drop=True)
    working["consensus_rank"] = np.arange(1, len(working) + 1)
    working["consensus_top25_all_core_axes"] = working["consensus_support_count"] == 3
    return working.head(top_k).copy()


def build_priority_bootstrap_stability_table(
    scored: pd.DataFrame,
    *,
    candidate_n: int = 50,
    top_k: int = 25,
    n_bootstrap: int = 1000,
    seed: int = 42,
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
    score_column: str = "priority_index",
    model_name: str | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Measure how often the highest-ranked candidates stay near the top under reference resampling."""
    n_bootstrap = max(int(n_bootstrap), 1)
    training = scored.loc[scored["member_count_train"].fillna(0).astype(int) > 0].copy()
    if model_name:
        training = training.loc[training["spread_label"].notna()].copy()
    if training.empty:
        return pd.DataFrame()

    score_key = "stability_score"
    if model_name:
        base_scores = fit_full_model_predictions(training, model_name=model_name).rename(
            columns={"prediction": score_key}
        )
        base = training.merge(base_scores, on="backbone_id", how="left", validate="1:1")
    else:
        base = training.copy()
        base[score_key] = base[score_column].fillna(0.0)
    base = base.sort_values(score_key, ascending=False).reset_index(drop=True)
    candidates = base.head(candidate_n)[["backbone_id", score_key]].copy()
    candidates["base_rank"] = np.arange(1, len(candidates) + 1)
    candidate_ids = candidates["backbone_id"].astype(str).tolist()
    stats: dict[str, _BootstrapCandidateStats] = {
        backbone_id: {
            "top_hits": 0,
            "top_hits_top10": 0,
            "top_hits_top25": 0,
            "ranks": [],
        }
        for backbone_id in candidate_ids
    }

    rng = np.random.default_rng(seed)
    sample_seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(n_bootstrap)]

    def _rank_lookup_from_seed(sample_seed: int) -> tuple[dict[str, int], int]:
        if model_name:
            bootstrap_train = training.sample(
                n=len(training),
                replace=True,
                random_state=sample_seed,
            )
            rescored = fit_predict_model_holdout(
                bootstrap_train,
                training,
                model_name=model_name,
            ).rename(columns={"prediction": score_key})
            if rescored.empty:
                return {}, 0
            rescored = rescored.sort_values(score_key, ascending=False)
        else:
            reference = training.sample(n=len(training), replace=True, random_state=sample_seed)
            rescored = recompute_priority_from_reference(
                training,
                reference,
                normalization_method=normalization_method,
            )
            rescored[score_key] = rescored[score_column].fillna(0.0)
            rescored = rescored.sort_values(score_key, ascending=False)
        return (
            {
                backbone_id: rank
                for rank, backbone_id in enumerate(rescored["backbone_id"].astype(str), start=1)
            },
            int(len(rescored)),
        )

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(sample_seeds), cap=8)
    if jobs > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                rank_lookups = list(executor.map(_rank_lookup_from_seed, sample_seeds))
    else:
        rank_lookups = [_rank_lookup_from_seed(sample_seed) for sample_seed in sample_seeds]

    for rank_lookup, ranked_size in rank_lookups:
        if ranked_size <= 0:
            continue
        for backbone_id in candidate_ids:
            rank = int(rank_lookup.get(backbone_id, ranked_size + 1))
            stats[backbone_id]["ranks"].append(rank)
            stats[backbone_id]["top_hits"] += int(rank <= top_k)
            stats[backbone_id]["top_hits_top10"] += int(rank <= min(10, ranked_size))
            stats[backbone_id]["top_hits_top25"] += int(rank <= min(25, ranked_size))

    rows = []
    for row in candidates.to_dict(orient="records"):
        ranks = np.asarray(stats[str(row["backbone_id"])]["ranks"], dtype=float)
        rows.append(
            {
                "backbone_id": row["backbone_id"],
                "base_priority_index": float(row[score_key]),
                "base_rank": int(row["base_rank"]),
                "top_k": int(top_k),
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_top_k_frequency": float(
                    stats[str(row["backbone_id"])]["top_hits"] / n_bootstrap
                ),
                "bootstrap_top_10_frequency": float(
                    stats[str(row["backbone_id"])]["top_hits_top10"] / n_bootstrap
                ),
                "bootstrap_top_25_frequency": float(
                    stats[str(row["backbone_id"])]["top_hits_top25"] / n_bootstrap
                ),
                "bootstrap_mean_rank": float(ranks.mean()),
                "bootstrap_median_rank": float(np.median(ranks)),
                "bootstrap_rank_std": float(ranks.std()),
            }
        )
    return pd.DataFrame(rows).sort_values("base_rank").reset_index(drop=True)


def build_variant_rank_consistency_table(
    base_scored: pd.DataFrame,
    variant_frames: dict[str, pd.DataFrame],
    *,
    candidate_n: int = 50,
    top_k: int = 25,
    score_column: str = "priority_index",
    model_name: str | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Summarize how often top candidates remain in the top set across robustness variants."""
    base = base_scored.loc[base_scored["member_count_train"].fillna(0).astype(int) > 0].copy()
    if model_name:
        base = base.loc[base["spread_label"].notna()].copy()
    if base.empty:
        return pd.DataFrame()
    score_key = "variant_score"
    if model_name:
        base_scores = fit_full_model_predictions(base, model_name=model_name).rename(
            columns={"prediction": score_key}
        )
        base = base.merge(base_scores, on="backbone_id", how="left", validate="1:1")
    else:
        base[score_key] = base[score_column].fillna(0.0)
    base = base.sort_values(score_key, ascending=False).reset_index(drop=True)
    candidates = base.head(candidate_n)[["backbone_id", score_key]].copy()
    candidates["base_rank"] = np.arange(1, len(candidates) + 1)
    tasks = [
        (name, frame)
        for name, frame in variant_frames.items()
        if not frame.empty
        and "backbone_id" in frame.columns
        and "member_count_train" in frame.columns
    ]

    def _evaluate_variant(
        task: tuple[str, pd.DataFrame],
    ) -> tuple[str, tuple[dict[str, int], int] | None]:
        name, frame = task
        working = frame.loc[frame["member_count_train"].fillna(0).astype(int) > 0].copy()
        if model_name:
            working = working.loc[working["spread_label"].notna()].copy()
        if working.empty:
            return name, None
        if model_name:
            predictions = fit_full_model_predictions(working, model_name=model_name).rename(
                columns={"prediction": score_key}
            )
            working = working.merge(predictions, on="backbone_id", how="left", validate="1:1")
        elif score_column in working.columns:
            working[score_key] = working[score_column].fillna(0.0)
        else:
            return name, None
        ranked = working.sort_values(score_key, ascending=False).reset_index(drop=True)
        rank_lookup = {
            backbone_id: rank
            for rank, backbone_id in enumerate(ranked["backbone_id"].astype(str), start=1)
        }
        return name, (rank_lookup, int(len(ranked)))

    valid_variants: dict[str, tuple[dict[str, int], int]] = {}
    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(tasks))
    if jobs > 1 and tasks:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                evaluated = list(executor.map(_evaluate_variant, tasks))
    else:
        evaluated = [_evaluate_variant(task) for task in tasks]
    for name, payload in evaluated:
        if payload is not None:
            valid_variants[name] = payload
    if not valid_variants:
        return pd.DataFrame()

    rows = []
    for row in candidates.to_dict(orient="records"):
        backbone_id = str(row["backbone_id"])
        top_hits = 0
        top_hits_top10 = 0
        top_hits_top25 = 0
        ranks: list[int] = []
        for rank_lookup, ranked_size in valid_variants.values():
            rank = int(rank_lookup.get(backbone_id, ranked_size + 1))
            ranks.append(rank)
            top_hits += int(rank <= top_k)
            top_hits_top10 += int(rank <= min(10, ranked_size))
            top_hits_top25 += int(rank <= min(25, ranked_size))
        rank_array = np.asarray(ranks, dtype=float)
        rows.append(
            {
                "backbone_id": backbone_id,
                "base_priority_index": float(row[score_key]),
                "base_rank": int(row["base_rank"]),
                "top_k": int(top_k),
                "n_variants_evaluated": int(len(valid_variants)),
                "variant_top_k_frequency": float(top_hits / len(valid_variants)),
                "variant_top_10_frequency": float(top_hits_top10 / len(valid_variants)),
                "variant_top_25_frequency": float(top_hits_top25 / len(valid_variants)),
                "variant_mean_rank": float(rank_array.mean()),
                "variant_median_rank": float(np.median(rank_array)),
                "variant_rank_std": float(rank_array.std()),
            }
        )
    return pd.DataFrame(rows).sort_values("base_rank").reset_index(drop=True)


def build_group_holdout_performance(
    scored: pd.DataFrame,
    *,
    model_names: list[str],
    group_columns: list[str],
    min_group_size: int = 25,
    max_groups_per_column: int = 8,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Evaluate selected models on strict held-out groups such as source or dominant genus."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["spread_label"] = eligible["spread_label"].astype(int)

    rows: list[dict[str, object]] = []
    tasks: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame, str]] = []
    for group_column in group_columns:
        if group_column not in eligible.columns:
            continue
        working = eligible.copy()
        working[group_column] = working[group_column].fillna("unknown").astype(str)
        counts = working[group_column].value_counts()
        if group_column == "dominant_source":
            selected_groups = counts.index.tolist()
        else:
            selected_groups = (
                counts.loc[counts >= min_group_size].head(max_groups_per_column).index.tolist()
            )
        for group_value in selected_groups:
            test = working.loc[working[group_column] == group_value].copy()
            train = working.loc[working[group_column] != group_value].copy()
            n_positive = int(test["spread_label"].sum())
            base_row = {
                "group_column": group_column,
                "group_value": str(group_value),
                "n_test_backbones": int(len(test)),
                "n_test_positive": n_positive,
                "n_train_backbones": int(len(train)),
            }
            for model_name in model_names:
                if (
                    len(test) < min_group_size
                    or test["spread_label"].nunique() < 2
                    or train["spread_label"].nunique() < 2
                ):
                    rows.append(
                        {
                            **base_row,
                            "model_name": model_name,
                            "roc_auc": np.nan,
                            "average_precision": np.nan,
                            "brier_score": np.nan,
                            "status": "skipped_insufficient_label_variation",
                        }
                    )
                    continue
                tasks.append((base_row, train, test, model_name))

    def _evaluate_holdout_task(
        task: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, str],
    ) -> dict[str, object]:
        base_row, train, test, model_name = task
        prediction_table = fit_predict_model_holdout(train, test, model_name=model_name)
        if prediction_table.empty or prediction_table["spread_label"].nunique() < 2:
            return {
                **base_row,
                "model_name": model_name,
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "brier_score": np.nan,
                "status": "skipped_fit_failure",
            }
        y = prediction_table["spread_label"].to_numpy(dtype=int)
        preds = prediction_table["prediction"].to_numpy(dtype=float)
        prevalence = positive_prevalence(y)
        top_k = max(int(np.ceil(len(preds) * 0.10)), 1)
        ranked = prediction_table.sort_values("prediction", ascending=False).head(top_k)
        return {
            **base_row,
            "model_name": model_name,
            "roc_auc": roc_auc_score(y, preds),
            "average_precision": average_precision(y, preds),
            "average_precision_lift": average_precision_lift(y, preds),
            "brier_score": brier_score(y, preds),
            "positive_prevalence": prevalence,
            "precision_at_top_10pct": float(ranked["spread_label"].astype(float).mean()),
            "extreme_prevalence_caution": bool(prevalence <= 0.10 or prevalence >= 0.90),
            "status": "ok",
        }

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(tasks))
    if jobs > 1 and tasks:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                rows.extend(executor.map(_evaluate_holdout_task, tasks))
    else:
        rows.extend(_evaluate_holdout_task(task) for task in tasks)
    return pd.DataFrame(rows)


def build_blocked_holdout_summary(
    group_holdout: pd.DataFrame,
    *,
    blocked_group_columns: list[str] | tuple[str, ...] = (
        "dominant_source",
        "dominant_region_train",
    ),
) -> pd.DataFrame:
    """Summarize strict source and spatial blocked holdout performance by model.

    Note: This reports separate pooled summaries for each blocked axis (source, region).
    The counts are not disjoint across axes - a backbone may be held out on both source
    and region. This is explicitly marked via the pooled_overlap_summary flag to prevent
    misinterpretation as a single disjoint cohort.
    """
    if group_holdout.empty:
        return pd.DataFrame()
    working = group_holdout.loc[
        group_holdout.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
        & group_holdout.get("group_column", pd.Series(dtype=str))
        .astype(str)
        .isin([str(column) for column in blocked_group_columns])
    ].copy()
    if working.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (model_name, group_column), frame in working.groupby(
        ["model_name", "group_column"], sort=False
    ):
        weights = pd.to_numeric(
            frame.get("n_test_backbones", pd.Series(0.0, index=frame.index)), errors="coerce"
        ).fillna(0.0)
        aucs = pd.to_numeric(
            frame.get("roc_auc", pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
        valid = aucs.notna()
        if not valid.any():
            continue
        valid_weights = np.clip(weights.loc[valid].to_numpy(dtype=float), 1.0, None)
        valid_aucs = aucs.loc[valid].to_numpy(dtype=float)
        if valid_weights.size and np.isfinite(valid_weights).any():
            weighted_auc = float(np.average(valid_aucs, weights=valid_weights))
        else:
            weighted_auc = float(np.mean(valid_aucs))
        best_idx = aucs.idxmax()
        worst_idx = aucs.idxmin()
        rows.append(
            {
                "model_name": str(model_name),
                "blocked_holdout_group_columns": str(group_column),
                "blocked_holdout_roc_auc": weighted_auc,
                "blocked_holdout_macro_roc_auc": float(aucs.mean()),
                "blocked_holdout_n_backbones": int(weights.sum()),
                "blocked_holdout_group_count": int(frame["group_value"].nunique(dropna=True)),
                "best_blocked_holdout_group": (
                    f"{frame.loc[best_idx, 'group_column']}:{frame.loc[best_idx, 'group_value']}"
                ),
                "best_blocked_holdout_group_roc_auc": float(aucs.loc[best_idx]),
                "worst_blocked_holdout_group": (
                    f"{frame.loc[worst_idx, 'group_column']}:{frame.loc[worst_idx, 'group_value']}"
                ),
                "worst_blocked_holdout_group_roc_auc": float(aucs.loc[worst_idx]),
                "pooled_overlap_summary": True,
            }
        )
    return pd.DataFrame(rows)


def build_logistic_implementation_audit(
    scored: pd.DataFrame,
    *,
    model_name: str,
    columns: list[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare the custom logistic implementation against sklearn on identical folds."""
    eligible, X, y = _eligible_xy(scored, columns)
    if eligible.empty or len(np.unique(y)) < 2:
        return pd.DataFrame()

    fit_kwargs = _model_fit_kwargs(model_name)
    sample_weight = _compute_sample_weight(eligible, mode=_fit_kwarg_mode(fit_kwargs))
    custom_preds = _oof_predictions(
        X,
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        sample_weight=sample_weight,
        l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
        max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
    )

    sklearn_preds = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    l2_value = _fit_kwarg_float(fit_kwargs, "l2", 1.0)
    c_value = 1e6 if l2_value <= 0 else 1.0 / l2_value
    max_iter = max(_fit_kwarg_int(fit_kwargs, "max_iter", 100), 1000)
    for fold_indices in _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed):
        for test_idx in fold_indices:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            X_train, X_test = X[train_mask], X[test_idx]
            y_train = y[train_mask]
            train_weight = sample_weight[train_mask] if sample_weight is not None else None
            X_train_scaled, mean, std = _standardize_fit(X_train)
            X_test_scaled = _standardize_apply(X_test, mean, std)
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                fit_intercept=True,
                max_iter=max_iter,
            )
            model.fit(X_train_scaled, y_train, sample_weight=train_weight)
            sklearn_preds[test_idx] += model.predict_proba(X_test_scaled)[:, 1]
            counts[test_idx] += 1
    counts[counts == 0] = 1.0
    sklearn_preds = sklearn_preds / counts

    diff = np.abs(custom_preds - sklearn_preds)
    row = {
        "model_name": model_name,
        "n_backbones": int(len(y)),
        "custom_roc_auc": roc_auc_score(y, custom_preds),
        "sklearn_roc_auc": roc_auc_score(y, sklearn_preds),
        "custom_average_precision": average_precision(y, custom_preds),
        "sklearn_average_precision": average_precision_score(y, sklearn_preds),
        "custom_average_precision_lift": average_precision_lift(y, custom_preds),
        "sklearn_average_precision_lift": average_precision_score(y, sklearn_preds)
        - positive_prevalence(y),
        "pearson_prediction_correlation": float(
            pd.Series(custom_preds).corr(pd.Series(sklearn_preds), method="pearson")
        ),
        "spearman_prediction_correlation": float(
            pd.Series(custom_preds).corr(pd.Series(sklearn_preds), method="spearman")
        ),
        "mean_absolute_prediction_difference": float(diff.mean()),
        "max_absolute_prediction_difference": float(diff.max()),
    }
    return pd.DataFrame([row])


def build_model_simplicity_summary(
    model_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model_name: str,
    conservative_model_name: str,
    top_ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:
    """Summarize whether the simpler conservative model preserves the candidate ranking signal."""
    model_metrics = _active_model_metrics(model_metrics)
    available = model_metrics.set_index("model_name", drop=False)
    if primary_model_name not in available.index or conservative_model_name not in available.index:
        return pd.DataFrame()

    primary = predictions.loc[predictions["model_name"] == primary_model_name].copy()
    conservative = predictions.loc[predictions["model_name"] == conservative_model_name].copy()
    if primary.empty or conservative.empty:
        return pd.DataFrame()

    primary = primary.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
    conservative = conservative.sort_values("oof_prediction", ascending=False).reset_index(
        drop=True
    )
    primary_ids = primary["backbone_id"].astype(str).tolist()
    conservative_ids = conservative["backbone_id"].astype(str).tolist()

    row: dict[str, object] = {
        "primary_model_name": primary_model_name,
        "conservative_model_name": conservative_model_name,
        "primary_roc_auc": float(available.loc[primary_model_name, "roc_auc"]),
        "conservative_roc_auc": float(available.loc[conservative_model_name, "roc_auc"]),
        "roc_auc_delta_primary_minus_conservative": float(
            available.loc[primary_model_name, "roc_auc"]
            - available.loc[conservative_model_name, "roc_auc"]
        ),
        "primary_average_precision": float(available.loc[primary_model_name, "average_precision"]),
        "conservative_average_precision": float(
            available.loc[conservative_model_name, "average_precision"]
        ),
        "primary_brier_score": float(available.loc[primary_model_name, "brier_score"]),
        "conservative_brier_score": float(available.loc[conservative_model_name, "brier_score"]),
    }
    for top_k in top_ks:
        primary_top = set(primary_ids[:top_k])
        conservative_top = set(conservative_ids[:top_k])
        overlap = len(primary_top & conservative_top)
        union = len(primary_top | conservative_top)
        row[f"top_{top_k}_overlap_count"] = int(overlap)
        row[f"top_{top_k}_overlap_fraction"] = float(overlap / top_k) if top_k else 0.0
        row[f"top_{top_k}_jaccard"] = float(overlap / union) if union else 0.0
    return pd.DataFrame([row])


def build_temporal_drift_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize how record density and source composition change over time."""
    if records.empty or "resolved_year" not in records.columns:
        return pd.DataFrame()
    working = records.copy()
    working["resolved_year"] = pd.to_numeric(working["resolved_year"], errors="coerce")
    working = working.loc[working["resolved_year"].between(1900, 2100, inclusive="both")].copy()
    if working.empty:
        return pd.DataFrame()
    working["resolved_year"] = working["resolved_year"].astype(int)
    rows: list[dict[str, object]] = []
    for year, frame in working.groupby("resolved_year", sort=True):
        country_values = frame["country"].fillna("").astype(str).str.strip()
        genus_values = frame["genus"].fillna("").astype(str).str.strip()
        rows.append(
            {
                "resolved_year": int(year),
                "n_records": int(len(frame)),
                "n_backbones": int(frame["backbone_id"].nunique()),
                "n_countries": int(country_values.loc[country_values != ""].nunique()),
                "n_genera": int(genus_values.loc[genus_values != ""].nunique()),
                "refseq_record_fraction": float(frame["record_origin"].eq("refseq").mean())
                if "record_origin" in frame.columns
                else np.nan,
                "insd_record_fraction": float(frame["record_origin"].eq("insd").mean())
                if "record_origin" in frame.columns
                else np.nan,
                "mobilizable_fraction": float(
                    frame["is_mobilizable"].fillna(False).astype(bool).mean()
                )
                if "is_mobilizable" in frame.columns
                else np.nan,
                "conjugative_fraction": float(
                    frame["is_conjugative"].fillna(False).astype(bool).mean()
                )
                if "is_conjugative" in frame.columns
                else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values("resolved_year").reset_index(drop=True)
    if summary.empty:
        return summary
    summary["low_density_year"] = summary["n_records"].fillna(0).astype(int) < 20
    for column in [
        "n_records",
        "n_backbones",
        "refseq_record_fraction",
        "mobilizable_fraction",
        "conjugative_fraction",
    ]:
        summary[f"{column}_rolling3"] = (
            summary[column].astype(float).rolling(window=3, min_periods=1, center=True).mean()
        )
    return summary


def build_temporal_rank_stability_table(
    predictions: pd.DataFrame,
    *,
    split_year_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((2014, 2015),),
    top_ks: tuple[int, ...] = (10, 25),
) -> pd.DataFrame:
    """Compare ranking stability for the same model across split-year variants."""
    required = {"split_year", "model_name", "backbone_id", "oof_prediction"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    working = predictions.copy()
    working["split_year"] = pd.to_numeric(working["split_year"], errors="coerce")
    working["oof_prediction"] = pd.to_numeric(working["oof_prediction"], errors="coerce")
    working = working.loc[working["split_year"].notna() & working["oof_prediction"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    working["split_year"] = working["split_year"].astype(int)
    for split_year_a, split_year_b in split_year_pairs:
        pair = working.loc[
            working["split_year"].isin([int(split_year_a), int(split_year_b)])
        ].copy()
        for model_name, frame in pair.groupby("model_name", sort=False):
            left = frame.loc[
                frame["split_year"] == int(split_year_a), ["backbone_id", "oof_prediction"]
            ]
            right = frame.loc[
                frame["split_year"] == int(split_year_b), ["backbone_id", "oof_prediction"]
            ]
            if left.empty or right.empty:
                continue
            merged = left.merge(
                right,
                on="backbone_id",
                how="inner",
                suffixes=("_a", "_b"),
            )
            if len(merged) < 5:
                rows.append(
                    {
                        "model_name": str(model_name),
                        "split_year_a": int(split_year_a),
                        "split_year_b": int(split_year_b),
                        "n_common_backbones": int(len(merged)),
                        "kendall_tau": np.nan,
                        "status": "skipped_insufficient_overlap",
                    }
                )
                continue
            rank_a = merged["oof_prediction_a"].rank(method="average", ascending=False)
            rank_b = merged["oof_prediction_b"].rank(method="average", ascending=False)
            row: dict[str, object] = {
                "model_name": str(model_name),
                "split_year_a": int(split_year_a),
                "split_year_b": int(split_year_b),
                "n_common_backbones": int(len(merged)),
                "kendall_tau": float(rank_a.corr(rank_b, method="kendall")),
                "mean_abs_rank_shift": float(np.mean(np.abs(rank_a - rank_b))),
                "status": "ok",
            }
            left_ranked = (
                left.sort_values("oof_prediction", ascending=False)["backbone_id"]
                .astype(str)
                .tolist()
            )
            right_ranked = (
                right.sort_values("oof_prediction", ascending=False)["backbone_id"]
                .astype(str)
                .tolist()
            )
            for top_k in top_ks:
                left_top = set(left_ranked[:top_k])
                right_top = set(right_ranked[:top_k])
                overlap = len(left_top & right_top)
                union = len(left_top | right_top)
                row[f"top_{top_k}_overlap_count"] = int(overlap)
                row[f"top_{top_k}_overlap_fraction"] = float(overlap / max(int(top_k), 1))
                row[f"top_{top_k}_jaccard"] = float(overlap / union) if union else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def build_sleeper_threat_table(model_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether models over-index on low-knownness positives."""
    required = {"model_name", "average_precision", "novelty_adjusted_average_precision"}
    if model_metrics.empty or not required.issubset(model_metrics.columns):
        return pd.DataFrame()
    working = model_metrics.copy()
    ap = pd.to_numeric(working["average_precision"], errors="coerce")
    naap = pd.to_numeric(working["novelty_adjusted_average_precision"], errors="coerce")
    result = pd.DataFrame(
        {
            "model_name": working["model_name"].astype(str),
            "average_precision": ap.astype(float),
            "novelty_adjusted_average_precision": naap.astype(float),
            "naap_minus_ap": (naap - ap).astype(float),
            "naap_to_ap_ratio": np.where(ap > 0.0, (naap / ap), np.nan),
            "sleeper_threat_advantage": np.where(
                naap > ap,
                "favors_low_knownness_positives",
                "rides_knownness_bias",
            ),
        }
    )
    return result.sort_values(
        ["naap_minus_ap", "novelty_adjusted_average_precision"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_magic_number_sensitivity_table(
    sensitivity_results: pd.DataFrame,
    *,
    baseline_variant: str = "default",
    tolerance_fraction: float = 0.05,
) -> pd.DataFrame:
    """Normalize parameter-perturbation experiments into an audit-ready sensitivity table."""
    required = {"variant", "roc_auc"}
    if sensitivity_results.empty or not required.issubset(sensitivity_results.columns):
        return pd.DataFrame()
    working = sensitivity_results.copy()
    working["variant"] = working["variant"].astype(str)
    working["roc_auc"] = pd.to_numeric(working["roc_auc"], errors="coerce")
    if "parameter_name" not in working.columns:
        working["parameter_name"] = "global"
    if "parameter_value" not in working.columns:
        working["parameter_value"] = working["variant"]
    baseline_row = working.loc[working["variant"] == str(baseline_variant)]
    if baseline_row.empty:
        baseline_auc = (
            float(working["roc_auc"].dropna().max()) if working["roc_auc"].notna().any() else np.nan
        )
        baseline_variant = (
            str(working.loc[working["roc_auc"].idxmax(), "variant"])
            if working["roc_auc"].notna().any()
            else str(baseline_variant)
        )
    else:
        baseline_auc = float(baseline_row.iloc[0]["roc_auc"])
    working["baseline_variant"] = str(baseline_variant)
    working["baseline_roc_auc"] = float(baseline_auc)
    working["delta_roc_auc"] = working["roc_auc"].astype(float) - float(baseline_auc)
    working["abs_relative_auc_delta"] = np.where(
        np.isfinite(float(baseline_auc)) and float(baseline_auc) > 0.0,
        working["delta_roc_auc"].abs() / float(baseline_auc),
        np.nan,
    )
    working["passes_auc_tolerance"] = working["abs_relative_auc_delta"].le(
        float(tolerance_fraction)
    )
    return working.sort_values(
        ["parameter_name", "passes_auc_tolerance", "abs_relative_auc_delta", "variant"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_candidate_universe_table(
    *,
    scored: pd.DataFrame,
    consensus_candidates: pd.DataFrame,
    candidate_dossiers: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    novelty_watchlist: pd.DataFrame,
    prospective_freeze: pd.DataFrame,
    high_confidence_candidates: pd.DataFrame,
    candidate_risk: pd.DataFrame,
) -> pd.DataFrame:
    """Unify all candidate universes into one reviewer-friendly map."""
    candidate_sets = []
    for frame in (
        consensus_candidates,
        candidate_dossiers,
        candidate_portfolio,
        novelty_watchlist,
        prospective_freeze,
        high_confidence_candidates,
        candidate_risk,
    ):
        if not frame.empty and "backbone_id" in frame.columns:
            candidate_sets.append(frame["backbone_id"].astype(str))
    if not candidate_sets:
        return pd.DataFrame()
    candidate_ids = pd.Index(
        pd.concat(candidate_sets, ignore_index=True).drop_duplicates().tolist(), dtype=object
    )
    base = pd.DataFrame({"backbone_id": candidate_ids.astype(str)})

    scored_columns = [
        "backbone_id",
        "member_count_train",
        "n_countries_train",
        "n_new_countries",
        "spread_label",
        "priority_index",
        "bio_priority_index",
        "evidence_support_index",
        "coherence_score",
    ]
    scored_payload = (
        scored[[column for column in scored_columns if column in scored.columns]].copy()
        if not scored.empty
        else pd.DataFrame()
    )
    if not scored_payload.empty:
        base = coalescing_left_merge(base, scored_payload, on="backbone_id")

    if not consensus_candidates.empty:
        consensus_payload = consensus_candidates[
            [
                column
                for column in [
                    "backbone_id",
                    "consensus_rank",
                    "consensus_candidate_score",
                    "consensus_support_count",
                    "primary_rank",
                    "conservative_rank",
                    "rank_disagreement_primary_vs_conservative",
                ]
                if column in consensus_candidates.columns
            ]
        ].copy()
        consensus_payload["in_consensus_top50"] = True
        base = coalescing_left_merge(base, consensus_payload, on="backbone_id")
    if not candidate_dossiers.empty:
        dossier_payload = candidate_dossiers[
            [
                column
                for column in [
                    "backbone_id",
                    "freeze_rank",
                    "candidate_confidence_tier",
                    "recommended_monitoring_tier",
                    "primary_model_candidate_score",
                    "baseline_both_candidate_score",
                    "conservative_model_candidate_score",
                    "novelty_margin_vs_baseline",
                    "knownness_score",
                    "knownness_half",
                    "external_support_modalities_count",
                    "module_f_enriched_signature_count",
                    "operational_risk_score",
                    "risk_uncertainty",
                    "uncertainty_review_tier",
                    "risk_decision_tier",
                    "risk_abstain_flag",
                ]
                if column in candidate_dossiers.columns
            ]
        ].copy()
        dossier_payload["in_candidate_dossier_top25"] = True
        base = coalescing_left_merge(base, dossier_payload, on="backbone_id")
    if not candidate_portfolio.empty:
        portfolio_payload = candidate_portfolio[
            [
                column
                for column in [
                    "backbone_id",
                    "portfolio_track",
                    "track_rank",
                    "candidate_confidence_tier",
                    "recommended_monitoring_tier",
                    "operational_risk_score",
                    "risk_uncertainty",
                    "uncertainty_review_tier",
                    "risk_decision_tier",
                    "risk_abstain_flag",
                ]
                if column in candidate_portfolio.columns
            ]
        ].copy()
        portfolio_payload["in_candidate_portfolio"] = True
        base = coalescing_left_merge(base, portfolio_payload, on="backbone_id")
    if not novelty_watchlist.empty:
        novelty_payload = novelty_watchlist[
            [
                column
                for column in [
                    "backbone_id",
                    "novelty_margin_vs_baseline",
                    "knownness_score",
                    "knownness_half",
                    "primary_model_candidate_score",
                    "operational_risk_score",
                    "risk_uncertainty",
                    "uncertainty_review_tier",
                    "risk_decision_tier",
                    "risk_abstain_flag",
                ]
                if column in novelty_watchlist.columns
            ]
        ].copy()
        novelty_payload["in_novelty_watchlist"] = True
        base = coalescing_left_merge(base, novelty_payload, on="backbone_id")
    if not prospective_freeze.empty:
        freeze_payload = prospective_freeze[
            [
                column
                for column in [
                    "backbone_id",
                    "freeze_rank",
                    "freeze_candidate_score",
                ]
                if column in prospective_freeze.columns
            ]
        ].copy()
        freeze_payload["in_prospective_freeze"] = True
        base = coalescing_left_merge(base, freeze_payload, on="backbone_id")
    if not high_confidence_candidates.empty:
        high_conf_payload = high_confidence_candidates[
            [
                column
                for column in [
                    "backbone_id",
                    "candidate_confidence_tier",
                    "false_positive_risk_tier",
                ]
                if column in high_confidence_candidates.columns
            ]
        ].copy()
        high_conf_payload["in_higher_confidence_shortlist"] = True
        base = coalescing_left_merge(base, high_conf_payload, on="backbone_id")
    if not candidate_risk.empty:
        risk_payload = candidate_risk[
            [
                column
                for column in [
                    "backbone_id",
                    "false_positive_risk_tier",
                    "risk_flag_count",
                    "risk_flags",
                ]
                if column in candidate_risk.columns
            ]
        ].copy()
        base = coalescing_left_merge(base, risk_payload, on="backbone_id")

    base = base.copy()

    flag_columns = (
        "in_consensus_top50",
        "in_candidate_dossier_top25",
        "in_candidate_portfolio",
        "in_novelty_watchlist",
        "in_prospective_freeze",
        "in_higher_confidence_shortlist",
    )
    flag_updates = {
        flag_column: (
            base[flag_column].fillna(False).astype(bool)
            if flag_column in base.columns
            else pd.Series(False, index=base.index)
        )
        for flag_column in flag_columns
    }
    base_updates = pd.DataFrame(
        {
            **flag_updates,
            "evidence_tier": base.get(
                "candidate_confidence_tier", pd.Series("unknown", index=base.index)
            ).fillna("unknown"),
            "action_tier": base.get(
                "recommended_monitoring_tier", pd.Series("unassigned", index=base.index)
            ).fillna("unassigned"),
            "main_outcome_status": np.select(
                [
                    base.get("spread_label", pd.Series(np.nan, index=base.index)).isna(),
                    base.get("spread_label", pd.Series(0.0, index=base.index))
                    .fillna(0.0)
                    .astype(float)
                    >= 1.0,
                ],
                ["not_evaluable", "positive"],
                default="negative",
            ),
        },
        index=base.index,
    )
    base = pd.concat(
        [base.drop(columns=list(base_updates.columns), errors="ignore"), base_updates], axis=1
    )
    base["candidate_universe_origin"] = np.select(
        [
            base["in_candidate_portfolio"]
            & base.get("portfolio_track", pd.Series("", index=base.index))
            .fillna("")
            .eq("established_high_risk"),
            base["in_candidate_portfolio"]
            & base.get("portfolio_track", pd.Series("", index=base.index))
            .fillna("")
            .eq("novel_signal"),
            base["in_higher_confidence_shortlist"],
            base["in_consensus_top50"],
            base["in_novelty_watchlist"],
            base["in_prospective_freeze"],
        ],
        [
            "portfolio_established",
            "portfolio_novel",
            "higher_confidence_shortlist",
            "consensus_only",
            "novelty_watchlist_only",
            "prospective_freeze_only",
        ],
        default="other_candidate_context",
    )
    sort_columns = [
        "in_candidate_portfolio",
        "in_higher_confidence_shortlist",
        "in_consensus_top50",
        "consensus_rank",
        "freeze_rank",
        "primary_model_candidate_score",
        "priority_index",
    ]
    available = [column for column in sort_columns if column in base.columns]
    ascending = [False, False, False, True, True, False, False][: len(available)]
    return base.sort_values(available, ascending=ascending).reset_index(drop=True)


def build_primary_model_selection_summary(
    model_metrics: pd.DataFrame,
    *,
    primary_model_name: str,
    conservative_model_name: str,
    governance_model_name: str | None = None,
    predictions: pd.DataFrame | None = None,
    decision_yield: pd.DataFrame | None = None,
    blocked_holdout_calibration_summary: pd.DataFrame | None = None,
    family_summary: pd.DataFrame | None = None,
    simplicity_summary: pd.DataFrame | None = None,
    model_selection_scorecard: pd.DataFrame | None = None,
    top_ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:
    """Make the publication choice of the primary model explicit for reviewers."""
    model_metrics = _active_model_metrics(model_metrics)
    if model_metrics.empty or primary_model_name not in set(
        model_metrics["model_name"].astype(str)
    ):
        return pd.DataFrame()
    available = model_metrics.set_index("model_name", drop=False)
    primary = available.loc[primary_model_name]
    strongest = model_metrics.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    strongest_model_name = str(strongest["model_name"])
    conservative = (
        available.loc[conservative_model_name]
        if conservative_model_name in available.index
        else None
    )
    governance_scorecard = pd.Series(dtype=object)
    if model_selection_scorecard is not None and not model_selection_scorecard.empty:
        if governance_model_name is not None and str(governance_model_name).strip():
            governance_scorecard = model_selection_scorecard.loc[
                model_selection_scorecard.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(governance_model_name))
            ].head(1)
            if not governance_scorecard.empty:
                governance_scorecard = governance_scorecard.iloc[0]
        if governance_scorecard.empty:
            governance_scorecard = _select_governance_scorecard_row(model_selection_scorecard)
    inferred_governance_name = str(governance_scorecard.get("model_name", "")).strip()
    if governance_model_name is None or not str(governance_model_name).strip():
        governance_model_name = inferred_governance_name
    governance = (
        available.loc[governance_model_name] if governance_model_name in available.index else None
    )
    governance_strict_acceptance = bool(
        governance_scorecard.get("strict_knownness_acceptance_flag", False)
    )
    governance_scientific_acceptance_status = str(
        governance_scorecard.get("scientific_acceptance_status", "")
    ).strip()
    governance_scientific_acceptance = (
        bool(governance_scorecard.get("scientific_acceptance_flag", False))
        if governance_scientific_acceptance_status in {"pass", "fail"}
        else governance_strict_acceptance
    )

    if (
        family_summary is None
        or family_summary.empty
        or "model_name" not in family_summary.columns
        or "evidence_role" not in family_summary.columns
        or primary_model_name not in set(family_summary["model_name"].astype(str))
        or strongest_model_name not in set(family_summary["model_name"].astype(str))
    ):
        family_summary = build_model_family_summary(model_metrics)
    model_selection_scorecard = (
        model_selection_scorecard if model_selection_scorecard is not None else pd.DataFrame()
    )
    decision_yield = decision_yield if decision_yield is not None else pd.DataFrame()
    blocked_holdout_calibration_summary = (
        blocked_holdout_calibration_summary
        if blocked_holdout_calibration_summary is not None
        else pd.DataFrame()
    )

    row: dict[str, object] = {
        "published_primary_model": primary_model_name,
        "published_primary_track": "discovery",
        "published_primary_roc_auc": float(primary["roc_auc"]),
        "published_primary_average_precision": float(primary["average_precision"]),
        "strongest_metric_model": strongest_model_name,
        "strongest_metric_model_roc_auc": float(strongest["roc_auc"]),
        "strongest_metric_model_average_precision": float(strongest["average_precision"]),
        "primary_minus_strongest_roc_auc": float(primary["roc_auc"] - strongest["roc_auc"]),
        "primary_minus_strongest_average_precision": float(
            primary["average_precision"] - strongest["average_precision"]
        ),
        "conservative_model_name": conservative_model_name,
        "conservative_model_track": "control",
        "conservative_roc_auc": float(conservative["roc_auc"])
        if conservative is not None
        else np.nan,
        "conservative_average_precision": float(conservative["average_precision"])
        if conservative is not None
        else np.nan,
        "governance_primary_model": governance_model_name if governance is not None else "",
        "governance_primary_track": (
            "governance_headline" if governance_scientific_acceptance else "governance_watch_only"
        )
        if governance is not None
        else "",
        "governance_primary_roc_auc": float(governance["roc_auc"])
        if governance is not None
        else np.nan,
        "governance_primary_average_precision": float(governance["average_precision"])
        if governance is not None
        else np.nan,
        "governance_primary_selection_rank": int(governance_scorecard["selection_rank"])
        if not governance_scorecard.empty
        and pd.notna(governance_scorecard.get("selection_rank", np.nan))
        else np.nan,
        "governance_primary_strict_knownness_acceptance_flag": bool(
            governance_scorecard.get("strict_knownness_acceptance_flag", False)
        )
        if not governance_scorecard.empty
        else np.nan,
        "governance_primary_scientific_acceptance_flag": bool(
            governance_scorecard.get("scientific_acceptance_flag", False)
        )
        if not governance_scorecard.empty
        and governance_scientific_acceptance_status in {"pass", "fail"}
        else np.nan,
        "governance_primary_scientific_acceptance_status": governance_scientific_acceptance_status
        if governance_scientific_acceptance_status
        else "not_scored",
        "governance_primary_knownness_matched_gap": float(
            governance_scorecard.get("knownness_matched_gap", np.nan)
        )
        if not governance_scorecard.empty
        else np.nan,
        "governance_primary_source_holdout_gap": float(
            governance_scorecard.get("source_holdout_gap", np.nan)
        )
        if not governance_scorecard.empty
        else np.nan,
        "governance_primary_guardrail_loss": float(
            governance_scorecard.get("guardrail_loss", np.nan)
        )
        if not governance_scorecard.empty
        else np.nan,
        "governance_primary_governance_priority_score": float(
            governance_scorecard.get("governance_priority_score", np.nan)
        )
        if not governance_scorecard.empty
        else np.nan,
        "governance_primary_leakage_review_required": bool(
            governance_scorecard.get("leakage_review_required", False)
        )
        if not governance_scorecard.empty
        else False,
        "governance_primary_benchmark_status": (
            "governance_headline"
            if governance_scientific_acceptance
            else "governance_watch_only"
            if not governance_scorecard.empty
            else "not_scored"
        ),
        "selection_rationale": (
            "discovery track keeps the headline benchmark explicit while the governance track is selected separately by matched-knownness/source-holdout loss; the report does not conflate discrimination with deployment safety"
            if primary_model_name == strongest_model_name
            else "current primary retained as the headline benchmark despite a marginally stronger audited alternative; the governance track is still selected separately by matched-knownness/source-holdout loss so discrimination and stability remain distinct decisions"
        ),
    }

    def _scorecard_for(model_name: str) -> pd.Series:
        if model_selection_scorecard.empty or "model_name" not in model_selection_scorecard.columns:
            return pd.Series(dtype=object)
        match = model_selection_scorecard.loc[
            model_selection_scorecard["model_name"].astype(str) == str(model_name)
        ].head(1)
        return match.iloc[0] if not match.empty else pd.Series(dtype=object)

    def _sorted_predictions(model_name: str) -> pd.DataFrame:
        if predictions is None or predictions.empty:
            return pd.DataFrame()
        frame = predictions.loc[predictions["model_name"].astype(str) == str(model_name)].copy()
        if frame.empty:
            return frame
        return frame.sort_values("oof_prediction", ascending=False).reset_index(drop=True)

    def _decision_yield_row(model_name: str, top_k: int) -> pd.Series:
        if decision_yield.empty:
            return pd.Series(dtype=object)
        model_series = decision_yield.get("model_name", pd.Series(dtype=str)).astype(str)
        top_k_series = pd.to_numeric(
            decision_yield.get("top_k", pd.Series(np.nan, index=decision_yield.index, dtype=float)),
            errors="coerce",
        ).astype("Int64")
        match = decision_yield.loc[
            model_series.eq(str(model_name)) & top_k_series.eq(int(top_k))
        ].head(1)
        return match.iloc[0] if not match.empty else pd.Series(dtype=object)

    def _blocked_holdout_row(model_name: str) -> pd.Series:
        if blocked_holdout_calibration_summary.empty:
            return pd.Series(dtype=object)
        match = blocked_holdout_calibration_summary.loc[
            blocked_holdout_calibration_summary.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(str(model_name))
        ].head(1)
        if match.empty:
            return pd.Series(dtype=object)
        raw = blocked_holdout_calibration_summary.loc[
            (
                blocked_holdout_calibration_summary.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            )
            & blocked_holdout_calibration_summary.get("calibration_method", pd.Series(dtype=str))
            .astype(str)
            .eq("raw")
        ].head(1)
        if raw.empty:
            raw_row = pd.Series(dtype=object)
        else:
            raw_row = raw.iloc[0]
        non_raw = blocked_holdout_calibration_summary.loc[
            (
                blocked_holdout_calibration_summary.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            )
            & ~blocked_holdout_calibration_summary.get("calibration_method", pd.Series(dtype=str))
            .astype(str)
            .eq("raw")
        ].copy()
        if non_raw.empty:
            best_row = pd.Series(dtype=object)
        else:
            best_row = non_raw.sort_values(
                ["brier_score", "ece"], ascending=[True, True], kind="mergesort"
            ).iloc[0]
        combined = pd.Series(dtype=object)
        if not raw_row.empty:
            combined["raw_brier_score"] = float(raw_row.get("brier_score", np.nan))
            combined["raw_ece"] = float(raw_row.get("ece", np.nan))
        if not best_row.empty:
            combined["best_calibration_method"] = str(best_row.get("calibration_method", ""))
            combined["best_brier_score"] = float(best_row.get("brier_score", np.nan))
            combined["best_ece"] = float(best_row.get("ece", np.nan))
            combined["best_calibration_gain_vs_raw_brier"] = float(
                best_row.get("calibration_gain_vs_raw_brier", np.nan)
            )
            combined["best_calibration_gain_vs_raw_ece"] = float(
                best_row.get("calibration_gain_vs_raw_ece", np.nan)
            )
        return combined

    def _add_topk_overlap(prefix: str, comparison_model_name: str) -> None:
        primary_predictions = _sorted_predictions(primary_model_name)
        comparison_predictions = _sorted_predictions(comparison_model_name)
        if primary_predictions.empty or comparison_predictions.empty:
            return
        primary_ids = primary_predictions["backbone_id"].astype(str).tolist()
        comparison_ids = comparison_predictions["backbone_id"].astype(str).tolist()
        for top_k in top_ks:
            primary_top = set(primary_ids[:top_k])
            comparison_top = set(comparison_ids[:top_k])
            overlap = len(primary_top & comparison_top)
            union = len(primary_top | comparison_top)
            row[f"primary_vs_{prefix}_top_{top_k}_overlap_count"] = int(overlap)
            row[f"primary_vs_{prefix}_top_{top_k}_overlap_fraction"] = (
                float(overlap / top_k) if top_k else 0.0
            )
            row[f"primary_vs_{prefix}_top_{top_k}_jaccard"] = (
                float(overlap / union) if union else 0.0
            )

    def _add_top10_yield(prefix: str, model_name: str) -> None:
        frame = _sorted_predictions(model_name)
        if frame.empty:
            return
        subset = frame.head(min(10, len(frame))).copy()
        total_positive = int(frame["spread_label"].sum())
        selected_positive = int(subset["spread_label"].sum()) if not subset.empty else 0
        row[f"{prefix}_top_10_precision"] = (
            float(selected_positive / len(subset)) if len(subset) else np.nan
        )
        row[f"{prefix}_top_10_recall"] = (
            float(selected_positive / total_positive) if total_positive > 0 else np.nan
        )

    def _add_topk_yield(prefix: str, model_name: str) -> None:
        for top_k in top_ks:
            yield_row = _decision_yield_row(model_name, top_k)
            if yield_row.empty:
                continue
            row[f"{prefix}_top_{top_k}_precision"] = float(yield_row.get("precision_at_k", np.nan))
            row[f"{prefix}_top_{top_k}_recall"] = float(yield_row.get("recall_at_k", np.nan))

    def _add_utility(prefix: str, model_name: str) -> None:
        scorecard_row = _scorecard_for(model_name)
        if scorecard_row.empty:
            return
        row[f"{prefix}_decision_utility_score"] = float(
            scorecard_row.get("decision_utility_score", np.nan)
        )
        row[f"{prefix}_decision_utility_cost"] = float(
            scorecard_row.get("decision_utility_cost", np.nan)
        )
        row[f"{prefix}_optimal_decision_threshold"] = float(
            scorecard_row.get("optimal_decision_threshold", np.nan)
        )
        row[f"{prefix}_decision_utility_precision"] = float(
            scorecard_row.get("decision_utility_precision", np.nan)
        )
        row[f"{prefix}_decision_utility_recall"] = float(
            scorecard_row.get("decision_utility_recall", np.nan)
        )
        row[f"{prefix}_decision_utility_positive_rate"] = float(
            scorecard_row.get("decision_utility_positive_rate", np.nan)
        )

    _add_topk_overlap("strongest", strongest_model_name)
    _add_topk_overlap("conservative", conservative_model_name)
    _add_top10_yield("published_primary", primary_model_name)
    _add_top10_yield("strongest", strongest_model_name)
    _add_top10_yield("conservative", conservative_model_name)
    if governance_model_name and governance is not None:
        _add_top10_yield("governance_primary", governance_model_name)
    _add_topk_yield("published_primary", primary_model_name)
    _add_topk_yield("strongest", strongest_model_name)
    _add_topk_yield("conservative", conservative_model_name)
    if governance_model_name and governance is not None:
        _add_topk_yield("governance_primary", governance_model_name)
    _add_utility("published_primary", primary_model_name)
    _add_utility("strongest", strongest_model_name)
    _add_utility("conservative", conservative_model_name)
    if governance_model_name and governance is not None:
        _add_utility("governance_primary", governance_model_name)

    strongest_overlap = row.get("primary_vs_strongest_top_10_overlap_count")
    strongest_overlap_25 = row.get("primary_vs_strongest_top_25_overlap_count")
    strongest_overlap_50 = row.get("primary_vs_strongest_top_50_overlap_count")
    if isinstance(strongest_overlap, (int, np.integer)) and strongest_overlap <= 4:
        overlap_25_text = (
            f", recovering to {int(strongest_overlap_25)}/25"
            if isinstance(strongest_overlap_25, (int, np.integer))
            else ""
        )
        overlap_50_text = (
            f" and {int(strongest_overlap_50)}/50"
            if isinstance(strongest_overlap_50, (int, np.integer))
            else ""
        )
        row["selection_rationale"] = (
            "current primary retained as the headline benchmark while a higher-metric audit view remains explicit; "
            f"the strongest audited alternative overlaps on only {int(strongest_overlap)}/10 top candidates{overlap_25_text}{overlap_50_text}, "
            "so the audit keeps both views explicit"
        )

    primary_scorecard = _scorecard_for(primary_model_name)
    strongest_scorecard = _scorecard_for(strongest_model_name)
    if not primary_scorecard.empty:
        if pd.notna(primary_scorecard.get("selection_rank", np.nan)):
            row["published_primary_selection_rank"] = int(primary_scorecard["selection_rank"])
        row["published_primary_strict_knownness_acceptance_flag"] = bool(
            primary_scorecard.get("strict_knownness_acceptance_flag", False)
        )
        row["published_primary_scientific_acceptance_flag"] = bool(
            primary_scorecard.get("scientific_acceptance_flag", False)
        )
        row["published_primary_scientific_acceptance_status"] = str(
            primary_scorecard.get("scientific_acceptance_status", "not_scored")
        )
        row["published_primary_knownness_matched_gap"] = float(
            primary_scorecard.get("knownness_matched_gap", np.nan)
        )
        row["published_primary_source_holdout_gap"] = float(
            primary_scorecard.get("source_holdout_gap", np.nan)
        )
        row["published_primary_leakage_review_required"] = bool(
            primary_scorecard.get("leakage_review_required", False)
        )
    primary_calibration_row = _blocked_holdout_row(primary_model_name)
    if not primary_calibration_row.empty:
        row["published_primary_blocked_holdout_raw_brier_score"] = float(
            primary_calibration_row.get("raw_brier_score", np.nan)
        )
        row["published_primary_blocked_holdout_raw_ece"] = float(
            primary_calibration_row.get("raw_ece", np.nan)
        )
        row["published_primary_blocked_holdout_best_calibration_method"] = str(
            primary_calibration_row.get("best_calibration_method", "")
        )
        row["published_primary_blocked_holdout_best_brier_score"] = float(
            primary_calibration_row.get("best_brier_score", np.nan)
        )
        row["published_primary_blocked_holdout_best_ece"] = float(
            primary_calibration_row.get("best_ece", np.nan)
        )
        row["published_primary_blocked_holdout_best_calibration_gain_vs_raw_brier"] = float(
            primary_calibration_row.get("best_calibration_gain_vs_raw_brier", np.nan)
        )
        row["published_primary_blocked_holdout_best_calibration_gain_vs_raw_ece"] = float(
            primary_calibration_row.get("best_calibration_gain_vs_raw_ece", np.nan)
        )
    if not strongest_scorecard.empty:
        if pd.notna(strongest_scorecard.get("selection_rank", np.nan)):
            row["strongest_metric_model_selection_rank"] = int(
                strongest_scorecard["selection_rank"]
            )
        row["strongest_metric_model_strict_knownness_acceptance_flag"] = bool(
            strongest_scorecard.get("strict_knownness_acceptance_flag", False)
        )
        row["strongest_metric_model_scientific_acceptance_flag"] = bool(
            strongest_scorecard.get("scientific_acceptance_flag", False)
        )
        row["strongest_metric_model_scientific_acceptance_status"] = str(
            strongest_scorecard.get("scientific_acceptance_status", "not_scored")
        )
        row["strongest_metric_model_knownness_matched_gap"] = float(
            strongest_scorecard.get("knownness_matched_gap", np.nan)
        )
        row["strongest_metric_model_source_holdout_gap"] = float(
            strongest_scorecard.get("source_holdout_gap", np.nan)
        )
        row["strongest_metric_model_leakage_review_required"] = bool(
            strongest_scorecard.get("leakage_review_required", False)
        )
    strongest_calibration_row = _blocked_holdout_row(strongest_model_name)
    if not strongest_calibration_row.empty:
        row["strongest_metric_model_blocked_holdout_raw_brier_score"] = float(
            strongest_calibration_row.get("raw_brier_score", np.nan)
        )
        row["strongest_metric_model_blocked_holdout_raw_ece"] = float(
            strongest_calibration_row.get("raw_ece", np.nan)
        )
        row["strongest_metric_model_blocked_holdout_best_calibration_method"] = str(
            strongest_calibration_row.get("best_calibration_method", "")
        )
        row["strongest_metric_model_blocked_holdout_best_brier_score"] = float(
            strongest_calibration_row.get("best_brier_score", np.nan)
        )
        row["strongest_metric_model_blocked_holdout_best_ece"] = float(
            strongest_calibration_row.get("best_ece", np.nan)
        )
        row["strongest_metric_model_blocked_holdout_best_calibration_gain_vs_raw_brier"] = float(
            strongest_calibration_row.get("best_calibration_gain_vs_raw_brier", np.nan)
        )
        row["strongest_metric_model_blocked_holdout_best_calibration_gain_vs_raw_ece"] = float(
            strongest_calibration_row.get("best_calibration_gain_vs_raw_ece", np.nan)
        )
    conservative_calibration_row = _blocked_holdout_row(conservative_model_name)
    if not conservative_calibration_row.empty:
        row["conservative_blocked_holdout_raw_brier_score"] = float(
            conservative_calibration_row.get("raw_brier_score", np.nan)
        )
        row["conservative_blocked_holdout_raw_ece"] = float(
            conservative_calibration_row.get("raw_ece", np.nan)
        )
        row["conservative_blocked_holdout_best_calibration_method"] = str(
            conservative_calibration_row.get("best_calibration_method", "")
        )
        row["conservative_blocked_holdout_best_brier_score"] = float(
            conservative_calibration_row.get("best_brier_score", np.nan)
        )
        row["conservative_blocked_holdout_best_ece"] = float(
            conservative_calibration_row.get("best_ece", np.nan)
        )
        row["conservative_blocked_holdout_best_calibration_gain_vs_raw_brier"] = float(
            conservative_calibration_row.get("best_calibration_gain_vs_raw_brier", np.nan)
        )
        row["conservative_blocked_holdout_best_calibration_gain_vs_raw_ece"] = float(
            conservative_calibration_row.get("best_calibration_gain_vs_raw_ece", np.nan)
        )
    if governance_model_name and governance is not None:
        governance_calibration_row = _blocked_holdout_row(governance_model_name)
        if not governance_calibration_row.empty:
            row["governance_primary_blocked_holdout_raw_brier_score"] = float(
                governance_calibration_row.get("raw_brier_score", np.nan)
            )
            row["governance_primary_blocked_holdout_raw_ece"] = float(
                governance_calibration_row.get("raw_ece", np.nan)
            )
            row["governance_primary_blocked_holdout_best_calibration_method"] = str(
                governance_calibration_row.get("best_calibration_method", "")
            )
            row["governance_primary_blocked_holdout_best_brier_score"] = float(
                governance_calibration_row.get("best_brier_score", np.nan)
            )
            row["governance_primary_blocked_holdout_best_ece"] = float(
                governance_calibration_row.get("best_ece", np.nan)
            )
            row["governance_primary_blocked_holdout_best_calibration_gain_vs_raw_brier"] = float(
                governance_calibration_row.get("best_calibration_gain_vs_raw_brier", np.nan)
            )
            row["governance_primary_blocked_holdout_best_calibration_gain_vs_raw_ece"] = float(
                governance_calibration_row.get("best_calibration_gain_vs_raw_ece", np.nan)
            )
    if (
        strongest_model_name != primary_model_name
        and not primary_scorecard.empty
        and not strongest_scorecard.empty
    ):
        primary_guardrail = bool(primary_scorecard.get("strict_knownness_acceptance_flag", False))
        primary_scientific_status = str(
            primary_scorecard.get("scientific_acceptance_status", "")
        ).strip()
        primary_guardrail = (
            bool(primary_scorecard.get("scientific_acceptance_flag", False))
            if primary_scientific_status in {"pass", "fail"}
            else primary_guardrail
        )
        strongest_guardrail = bool(
            strongest_scorecard.get("strict_knownness_acceptance_flag", False)
        )
        strongest_scientific_status = str(
            strongest_scorecard.get("scientific_acceptance_status", "")
        ).strip()
        strongest_guardrail = (
            bool(strongest_scorecard.get("scientific_acceptance_flag", False))
            if strongest_scientific_status in {"pass", "fail"}
            else strongest_guardrail
        )
        if primary_guardrail and not strongest_guardrail:
            row["selection_rationale"] = (
                "current primary retained as the headline benchmark because it passes strict matched-knownness and source-holdout guardrails and clears the frozen scientific acceptance gate while the strongest audited alternative does not; the discovery and governance tracks are kept explicit rather than collapsed into one headline claim"
            )
        elif not primary_guardrail:
            row["selection_rationale"] = (
                f"{row['selection_rationale']}; however, the published primary does not clear the frozen scientific acceptance gate and should be treated as a conditional discovery benchmark rather than a deployment-safe governance claim"
            )

    if governance_model_name and governance is not None:
        governance_guardrail = (
            governance_scientific_acceptance if not governance_scorecard.empty else False
        )
        if governance_guardrail:
            row["governance_selection_rationale"] = (
                f"governance benchmark `{governance_model_name}` clears the frozen scientific acceptance gate and is therefore the current governance headline"
            )
        else:
            row["governance_selection_rationale"] = (
                f"governance benchmark `{governance_model_name}` is selected as the best available guardrail-aware candidate, but frozen scientific acceptance still fails and it must remain a watch-only governance track"
            )
    else:
        row["governance_selection_rationale"] = (
            "no separate governance benchmark could be resolved from the available scorecard, so the report should treat governance as unresolved rather than silently reusing the discovery track"
        )

    if family_summary is not None and not family_summary.empty:
        family_index = family_summary.set_index("model_name", drop=False)
        if primary_model_name in family_index.index:
            row["published_primary_evidence_role"] = str(
                family_index.loc[primary_model_name, "evidence_role"]
            )
            row["published_primary_evidence_summary"] = str(
                family_index.loc[primary_model_name, "evidence_summary"]
            )
        if strongest_model_name in family_index.index:
            row["strongest_metric_model_evidence_role"] = str(
                family_index.loc[strongest_model_name, "evidence_role"]
            )
            row["strongest_metric_model_evidence_summary"] = str(
                family_index.loc[strongest_model_name, "evidence_summary"]
            )
        if conservative_model_name in family_index.index:
            row["conservative_model_evidence_role"] = str(
                family_index.loc[conservative_model_name, "evidence_role"]
            )
            row["conservative_model_evidence_summary"] = str(
                family_index.loc[conservative_model_name, "evidence_summary"]
            )
        if governance_model_name and governance_model_name in family_index.index:
            row["governance_primary_evidence_role"] = str(
                family_index.loc[governance_model_name, "evidence_role"]
            )
            row["governance_primary_evidence_summary"] = str(
                family_index.loc[governance_model_name, "evidence_summary"]
            )

    if simplicity_summary is not None and not simplicity_summary.empty:
        summary_row = simplicity_summary.iloc[0]
        for top_k in top_ks:
            legacy_count = f"top_{top_k}_overlap_count"
            legacy_fraction = f"top_{top_k}_overlap_fraction"
            legacy_jaccard = f"top_{top_k}_jaccard"
            if legacy_count in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_overlap_count"] = int(
                    summary_row[legacy_count]
                )
            if legacy_fraction in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_overlap_fraction"] = float(
                    summary_row[legacy_fraction]
                )
            if legacy_jaccard in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_jaccard"] = float(
                    summary_row[legacy_jaccard]
                )

    return pd.DataFrame([row])


def build_benchmark_protocol_table(
    model_metrics: pd.DataFrame,
    model_selection_summary: pd.DataFrame,
    *,
    adaptive_gated_metrics: pd.DataFrame | None = None,
    gate_consistency_audit: pd.DataFrame | None = None,
    model_selection_scorecard: pd.DataFrame | None = None,
    governance_model_name: str | None = None,
) -> pd.DataFrame:
    """Summarize which model is official, supporting, audit-only, or control."""
    if model_metrics.empty or model_selection_summary.empty:
        return pd.DataFrame()
    adaptive_gated_metrics = (
        adaptive_gated_metrics if adaptive_gated_metrics is not None else pd.DataFrame()
    )
    gate_consistency_audit = (
        gate_consistency_audit if gate_consistency_audit is not None else pd.DataFrame()
    )
    model_selection_scorecard = (
        model_selection_scorecard if model_selection_scorecard is not None else pd.DataFrame()
    )

    metrics_index = model_metrics.set_index("model_name", drop=False)
    selection_row = model_selection_summary.iloc[0]
    primary_model_name = str(selection_row.get("published_primary_model", ""))
    conservative_model_name = str(selection_row.get("conservative_model_name", ""))
    strongest_model_name = str(selection_row.get("strongest_metric_model", ""))
    governance_model_name = str(
        governance_model_name
        or selection_row.get("governance_primary_model", "")
        or _select_governance_scorecard_row(model_selection_scorecard).get("model_name", "")
    ).strip()

    rows: list[dict[str, object]] = []

    def _scorecard_for(model_name: str) -> pd.Series:
        if model_selection_scorecard.empty or "model_name" not in model_selection_scorecard.columns:
            return pd.Series(dtype=object)
        match = model_selection_scorecard.loc[
            model_selection_scorecard["model_name"].astype(str) == str(model_name)
        ].head(1)
        return match.iloc[0] if not match.empty else pd.Series(dtype=object)

    def _safe_float(value: object) -> float:
        coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(coerced) if pd.notna(coerced) else float("nan")

    def _append_single(
        model_name: str, *, role: str, status: str, track: str, rationale: str
    ) -> None:
        if model_name not in metrics_index.index:
            return
        metric_row = metrics_index.loc[model_name]
        scorecard_row = _scorecard_for(model_name)
        strict_flag = (
            bool(scorecard_row.get("strict_knownness_acceptance_flag", False))
            if not scorecard_row.empty
            and pd.notna(scorecard_row.get("strict_knownness_acceptance_flag", np.nan))
            else np.nan
        )
        benchmark_row: dict[str, object] = {
            "model_name": model_name,
            "benchmark_role": role,
            "benchmark_status": status,
            "benchmark_track": track,
            "model_family": "single_model",
            "roc_auc": float(metric_row["roc_auc"]),
            "average_precision": float(metric_row["average_precision"]),
            "selection_rank": _safe_float(scorecard_row.get("selection_rank", np.nan))
            if not scorecard_row.empty
            else np.nan,
            "strict_knownness_acceptance_flag": strict_flag,
            "scientific_acceptance_flag": bool(
                scorecard_row.get("scientific_acceptance_flag", False)
            )
            if not scorecard_row.empty
            and str(scorecard_row.get("scientific_acceptance_status", "")).strip()
            in {"pass", "fail"}
            else np.nan,
            "scientific_acceptance_status": str(
                scorecard_row.get("scientific_acceptance_status", "not_scored")
            )
            if not scorecard_row.empty
            else "not_scored",
            "knownness_matched_gap": _safe_float(scorecard_row.get("knownness_matched_gap", np.nan))
            if not scorecard_row.empty
            else np.nan,
            "source_holdout_gap": _safe_float(scorecard_row.get("source_holdout_gap", np.nan))
            if not scorecard_row.empty
            else np.nan,
            "leakage_review_required": bool(scorecard_row.get("leakage_review_required", False))
            if not scorecard_row.empty
            else False,
            "benchmark_guardrail_status": (
                "passes_strict_acceptance"
                if strict_flag is True
                else "fails_strict_acceptance"
                if strict_flag is False
                else "not_scored"
            ),
            "gate_consistency_tier": np.nan,
            "specialist_weight_lower_half": np.nan,
            "selection_rationale": rationale,
        }
        prefixes: list[str] = []
        if model_name == primary_model_name:
            prefixes.append("published_primary")
        if model_name == conservative_model_name:
            prefixes.append("conservative")
        if model_name == strongest_model_name:
            prefixes.append("strongest_metric_model")
        if model_name == "baseline_both":
            prefixes.append("baseline_both")
        if model_name == "source_only":
            prefixes.append("source_only")
        if governance_model_name and model_name == governance_model_name:
            prefixes.append("governance_primary")
        for prefix in prefixes:
            selection_prefix = f"{prefix}_"
            for key, value in selection_row.items():
                if isinstance(key, str) and key.startswith(selection_prefix):
                    benchmark_row[key] = value
        rows.append(benchmark_row)

    _append_single(
        primary_model_name,
        role="primary_benchmark",
        status="headline",
        track="discovery",
        rationale="Current official single-model benchmark used for the main headline claims.",
    )
    if governance_model_name:
        governance_scorecard = _scorecard_for(governance_model_name)
        governance_status = (
            "governance_headline"
            if bool(governance_scorecard.get("strict_knownness_acceptance_flag", False))
            else "governance_watch"
            if not governance_scorecard.empty
            else "not_scored"
        )
        governance_rationale = (
            "Separate governance track chosen to minimize matched-knownness and source-holdout loss rather than to maximize discrimination."
            if not governance_scorecard.empty
            else "Separate governance track could not be resolved from the available scorecard."
        )
        if governance_model_name == primary_model_name:
            governance_rationale = "Discovery and governance currently coincide on the same model, but the report keeps the two policy tracks separate so the strict guardrail status remains visible."
        _append_single(
            governance_model_name,
            role="governance_benchmark",
            status=governance_status,
            track="governance",
            rationale=governance_rationale,
        )
    _append_single(
        conservative_model_name,
        role="conservative_benchmark",
        status="headline_supporting",
        track="control",
        rationale="Supportive conservative benchmark kept for interpretability and proxy-light comparison.",
    )
    if strongest_model_name and strongest_model_name != primary_model_name:
        _append_single(
            strongest_model_name,
            role="strongest_single_model",
            status="audit_only",
            track="audit",
            rationale="Highest-metric single model retained as an audited alternative rather than the headline benchmark.",
        )
    for control_model_name, role, rationale in (
        (
            "baseline_both",
            "counts_baseline",
            "Counts-only baseline used to measure incremental value beyond simple popularity signals.",
        ),
        (
            "source_only",
            "source_control",
            "Weak source-only control used to show that source composition alone does not explain the signal.",
        ),
    ):
        if control_model_name in metrics_index.index:
            _append_single(
                control_model_name,
                role=role,
                status="control",
                track="control",
                rationale=rationale,
            )

    if not adaptive_gated_metrics.empty:
        gated = adaptive_gated_metrics.loc[
            adaptive_gated_metrics.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ].copy()
        if not gated.empty:
            preferred_name = None
            if not gate_consistency_audit.empty:
                stable_names = gate_consistency_audit.loc[
                    gate_consistency_audit.get(
                        "gate_consistency_tier", pd.Series(dtype=str)
                    ).astype(str)
                    == "stable",
                    "model_name",
                ].astype(str)
                preferred = (
                    gated.loc[gated["model_name"].astype(str).isin(set(stable_names))]
                    .sort_values(
                        ["roc_auc", "average_precision"],
                        ascending=False,
                        kind="mergesort",
                    )
                    .head(1)
                )
                if not preferred.empty:
                    preferred_name = str(preferred.iloc[0]["model_name"])
                    gate_row = gate_consistency_audit.loc[
                        gate_consistency_audit["model_name"].astype(str) == preferred_name
                    ].head(1)
                    rows.append(
                        {
                            "model_name": preferred_name,
                            "benchmark_role": "preferred_adaptive_audit",
                            "benchmark_status": "audit_preferred",
                            "benchmark_track": "audit",
                            "model_family": "adaptive_routing",
                            "roc_auc": float(preferred.iloc[0]["roc_auc"]),
                            "average_precision": float(preferred.iloc[0]["average_precision"]),
                            "selection_rank": np.nan,
                            "strict_knownness_acceptance_flag": np.nan,
                            "knownness_matched_gap": np.nan,
                            "source_holdout_gap": np.nan,
                            "leakage_review_required": False,
                            "benchmark_guardrail_status": "adaptive_not_scored",
                            "gate_consistency_tier": str(gate_row.iloc[0]["gate_consistency_tier"])
                            if not gate_row.empty
                            else np.nan,
                            "specialist_weight_lower_half": float(
                                preferred.iloc[0].get("specialist_weight_lower_half", np.nan)
                            ),
                            "selection_rationale": "Preferred adaptive audit because it preserves gate stability while improving low-knownness performance.",
                        }
                    )
            strongest_adaptive = gated.sort_values(
                ["roc_auc", "average_precision"],
                ascending=False,
                kind="mergesort",
            ).head(1)
            if not strongest_adaptive.empty:
                strongest_adaptive_name = str(strongest_adaptive.iloc[0]["model_name"])
                if strongest_adaptive_name != preferred_name:
                    gate_row = (
                        gate_consistency_audit.loc[
                            gate_consistency_audit.get("model_name", pd.Series(dtype=str)).astype(
                                str
                            )
                            == strongest_adaptive_name
                        ].head(1)
                        if not gate_consistency_audit.empty
                        else pd.DataFrame()
                    )
                    rows.append(
                        {
                            "model_name": strongest_adaptive_name,
                            "benchmark_role": "strongest_adaptive_upper_bound",
                            "benchmark_status": "audit_upper_bound",
                            "benchmark_track": "audit",
                            "model_family": "adaptive_routing",
                            "roc_auc": float(strongest_adaptive.iloc[0]["roc_auc"]),
                            "average_precision": float(
                                strongest_adaptive.iloc[0]["average_precision"]
                            ),
                            "selection_rank": np.nan,
                            "strict_knownness_acceptance_flag": np.nan,
                            "knownness_matched_gap": np.nan,
                            "source_holdout_gap": np.nan,
                            "leakage_review_required": False,
                            "benchmark_guardrail_status": "adaptive_not_scored",
                            "gate_consistency_tier": str(gate_row.iloc[0]["gate_consistency_tier"])
                            if not gate_row.empty
                            else np.nan,
                            "specialist_weight_lower_half": float(
                                strongest_adaptive.iloc[0].get(
                                    "specialist_weight_lower_half", np.nan
                                )
                            ),
                            "selection_rationale": "Highest-metric adaptive routing result kept as an upper-bound audit, not as the headline benchmark.",
                        }
                    )

    protocol = pd.DataFrame(rows)
    if protocol.empty:
        return protocol
    status_order = {
        "headline": 0,
        "headline_supporting": 1,
        "audit_preferred": 2,
        "audit_upper_bound": 3,
        "audit_only": 4,
        "control": 5,
    }
    protocol["_status_order"] = (
        protocol["benchmark_status"].map(status_order).fillna(99).astype(int)
    )
    protocol = (
        protocol.sort_values(
            ["_status_order", "roc_auc", "average_precision"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )
    return protocol


def build_official_benchmark_panel(benchmark_protocol: pd.DataFrame) -> pd.DataFrame:
    """Extract the compact public-facing benchmark panel from the full protocol."""
    if benchmark_protocol.empty:
        return pd.DataFrame()
    if "benchmark_role" not in benchmark_protocol.columns:
        return benchmark_protocol.copy().reset_index(drop=True)
    official_roles = {
        "primary_benchmark",
        "governance_benchmark",
        "conservative_benchmark",
        "counts_baseline",
        "source_control",
    }
    panel = benchmark_protocol.loc[
        benchmark_protocol["benchmark_role"].astype(str).isin(official_roles)
    ].copy()
    if panel.empty:
        return panel
    role_order = {
        "primary_benchmark": 0,
        "governance_benchmark": 1,
        "conservative_benchmark": 2,
        "counts_baseline": 3,
        "source_control": 4,
    }
    panel["_official_benchmark_rank"] = panel["benchmark_role"].map(role_order).fillna(99)
    sort_columns = ["_official_benchmark_rank"]
    if "benchmark_track" in panel.columns:
        sort_columns.append("benchmark_track")
    if "benchmark_status" in panel.columns:
        sort_columns.append("benchmark_status")
    panel = (
        panel.sort_values(sort_columns, kind="mergesort")
        .drop(columns="_official_benchmark_rank")
        .reset_index(drop=True)
    )
    return panel


def build_model_selection_scorecard(
    model_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    knownness_matched_validation: pd.DataFrame | None = None,
    group_holdout: pd.DataFrame | None = None,
    single_model_finalist_audit: pd.DataFrame | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Assemble a multi-objective model selection table beyond overall ROC AUC."""
    if model_metrics.empty or predictions.empty or scored.empty:
        return pd.DataFrame()
    model_metrics = _active_model_metrics(model_metrics)
    knownness_matched_validation = (
        knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    )
    group_holdout = group_holdout if group_holdout is not None else pd.DataFrame()
    single_model_finalist_audit = (
        single_model_finalist_audit if single_model_finalist_audit is not None else pd.DataFrame()
    )
    available_metrics = model_metrics.set_index("model_name", drop=False)
    requested_models = [str(name) for name in (model_names or available_metrics.index.tolist())]
    if "spread_label" in scored.columns:
        eligible_scored = scored.loc[
            pd.to_numeric(scored["spread_label"], errors="coerce").notna()
        ].copy()
    else:
        eligible_scored = scored.copy()
    annotated = annotate_knownness_metadata(
        eligible_scored[
            [
                column
                for column in (
                    "backbone_id",
                    "log1p_member_count_train",
                    "log1p_n_countries_train",
                    "refseq_share_train",
                )
                if column in eligible_scored.columns
            ]
        ].drop_duplicates("backbone_id")
    )
    annotated["backbone_id"] = annotated["backbone_id"].astype(str)

    rows: list[dict[str, object]] = []
    for model_name in requested_models:
        if model_name not in available_metrics.index:
            continue
        frame = predictions.loc[predictions["model_name"].astype(str) == model_name].copy()
        if frame.empty:
            continue
        frame["backbone_id"] = frame["backbone_id"].astype(str)
        frame = frame.merge(
            annotated[["backbone_id", "knownness_score", "knownness_half", "knownness_quartile"]],
            on="backbone_id",
            how="left",
            validate="m:1",
        )
        valid = frame.loc[frame["spread_label"].notna() & frame["oof_prediction"].notna()].copy()
        if valid.empty:
            continue
        valid["spread_label"] = valid["spread_label"].astype(int)
        finalist_row = single_model_finalist_audit.loc[
            single_model_finalist_audit.get("model_name", pd.Series(dtype=str)).astype(str)
            == model_name
        ].head(1)
        finalist_metrics = (
            finalist_row.iloc[0] if not finalist_row.empty else pd.Series(dtype=object)
        )
        row = {
            "model_name": model_name,
            "model_track": _safe_model_track(model_name),
            "roc_auc": float(available_metrics.loc[model_name, "roc_auc"]),
            "average_precision": float(available_metrics.loc[model_name, "average_precision"]),
            "prediction_vs_knownness_spearman": _safe_spearman(
                valid["oof_prediction"], valid["knownness_score"]
            ),
        }
        for extra_metric in (
            "ece",
            "expected_calibration_error",
            "max_calibration_error",
            "weighted_classification_cost",
            "decision_utility_score",
            "decision_utility_cost",
            "optimal_decision_threshold",
            "decision_utility_precision",
            "decision_utility_recall",
            "decision_utility_positive_rate",
            "spatial_holdout_roc_auc",
            "selection_adjusted_empirical_p_roc_auc",
        ):
            if extra_metric in available_metrics.columns:
                row[extra_metric] = available_metrics.loc[model_name, extra_metric]
        if pd.isna(row.get("decision_utility_score", np.nan)):
            weighted_cost = pd.to_numeric(
                pd.Series([row.get("weighted_classification_cost", np.nan)]), errors="coerce"
            ).iloc[0]
            if pd.notna(weighted_cost):
                row["decision_utility_score"] = float(-weighted_cost)
        if pd.isna(row.get("decision_utility_cost", np.nan)):
            utility_score = pd.to_numeric(
                pd.Series([row.get("decision_utility_score", np.nan)]), errors="coerce"
            ).iloc[0]
            if pd.notna(utility_score):
                row["decision_utility_cost"] = float(-utility_score)
        for cohort_name, mask in (
            ("lower_half_knownness", valid["knownness_half"].astype(str).eq("lower_half")),
            ("lowest_knownness_quartile", valid["knownness_quartile"].astype(str).eq("q1_lowest")),
        ):
            cohort = valid.loc[mask].copy()
            row[f"{cohort_name}_n_backbones"] = int(len(cohort))
            if not cohort.empty and cohort["spread_label"].nunique() >= 2:
                row[f"{cohort_name}_roc_auc"] = roc_auc_score(
                    cohort["spread_label"], cohort["oof_prediction"]
                )
            else:
                row[f"{cohort_name}_roc_auc"] = np.nan
        matched_row = knownness_matched_validation.loc[
            (
                knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str)).astype(
                    str
                )
                == "__weighted_overall__"
            )
            & (
                knownness_matched_validation.get("model_name", pd.Series(dtype=str)).astype(str)
                == model_name
            )
        ]
        if not matched_row.empty:
            if "weighted_mean_roc_auc" in matched_row.columns and pd.notna(
                matched_row.iloc[0]["weighted_mean_roc_auc"]
            ):
                row["matched_knownness_weighted_roc_auc"] = float(
                    matched_row.iloc[0]["weighted_mean_roc_auc"]
                )
            elif "roc_auc" in matched_row.columns and pd.notna(matched_row.iloc[0]["roc_auc"]):
                row["matched_knownness_weighted_roc_auc"] = float(matched_row.iloc[0]["roc_auc"])
            else:
                row["matched_knownness_weighted_roc_auc"] = np.nan
        else:
            row["matched_knownness_weighted_roc_auc"] = np.nan
        if pd.isna(row["matched_knownness_weighted_roc_auc"]) and not finalist_metrics.empty:
            row["matched_knownness_weighted_roc_auc"] = pd.to_numeric(
                pd.Series([finalist_metrics.get("matched_knownness_weighted_roc_auc", np.nan)]),
                errors="coerce",
            ).iloc[0]
        source_rows = group_holdout.loc[
            (
                group_holdout.get("group_column", pd.Series(dtype=str)).astype(str)
                == "dominant_source"
            )
            & (group_holdout.get("model_name", pd.Series(dtype=str)).astype(str) == model_name)
            & (group_holdout.get("status", pd.Series(dtype=str)).astype(str) == "ok")
        ].copy()
        if not source_rows.empty:
            weights = pd.to_numeric(source_rows["n_test_backbones"], errors="coerce").fillna(0.0)
            aucs = pd.to_numeric(source_rows["roc_auc"], errors="coerce")
            if float(weights.sum()) > 0:
                row["source_holdout_weighted_roc_auc"] = float(np.average(aucs, weights=weights))
            else:
                row["source_holdout_weighted_roc_auc"] = float(aucs.mean())
            row["source_holdout_macro_roc_auc"] = float(aucs.mean())
        else:
            row["source_holdout_weighted_roc_auc"] = np.nan
            row["source_holdout_macro_roc_auc"] = np.nan
        if pd.isna(row["source_holdout_weighted_roc_auc"]) and not finalist_metrics.empty:
            row["source_holdout_weighted_roc_auc"] = pd.to_numeric(
                pd.Series([finalist_metrics.get("source_holdout_weighted_roc_auc", np.nan)]),
                errors="coerce",
            ).iloc[0]
        if pd.isna(row.get("spatial_holdout_roc_auc", np.nan)) and not finalist_metrics.empty:
            row["spatial_holdout_roc_auc"] = pd.to_numeric(
                pd.Series([finalist_metrics.get("spatial_holdout_weighted_roc_auc", np.nan)]),
                errors="coerce",
            ).iloc[0]
        if pd.isna(row.get("ece", np.nan)) and not finalist_metrics.empty:
            row["ece"] = pd.to_numeric(
                pd.Series([finalist_metrics.get("ece", np.nan)]),
                errors="coerce",
            ).iloc[0]
        if (
            pd.isna(row.get("selection_adjusted_empirical_p_roc_auc", np.nan))
            and not finalist_metrics.empty
        ):
            row["selection_adjusted_empirical_p_roc_auc"] = pd.to_numeric(
                pd.Series([finalist_metrics.get("selection_adjusted_empirical_p_roc_auc", np.nan)]),
                errors="coerce",
            ).iloc[0]
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard

    scoring_directions = {
        "roc_auc": False,
        "average_precision": False,
        "lower_half_knownness_roc_auc": False,
        "lowest_knownness_quartile_roc_auc": False,
        "matched_knownness_weighted_roc_auc": False,
        "source_holdout_macro_roc_auc": False,
        "prediction_vs_knownness_spearman": True,
        "decision_utility_score": False,
    }
    component_columns: list[str] = []
    for column, ascending in scoring_directions.items():
        values = pd.to_numeric(
            scorecard.get(column, pd.Series(np.nan, index=scorecard.index)), errors="coerce"
        )
        valid = values.notna()
        component_column = f"{column}_component_score"
        scorecard[component_column] = np.nan
        if valid.any():
            ranks = values.loc[valid].rank(method="average", ascending=ascending)
            if valid.sum() == 1:
                component_scores = pd.Series(1.0, index=ranks.index)
            else:
                component_scores = 1.0 - ((ranks - 1.0) / (valid.sum() - 1.0))
            scorecard.loc[valid, component_column] = component_scores.astype(float)
        component_columns.append(component_column)

    available_component_columns = [
        column for column in component_columns if scorecard[column].notna().any()
    ]
    if available_component_columns:
        scorecard["selection_composite_score"] = (
            scorecard[available_component_columns].fillna(0.0).mean(axis=1)
        )
        scorecard["selection_metric_count"] = (
            scorecard[available_component_columns].notna().sum(axis=1).astype(int)
        )
        scorecard["selection_missing_metric_count"] = (
            len(available_component_columns) - scorecard["selection_metric_count"]
        ).astype(int)
    else:
        scorecard["selection_composite_score"] = np.nan
        scorecard["selection_metric_count"] = 0
        scorecard["selection_missing_metric_count"] = 0
    scorecard["leakage_review_required"] = (
        pd.to_numeric(
            scorecard.get("roc_auc", pd.Series(np.nan, index=scorecard.index)), errors="coerce"
        )
        .fillna(0.0)
        .ge(0.90)
    )
    scorecard["leakage_review_reason"] = np.where(
        scorecard["leakage_review_required"],
        "roc_auc_ge_0p90_on_current_feature_universe",
        "",
    )
    matched_gap = pd.to_numeric(
        scorecard.get(
            "matched_knownness_weighted_roc_auc", pd.Series(np.nan, index=scorecard.index)
        ),
        errors="coerce",
    ) - pd.to_numeric(
        scorecard.get("roc_auc", pd.Series(np.nan, index=scorecard.index)), errors="coerce"
    )
    source_gap = pd.to_numeric(
        scorecard.get("source_holdout_weighted_roc_auc", pd.Series(np.nan, index=scorecard.index)),
        errors="coerce",
    ) - pd.to_numeric(
        scorecard.get("roc_auc", pd.Series(np.nan, index=scorecard.index)), errors="coerce"
    )
    scorecard["knownness_matched_gap"] = matched_gap
    scorecard["source_holdout_gap"] = source_gap
    knownness_gap_loss = matched_gap.abs().fillna(1.0)
    source_gap_loss = source_gap.abs().fillna(1.0)
    scorecard["guardrail_loss"] = knownness_gap_loss + source_gap_loss
    scorecard["governance_priority_score"] = (
        pd.to_numeric(
            scorecard.get("roc_auc", pd.Series(np.nan, index=scorecard.index)), errors="coerce"
        ).fillna(0.0)
        - scorecard["guardrail_loss"].fillna(1.0)
        - 0.25 * scorecard["leakage_review_required"].astype(float)
    )
    scorecard["strict_knownness_acceptance_flag"] = (
        scorecard["matched_knownness_weighted_roc_auc"].notna()
        & scorecard["source_holdout_weighted_roc_auc"].notna()
        & scorecard["matched_knownness_weighted_roc_auc"].ge(scorecard["roc_auc"] - 0.005)
        & scorecard["source_holdout_weighted_roc_auc"].ge(scorecard["roc_auc"] - 0.005)
    )

    ece_series = pd.to_numeric(
        scorecard.get(
            "ece",
            scorecard.get("expected_calibration_error", pd.Series(np.nan, index=scorecard.index)),
        ),
        errors="coerce",
    )
    spatial_holdout_auc = pd.to_numeric(
        scorecard.get("spatial_holdout_roc_auc", pd.Series(np.nan, index=scorecard.index)),
        errors="coerce",
    )
    selection_adjusted_p = pd.to_numeric(
        scorecard.get(
            "selection_adjusted_empirical_p_roc_auc", pd.Series(np.nan, index=scorecard.index)
        ),
        errors="coerce",
    )
    scorecard["spatial_holdout_gap"] = spatial_holdout_auc - pd.to_numeric(
        scorecard.get("roc_auc", pd.Series(np.nan, index=scorecard.index)), errors="coerce"
    )
    scorecard["matched_knownness_gate_pass"] = scorecard[
        "matched_knownness_weighted_roc_auc"
    ].notna() & matched_gap.ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["matched_knownness_gap_min"])
    scorecard["source_holdout_gate_pass"] = scorecard[
        "source_holdout_weighted_roc_auc"
    ].notna() & source_gap.ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["source_holdout_gap_min"])
    scorecard["spatial_holdout_gate_pass"] = spatial_holdout_auc.notna() & scorecard[
        "spatial_holdout_gap"
    ].ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["spatial_holdout_gap_min"])
    scorecard["calibration_gate_pass"] = ece_series.notna() & ece_series.le(
        FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["ece_max"]
    )
    scorecard["selection_adjusted_gate_pass"] = selection_adjusted_p.notna() & (
        selection_adjusted_p.le(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["selection_adjusted_p_max"])
    )
    scorecard["leakage_review_gate_pass"] = ~scorecard["leakage_review_required"].fillna(
        False
    ).astype(bool)
    required_gate_inputs = {
        "matched_knownness_weighted_roc_auc": scorecard[
            "matched_knownness_weighted_roc_auc"
        ].notna(),
        "source_holdout_weighted_roc_auc": scorecard["source_holdout_weighted_roc_auc"].notna(),
        "spatial_holdout_roc_auc": spatial_holdout_auc.notna(),
        "ece": ece_series.notna(),
        "selection_adjusted_empirical_p_roc_auc": selection_adjusted_p.notna(),
    }
    scorecard["scientific_acceptance_scored"] = pd.DataFrame(required_gate_inputs).all(axis=1)
    scorecard["scientific_acceptance_flag"] = (
        scorecard["scientific_acceptance_scored"]
        & scorecard["matched_knownness_gate_pass"]
        & scorecard["source_holdout_gate_pass"]
        & scorecard["spatial_holdout_gate_pass"]
        & scorecard["calibration_gate_pass"]
        & scorecard["selection_adjusted_gate_pass"]
        & scorecard["leakage_review_gate_pass"]
    )

    def _scientific_acceptance_reason(row: pd.Series) -> str:
        missing_tokens: list[str] = []
        for column_name in required_gate_inputs:
            value = row.get(column_name, np.nan)
            if column_name == "ece":
                value = row.get("ece", row.get("expected_calibration_error", np.nan))
            if pd.isna(value):
                missing_tokens.append(column_name)
        if missing_tokens:
            return "missing:" + ",".join(sorted(missing_tokens))
        failed_tokens = [
            label
            for label, column_name in [
                ("matched_knownness", "matched_knownness_gate_pass"),
                ("source_holdout", "source_holdout_gate_pass"),
                ("spatial_holdout", "spatial_holdout_gate_pass"),
                ("calibration", "calibration_gate_pass"),
                ("selection_adjusted_null", "selection_adjusted_gate_pass"),
                ("leakage_review", "leakage_review_gate_pass"),
            ]
            if not bool(row.get(column_name, False))
        ]
        if failed_tokens:
            return "fail:" + ",".join(failed_tokens)
        return "pass"

    scorecard["scientific_acceptance_status"] = np.where(
        scorecard["scientific_acceptance_scored"],
        np.where(scorecard["scientific_acceptance_flag"], "pass", "fail"),
        "not_scored",
    )
    scorecard["selection_metrics_complete"] = (
        scorecard["matched_knownness_weighted_roc_auc"].notna()
        & scorecard["source_holdout_weighted_roc_auc"].notna()
    )
    spatial_holdout_values = pd.to_numeric(
        scorecard.get("spatial_holdout_roc_auc", pd.Series(np.nan, index=scorecard.index)),
        errors="coerce",
    )
    if spatial_holdout_values.notna().any():
        scorecard["selection_metrics_complete"] &= spatial_holdout_values.notna()
    ece_values = pd.to_numeric(
        scorecard.get(
            "ece",
            scorecard.get("expected_calibration_error", pd.Series(np.nan, index=scorecard.index)),
        ),
        errors="coerce",
    )
    if ece_values.notna().any():
        scorecard["selection_metrics_complete"] &= ece_values.notna()
    selection_adjusted_values = pd.to_numeric(
        scorecard.get(
            "selection_adjusted_empirical_p_roc_auc",
            pd.Series(np.nan, index=scorecard.index),
        ),
        errors="coerce",
    )
    if selection_adjusted_values.notna().any():
        scorecard["selection_metrics_complete"] &= selection_adjusted_values.notna()
    scorecard["scientific_acceptance_failed_criteria"] = scorecard.apply(
        _scientific_acceptance_reason, axis=1
    )
    scorecard = scorecard.sort_values(
        [
            "selection_metrics_complete",
            "selection_composite_score",
            "roc_auc",
            "average_precision",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    scorecard["selection_rank"] = pd.Series(np.nan, index=scorecard.index, dtype="Int64")
    rankable = scorecard["selection_metrics_complete"].fillna(False).astype(bool)
    if rankable.any():
        scorecard.loc[rankable, "selection_rank"] = pd.Series(
            np.arange(1, int(rankable.sum()) + 1), index=scorecard.index[rankable], dtype="Int64"
        )
    scorecard["track_rank"] = (
        scorecard.groupby(scorecard["model_track"].astype(str), sort=False).cumcount() + 1
    ).astype(int)

    def _assign_track_specific_rank(
        column_name: str,
        mask: pd.Series,
        sort_columns: list[str],
        ascending: list[bool],
    ) -> None:
        scorecard[column_name] = pd.Series(np.nan, index=scorecard.index, dtype="Int64")
        if not mask.any():
            return
        ranked_index = (
            scorecard.loc[mask]
            .sort_values(
                sort_columns,
                ascending=ascending,
                kind="mergesort",
            )
            .index
        )
        scorecard.loc[ranked_index, column_name] = pd.Series(
            np.arange(1, len(ranked_index) + 1),
            index=ranked_index,
            dtype="Int64",
        )

    track_values = scorecard["model_track"].astype(str)
    _assign_track_specific_rank(
        "discovery_track_rank",
        track_values.eq("discovery") & rankable,
        ["selection_composite_score", "roc_auc", "average_precision"],
        [False, False, False],
    )
    _assign_track_specific_rank(
        "governance_track_rank",
        track_values.eq("governance") & rankable,
        ["governance_priority_score", "guardrail_loss", "roc_auc", "average_precision"],
        [False, True, False, False],
    )
    _assign_track_specific_rank(
        "baseline_track_rank",
        track_values.eq("baseline") & rankable,
        ["selection_composite_score", "roc_auc", "average_precision"],
        [False, False, False],
    )
    scorecard["governance_rank"] = (
        scorecard["governance_priority_score"]
        .where(rankable)
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return scorecard


def build_frozen_scientific_acceptance_audit(
    model_selection_scorecard: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Summarize the frozen acceptance gate for official benchmark surfaces."""
    if model_selection_scorecard.empty:
        return pd.DataFrame()
    working = model_selection_scorecard.copy()
    if "model_track" not in working.columns:
        working["model_track"] = working["model_name"].map(_safe_model_track)
    if model_names is None:
        working = working.loc[
            working["model_track"].astype(str).isin({"baseline", "discovery", "governance"})
        ].copy()
    else:
        working = working.loc[
            working["model_name"].astype(str).isin({str(name) for name in model_names})
        ].copy()
    if working.empty:
        return pd.DataFrame()
    for column, value in FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS.items():
        working[column] = float(value)
    ordered_columns = [
        "model_name",
        "model_track",
        "selection_rank",
        "roc_auc",
        "average_precision",
        "matched_knownness_weighted_roc_auc",
        "knownness_matched_gap",
        "source_holdout_weighted_roc_auc",
        "source_holdout_gap",
        "spatial_holdout_roc_auc",
        "spatial_holdout_gap",
        "ece",
        "selection_adjusted_empirical_p_roc_auc",
        "matched_knownness_gate_pass",
        "source_holdout_gate_pass",
        "spatial_holdout_gate_pass",
        "calibration_gate_pass",
        "selection_adjusted_gate_pass",
        "leakage_review_gate_pass",
        "scientific_acceptance_scored",
        "scientific_acceptance_flag",
        "scientific_acceptance_status",
        "scientific_acceptance_failed_criteria",
        "matched_knownness_gap_min",
        "source_holdout_gap_min",
        "spatial_holdout_gap_min",
        "ece_max",
        "selection_adjusted_p_max",
    ]
    for column in ordered_columns:
        if column not in working.columns:
            working[column] = np.nan
    working = working[ordered_columns].sort_values(
        ["model_track", "selection_rank", "roc_auc", "average_precision"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    return working.reset_index(drop=True)


def _coerce_feature_set(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    if text.startswith(("(", "[")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return tuple(str(item) for item in parsed)
    return tuple(token.strip() for token in text.split(",") if token.strip())


def build_single_model_pareto_finalists(
    single_model_pareto_screen: pd.DataFrame,
    *,
    max_finalists: int = 3,
) -> pd.DataFrame:
    """Keep a bounded finalist set from the Pareto screen."""
    if single_model_pareto_screen.empty:
        return pd.DataFrame()
    shortlist = build_pareto_shortlist(single_model_pareto_screen)
    if shortlist.empty:
        return shortlist
    working = shortlist.sort_values(
        [
            "weighted_objective_score",
            "reliability_score",
            "roc_auc",
            "average_precision",
            "model_name",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if max_finalists > 0:
        working = working.head(int(max_finalists)).copy()
    return working.reset_index(drop=True)


def build_single_model_selection_adjusted_permutation_null(
    scored: pd.DataFrame,
    finalists: pd.DataFrame,
    *,
    n_permutations: int = 200,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a finalist-scoped selection-adjusted null for arbitrary candidate feature sets."""
    if finalists.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible_mask = scored.get("spread_label", pd.Series(index=scored.index)).notna()
    if not bool(eligible_mask.any()):
        return pd.DataFrame(), pd.DataFrame()
    y_true = scored.loc[eligible_mask, "spread_label"].astype(int).to_numpy(dtype=int)
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(), pd.DataFrame()

    prepared_inputs: dict[str, tuple[pd.DataFrame, list[str], dict[str, object]]] = {}
    for finalist in finalists.to_dict("records"):
        model_name = str(finalist.get("model_name", "")).strip()
        if not model_name:
            continue
        feature_set = list(_coerce_feature_set(finalist.get("feature_set")))
        if not feature_set:
            continue
        parent_model_name = str(finalist.get("parent_model_name", model_name))
        fit_kwargs = _model_fit_kwargs(parent_model_name)
        eligible = _ensure_feature_columns(scored, feature_set).loc[eligible_mask].copy()
        eligible["spread_label"] = eligible["spread_label"].astype(int)
        prepared_inputs[model_name] = (eligible, feature_set, fit_kwargs)
    if not prepared_inputs:
        return pd.DataFrame(), pd.DataFrame()

    observed_metrics: dict[str, tuple[float, float]] = {}
    for model_name, (eligible, feature_set, fit_kwargs) in prepared_inputs.items():
        preds, y = _oof_predictions_from_eligible(
            eligible,
            columns=feature_set,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
        )
        observed_metrics[model_name] = (
            float(roc_auc_score(y, preds)),
            float(average_precision(y, preds)),
        )

    rng = np.random.default_rng(seed)
    detail_rows: list[dict[str, object]] = []
    selected_null_aucs: list[float] = []
    selected_null_aps: list[float] = []
    selected_model_names: list[str] = []

    for permutation_index in range(1, max(int(n_permutations), 0) + 1):
        permuted_y = rng.permutation(y_true)
        fold_groups = _stratified_folds(
            permuted_y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        permutation_rows: list[dict[str, object]] = []
        for model_name, (eligible, feature_set, fit_kwargs) in prepared_inputs.items():
            preds, _ = _oof_predictions_from_eligible(
                eligible,
                columns=feature_set,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                fit_kwargs=fit_kwargs,
                y_override=permuted_y,
                folds_per_repeat=fold_groups,
            )
            permutation_rows.append(
                {
                    "model_name": model_name,
                    "null_roc_auc": float(roc_auc_score(permuted_y, preds)),
                    "null_average_precision": float(average_precision(permuted_y, preds)),
                }
            )
        permutation_frame = pd.DataFrame(permutation_rows).sort_values(
            ["null_roc_auc", "null_average_precision", "model_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected_row = permutation_frame.iloc[0]
        selected_null_aucs.append(float(selected_row["null_roc_auc"]))
        selected_null_aps.append(float(selected_row["null_average_precision"]))
        selected_model_names.append(str(selected_row["model_name"]))
        detail_rows.append(
            {
                "permutation_index": permutation_index,
                "selection_scope": "single_model_pareto_finalists",
                "n_models_in_scope": int(len(prepared_inputs)),
                "selected_model_name": str(selected_row["model_name"]),
                "selected_null_roc_auc": float(selected_row["null_roc_auc"]),
                "selected_null_average_precision": float(selected_row["null_average_precision"]),
            }
        )

    if not selected_null_aucs:
        return pd.DataFrame(detail_rows), pd.DataFrame()

    winner_counts = pd.Series(selected_model_names, dtype=object).value_counts()
    modal_selected_model = str(winner_counts.index[0]) if not winner_counts.empty else ""
    modal_selected_share = (
        float(winner_counts.iloc[0] / max(len(selected_model_names), 1))
        if not winner_counts.empty
        else 0.0
    )

    summary_rows: list[dict[str, object]] = []
    for model_name, (observed_auc, observed_ap) in observed_metrics.items():
        summary_rows.append(
            {
                "model_name": model_name,
                "null_protocol": "selection_adjusted_single_model_finalist_refit",
                "selection_scope": "single_model_pareto_finalists",
                "n_models_in_scope": int(len(prepared_inputs)),
                "n_permutations": int(n_permutations),
                "observed_roc_auc": observed_auc,
                "observed_average_precision": observed_ap,
                "null_roc_auc_mean": float(np.mean(selected_null_aucs)),
                "null_roc_auc_std": float(np.std(selected_null_aucs)),
                "null_roc_auc_q975": float(np.quantile(selected_null_aucs, 0.975)),
                "selection_adjusted_empirical_p_roc_auc": float(
                    (1 + sum(value >= observed_auc for value in selected_null_aucs))
                    / (len(selected_null_aucs) + 1)
                ),
                "null_average_precision_mean": float(np.mean(selected_null_aps)),
                "null_average_precision_std": float(np.std(selected_null_aps)),
                "null_average_precision_q975": float(np.quantile(selected_null_aps, 0.975)),
                "selection_adjusted_empirical_p_average_precision": float(
                    (1 + sum(value >= observed_ap for value in selected_null_aps))
                    / (len(selected_null_aps) + 1)
                ),
                "modal_selected_model_name": modal_selected_model,
                "modal_selected_model_share": modal_selected_share,
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_single_model_finalist_audit(
    scored: pd.DataFrame,
    finalists: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    selection_adjusted_n_permutations: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Re-score Pareto finalists under a heavier audit protocol."""
    if finalists.empty:
        return pd.DataFrame()
    heavy = build_single_model_pareto_screen(
        scored,
        family=finalists,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )
    if heavy.empty:
        return heavy
    _, selection_adjusted_summary = build_single_model_selection_adjusted_permutation_null(
        scored,
        finalists,
        n_permutations=selection_adjusted_n_permutations,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )
    if not selection_adjusted_summary.empty:
        heavy = heavy.merge(
            selection_adjusted_summary[
                [
                    "model_name",
                    "selection_adjusted_empirical_p_roc_auc",
                    "n_permutations",
                ]
            ],
            on="model_name",
            how="left",
        )
    else:
        heavy["selection_adjusted_empirical_p_roc_auc"] = np.nan
        heavy["n_permutations"] = 0

    heavy["matched_knownness_gate_pass"] = pd.to_numeric(
        heavy["knownness_matched_gap"], errors="coerce"
    ).ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["matched_knownness_gap_min"])
    heavy["source_holdout_gate_pass"] = pd.to_numeric(
        heavy["source_holdout_gap"], errors="coerce"
    ).ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["source_holdout_gap_min"])
    heavy["spatial_holdout_gate_pass"] = pd.to_numeric(
        heavy["spatial_holdout_gap"], errors="coerce"
    ).ge(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["spatial_holdout_gap_min"])
    heavy["calibration_gate_pass"] = pd.to_numeric(heavy["ece"], errors="coerce").le(
        FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["ece_max"]
    )
    heavy["selection_adjusted_gate_pass"] = pd.to_numeric(
        heavy["selection_adjusted_empirical_p_roc_auc"], errors="coerce"
    ).le(FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["selection_adjusted_p_max"])
    required_columns = [
        "knownness_matched_gap",
        "source_holdout_gap",
        "spatial_holdout_gap",
        "ece",
        "selection_adjusted_empirical_p_roc_auc",
    ]
    heavy["scientific_acceptance_scored"] = heavy[required_columns].notna().all(axis=1)
    heavy["scientific_acceptance_flag"] = (
        heavy["scientific_acceptance_scored"]
        & heavy["matched_knownness_gate_pass"]
        & heavy["source_holdout_gate_pass"]
        & heavy["spatial_holdout_gate_pass"]
        & heavy["calibration_gate_pass"]
        & heavy["selection_adjusted_gate_pass"]
    )

    def _single_model_acceptance_reason(row: pd.Series) -> str:
        missing = [column for column in required_columns if pd.isna(row.get(column, np.nan))]
        if missing:
            return "missing:" + ",".join(sorted(missing))
        failed = [
            label
            for label, column_name in [
                ("matched_knownness", "matched_knownness_gate_pass"),
                ("source_holdout", "source_holdout_gate_pass"),
                ("spatial_holdout", "spatial_holdout_gate_pass"),
                ("calibration", "calibration_gate_pass"),
                ("selection_adjusted_null", "selection_adjusted_gate_pass"),
            ]
            if not bool(row.get(column_name, False))
        ]
        if failed:
            return "fail:" + ",".join(failed)
        return "pass"

    heavy["scientific_acceptance_status"] = np.where(
        heavy["scientific_acceptance_scored"],
        np.where(heavy["scientific_acceptance_flag"], "pass", "fail"),
        "not_scored",
    )
    heavy["scientific_acceptance_failed_criteria"] = heavy.apply(
        _single_model_acceptance_reason,
        axis=1,
    )
    heavy = add_failure_severity(heavy)
    return heavy.sort_values(
        ["failure_severity", "roc_auc", "average_precision", "model_name"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_single_model_official_decision(finalists: pd.DataFrame) -> pd.DataFrame:
    """Choose one official model from finalists using scientific-first ordering."""
    if finalists.empty:
        return pd.DataFrame()
    working = finalists.copy()
    if "failure_severity" not in working.columns:
        working = add_failure_severity(working)
    working["acceptance_sort"] = (
        working.get("scientific_acceptance_status", pd.Series(index=working.index, dtype=object))
        .astype(str)
        .eq("pass")
        .astype(int)
    )
    ranked = working.sort_values(
        [
            "acceptance_sort",
            "failure_severity",
            "roc_auc",
            "average_precision",
            "compute_efficiency_score",
            "weighted_objective_score",
            "screen_fit_seconds",
            "model_name",
        ],
        ascending=[False, True, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    winner = ranked.iloc[0]
    acceptance_status = str(winner.get("scientific_acceptance_status", "not_scored"))
    if acceptance_status == "pass":
        decision_reason = "accepted_with_best_reliability_power_tradeoff"
    elif str(winner.get("scientific_acceptance_failed_criteria", "")).startswith("fail:"):
        decision_reason = "lowest_failure_severity_with_competitive_auc"
    else:
        decision_reason = "best_available_weighted_tradeoff_pending_full_acceptance"
    return pd.DataFrame(
        [
            {
                "official_model_name": str(winner.get("model_name", "")),
                "decision_reason": decision_reason,
                "scientific_acceptance_status": acceptance_status,
                "scientific_acceptance_failed_criteria": str(
                    winner.get("scientific_acceptance_failed_criteria", "")
                ),
                "failure_severity": float(winner.get("failure_severity", np.nan)),
                "roc_auc": float(winner.get("roc_auc", np.nan)),
                "average_precision": float(winner.get("average_precision", np.nan)),
                "weighted_objective_score": float(winner.get("weighted_objective_score", np.nan)),
                "screen_fit_seconds": float(winner.get("screen_fit_seconds", np.nan)),
                "compute_efficiency_score": float(winner.get("compute_efficiency_score", np.nan)),
                "selected_from_n_finalists": int(len(ranked)),
            }
        ]
    )
