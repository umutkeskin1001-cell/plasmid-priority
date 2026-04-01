"""Model audit tables for jury-facing validation and explainability."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    annotate_knownness_metadata,
    evaluate_feature_columns,
    evaluate_model_name,
    fit_full_model_predictions,
    fit_predict_model_holdout,
    get_primary_model_name,
)
from plasmid_priority.modeling.module_a import (
    _compute_sample_weight,
    _eligible_xy,
    _model_fit_kwargs,
    _oof_predictions,
    _standardize_apply,
    _standardize_fit,
    _stratified_folds,
)
from plasmid_priority.scoring import recompute_priority_from_reference
from plasmid_priority.utils.dataframe import coalescing_left_merge
from plasmid_priority.validation import (
    average_precision,
    average_precision_enrichment,
    average_precision_lift,
    brier_score,
    expected_calibration_error,
    paired_auc_delong,
    paired_bootstrap_deltas,
    paired_bootstrap_delta,
    positive_prevalence,
    roc_auc_score,
)


def _stable_unit_interval(text: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    return float(int.from_bytes(digest[:8], "big") / 2**64)


def build_model_family_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the most decision-relevant models and label their evidence role."""
    selected = [
        ("source_only", "easy_proxy", "source composition only"),
        ("baseline_both", "easy_proxy", "training visibility counts only"),
        ("full_priority", "handcrafted_score", "counts plus arithmetic conservative priority index"),
        ("T_plus_H_plus_A", "legacy_biological_core", "legacy support-adjusted T/H/A components"),
        ("bio_clean_priority", "biological_core", "raw biological T/A plus host specialization and oriT support"),
        ("parsimonious_priority", "legacy_published_primary", "support-adjusted T/H/A plus coherence; retained as the legacy interpretable benchmark"),
        ("natural_auc_priority", "augmented_biological_core", "biological core plus external host-range, backbone purity, assignment confidence, mash-based novelty, and replicon architecture"),
        ("phylogeny_aware_priority", "phylogeny_augmented_core", "augmented biological core with host specialization replaced by taxonomy-aware phylogenetic host breadth"),
        ("structured_signal_priority", "structure_augmented_core", "phylogeny-aware biological core plus host evenness and recurrent AMR structure"),
        ("ecology_clinical_priority", "eco_clinical_augmented_core", "augmented biological core plus clinical-context prevalence and ecological-context diversity"),
        ("knownness_robust_priority", "knownness_robust_core", "augmented biological core plus clinical-context prevalence, ecological-context diversity, recurrent AMR structure, and pMLST coherence under class+knownness balancing"),
        ("support_calibrated_priority", "support_calibrated_core", "knownness-robust biological core plus explicit host-range support, pMLST presence, and AMR support depth for sparse-annotation error recovery"),
        ("support_synergy_priority", "published_primary", "support-calibrated biological core plus metadata support depth, external host-range magnitude, and host-range x transfer synergy; chosen as the current primary benchmark"),
        ("host_transfer_synergy_priority", "error_focused_augmented_core", "knownness-robust biological core plus explicit host-transfer synergy and external host-range support for sparse-backbone error recovery"),
        ("threat_architecture_priority", "threat_architecture_audit", "host-transfer augmented biological core plus AMR clinical-threat burden and replicon multiplicity for sparse-backbone error recovery"),
        ("contextual_bio_priority", "contextual_biological_core", "augmented biological core plus PMLST coherence and eco-clinical context diversity"),
        ("visibility_adjusted_priority", "deconfounded_evidence", "raw biological T/A plus host specialization and visibility-adjusted support residuals"),
        ("balanced_evidence_priority", "evidence_aware", "raw biological axes plus axis-specific evidence depth"),
        ("evidence_aware_priority", "support_heavy_evidence", "raw mobility plus global evidence-depth support"),
        ("proxy_light_priority", "legacy_integrated", "legacy support-adjusted integrated model"),
        ("enhanced_priority", "proxy_integrated", "support-adjusted model plus explicit count proxies"),
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
    reference_auc = float(available.loc[reference_model, "roc_auc"]) if reference_model in available.index else np.nan
    enhanced_auc = float(available.loc["enhanced_priority", "roc_auc"]) if "enhanced_priority" in available.index else np.nan
    for model_name, evidence_role, evidence_summary in selected:
        if model_name not in available.index:
            continue
        row = available.loc[model_name].to_dict()
        row["evidence_role"] = evidence_role
        row["evidence_summary"] = evidence_summary
        row["primary_reference_model"] = reference_model
        row["delta_auc_vs_primary_reference"] = float(row["roc_auc"]) - reference_auc if np.isfinite(reference_auc) else np.nan
        row["delta_auc_vs_enhanced_priority"] = float(row["roc_auc"]) - enhanced_auc if np.isfinite(enhanced_auc) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


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
    h_feature_names = {"H_eff_norm", "H_breadth_norm", "H_specialization_norm", "H_support_norm", "H_support_norm_residual"}
    row: dict[str, object] = {
        "n_scored_backbones": int(len(scored)),
        "n_eligible_backbones": int(len(eligible)),
        "nonzero_h_fraction_all": float(scored["H_eff_norm"].fillna(0.0).gt(0.0).mean()),
        "nonzero_h_fraction_eligible": float(eligible["H_eff_norm"].fillna(0.0).gt(0.0).mean()),
        "h_eff_norm_mean_positive": float(eligible.loc[eligible["spread_label"] == 1, "H_eff_norm"].fillna(0.0).mean()),
        "h_eff_norm_mean_negative": float(eligible.loc[eligible["spread_label"] == 0, "H_eff_norm"].fillna(0.0).mean()),
        "h_eff_norm_vs_spread_label_spearman": _safe_spearman(eligible["H_eff_norm"], eligible["spread_label"]),
        "h_eff_norm_vs_member_count_train_spearman": _safe_spearman(
            eligible["H_eff_norm"],
            eligible["log1p_member_count_train"],
        ),
        "h_eff_norm_vs_n_countries_train_spearman": _safe_spearman(
            eligible["H_eff_norm"],
            eligible["log1p_n_countries_train"],
        ),
        "h_eff_norm_top100_mean": float(scored.sort_values(ranking_column, ascending=False).head(100)["H_eff_norm"].fillna(0.0).mean()),
        "h_eff_norm_bottom100_mean": float(scored.sort_values(ranking_column, ascending=True).head(100)["H_eff_norm"].fillna(0.0).mean()),
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
            scored.sort_values(ranking_column, ascending=False).head(100)["H_breadth_norm"].fillna(0.0).mean()
        )
        row["h_breadth_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True).head(100)["H_breadth_norm"].fillna(0.0).mean()
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
            scored.sort_values(ranking_column, ascending=False).head(100)["H_specialization_norm"].fillna(0.0).mean()
        )
        row["h_specialization_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True).head(100)["H_specialization_norm"].fillna(0.0).mean()
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
        row["h_raw_vs_spread_label_spearman"] = _safe_spearman(eligible["H_raw"], eligible["spread_label"])
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
            scored.sort_values(ranking_column, ascending=False).head(100)["H_phylogenetic_norm"].fillna(0.0).mean()
        )
        row["h_phylogenetic_norm_bottom100_mean"] = float(
            scored.sort_values(ranking_column, ascending=True).head(100)["H_phylogenetic_norm"].fillna(0.0).mean()
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
        matches = coefficient_table.loc[coefficient_table["feature_name"].isin(h_feature_names)].copy()
        if "abs_coefficient" not in matches.columns and "coefficient" in matches.columns:
            matches["abs_coefficient"] = matches["coefficient"].astype(float).abs()
        row["primary_model_uses_h"] = bool(len(matches))
        row["primary_model_h_features"] = ",".join(matches["feature_name"].astype(str).tolist()) if not matches.empty else ""
        row["primary_model_h_total_abs_coefficient"] = float(matches["abs_coefficient"].sum()) if not matches.empty else np.nan
        if not matches.empty:
            dominant = matches.sort_values("abs_coefficient", ascending=False).iloc[0]
            row["primary_model_h_primary_feature"] = str(dominant["feature_name"])
            row["primary_model_h_coefficient"] = float(dominant["coefficient"])
    else:
        row["primary_model_uses_h"] = False
    if dropout_table is not None and not dropout_table.empty:
        matches = dropout_table.loc[dropout_table["feature_name"].isin(h_feature_names), "roc_auc_drop_vs_full"]
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
            & mobsuite_detail["mobsuite_reported_host_range_taxid_count"].fillna(0).astype(float).gt(0.0)
        ].copy()
        row["mobsuite_supported_backbones"] = int(len(supported))
        if not supported.empty and "priority_group" in supported.columns:
            group_summary = supported.groupby("priority_group", as_index=True)["mobsuite_reported_host_range_taxid_count"].agg(["count", "mean"])
            row["mobsuite_high_literature_supported_n"] = int(group_summary.loc["high", "count"]) if "high" in group_summary.index else 0
            row["mobsuite_low_literature_supported_n"] = int(group_summary.loc["low", "count"]) if "low" in group_summary.index else 0
            row["mobsuite_high_mean_reported_host_range_taxid_count"] = float(group_summary.loc["high", "mean"]) if "high" in group_summary.index else np.nan
            row["mobsuite_low_mean_reported_host_range_taxid_count"] = float(group_summary.loc["low", "mean"]) if "low" in group_summary.index else np.nan
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
    merged = merged.merge(scored[available_meta_columns], on="backbone_id", how="left", validate="1:1")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    for column in ("operational_priority_index", "bio_priority_index", "evidence_support_index", "priority_index"):
        if column not in merged.columns:
            merged[column] = np.nan

    merged["operational_priority_index"] = merged["operational_priority_index"].fillna(merged["priority_index"])
    merged = annotate_knownness_metadata(merged)

    summary_row: dict[str, object] = {
        "primary_model_name": primary_model_name,
        "baseline_model_name": baseline_model_name,
        "n_backbones": int(len(merged)),
        "n_positive": int(merged["spread_label"].sum()),
        "overall_primary_roc_auc": roc_auc_score(merged["spread_label"], merged["primary_prediction"]),
        "overall_baseline_roc_auc": roc_auc_score(merged["spread_label"], merged["baseline_prediction"]),
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
        "lowest_knownness_quartile_supported": bool((merged["knownness_quartile"] == "q1_lowest").any()),
        "lowest_knownness_quartile_n_backbones": 0,
        "lowest_knownness_quartile_n_positive": 0,
        "lowest_knownness_quartile_primary_roc_auc": np.nan,
        "lowest_knownness_quartile_baseline_roc_auc": np.nan,
        "lowest_knownness_quartile_delta_roc_auc": np.nan,
    }
    summary_row["priority_index_vs_knownness_spearman"] = summary_row["operational_priority_index_vs_knownness_spearman"]

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
        summary_row[f"{label}_delta_roc_auc"] = (
            summary_row[f"{label}_primary_roc_auc"] - summary_row[f"{label}_baseline_roc_auc"]
        )

    top_candidates = merged.sort_values("primary_prediction", ascending=False).head(top_k).copy()
    summary_row["top_k"] = int(top_k)
    summary_row["top_k_mean_knownness_score"] = float(top_candidates["knownness_score"].mean())
    summary_row["eligible_mean_knownness_score"] = float(merged["knownness_score"].mean())
    summary_row["top_k_lower_half_knownness_count"] = int((top_candidates["knownness_half"] == "lower_half").sum())
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
        weighted_primary = float(np.average(strata["primary_roc_auc"], weights=strata["n_backbones"]))
        weighted_baseline = float(np.average(strata["baseline_roc_auc"], weights=strata["n_backbones"]))
        summary_row["matched_strata_count"] = int(len(strata))
        summary_row["matched_strata_n_backbones"] = int(strata["n_backbones"].sum())
        summary_row["matched_strata_primary_weighted_roc_auc"] = weighted_primary
        summary_row["matched_strata_baseline_weighted_roc_auc"] = weighted_baseline
        summary_row["matched_strata_weighted_delta_roc_auc"] = weighted_primary - weighted_baseline

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
        "novelty_margin_overall_roc_auc": roc_auc_score(merged["spread_label"], merged["novelty_margin"]),
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
    row["watchlist_positive_count"] = int(watchlist["spread_label"].sum()) if not watchlist.empty else 0
    row["watchlist_positive_fraction"] = float(watchlist["spread_label"].mean()) if not watchlist.empty else np.nan
    row["watchlist_mean_novelty_margin"] = float(watchlist["novelty_margin"].mean()) if not watchlist.empty else np.nan
    row["watchlist_mean_primary_prediction"] = float(watchlist["primary_prediction"].mean()) if not watchlist.empty else np.nan
    row["watchlist_mean_baseline_prediction"] = float(watchlist["baseline_prediction"].mean()) if not watchlist.empty else np.nan
    row["watchlist_mean_knownness_score"] = float(watchlist["knownness_score"].mean()) if not watchlist.empty else np.nan
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
    near = working.sort_values(["distance_to_gate", "knownness_score"], ascending=[True, True], kind="mergesort").head(near_n).copy()
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
        lower_mask = frame.get("knownness_half", pd.Series("", index=frame.index)).astype(str).eq("lower_half")
        upper_mask = frame.get("knownness_half", pd.Series("", index=frame.index)).astype(str).eq("upper_half")
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


def build_candidate_portfolio_table(
    candidate_dossiers: pd.DataFrame,
    novelty_watchlist: pd.DataFrame,
    *,
    established_n: int = 10,
    novel_n: int = 10,
) -> pd.DataFrame:
    """Create a two-track candidate portfolio: established high-risk and novel lower-knownness signals."""
    established = pd.DataFrame()
    if not candidate_dossiers.empty:
        established_pool = candidate_dossiers.copy()
        if "eligible_for_oof" in established_pool.columns:
            established_pool = established_pool.loc[established_pool["eligible_for_oof"].fillna(False)].copy()
        positive_column = next(
            (column for column in ("visibility_expansion_label", "spread_label") if column in established_pool.columns),
            None,
        )
        if positive_column is not None:
            positive_pool = established_pool.loc[
                established_pool[positive_column].fillna(0).astype(float) >= 1.0
            ].copy()
            non_positive_pool = established_pool.loc[~established_pool.index.isin(positive_pool.index)].copy()
        else:
            positive_pool = established_pool.copy()
            non_positive_pool = established_pool.iloc[0:0].copy()
        if "false_positive_risk_tier" in established_pool.columns:
            preferred = positive_pool.loc[
                positive_pool["false_positive_risk_tier"].fillna("high").ne("high")
            ].copy()
            fallback = pd.concat(
                [
                    positive_pool.loc[positive_pool["false_positive_risk_tier"].fillna("high").eq("high")].copy(),
                    non_positive_pool.copy(),
                ],
                ignore_index=True,
            )
        else:
            preferred = positive_pool.copy()
            fallback = non_positive_pool.copy()
        confidence_order = {"tier_a": 0, "tier_b": 1, "watchlist": 2}
        for frame in (preferred, fallback):
            if "candidate_confidence_tier" in frame.columns:
                frame["confidence_sort"] = frame["candidate_confidence_tier"].map(confidence_order).fillna(3)
            else:
                frame["confidence_sort"] = 3
        sort_columns = [column for column in ["confidence_sort", "consensus_rank"] if column in preferred.columns]
        ascending = [True] * len(sort_columns)
        if "primary_model_candidate_score" in preferred.columns:
            sort_columns.append("primary_model_candidate_score")
            ascending.append(False)
        if sort_columns:
            preferred = preferred.sort_values(sort_columns, ascending=ascending)
            fallback = fallback.sort_values(sort_columns, ascending=ascending)
        established = pd.concat(
            [preferred.head(established_n), fallback.head(max(established_n - len(preferred.head(established_n)), 0))],
            ignore_index=True,
        ).drop_duplicates(subset=["backbone_id"], keep="first").head(established_n)
    if not established.empty:
        established = established.assign(
            portfolio_track="established_high_risk",
            track_rank=np.arange(1, len(established) + 1),
        )

    novel = pd.DataFrame()
    if not novelty_watchlist.empty:
        novel_pool = novelty_watchlist.copy()
        novelty_mask = novel_pool.get("novelty_margin_vs_baseline", pd.Series(0.0, index=novel_pool.index)).fillna(0.0) > 0
        support_mask = novel_pool.get(
            "external_support_modalities_count",
            pd.Series(0, index=novel_pool.index),
        ).fillna(0).astype(int) > 0
        train_support_mask = novel_pool.get(
            "member_count_train",
            pd.Series(0, index=novel_pool.index),
        ).fillna(0).astype(int) >= 2
        positive_column = next(
            (column for column in ("visibility_expansion_label", "spread_label") if column in novel_pool.columns),
            None,
        )
        positive_mask = (
            novel_pool[positive_column].fillna(0).astype(float) >= 1.0
            if positive_column is not None
            else pd.Series(True, index=novel_pool.index)
        )
        preferred = novel_pool.loc[positive_mask & novelty_mask & (support_mask | train_support_mask)].copy()
        fallback = novel_pool.loc[~novel_pool.index.isin(preferred.index)].copy()
        sort_columns = []
        ascending = []
        if "novelty_margin_vs_baseline" in preferred.columns:
            sort_columns.append("novelty_margin_vs_baseline")
            ascending.append(False)
        if "primary_model_candidate_score" in preferred.columns:
            sort_columns.append("primary_model_candidate_score")
            ascending.append(False)
        if sort_columns:
            preferred = preferred.sort_values(sort_columns, ascending=ascending)
            fallback = fallback.sort_values(sort_columns, ascending=ascending)
        novel = pd.concat(
            [preferred.head(novel_n), fallback.head(max(novel_n - len(preferred.head(novel_n)), 0))],
            ignore_index=True,
        ).drop_duplicates(subset=["backbone_id"], keep="first").head(novel_n)
    if not novel.empty:
        novel = novel.assign(
            portfolio_track="novel_signal",
            track_rank=np.arange(1, len(novel) + 1),
            candidate_confidence_tier=novel.get("candidate_confidence_tier", "novelty_watchlist"),
        )

    frames = [frame for frame in (established, novel) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "primary_model_candidate_score" not in combined.columns and "primary_model_oof_prediction" in combined.columns:
        combined["primary_model_candidate_score"] = combined["primary_model_oof_prediction"]
    if "baseline_both_candidate_score" not in combined.columns and "baseline_both_oof_prediction" in combined.columns:
        combined["baseline_both_candidate_score"] = combined["baseline_both_oof_prediction"]
    combined["in_consensus_top50"] = combined.get("consensus_rank", pd.Series(np.nan, index=combined.index)).notna()
    if "candidate_prediction_source" not in combined.columns:
        combined["candidate_prediction_source"] = np.where(
            combined.get("primary_model_oof_prediction", pd.Series(np.nan, index=combined.index)).notna(),
            "oof",
            "missing",
        )
    if "eligible_for_oof" not in combined.columns:
        combined["eligible_for_oof"] = combined.get("primary_model_oof_prediction", pd.Series(np.nan, index=combined.index)).notna()
    support_columns = [
        "who_mia_any_support",
        "card_any_support",
        "mobsuite_any_literature_support",
        "pd_any_support",
    ]
    available_support_columns = [column for column in support_columns if column in combined.columns]
    for column in available_support_columns:
        combined[column] = combined[column].astype("boolean")
    if available_support_columns:
        support_matrix = pd.DataFrame({column: combined[column] for column in available_support_columns}, index=combined.index)
        combined["support_profile_available"] = support_matrix.notna().any(axis=1)
        combined["external_support_modalities_count"] = support_matrix.fillna(False).sum(axis=1).astype(int)
    else:
        combined["support_profile_available"] = False
        combined["external_support_modalities_count"] = 0
    dominant_source_share = combined[["refseq_share_train", "insd_share_train"]].fillna(0.0).max(axis=1) if {"refseq_share_train", "insd_share_train"}.issubset(combined.columns) else pd.Series(0.0, index=combined.index)
    cross_source_share = combined[["refseq_share_train", "insd_share_train"]].fillna(0.0).min(axis=1) if {"refseq_share_train", "insd_share_train"}.issubset(combined.columns) else pd.Series(0.0, index=combined.index)
    combined["source_support_tier"] = np.select(
        [
            cross_source_share >= 0.15,
            combined.get("refseq_share_train", pd.Series(0.0, index=combined.index)).fillna(0.0) >= 0.85,
            combined.get("insd_share_train", pd.Series(0.0, index=combined.index)).fillna(0.0) >= 0.85,
            dominant_source_share >= 0.60,
        ],
        ["cross_source_supported", "refseq_dominant", "insd_dominant", "source_mixed"],
        default="source_sparse",
    )
    bootstrap_top10 = combined.get("bootstrap_top_10_frequency", pd.Series(0.0, index=combined.index)).fillna(0.0)
    risk_tier = combined.get("false_positive_risk_tier", pd.Series("unknown", index=combined.index)).fillna("unknown").astype(str)
    combined["recommended_monitoring_tier"] = np.select(
        [
            (bootstrap_top10 >= 0.80) & risk_tier.isin(["low", "medium"]),
            (bootstrap_top10 >= 0.50) & risk_tier.ne("high"),
        ],
        ["core_surveillance", "extended_watchlist"],
        default="low_confidence_backlog",
    )
    combined["evidence_tier"] = combined.get("candidate_confidence_tier", pd.Series("unknown", index=combined.index)).fillna("unknown")
    combined["action_tier"] = combined.get("recommended_monitoring_tier", pd.Series("unassigned", index=combined.index)).fillna("unassigned")
    keep_columns = [
        "portfolio_track",
        "track_rank",
        "backbone_id",
        "candidate_confidence_tier",
        "evidence_tier",
        "recommended_monitoring_tier",
        "action_tier",
        "in_consensus_top50",
        "consensus_rank",
        "consensus_candidate_score",
        "consensus_support_count",
        "rank_disagreement_primary_vs_conservative",
        "priority_index",
        "operational_priority_index",
        "bio_priority_index",
        "evidence_support_index",
        "primary_model_candidate_score",
        "baseline_both_candidate_score",
        "conservative_model_candidate_score",
        "primary_model_oof_prediction",
        "baseline_both_oof_prediction",
        "novelty_margin_vs_baseline",
        "candidate_prediction_source",
        "eligible_for_oof",
        "knownness_score",
        "knownness_half",
        "spread_label",
        "bootstrap_top_k_frequency",
        "bootstrap_top_10_frequency",
        "variant_top_k_frequency",
        "false_positive_risk_tier",
        "risk_flag_count",
        "risk_flags",
        "member_count_train",
        "n_countries_train",
        "n_new_countries",
        "coherence_score",
        "source_support_tier",
        "support_profile_available",
        "external_support_modalities_count",
        "module_f_enriched_signature_count",
        "module_f_enriched_signatures",
        "who_mia_any_support",
        "card_any_support",
        "mobsuite_any_literature_support",
        "pd_any_support",
        "amrfinder_any_hit",
    ]
    available_columns = [column for column in keep_columns if column in combined.columns]
    return combined[available_columns].reset_index(drop=True)


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
    dominant_floor = component_frame.idxmin(axis=1).map({value: key for key, value in component_columns.items()})
    working["dominant_floor_component"] = dominant_floor.fillna("unknown")

    segments = {
        "all_backbones": pd.Series(True, index=working.index),
        "training_supported": working["member_count_train"].fillna(0).astype(int) > 0,
        "eligible_candidate_cohort": working["spread_label"].notna() if "spread_label" in working.columns else pd.Series(False, index=working.index),
        "no_training_support": working["member_count_train"].fillna(0).astype(int) == 0,
        "low_score_cluster": working["priority_index"].fillna(0.0) < low_score_threshold,
        "low_score_supported": (working["priority_index"].fillna(0.0) < low_score_threshold) & working["member_count_train"].fillna(0).astype(int).gt(0),
        "low_score_supported_eligible": (
            (working["priority_index"].fillna(0.0) < low_score_threshold)
            & working["member_count_train"].fillna(0).astype(int).gt(0)
            & (working["spread_label"].notna() if "spread_label" in working.columns else False)
        ),
        "low_score_no_training_support": (working["priority_index"].fillna(0.0) < low_score_threshold) & working["member_count_train"].fillna(0).astype(int).eq(0),
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
                    frame.get("operational_priority_index", frame["priority_index"]).fillna(0.0).mean()
                ),
                "mean_bio_priority_index": float(frame.get("bio_priority_index", pd.Series(0.0, index=frame.index)).fillna(0.0).mean()),
                "mean_evidence_support_index": float(
                    frame.get("evidence_support_index", pd.Series(0.0, index=frame.index)).fillna(0.0).mean()
                ),
                "median_priority_index": float(frame["priority_index"].fillna(0.0).median()),
                "mean_member_count_train": float(frame["member_count_train"].fillna(0.0).mean()),
                "median_member_count_train": float(frame["member_count_train"].fillna(0.0).median()),
                "zero_training_support_fraction": float(frame["member_count_train"].fillna(0).astype(int).eq(0).mean()),
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
    eligible["operational_priority_index"] = eligible.get("operational_priority_index", eligible.get("priority_index", 0.0)).fillna(
        eligible.get("priority_index", 0.0)
    )
    if "bio_priority_index" not in eligible.columns:
        eligible["bio_priority_index"] = np.nan
    if "evidence_support_index" not in eligible.columns:
        eligible["evidence_support_index"] = np.nan
    if "H_specialization_norm" not in eligible.columns and "H_breadth_norm" in eligible.columns:
        eligible["H_specialization_norm"] = (1.0 - eligible["H_breadth_norm"].fillna(0.0)).clip(lower=0.0, upper=1.0)

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
                "average_precision_lift": average_precision_lift(frame["spread_label"], frame[column]),
                "average_precision_enrichment": average_precision_enrichment(frame["spread_label"], frame[column]),
                "knownness_spearman": _safe_spearman(frame[column], frame["knownness_score"]),
                "mean_value": float(frame[column].mean()),
                "median_value": float(frame[column].median()),
                "zero_fraction": float(frame[column].eq(0.0).mean()),
                "lower_quartile_fraction": float(frame[column].le(frame[column].quantile(0.25)).mean()),
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
                "zero_fraction_training_reference": float(training_zero.sum() / training_supported.sum()) if training_supported.any() else np.nan,
                "zero_fraction_eligible": float(eligible_zero.sum() / eligible.sum()) if eligible.any() else np.nan,
                "normalized_value_when_raw_zero_min": float(norm.loc[zero_mask].min()) if zero_mask.any() else np.nan,
                "normalized_value_when_raw_zero_median": float(norm.loc[zero_mask].median()) if zero_mask.any() else np.nan,
                "normalized_value_when_raw_zero_max": float(norm.loc[zero_mask].max()) if zero_mask.any() else np.nan,
                "normalized_value_when_raw_positive_min": float(norm.loc[raw > 0.0].min()) if (raw > 0.0).any() else np.nan,
                "normalized_value_when_raw_positive_median": float(norm.loc[raw > 0.0].median()) if (raw > 0.0).any() else np.nan,
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
    working["amrfinder_hit_fraction"] = working["n_with_amrfinder_hits"] / working["n_sequences"].replace(0, np.nan)
    working["amr_evidence_fraction"] = working["n_with_any_amr_evidence"] / working["n_sequences"].replace(0, np.nan)
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
    backbone_meta["country_count_band"] = backbone_meta["n_countries_train"].fillna(0).astype(int).astype(str)

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
    required_models = [primary_model_name] + [name for name in comparison_model_names if name != primary_model_name]
    wide = (
        predictions.loc[predictions["model_name"].isin(required_models), ["backbone_id", "model_name", "oof_prediction", "spread_label"]]
        .pivot_table(index="backbone_id", columns="model_name", values="oof_prediction", aggfunc="first")
        .reset_index()
    )
    primary_labels = (
        predictions.loc[predictions["model_name"] == primary_model_name, ["backbone_id", "spread_label"]]
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
                "delta_average_precision_ci_lower": discrimination_deltas["average_precision"]["lower"],
                "delta_average_precision_ci_upper": discrimination_deltas["average_precision"]["upper"],
                "delta_brier_improvement": brier_delta["delta"],
                "delta_brier_ci_lower": brier_delta["lower"],
                "delta_brier_ci_upper": brier_delta["upper"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False).reset_index(drop=True)


def build_calibration_metric_table(predictions: pd.DataFrame, *, model_names: list[str]) -> pd.DataFrame:
    """Compute compact calibration-quality summaries for selected models."""
    rows = []
    for model_name in model_names:
        frame = predictions.loc[predictions["model_name"] == model_name].copy()
        if frame.empty:
            continue
        y = frame["spread_label"].to_numpy(dtype=int)
        preds = frame["oof_prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "n_backbones": int(len(frame)),
                "mean_prediction": float(preds.mean()),
                "observed_rate": float(y.mean()),
                "brier_score": brier_score(y, preds),
                "expected_calibration_error": expected_calibration_error(y, preds),
            }
        )
    return pd.DataFrame(rows)


def build_source_balance_resampling_table(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_resamples: int = 20,
    seed: int = 42,
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
    rows = []
    for resample_index in range(1, n_resamples + 1):
        sampled_frames = []
        sample_seed = int(rng.integers(0, 1_000_000_000))
        for frame in grouped_frames:
            sampled_frames.append(frame.sample(n=n_per_group, random_state=sample_seed))
        sampled = pd.concat(sampled_frames, ignore_index=True).drop(columns=["dominant_source"])
        result = evaluate_model_name(sampled, model_name=model_name, n_repeats=2, seed=sample_seed, include_ci=False)
        rows.append(
            {
                "model_name": model_name,
                "resample_index": resample_index,
                "sample_seed": sample_seed,
                "n_backbones": int(len(sampled)),
                "n_per_source_group": n_per_group,
                "roc_auc": result.metrics["roc_auc"],
                "average_precision": result.metrics["average_precision"],
                "brier_score": result.metrics["brier_score"],
            }
        )
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
    working["negative_control_noise_a"] = backbone_ids.map(lambda value: _stable_unit_interval(value, salt="noise_a"))
    working["negative_control_noise_b"] = backbone_ids.map(lambda value: _stable_unit_interval(value, salt="noise_b"))
    working["negative_control_length"] = backbone_ids.str.len().astype(float)

    # Use the same fit parameters as the primary model for all specs so that
    # the "primary_model" baseline row matches the official evaluation and the
    # delta columns are computed against a consistent reference.
    fit_kwargs = _model_fit_kwargs(primary_model_name)

    specs = [
        ("primary_model", MODULE_A_FEATURE_SETS[primary_model_name]),
        ("negative_control_noise_a_only", ["negative_control_noise_a"]),
        ("negative_control_noise_ab_only", ["negative_control_noise_a", "negative_control_noise_b"]),
        ("negative_control_length_only", ["negative_control_length"]),
        ("primary_plus_negative_control_a", MODULE_A_FEATURE_SETS[primary_model_name] + ["negative_control_noise_a"]),
        (
            "primary_plus_negative_control_ab",
            MODULE_A_FEATURE_SETS[primary_model_name] + ["negative_control_noise_a", "negative_control_noise_b"],
        ),
    ]

    rows: list[dict[str, object]] = []
    primary_metrics: dict[str, float] | None = None
    for audit_name, columns in specs:
        eligible, X, y = _eligible_xy(working, columns)
        sample_weight = _compute_sample_weight(eligible, mode=fit_kwargs.get("sample_weight_mode"))
        preds = _oof_predictions(
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            sample_weight=sample_weight,
            l2=float(fit_kwargs.get("l2", 1.0)),
            max_iter=int(fit_kwargs.get("max_iter", 100)),
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
        audit["delta_average_precision_vs_primary"] = audit["average_precision"] - float(primary_metrics["average_precision"])
        audit["delta_brier_vs_primary"] = audit["brier_score"] - float(primary_metrics["brier_score"])
    return audit


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
            permuted_batch = np.vstack([rng.permutation(y) for _ in range(current_batch)]).astype(int, copy=False)
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
                "empirical_p_roc_auc": float((1 + sum(value >= observed_auc for value in null_aucs)) / (n_permutations + 1)),
                "observed_average_precision": observed_ap,
                "observed_average_precision_lift": average_precision_lift(y, preds),
                "observed_average_precision_enrichment": average_precision_enrichment(y, preds),
                "null_average_precision_mean": float(np.mean(null_aps)),
                "null_average_precision_std": float(np.std(null_aps)),
                "null_average_precision_q975": float(np.quantile(null_aps, 0.975)),
                "empirical_p_average_precision": float((1 + sum(value >= observed_ap for value in null_aps)) / (n_permutations + 1)),
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
    working["primary_rank"] = working["primary_candidate_score"].rank(method="average", ascending=False)
    working["conservative_rank"] = working["conservative_candidate_score"].rank(method="average", ascending=False)
    working["bio_rank"] = working["bio_priority_index"].fillna(0.0).rank(method="average", ascending=False)
    working["consensus_rank_mean"] = (
        working["primary_rank"] + working["conservative_rank"]
    ) / 2.0
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
    n_bootstrap: int = 200,
    seed: int = 42,
    normalization_method: str = "rank_percentile",
    score_column: str = "priority_index",
    model_name: str | None = None,
) -> pd.DataFrame:
    """Measure how often the highest-ranked candidates stay near the top under reference resampling."""
    training = scored.loc[scored["member_count_train"].fillna(0).astype(int) > 0].copy()
    if model_name:
        training = training.loc[training["spread_label"].notna()].copy()
    if training.empty:
        return pd.DataFrame()

    score_key = "stability_score"
    if model_name:
        base_scores = fit_full_model_predictions(training, model_name=model_name).rename(columns={"prediction": score_key})
        base = training.merge(base_scores, on="backbone_id", how="left", validate="1:1")
    else:
        base = training.copy()
        base[score_key] = base[score_column].fillna(0.0)
    base = base.sort_values(score_key, ascending=False).reset_index(drop=True)
    candidates = base.head(candidate_n)[["backbone_id", score_key]].copy()
    candidates["base_rank"] = np.arange(1, len(candidates) + 1)
    candidate_ids = candidates["backbone_id"].astype(str).tolist()
    stats = {
        backbone_id: {
            "top_hits": 0,
            "top_hits_top10": 0,
            "top_hits_top25": 0,
            "ranks": [],
        }
        for backbone_id in candidate_ids
    }

    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sample_seed = int(rng.integers(0, 1_000_000_000))
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
                continue
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
        rank_lookup = {backbone_id: rank for rank, backbone_id in enumerate(rescored["backbone_id"].astype(str), start=1)}
        for backbone_id in candidate_ids:
            rank = int(rank_lookup.get(backbone_id, len(rescored) + 1))
            stats[backbone_id]["ranks"].append(rank)
            stats[backbone_id]["top_hits"] += int(rank <= top_k)
            stats[backbone_id]["top_hits_top10"] += int(rank <= min(10, len(rescored)))
            stats[backbone_id]["top_hits_top25"] += int(rank <= min(25, len(rescored)))

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
                "bootstrap_top_k_frequency": float(stats[str(row["backbone_id"])]["top_hits"] / n_bootstrap),
                "bootstrap_top_10_frequency": float(stats[str(row["backbone_id"])]["top_hits_top10"] / n_bootstrap),
                "bootstrap_top_25_frequency": float(stats[str(row["backbone_id"])]["top_hits_top25"] / n_bootstrap),
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
) -> pd.DataFrame:
    """Summarize how often top candidates remain in the top set across robustness variants."""
    base = base_scored.loc[base_scored["member_count_train"].fillna(0).astype(int) > 0].copy()
    if model_name:
        base = base.loc[base["spread_label"].notna()].copy()
    if base.empty:
        return pd.DataFrame()
    score_key = "variant_score"
    if model_name:
        base_scores = fit_full_model_predictions(base, model_name=model_name).rename(columns={"prediction": score_key})
        base = base.merge(base_scores, on="backbone_id", how="left", validate="1:1")
    else:
        base[score_key] = base[score_column].fillna(0.0)
    base = base.sort_values(score_key, ascending=False).reset_index(drop=True)
    candidates = base.head(candidate_n)[["backbone_id", score_key]].copy()
    candidates["base_rank"] = np.arange(1, len(candidates) + 1)
    candidate_ids = candidates["backbone_id"].astype(str).tolist()

    valid_variants: dict[str, tuple[dict[str, int], int]] = {}
    for name, frame in variant_frames.items():
        if frame.empty or "backbone_id" not in frame.columns or "member_count_train" not in frame.columns:
            continue
        working = frame.loc[frame["member_count_train"].fillna(0).astype(int) > 0].copy()
        if model_name:
            working = working.loc[working["spread_label"].notna()].copy()
        if working.empty:
            continue
        if model_name:
            predictions = fit_full_model_predictions(working, model_name=model_name).rename(columns={"prediction": score_key})
            working = working.merge(predictions, on="backbone_id", how="left", validate="1:1")
        elif score_column in working.columns:
            working[score_key] = working[score_column].fillna(0.0)
        else:
            continue
        ranked = working.sort_values(score_key, ascending=False).reset_index(drop=True)
        rank_lookup = {
            backbone_id: rank
            for rank, backbone_id in enumerate(ranked["backbone_id"].astype(str), start=1)
        }
        valid_variants[name] = (rank_lookup, int(len(ranked)))
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
) -> pd.DataFrame:
    """Evaluate selected models on strict held-out groups such as source or dominant genus."""
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["spread_label"] = eligible["spread_label"].astype(int)

    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        if group_column not in eligible.columns:
            continue
        working = eligible.copy()
        working[group_column] = working[group_column].fillna("unknown").astype(str)
        counts = working[group_column].value_counts()
        if group_column == "dominant_source":
            selected_groups = counts.index.tolist()
        else:
            selected_groups = counts.loc[counts >= min_group_size].head(max_groups_per_column).index.tolist()
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
                if len(test) < min_group_size or test["spread_label"].nunique() < 2 or train["spread_label"].nunique() < 2:
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
                prediction_table = fit_predict_model_holdout(train, test, model_name=model_name)
                if prediction_table.empty or prediction_table["spread_label"].nunique() < 2:
                    rows.append(
                        {
                            **base_row,
                            "model_name": model_name,
                            "roc_auc": np.nan,
                            "average_precision": np.nan,
                            "brier_score": np.nan,
                            "status": "skipped_fit_failure",
                        }
                    )
                    continue
                y = prediction_table["spread_label"].to_numpy(dtype=int)
                preds = prediction_table["prediction"].to_numpy(dtype=float)
                prevalence = positive_prevalence(y)
                top_k = max(int(np.ceil(len(preds) * 0.10)), 1)
                ranked = prediction_table.sort_values("prediction", ascending=False).head(top_k)
                rows.append(
                    {
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
    sample_weight = _compute_sample_weight(eligible, mode=fit_kwargs.get("sample_weight_mode"))
    custom_preds = _oof_predictions(
        X,
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        sample_weight=sample_weight,
        l2=float(fit_kwargs.get("l2", 1.0)),
        max_iter=int(fit_kwargs.get("max_iter", 100)),
    )

    sklearn_preds = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    c_value = 1e6 if float(fit_kwargs.get("l2", 1.0)) <= 0 else 1.0 / float(fit_kwargs.get("l2", 1.0))
    max_iter = max(int(fit_kwargs.get("max_iter", 100)), 1000)
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
        "sklearn_average_precision": average_precision(y, sklearn_preds),
        "custom_average_precision_lift": average_precision_lift(y, custom_preds),
        "sklearn_average_precision_lift": average_precision_lift(y, sklearn_preds),
        "pearson_prediction_correlation": float(pd.Series(custom_preds).corr(pd.Series(sklearn_preds), method="pearson")),
        "spearman_prediction_correlation": float(pd.Series(custom_preds).corr(pd.Series(sklearn_preds), method="spearman")),
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
    available = model_metrics.set_index("model_name", drop=False)
    if primary_model_name not in available.index or conservative_model_name not in available.index:
        return pd.DataFrame()

    primary = predictions.loc[predictions["model_name"] == primary_model_name].copy()
    conservative = predictions.loc[predictions["model_name"] == conservative_model_name].copy()
    if primary.empty or conservative.empty:
        return pd.DataFrame()

    primary = primary.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
    conservative = conservative.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
    primary_ids = primary["backbone_id"].astype(str).tolist()
    conservative_ids = conservative["backbone_id"].astype(str).tolist()

    row: dict[str, object] = {
        "primary_model_name": primary_model_name,
        "conservative_model_name": conservative_model_name,
        "primary_roc_auc": float(available.loc[primary_model_name, "roc_auc"]),
        "conservative_roc_auc": float(available.loc[conservative_model_name, "roc_auc"]),
        "roc_auc_delta_primary_minus_conservative": float(
            available.loc[primary_model_name, "roc_auc"] - available.loc[conservative_model_name, "roc_auc"]
        ),
        "primary_average_precision": float(available.loc[primary_model_name, "average_precision"]),
        "conservative_average_precision": float(available.loc[conservative_model_name, "average_precision"]),
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
                "refseq_record_fraction": float(frame["record_origin"].eq("refseq").mean()) if "record_origin" in frame.columns else np.nan,
                "insd_record_fraction": float(frame["record_origin"].eq("insd").mean()) if "record_origin" in frame.columns else np.nan,
                "mobilizable_fraction": float(frame["is_mobilizable"].fillna(False).astype(bool).mean()) if "is_mobilizable" in frame.columns else np.nan,
                "conjugative_fraction": float(frame["is_conjugative"].fillna(False).astype(bool).mean()) if "is_conjugative" in frame.columns else np.nan,
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
            summary[column]
            .astype(float)
            .rolling(window=3, min_periods=1, center=True)
            .mean()
        )
    return summary


def build_candidate_dossier_table(
    base_candidates: pd.DataFrame,
    *,
    candidate_stability: pd.DataFrame,
    predictions: pd.DataFrame,
    primary_model_name: str,
    conservative_model_name: str,
    who_detail: pd.DataFrame,
    card_detail: pd.DataFrame,
    mobsuite_detail: pd.DataFrame,
    pathogen_support: pd.DataFrame,
    amrfinder_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a compact candidate dossier with stability, support, and prediction context."""
    dossier = base_candidates.copy()
    if dossier.empty:
        return dossier
    if "freeze_rank" not in dossier.columns:
        dossier = dossier.sort_values("priority_index", ascending=False).reset_index(drop=True)
        dossier["freeze_rank"] = np.arange(1, len(dossier) + 1)

    if not candidate_stability.empty:
        stability_columns = [
            "backbone_id",
            "base_rank",
            "top_k",
            "n_bootstrap",
            "bootstrap_top_k_frequency",
            "bootstrap_top_10_frequency",
            "bootstrap_top_25_frequency",
            "bootstrap_mean_rank",
            "bootstrap_median_rank",
            "bootstrap_rank_std",
            "n_variants_evaluated",
            "variant_top_k_frequency",
            "variant_top_10_frequency",
            "variant_top_25_frequency",
            "variant_mean_rank",
            "variant_median_rank",
            "variant_rank_std",
            "high_confidence_candidate",
            "knownness_score",
            "knownness_half",
            "baseline_both_oof_prediction",
            "primary_model_full_fit_prediction",
            "baseline_both_full_fit_prediction",
            "conservative_model_full_fit_prediction",
            "novelty_margin_vs_baseline",
        ]
        stability_payload = candidate_stability[[column for column in stability_columns if column in candidate_stability.columns]].copy()
        dossier = coalescing_left_merge(dossier, stability_payload, on="backbone_id")

    for model_name, output_column in (
        (primary_model_name, "primary_model_oof_prediction"),
        (conservative_model_name, "conservative_model_oof_prediction"),
    ):
        if output_column in dossier.columns:
            continue
        if model_name in set(predictions["model_name"].astype(str)):
            model_predictions = predictions.loc[
                predictions["model_name"] == model_name,
                ["backbone_id", "oof_prediction"],
            ].rename(columns={"oof_prediction": output_column})
            dossier = coalescing_left_merge(dossier, model_predictions, on="backbone_id")

    if not who_detail.empty:
        dossier = coalescing_left_merge(
            dossier,
            who_detail[["backbone_id", "who_mia_any_support", "who_mia_any_hpecia", "who_mia_mapped_fraction"]],
            on="backbone_id",
        )
    if not card_detail.empty:
        dossier = coalescing_left_merge(
            dossier,
            card_detail[["backbone_id", "card_any_support", "card_match_fraction"]],
            on="backbone_id",
        )
    if not mobsuite_detail.empty:
        dossier = coalescing_left_merge(
            dossier,
            mobsuite_detail[["backbone_id", "mobsuite_any_literature_support", "mobsuite_any_cluster_support"]],
            on="backbone_id",
        )
    if not pathogen_support.empty and "pathogen_dataset" in pathogen_support.columns:
        combined = pathogen_support.loc[
            pathogen_support["pathogen_dataset"] == "combined",
            ["backbone_id", "pd_any_support", "pd_matching_fraction"],
        ]
        dossier = coalescing_left_merge(dossier, combined, on="backbone_id")
    if not amrfinder_detail.empty:
        amrfinder_summary = (
            amrfinder_detail.groupby("backbone_id", as_index=False)
            .agg(
                amrfinder_any_hit=("amrfinder_any_hit", "max"),
                amrfinder_mean_gene_jaccard=("gene_jaccard", "mean"),
                amrfinder_mean_class_jaccard=("class_jaccard", "mean"),
            )
        )
        dossier = coalescing_left_merge(dossier, amrfinder_summary, on="backbone_id")

    support_columns = [
        "who_mia_any_support",
        "card_any_support",
        "mobsuite_any_literature_support",
        "pd_any_support",
    ]
    for column in support_columns:
        if column not in dossier.columns:
            dossier[column] = pd.Series(pd.NA, index=dossier.index, dtype="boolean")
        else:
            dossier[column] = dossier[column].astype("boolean")
    support_matrix = pd.DataFrame({column: dossier[column] for column in support_columns}, index=dossier.index)
    dossier["support_profile_available"] = support_matrix.notna().any(axis=1)
    dossier["external_support_modalities_count"] = support_matrix.fillna(False).sum(axis=1).astype(int)
    primary_oof = dossier.get("primary_model_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)
    primary_full = dossier.get("primary_model_full_fit_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)
    baseline_oof = dossier.get("baseline_both_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)
    baseline_full = dossier.get("baseline_both_full_fit_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)
    conservative_oof = dossier.get("conservative_model_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)
    conservative_full = dossier.get("conservative_model_full_fit_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)).astype(float)

    dossier["primary_model_candidate_score"] = primary_oof.fillna(primary_full)
    dossier["baseline_both_candidate_score"] = baseline_oof.fillna(baseline_full)
    dossier["conservative_model_candidate_score"] = conservative_oof.fillna(conservative_full)
    dossier["candidate_prediction_source"] = np.where(
        primary_oof.notna(),
        "oof",
        np.where(primary_full.notna(), "full_fit", "missing"),
    )
    dossier["eligible_for_oof"] = primary_oof.notna()
    dossier["primary_minus_conservative_prediction"] = (
        dossier["primary_model_candidate_score"].fillna(0.0)
        - dossier["conservative_model_candidate_score"].fillna(0.0)
    )
    dossier["novelty_margin_vs_baseline"] = (
        dossier["primary_model_candidate_score"].fillna(0.0)
        - dossier["baseline_both_candidate_score"].fillna(0.0)
    )

    bootstrap = dossier.get("bootstrap_top_k_frequency", pd.Series(0.0, index=dossier.index)).fillna(0.0)
    variant = dossier.get("variant_top_k_frequency", pd.Series(0.0, index=dossier.index)).fillna(0.0)
    coherence = dossier.get("coherence_score", pd.Series(0.0, index=dossier.index)).fillna(0.0)
    consensus_support = dossier.get("consensus_support_count", pd.Series(0, index=dossier.index)).fillna(0).astype(int)
    disagreement = dossier.get(
        "rank_disagreement_primary_vs_conservative",
        pd.Series(np.inf, index=dossier.index, dtype=float),
    ).fillna(np.inf)
    stability_available = dossier.get("bootstrap_top_k_frequency", pd.Series(np.nan, index=dossier.index)).notna() | dossier.get(
        "variant_top_k_frequency",
        pd.Series(np.nan, index=dossier.index),
    ).notna()
    dossier["candidate_confidence_tier"] = np.select(
        [
            (coherence >= 0.60)
            & (consensus_support >= 2)
            & (((bootstrap >= 0.85) & (variant >= 0.75)) | ((~stability_available) & (disagreement <= 50))),
            (coherence >= 0.50)
            & (consensus_support >= 2)
            & (((bootstrap >= 0.70) & (variant >= 0.60)) | (~stability_available)),
        ],
        ["tier_a", "tier_b"],
        default="watchlist",
    )
    return annotate_candidate_explanation_fields(dossier)


def annotate_candidate_explanation_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact mechanistic and monitoring rationales for reviewer-facing candidate tables."""
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    signal_specs = [
        ("T_raw_norm", "mobility"),
        ("H_specialization_norm", "host_breadth"),
        ("A_raw_norm", "amr_load"),
        ("coherence_score", "coherence"),
        ("H_external_host_range_support", "external_host_support"),
        ("replicon_architecture_norm", "replicon_complexity"),
        ("mash_novelty_norm", "graph_novelty"),
        ("backbone_purity_score", "backbone_purity"),
    ]
    available_signals: list[tuple[str, pd.Series]] = []
    for column, label in signal_specs:
        if column in working.columns:
            available_signals.append(
                (
                    label,
                    pd.to_numeric(working[column], errors="coerce"),
                )
            )

    primary_axes: list[str] = []
    secondary_axes: list[str] = []
    rationale_text: list[str] = []
    signal_counts: list[int] = []
    monitoring_text: list[str] = []

    external_support_count = pd.to_numeric(
        working.get("external_support_modalities_count", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    novelty_margin = pd.to_numeric(
        working.get("novelty_margin_vs_baseline", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    bootstrap_top10 = pd.to_numeric(
        working.get("bootstrap_top_10_frequency", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    variant_top10 = pd.to_numeric(
        working.get("variant_top_10_frequency", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    confidence_tier = working.get("candidate_confidence_tier", pd.Series("", index=working.index)).fillna("").astype(str)
    false_positive_risk = working.get("false_positive_risk_tier", pd.Series("", index=working.index)).fillna("").astype(str)
    prediction_source = working.get("candidate_prediction_source", pd.Series("", index=working.index)).fillna("").astype(str)

    for idx in working.index:
        ranked_signals: list[tuple[float, str]] = []
        for label, series in available_signals:
            value = series.loc[idx]
            if pd.notna(value) and float(value) >= 0.55:
                ranked_signals.append((float(value), label))
        ranked_signals.sort(key=lambda item: item[0], reverse=True)
        labels = [label for _, label in ranked_signals[:3]]
        primary_axes.append(labels[0] if labels else "mixed_signal")
        secondary_axes.append(labels[1] if len(labels) > 1 else "")
        rationale_text.append(",".join(labels) if labels else "limited_distinct_signal")
        signal_counts.append(len(ranked_signals))

        monitoring_flags: list[str] = []
        if confidence_tier.loc[idx] in {"tier_a", "tier_b"}:
            monitoring_flags.append("stable_internal_signal")
        if external_support_count.loc[idx] >= 2:
            monitoring_flags.append("multi_modal_support")
        if novelty_margin.loc[idx] >= 0.10:
            monitoring_flags.append("outperforms_counts_baseline")
        if max(float(bootstrap_top10.loc[idx]), float(variant_top10.loc[idx])) >= 0.60:
            monitoring_flags.append("multiverse_stable")
        if false_positive_risk.loc[idx] == "low":
            monitoring_flags.append("manageable_false_positive_risk")
        if prediction_source.loc[idx] == "oof":
            monitoring_flags.append("oof_supported")
        monitoring_text.append(",".join(monitoring_flags[:4]) if monitoring_flags else "limited_monitoring_context")

    working["primary_driver_axis"] = primary_axes
    working["secondary_driver_axis"] = secondary_axes
    working["mechanistic_rationale"] = rationale_text
    working["mechanistic_signal_count"] = pd.Series(signal_counts, index=working.index, dtype=int)
    working["monitoring_rationale"] = monitoring_text
    return working


def build_candidate_risk_table(dossier: pd.DataFrame) -> pd.DataFrame:
    """Flag candidate-specific false-positive risks without changing the ranking itself."""
    if dossier.empty:
        return pd.DataFrame()
    working = dossier.copy()
    coherence = working.get("coherence_score", pd.Series(np.nan, index=working.index, dtype=float))
    member_count_train = working.get("member_count_train", pd.Series(np.nan, index=working.index, dtype=float))
    n_countries_train = working.get("n_countries_train", pd.Series(np.nan, index=working.index, dtype=float))
    refseq_share_train = working.get("refseq_share_train", pd.Series(np.nan, index=working.index, dtype=float))
    insd_share_train = working.get("insd_share_train", pd.Series(np.nan, index=working.index, dtype=float))

    working["low_coherence_risk"] = coherence.fillna(0.0) < 0.50
    working["sparse_training_support_risk"] = member_count_train.fillna(0).astype(int) <= 2
    working["narrow_geography_risk"] = n_countries_train.fillna(0).astype(int) <= 1
    dominant_source_share = pd.concat([refseq_share_train, insd_share_train], axis=1).fillna(0.0).max(axis=1)
    source_info_available = refseq_share_train.notna() | insd_share_train.notna()
    working["source_concentration_risk"] = source_info_available & dominant_source_share.ge(0.90)
    support_available = working.get(
        "support_profile_available",
        pd.Series(False, index=working.index, dtype=bool),
    ).fillna(False).astype(bool)
    external_support_count = working.get(
        "external_support_modalities_count",
        pd.Series(0, index=working.index, dtype=float),
    ).fillna(0).astype(int)
    working["weak_external_support_risk"] = support_available & external_support_count.eq(0)
    bootstrap_series = working.get(
        "bootstrap_top_k_frequency",
        pd.Series(np.nan, index=working.index, dtype=float),
    )
    variant_series = working.get(
        "variant_top_k_frequency",
        pd.Series(np.nan, index=working.index, dtype=float),
    )
    bootstrap_frequency = bootstrap_series.fillna(0.0)
    variant_frequency = variant_series.fillna(0.0)
    stability_available = bootstrap_series.notna() | variant_series.notna()
    working["stability_risk"] = stability_available & ((bootstrap_frequency < 0.50) | (variant_frequency < 0.50))
    working["proxy_gap_risk"] = working.get(
        "primary_minus_conservative_prediction",
        pd.Series(0.0, index=working.index, dtype=float),
    ).fillna(0.0) >= 0.15

    risk_columns = [
        "low_coherence_risk",
        "sparse_training_support_risk",
        "narrow_geography_risk",
        "source_concentration_risk",
        "weak_external_support_risk",
        "stability_risk",
        "proxy_gap_risk",
    ]
    working["risk_flag_count"] = working[risk_columns].sum(axis=1).astype(int)
    working["false_positive_risk_tier"] = np.select(
        [
            working["risk_flag_count"] >= 3,
            working["risk_flag_count"] >= 1,
        ],
        ["high", "medium"],
        default="low",
    )
    working["risk_flags"] = working[risk_columns].apply(
        lambda row: ",".join(column for column, value in row.items() if bool(value)),
        axis=1,
    )
    columns = [
        "backbone_id",
        "freeze_rank",
        "candidate_confidence_tier",
        "false_positive_risk_tier",
        "risk_flag_count",
        "risk_flags",
        "coherence_score",
        "member_count_train",
        "n_countries_train",
        "external_support_modalities_count",
        "primary_minus_conservative_prediction",
    ] + risk_columns
    available = [column for column in columns if column in working.columns]
    sort_columns = [column for column in ["freeze_rank", "risk_flag_count"] if column in working.columns]
    ascending = [True, False][: len(sort_columns)]
    result = working[available]
    if sort_columns:
        result = result.sort_values(sort_columns, ascending=ascending)
    return result.reset_index(drop=True)


def build_decision_yield_table(
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    top_ks: tuple[int, ...] = (5, 10, 25, 50, 100),
) -> pd.DataFrame:
    """Summarize shortlist precision/recall tradeoffs at practical candidate-list sizes."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = predictions.loc[predictions["model_name"] == model_name].copy()
        if frame.empty:
            continue
        frame = frame.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
        y = frame["spread_label"].to_numpy(dtype=int)
        prevalence = positive_prevalence(y)
        total_positive = max(int(y.sum()), 0)
        for top_k in top_ks:
            subset = frame.head(min(top_k, len(frame))).copy()
            selected_positive = int(subset["spread_label"].sum()) if not subset.empty else 0
            selected_negative = int(len(subset) - selected_positive)
            precision_at_k = float(selected_positive / len(subset)) if len(subset) else np.nan
            recall_at_k = float(selected_positive / total_positive) if total_positive > 0 else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "top_k": int(top_k),
                    "n_backbones": int(len(frame)),
                    "n_selected": int(len(subset)),
                    "n_positive_total": int(total_positive),
                    "n_positive_selected": int(selected_positive),
                    "n_negative_selected": int(selected_negative),
                    "positive_prevalence": prevalence,
                    "precision_at_k": precision_at_k,
                    "recall_at_k": recall_at_k,
                    "precision_lift_vs_prevalence": float(precision_at_k - prevalence) if np.isfinite(precision_at_k) and np.isfinite(prevalence) else np.nan,
                    "mean_prediction_at_k": float(subset["oof_prediction"].mean()) if len(subset) else np.nan,
                    "min_prediction_at_k": float(subset["oof_prediction"].min()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_threshold_flip_table(
    scored: pd.DataFrame,
    *,
    candidate_ids: list[str] | None = None,
    thresholds: tuple[int, ...] = (1, 2, 3, 4),
    default_threshold: int = 3,
) -> pd.DataFrame:
    """Show how candidate status changes under alternate new-country thresholds."""
    if scored.empty:
        return pd.DataFrame()
    if default_threshold not in thresholds:
        raise ValueError("default_threshold must be one of the audited thresholds")

    def _coerce_int(value: object, *, default: int = 0) -> int:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return int(numeric) if pd.notna(numeric) else int(default)

    working = scored.copy()
    if candidate_ids is not None:
        candidate_id_set = {str(value) for value in candidate_ids}
        working = working.loc[working["backbone_id"].astype(str).isin(candidate_id_set)].copy()
    if working.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in working.to_dict(orient="records"):
        train_countries = _coerce_int(row.get("n_countries_train", 0))
        n_new = _coerce_int(row.get("n_new_countries", 0))
        status_by_threshold: dict[int, object] = {}
        for threshold in thresholds:
            status_by_threshold[threshold] = int(n_new >= threshold) if 1 <= train_countries <= 3 else np.nan
        default_status = status_by_threshold[default_threshold]
        finite_statuses = [value for value in status_by_threshold.values() if pd.notna(value)]
        flip_count = int(sum(value != default_status for value in finite_statuses)) if finite_statuses else 0
        result = {
            "backbone_id": str(row["backbone_id"]),
            "member_count_train": _coerce_int(row.get("member_count_train", 0)),
            "n_countries_train": train_countries,
            "n_new_countries": n_new,
            "priority_index": float(row.get("priority_index", np.nan)),
            "spread_label_default": default_status,
            "default_threshold": int(default_threshold),
            "threshold_flip_count": int(flip_count),
            "eligible_for_threshold_audit": bool(1 <= train_countries <= 3),
        }
        for threshold, value in status_by_threshold.items():
            result[f"label_ge_{threshold}"] = value
        rows.append(result)
    frame = pd.DataFrame(rows)
    sort_columns = [column for column in ["threshold_flip_count", "priority_index"] if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False, False][: len(sort_columns)])
    return frame.reset_index(drop=True)


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
    candidate_ids = pd.Index(pd.concat(candidate_sets, ignore_index=True).drop_duplicates().tolist(), dtype=object)
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
    scored_payload = scored[[column for column in scored_columns if column in scored.columns]].copy() if not scored.empty else pd.DataFrame()
    if not scored_payload.empty:
        base = coalescing_left_merge(base, scored_payload, on="backbone_id")

    if not consensus_candidates.empty:
        consensus_payload = consensus_candidates[[
            column for column in [
                "backbone_id",
                "consensus_rank",
                "consensus_candidate_score",
                "consensus_support_count",
                "primary_rank",
                "conservative_rank",
                "rank_disagreement_primary_vs_conservative",
            ] if column in consensus_candidates.columns
        ]].copy()
        consensus_payload["in_consensus_top50"] = True
        base = coalescing_left_merge(base, consensus_payload, on="backbone_id")
    if not candidate_dossiers.empty:
        dossier_payload = candidate_dossiers[[
            column for column in [
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
            ] if column in candidate_dossiers.columns
        ]].copy()
        dossier_payload["in_candidate_dossier_top25"] = True
        base = coalescing_left_merge(base, dossier_payload, on="backbone_id")
    if not candidate_portfolio.empty:
        portfolio_payload = candidate_portfolio[[
            column for column in [
                "backbone_id",
                "portfolio_track",
                "track_rank",
                "candidate_confidence_tier",
                "recommended_monitoring_tier",
            ] if column in candidate_portfolio.columns
        ]].copy()
        portfolio_payload["in_candidate_portfolio"] = True
        base = coalescing_left_merge(base, portfolio_payload, on="backbone_id")
    if not novelty_watchlist.empty:
        novelty_payload = novelty_watchlist[[
            column for column in [
                "backbone_id",
                "novelty_margin_vs_baseline",
                "knownness_score",
                "knownness_half",
                "primary_model_candidate_score",
            ] if column in novelty_watchlist.columns
        ]].copy()
        novelty_payload["in_novelty_watchlist"] = True
        base = coalescing_left_merge(base, novelty_payload, on="backbone_id")
    if not prospective_freeze.empty:
        freeze_payload = prospective_freeze[[
            column for column in [
                "backbone_id",
                "freeze_rank",
                "freeze_candidate_score",
            ] if column in prospective_freeze.columns
        ]].copy()
        freeze_payload["in_prospective_freeze"] = True
        base = coalescing_left_merge(base, freeze_payload, on="backbone_id")
    if not high_confidence_candidates.empty:
        high_conf_payload = high_confidence_candidates[[
            column for column in ["backbone_id", "candidate_confidence_tier", "false_positive_risk_tier"] if column in high_confidence_candidates.columns
        ]].copy()
        high_conf_payload["in_higher_confidence_shortlist"] = True
        base = coalescing_left_merge(base, high_conf_payload, on="backbone_id")
    if not candidate_risk.empty:
        risk_payload = candidate_risk[[
            column for column in [
                "backbone_id",
                "false_positive_risk_tier",
                "risk_flag_count",
                "risk_flags",
            ] if column in candidate_risk.columns
        ]].copy()
        base = coalescing_left_merge(base, risk_payload, on="backbone_id")

    for flag_column in (
        "in_consensus_top50",
        "in_candidate_dossier_top25",
        "in_candidate_portfolio",
        "in_novelty_watchlist",
        "in_prospective_freeze",
        "in_higher_confidence_shortlist",
    ):
        if flag_column not in base.columns:
            base[flag_column] = False
        base[flag_column] = base[flag_column].fillna(False).astype(bool)

    base["evidence_tier"] = base.get("candidate_confidence_tier", pd.Series("unknown", index=base.index)).fillna("unknown")
    base["action_tier"] = base.get("recommended_monitoring_tier", pd.Series("unassigned", index=base.index)).fillna("unassigned")
    base["main_outcome_status"] = np.select(
        [
            base.get("spread_label", pd.Series(np.nan, index=base.index)).isna(),
            base.get("spread_label", pd.Series(0.0, index=base.index)).fillna(0.0).astype(float) >= 1.0,
        ],
        ["not_evaluable", "positive"],
        default="negative",
    )
    base["candidate_universe_origin"] = np.select(
        [
            base["in_candidate_portfolio"] & base.get("portfolio_track", pd.Series("", index=base.index)).fillna("").eq("established_high_risk"),
            base["in_candidate_portfolio"] & base.get("portfolio_track", pd.Series("", index=base.index)).fillna("").eq("novel_signal"),
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
    predictions: pd.DataFrame | None = None,
    family_summary: pd.DataFrame | None = None,
    simplicity_summary: pd.DataFrame | None = None,
    top_ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:
    """Make the publication choice of the primary model explicit for reviewers."""
    if model_metrics.empty or primary_model_name not in set(model_metrics["model_name"].astype(str)):
        return pd.DataFrame()
    available = model_metrics.set_index("model_name", drop=False)
    primary = available.loc[primary_model_name]
    strongest = model_metrics.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    strongest_model_name = str(strongest["model_name"])
    conservative = available.loc[conservative_model_name] if conservative_model_name in available.index else None

    if (
        family_summary is None
        or family_summary.empty
        or "model_name" not in family_summary.columns
        or "evidence_role" not in family_summary.columns
        or primary_model_name not in set(family_summary["model_name"].astype(str))
        or strongest_model_name not in set(family_summary["model_name"].astype(str))
    ):
        family_summary = build_model_family_summary(model_metrics)

    row: dict[str, object] = {
        "published_primary_model": primary_model_name,
        "published_primary_roc_auc": float(primary["roc_auc"]),
        "published_primary_average_precision": float(primary["average_precision"]),
        "strongest_metric_model": strongest_model_name,
        "strongest_metric_model_roc_auc": float(strongest["roc_auc"]),
        "strongest_metric_model_average_precision": float(strongest["average_precision"]),
        "primary_minus_strongest_roc_auc": float(primary["roc_auc"] - strongest["roc_auc"]),
        "primary_minus_strongest_average_precision": float(primary["average_precision"] - strongest["average_precision"]),
        "conservative_model_name": conservative_model_name,
        "conservative_roc_auc": float(conservative["roc_auc"]) if conservative is not None else np.nan,
        "conservative_average_precision": float(conservative["average_precision"]) if conservative is not None else np.nan,
        "selection_rationale": (
            "current primary is also the strongest current single-model benchmark, so the headline and strongest audited metric model now coincide"
            if primary_model_name == strongest_model_name
            else "current primary retained as the headline benchmark despite a marginally stronger audited alternative; the difference is small enough that the report keeps both views explicit rather than silently collapsing them"
        ),
    }

    def _sorted_predictions(model_name: str) -> pd.DataFrame:
        if predictions is None or predictions.empty:
            return pd.DataFrame()
        frame = predictions.loc[predictions["model_name"].astype(str) == str(model_name)].copy()
        if frame.empty:
            return frame
        return frame.sort_values("oof_prediction", ascending=False).reset_index(drop=True)

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
            row[f"primary_vs_{prefix}_top_{top_k}_overlap_fraction"] = float(overlap / top_k) if top_k else 0.0
            row[f"primary_vs_{prefix}_top_{top_k}_jaccard"] = float(overlap / union) if union else 0.0

    def _add_top10_yield(prefix: str, model_name: str) -> None:
        frame = _sorted_predictions(model_name)
        if frame.empty:
            return
        subset = frame.head(min(10, len(frame))).copy()
        total_positive = int(frame["spread_label"].sum())
        selected_positive = int(subset["spread_label"].sum()) if not subset.empty else 0
        row[f"{prefix}_top_10_precision"] = float(selected_positive / len(subset)) if len(subset) else np.nan
        row[f"{prefix}_top_10_recall"] = float(selected_positive / total_positive) if total_positive > 0 else np.nan

    _add_topk_overlap("strongest", strongest_model_name)
    _add_topk_overlap("conservative", conservative_model_name)
    _add_top10_yield("published_primary", primary_model_name)
    _add_top10_yield("strongest", strongest_model_name)
    _add_top10_yield("conservative", conservative_model_name)

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

    if family_summary is not None and not family_summary.empty:
        family_index = family_summary.set_index("model_name", drop=False)
        if primary_model_name in family_index.index:
            row["published_primary_evidence_role"] = str(family_index.loc[primary_model_name, "evidence_role"])
            row["published_primary_evidence_summary"] = str(family_index.loc[primary_model_name, "evidence_summary"])
        if strongest_model_name in family_index.index:
            row["strongest_metric_model_evidence_role"] = str(family_index.loc[strongest_model_name, "evidence_role"])
            row["strongest_metric_model_evidence_summary"] = str(family_index.loc[strongest_model_name, "evidence_summary"])
        if conservative_model_name in family_index.index:
            row["conservative_model_evidence_role"] = str(family_index.loc[conservative_model_name, "evidence_role"])
            row["conservative_model_evidence_summary"] = str(family_index.loc[conservative_model_name, "evidence_summary"])

    if simplicity_summary is not None and not simplicity_summary.empty:
        summary_row = simplicity_summary.iloc[0]
        for top_k in top_ks:
            legacy_count = f"top_{top_k}_overlap_count"
            legacy_fraction = f"top_{top_k}_overlap_fraction"
            legacy_jaccard = f"top_{top_k}_jaccard"
            if legacy_count in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_overlap_count"] = int(summary_row[legacy_count])
            if legacy_fraction in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_overlap_fraction"] = float(summary_row[legacy_fraction])
            if legacy_jaccard in summary_row.index:
                row[f"primary_vs_conservative_top_{top_k}_jaccard"] = float(summary_row[legacy_jaccard])

    return pd.DataFrame([row])


def build_benchmark_protocol_table(
    model_metrics: pd.DataFrame,
    model_selection_summary: pd.DataFrame,
    *,
    adaptive_gated_metrics: pd.DataFrame | None = None,
    gate_consistency_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize which model is official, supporting, audit-only, or control."""
    if model_metrics.empty or model_selection_summary.empty:
        return pd.DataFrame()
    adaptive_gated_metrics = adaptive_gated_metrics if adaptive_gated_metrics is not None else pd.DataFrame()
    gate_consistency_audit = gate_consistency_audit if gate_consistency_audit is not None else pd.DataFrame()

    metrics_index = model_metrics.set_index("model_name", drop=False)
    selection_row = model_selection_summary.iloc[0]
    primary_model_name = str(selection_row.get("published_primary_model", ""))
    conservative_model_name = str(selection_row.get("conservative_model_name", ""))
    strongest_model_name = str(selection_row.get("strongest_metric_model", ""))

    rows: list[dict[str, object]] = []

    def _append_single(model_name: str, *, role: str, status: str, rationale: str) -> None:
        if model_name not in metrics_index.index:
            return
        metric_row = metrics_index.loc[model_name]
        rows.append(
            {
                "model_name": model_name,
                "benchmark_role": role,
                "benchmark_status": status,
                "model_family": "single_model",
                "roc_auc": float(metric_row["roc_auc"]),
                "average_precision": float(metric_row["average_precision"]),
                "gate_consistency_tier": np.nan,
                "specialist_weight_lower_half": np.nan,
                "selection_rationale": rationale,
            }
        )

    _append_single(
        primary_model_name,
        role="primary_benchmark",
        status="headline",
        rationale="Current official single-model benchmark used for the main headline claims.",
    )
    _append_single(
        conservative_model_name,
        role="conservative_benchmark",
        status="headline_supporting",
        rationale="Supportive conservative benchmark kept for interpretability and proxy-light comparison.",
    )
    if strongest_model_name and strongest_model_name != primary_model_name:
        _append_single(
            strongest_model_name,
            role="strongest_single_model",
            status="audit_only",
            rationale="Highest-metric single model retained as an audited alternative rather than the headline benchmark.",
        )
    for control_model_name, role, rationale in (
        ("baseline_both", "counts_baseline", "Counts-only baseline used to measure incremental value beyond simple popularity signals."),
        ("source_only", "source_control", "Weak source-only control used to show that source composition alone does not explain the signal."),
    ):
        if control_model_name in metrics_index.index:
            _append_single(
                control_model_name,
                role=role,
                status="control",
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
                    gate_consistency_audit.get("gate_consistency_tier", pd.Series(dtype=str)).astype(str) == "stable",
                    "model_name",
                ].astype(str)
                preferred = gated.loc[gated["model_name"].astype(str).isin(set(stable_names))].sort_values(
                    ["roc_auc", "average_precision"],
                    ascending=False,
                    kind="mergesort",
                ).head(1)
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
                            "model_family": "adaptive_routing",
                            "roc_auc": float(preferred.iloc[0]["roc_auc"]),
                            "average_precision": float(preferred.iloc[0]["average_precision"]),
                            "gate_consistency_tier": str(gate_row.iloc[0]["gate_consistency_tier"]) if not gate_row.empty else np.nan,
                            "specialist_weight_lower_half": float(preferred.iloc[0].get("specialist_weight_lower_half", np.nan)),
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
                    gate_row = gate_consistency_audit.loc[
                        gate_consistency_audit.get("model_name", pd.Series(dtype=str)).astype(str) == strongest_adaptive_name
                    ].head(1) if not gate_consistency_audit.empty else pd.DataFrame()
                    rows.append(
                        {
                            "model_name": strongest_adaptive_name,
                            "benchmark_role": "strongest_adaptive_upper_bound",
                            "benchmark_status": "audit_upper_bound",
                            "model_family": "adaptive_routing",
                            "roc_auc": float(strongest_adaptive.iloc[0]["roc_auc"]),
                            "average_precision": float(strongest_adaptive.iloc[0]["average_precision"]),
                            "gate_consistency_tier": str(gate_row.iloc[0]["gate_consistency_tier"]) if not gate_row.empty else np.nan,
                            "specialist_weight_lower_half": float(strongest_adaptive.iloc[0].get("specialist_weight_lower_half", np.nan)),
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
    protocol["_status_order"] = protocol["benchmark_status"].map(status_order).fillna(99).astype(int)
    protocol = protocol.sort_values(
        ["_status_order", "roc_auc", "average_precision"],
        ascending=[True, False, False],
        kind="mergesort",
    ).drop(columns="_status_order").reset_index(drop=True)
    return protocol


def build_model_selection_scorecard(
    model_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    knownness_matched_validation: pd.DataFrame | None = None,
    group_holdout: pd.DataFrame | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Assemble a multi-objective model selection table beyond overall ROC AUC."""
    if model_metrics.empty or predictions.empty or scored.empty:
        return pd.DataFrame()
    knownness_matched_validation = knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    group_holdout = group_holdout if group_holdout is not None else pd.DataFrame()
    available_metrics = model_metrics.set_index("model_name", drop=False)
    requested_models = [str(name) for name in (model_names or available_metrics.index.tolist())]
    if "spread_label" in scored.columns:
        eligible_scored = scored.loc[pd.to_numeric(scored["spread_label"], errors="coerce").notna()].copy()
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
        row = {
            "model_name": model_name,
            "roc_auc": float(available_metrics.loc[model_name, "roc_auc"]),
            "average_precision": float(available_metrics.loc[model_name, "average_precision"]),
            "prediction_vs_knownness_spearman": _safe_spearman(valid["oof_prediction"], valid["knownness_score"]),
        }
        for cohort_name, mask in (
            ("lower_half_knownness", valid["knownness_half"].astype(str).eq("lower_half")),
            ("lowest_knownness_quartile", valid["knownness_quartile"].astype(str).eq("q1_lowest")),
        ):
            cohort = valid.loc[mask].copy()
            row[f"{cohort_name}_n_backbones"] = int(len(cohort))
            if not cohort.empty and cohort["spread_label"].nunique() >= 2:
                row[f"{cohort_name}_roc_auc"] = roc_auc_score(cohort["spread_label"], cohort["oof_prediction"])
            else:
                row[f"{cohort_name}_roc_auc"] = np.nan
        matched_row = knownness_matched_validation.loc[
            (knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str)).astype(str) == "__weighted_overall__")
            & (knownness_matched_validation.get("model_name", pd.Series(dtype=str)).astype(str) == model_name)
        ]
        if not matched_row.empty:
            if "weighted_mean_roc_auc" in matched_row.columns and pd.notna(matched_row.iloc[0]["weighted_mean_roc_auc"]):
                row["matched_knownness_weighted_roc_auc"] = float(matched_row.iloc[0]["weighted_mean_roc_auc"])
            elif "roc_auc" in matched_row.columns and pd.notna(matched_row.iloc[0]["roc_auc"]):
                row["matched_knownness_weighted_roc_auc"] = float(matched_row.iloc[0]["roc_auc"])
            else:
                row["matched_knownness_weighted_roc_auc"] = np.nan
        else:
            row["matched_knownness_weighted_roc_auc"] = np.nan
        source_rows = group_holdout.loc[
            (group_holdout.get("group_column", pd.Series(dtype=str)).astype(str) == "dominant_source")
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
    }
    component_columns: list[str] = []
    n_models = max(len(scorecard), 1)
    for column, ascending in scoring_directions.items():
        values = pd.to_numeric(scorecard[column], errors="coerce")
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
        column for column in component_columns
        if scorecard[column].notna().any()
    ]
    if available_component_columns:
        scorecard["selection_composite_score"] = (
            scorecard[available_component_columns]
            .fillna(0.0)
            .mean(axis=1)
        )
        scorecard["selection_metric_count"] = scorecard[available_component_columns].notna().sum(axis=1).astype(int)
        scorecard["selection_missing_metric_count"] = (
            len(available_component_columns) - scorecard["selection_metric_count"]
        ).astype(int)
    else:
        scorecard["selection_composite_score"] = np.nan
        scorecard["selection_metric_count"] = 0
        scorecard["selection_missing_metric_count"] = 0
    scorecard = scorecard.sort_values(
        ["selection_composite_score", "roc_auc", "average_precision"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    scorecard["selection_rank"] = np.arange(1, len(scorecard) + 1)
    return scorecard
