"""Candidate portfolio, dossier, and risk tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from plasmid_priority.utils.dataframe import coalescing_left_merge


def _annotate_source_support_tier(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if working.empty:
        working["source_support_tier"] = pd.Series(dtype=str)
        return working
    dominant_source_share = (
        working[["refseq_share_train", "insd_share_train"]].fillna(0.0).max(axis=1)
        if {"refseq_share_train", "insd_share_train"}.issubset(working.columns)
        else pd.Series(0.0, index=working.index)
    )
    cross_source_share = (
        working[["refseq_share_train", "insd_share_train"]].fillna(0.0).min(axis=1)
        if {"refseq_share_train", "insd_share_train"}.issubset(working.columns)
        else pd.Series(0.0, index=working.index)
    )
    working["source_support_tier"] = np.select(
        [
            cross_source_share >= 0.15,
            working.get("refseq_share_train", pd.Series(0.0, index=working.index)).fillna(0.0)
            >= 0.85,
            working.get("insd_share_train", pd.Series(0.0, index=working.index)).fillna(0.0)
            >= 0.85,
            dominant_source_share >= 0.60,
        ],
        ["cross_source_supported", "refseq_dominant", "insd_dominant", "source_mixed"],
        default="source_sparse",
    )
    return working


def _select_diversified_head(
    frame: pd.DataFrame,
    n: int,
    *,
    tier_column: str = "source_support_tier",
    sort_columns: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    sort_columns = [column for column in (sort_columns or []) if column in working.columns]
    if sort_columns:
        sort_ascending = list(ascending or [])
        if len(sort_ascending) < len(sort_columns):
            sort_ascending.extend([True] * (len(sort_columns) - len(sort_ascending)))
        working = working.sort_values(
            sort_columns,
            ascending=sort_ascending[: len(sort_columns)],
            kind="mergesort",
        )
    if tier_column not in working.columns:
        return working.head(n).reset_index(drop=True)
    tier_order = [
        "cross_source_supported",
        "source_mixed",
        "refseq_dominant",
        "insd_dominant",
        "source_sparse",
    ]
    tier_values = working[tier_column].fillna("source_sparse").astype(str)
    unique_tiers = [tier for tier in tier_order if tier in set(tier_values)]
    extra_tiers = [tier for tier in tier_values.unique() if tier not in unique_tiers]
    tier_sequence = unique_tiers + extra_tiers
    grouped = {tier: working.loc[tier_values.eq(tier)].copy() for tier in tier_sequence}
    selected: list[pd.DataFrame] = []
    while len(selected) < n and any(not group.empty for group in grouped.values()):
        progressed = False
        for tier in tier_sequence:
            group = grouped.get(tier)
            if group is None or group.empty:
                continue
            selected.append(group.iloc[[0]].copy())
            grouped[tier] = group.iloc[1:].copy()
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    if not selected:
        return working.head(n).reset_index(drop=True)
    result = pd.concat(selected, ignore_index=True, sort=False)
    if "backbone_id" in result.columns:
        result = result.drop_duplicates(subset=["backbone_id"], keep="first")
    if len(result) < n:
        used_ids = (
            set(result["backbone_id"].astype(str)) if "backbone_id" in result.columns else set()
        )
        if "backbone_id" in working.columns:
            remaining = working.loc[~working["backbone_id"].astype(str).isin(used_ids)].copy()
        else:
            remaining = working.copy()
        if sort_columns:
            sort_ascending = list(ascending or [])
            if len(sort_ascending) < len(sort_columns):
                sort_ascending.extend([True] * (len(sort_columns) - len(sort_ascending)))
            remaining = remaining.sort_values(
                sort_columns,
                ascending=sort_ascending[: len(sort_columns)],
                kind="mergesort",
            )
        result = pd.concat([result, remaining.head(n - len(result))], ignore_index=True, sort=False)
    return result.head(n).reset_index(drop=True)


def _low_knownness_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if "knownness_quartile" in frame.columns:
        return frame["knownness_quartile"].astype(str).eq("q1_lowest")
    if "knownness_half" in frame.columns:
        return frame["knownness_half"].astype(str).eq("lower_half")
    return pd.Series(False, index=frame.index)


def _uncertainty_review_tier(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    risk_uncertainty = pd.to_numeric(
        frame.get("risk_uncertainty", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    risk_abstain = (
        frame.get("risk_abstain_flag", pd.Series(False, index=frame.index, dtype=bool))
        .fillna(False)
        .astype(bool)
    )
    confidence_tier = (
        frame.get("candidate_confidence_tier", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    false_positive_risk = (
        frame.get("false_positive_risk_tier", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    return pd.Series(
        np.select(
            [
                risk_abstain | risk_uncertainty.ge(0.20),
                risk_uncertainty.ge(0.10)
                | confidence_tier.eq("watchlist")
                | false_positive_risk.eq("high"),
            ],
            ["abstain", "review"],
            default="clear",
        ),
        index=frame.index,
    )


def _ensure_low_knownness_coverage(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    target_fraction: float = 0.2,
) -> pd.DataFrame:
    if selected.empty or pool.empty or target_fraction <= 0:
        return selected
    working = selected.copy()
    pool_working = pool.copy()
    if "backbone_id" in working.columns and "backbone_id" in pool_working.columns:
        pool_working = pool_working.loc[
            ~pool_working["backbone_id"].astype(str).isin(working["backbone_id"].astype(str))
        ].copy()
    low_selected = _low_knownness_mask(working)
    low_pool = _low_knownness_mask(pool_working)
    if not low_pool.any():
        return working
    target_count = max(1, int(math.ceil(len(working) * target_fraction)))
    deficit = target_count - int(low_selected.sum())
    if deficit <= 0:
        return working
    low_candidates = pool_working.loc[low_pool].copy()
    if low_candidates.empty:
        return working
    overflow = 0
    if len(working) + len(low_candidates.head(deficit)) > len(selected):
        overflow = len(working) + len(low_candidates.head(deficit)) - len(selected)
    additions = low_candidates.head(deficit).copy()
    if not additions.empty:
        working = pd.concat([working, additions], ignore_index=True, sort=False)
    if len(working) <= len(selected):
        return working.head(len(selected)).reset_index(drop=True)
    if overflow <= 0:
        overflow = len(working) - len(selected)
    if overflow <= 0:
        return working.head(len(selected)).reset_index(drop=True)
    drop_candidates = working.index[~_low_knownness_mask(working)]
    drop_indices = list(drop_candidates[-overflow:])
    if len(drop_indices) < overflow:
        remaining = [idx for idx in working.index if idx not in drop_indices]
        drop_indices.extend(remaining[-(overflow - len(drop_indices)) :])
    working = working.drop(index=drop_indices).reset_index(drop=True)
    if len(working) > len(selected):
        working = working.head(len(selected)).reset_index(drop=True)
    return working


def build_candidate_portfolio_table(
    candidate_dossiers: pd.DataFrame,
    novelty_watchlist: pd.DataFrame,
    *,
    established_n: int = 10,
    novel_n: int = 10,
) -> pd.DataFrame:
    """Create a two-track candidate portfolio with knownness coverage constraints."""
    established = pd.DataFrame()
    if not candidate_dossiers.empty:
        established_pool = candidate_dossiers.copy()
        if "eligible_for_oof" in established_pool.columns:
            established_pool = established_pool.loc[
                established_pool["eligible_for_oof"].fillna(False)
            ].copy()
        positive_column = next(
            (
                column
                for column in ("visibility_expansion_label", "spread_label")
                if column in established_pool.columns
            ),
            None,
        )
        if positive_column is not None:
            positive_pool = established_pool.loc[
                established_pool[positive_column].fillna(0).astype(float) >= 1.0
            ].copy()
            non_positive_pool = established_pool.loc[
                ~established_pool.index.isin(positive_pool.index)
            ].copy()
        else:
            positive_pool = established_pool.copy()
            non_positive_pool = established_pool.iloc[0:0].copy()
        if "false_positive_risk_tier" in established_pool.columns:
            preferred = positive_pool.loc[
                positive_pool["false_positive_risk_tier"].fillna("high").ne("high")
            ].copy()
            fallback = pd.concat(
                [
                    positive_pool.loc[
                        positive_pool["false_positive_risk_tier"].fillna("high").eq("high")
                    ].copy(),
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
                frame["confidence_sort"] = (
                    frame["candidate_confidence_tier"].map(confidence_order).fillna(3)
                )
            else:
                frame["confidence_sort"] = 3
        sort_columns = [
            column
            for column in ["confidence_sort", "consensus_rank"]
            if column in preferred.columns
        ]
        ascending = [True] * len(sort_columns)
        if "primary_model_candidate_score" in preferred.columns:
            sort_columns.append("primary_model_candidate_score")
            ascending.append(False)
        if sort_columns:
            preferred = preferred.sort_values(sort_columns, ascending=ascending)
            fallback = fallback.sort_values(sort_columns, ascending=ascending)
        preferred = _annotate_source_support_tier(preferred)
        fallback = _annotate_source_support_tier(fallback)
        preferred_selected = _select_diversified_head(
            preferred,
            established_n,
            tier_column="source_support_tier",
            sort_columns=sort_columns,
            ascending=ascending,
        )
        fallback_n = max(established_n - len(preferred_selected), 0)
        fallback_selected = _select_diversified_head(
            fallback,
            fallback_n,
            tier_column="source_support_tier",
            sort_columns=sort_columns,
            ascending=ascending,
        )
        established = pd.concat(
            [preferred_selected, fallback_selected], ignore_index=True, sort=False
        )
        if not established.empty and "backbone_id" in established.columns:
            established = established.drop_duplicates(subset=["backbone_id"], keep="first").head(
                established_n
            )
            established = _ensure_low_knownness_coverage(
                established,
                pd.concat([preferred, fallback], ignore_index=True, sort=False),
                target_fraction=0.2,
            ).head(established_n)
    if not established.empty:
        established = established.assign(
            portfolio_track="established_high_risk",
            track_rank=np.arange(1, len(established) + 1),
        )

    novel = pd.DataFrame()
    if not novelty_watchlist.empty:
        novel_pool = novelty_watchlist.copy()
        novelty_mask = (
            novel_pool.get(
                "novelty_margin_vs_baseline", pd.Series(0.0, index=novel_pool.index)
            ).fillna(0.0)
            > 0
        )
        support_mask = (
            novel_pool.get(
                "external_support_modalities_count",
                pd.Series(0, index=novel_pool.index),
            )
            .fillna(0)
            .astype(int)
            > 0
        )
        train_support_mask = (
            novel_pool.get(
                "member_count_train",
                pd.Series(0, index=novel_pool.index),
            )
            .fillna(0)
            .astype(int)
            >= 2
        )
        positive_column = next(
            (
                column
                for column in ("visibility_expansion_label", "spread_label")
                if column in novel_pool.columns
            ),
            None,
        )
        positive_mask = (
            novel_pool[positive_column].fillna(0).astype(float) >= 1.0
            if positive_column is not None
            else pd.Series(True, index=novel_pool.index)
        )
        preferred = novel_pool.loc[
            positive_mask & novelty_mask & (support_mask | train_support_mask)
        ].copy()
        fallback = novel_pool.loc[~novel_pool.index.isin(preferred.index)].copy()
        novel_sort_columns: list[str] = []
        novel_ascending: list[bool] = []
        if "novelty_margin_vs_baseline" in preferred.columns:
            novel_sort_columns.append("novelty_margin_vs_baseline")
            novel_ascending.append(False)
        if "primary_model_candidate_score" in preferred.columns:
            novel_sort_columns.append("primary_model_candidate_score")
            novel_ascending.append(False)
        if novel_sort_columns:
            preferred = preferred.sort_values(novel_sort_columns, ascending=novel_ascending)
            fallback = fallback.sort_values(novel_sort_columns, ascending=novel_ascending)
        preferred = _annotate_source_support_tier(preferred)
        fallback = _annotate_source_support_tier(fallback)
        preferred_selected = _select_diversified_head(
            preferred,
            novel_n,
            tier_column="source_support_tier",
            sort_columns=novel_sort_columns,
            ascending=novel_ascending,
        )
        fallback_n = max(novel_n - len(preferred_selected), 0)
        fallback_selected = _select_diversified_head(
            fallback,
            fallback_n,
            tier_column="source_support_tier",
            sort_columns=novel_sort_columns,
            ascending=novel_ascending,
        )
        novel = pd.concat([preferred_selected, fallback_selected], ignore_index=True, sort=False)
        if not novel.empty and "backbone_id" in novel.columns:
            novel = novel.drop_duplicates(subset=["backbone_id"], keep="first").head(novel_n)
            novel = _ensure_low_knownness_coverage(
                novel,
                pd.concat([preferred, fallback], ignore_index=True, sort=False),
                target_fraction=0.2,
            ).head(novel_n)
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
    if (
        "primary_model_candidate_score" not in combined.columns
        and "primary_model_oof_prediction" in combined.columns
    ):
        combined["primary_model_candidate_score"] = combined["primary_model_oof_prediction"]
    if (
        "baseline_both_candidate_score" not in combined.columns
        and "baseline_both_oof_prediction" in combined.columns
    ):
        combined["baseline_both_candidate_score"] = combined["baseline_both_oof_prediction"]
    combined = combined.copy()
    if "candidate_prediction_source" not in combined.columns:
        combined["candidate_prediction_source"] = np.where(
            combined.get(
                "primary_model_oof_prediction", pd.Series(np.nan, index=combined.index)
            ).notna(),
            "oof",
            "missing",
        )
    if "eligible_for_oof" not in combined.columns:
        combined["eligible_for_oof"] = combined.get(
            "primary_model_oof_prediction", pd.Series(np.nan, index=combined.index)
        ).notna()
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
        support_matrix = pd.DataFrame(
            {column: combined[column] for column in available_support_columns},
            index=combined.index,
        )
        combined["support_profile_available"] = support_matrix.notna().any(axis=1)
        combined["external_support_modalities_count"] = (
            support_matrix.fillna(False).sum(axis=1).astype(int)
        )
    else:
        combined["support_profile_available"] = False
        combined["external_support_modalities_count"] = 0
    dominant_source_share = (
        combined[["refseq_share_train", "insd_share_train"]].fillna(0.0).max(axis=1)
        if {"refseq_share_train", "insd_share_train"}.issubset(combined.columns)
        else pd.Series(0.0, index=combined.index)
    )
    cross_source_share = (
        combined[["refseq_share_train", "insd_share_train"]].fillna(0.0).min(axis=1)
        if {"refseq_share_train", "insd_share_train"}.issubset(combined.columns)
        else pd.Series(0.0, index=combined.index)
    )
    source_support_tier = np.select(
        [
            cross_source_share >= 0.15,
            combined.get("refseq_share_train", pd.Series(0.0, index=combined.index)).fillna(0.0)
            >= 0.85,
            combined.get("insd_share_train", pd.Series(0.0, index=combined.index)).fillna(0.0)
            >= 0.85,
            dominant_source_share >= 0.60,
        ],
        ["cross_source_supported", "refseq_dominant", "insd_dominant", "source_mixed"],
        default="source_sparse",
    )
    bootstrap_top10 = combined.get(
        "bootstrap_top_10_frequency", pd.Series(0.0, index=combined.index)
    ).fillna(0.0)
    risk_tier = (
        combined.get("false_positive_risk_tier", pd.Series("unknown", index=combined.index))
        .fillna("unknown")
        .astype(str)
    )
    recommended_monitoring_tier = np.select(
        [
            (bootstrap_top10 >= 0.80) & risk_tier.isin(["low", "medium"]),
            (bootstrap_top10 >= 0.50) & risk_tier.ne("high"),
        ],
        ["core_surveillance", "extended_watchlist"],
        default="low_confidence_backlog",
    )
    combined_updates = pd.DataFrame(
        {
            "in_consensus_top50": combined.get(
                "consensus_rank", pd.Series(np.nan, index=combined.index)
            ).notna(),
            "support_profile_available": (
                support_matrix.notna().any(axis=1)
                if available_support_columns
                else pd.Series(False, index=combined.index)
            ),
            "external_support_modalities_count": (
                support_matrix.fillna(False).sum(axis=1).astype(int)
                if available_support_columns
                else pd.Series(0, index=combined.index, dtype=int)
            ),
            "source_support_tier": source_support_tier,
            "recommended_monitoring_tier": recommended_monitoring_tier,
            "evidence_tier": combined.get(
                "candidate_confidence_tier", pd.Series("unknown", index=combined.index)
            ).fillna("unknown"),
            "action_tier": combined.get(
                "recommended_monitoring_tier", pd.Series("unassigned", index=combined.index)
            ).fillna("unassigned"),
        },
        index=combined.index,
    )
    combined = pd.concat(
        [combined.drop(columns=list(combined_updates.columns), errors="ignore"), combined_updates],
        axis=1,
    )
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
        "primary_model_full_fit_prediction_std",
        "primary_model_full_fit_prediction_ci_lower",
        "primary_model_full_fit_prediction_ci_upper",
        "conservative_model_full_fit_prediction_std",
        "conservative_model_full_fit_prediction_ci_lower",
        "conservative_model_full_fit_prediction_ci_upper",
        "assignment_confidence_score",
        "mash_graph_novelty_score",
        "mash_graph_bridge_fraction",
        "mash_graph_external_neighbor_count",
        "amr_agreement_score",
        "mean_amr_uncertainty_score",
        "model_prediction_uncertainty",
        "primary_minus_conservative_uncertainty",
        "novelty_margin_vs_baseline",
        "risk_spread_probability",
        "risk_spread_severity",
        "risk_macro_region_jump_3y",
        "risk_event_within_3y",
        "risk_three_countries_within_5y",
        "operational_risk_score",
        "risk_component_std",
        "risk_uncertainty",
        "risk_abstain_flag",
        "risk_decision_tier",
        "uncertainty_review_tier",
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
            "primary_model_full_fit_prediction_std",
            "primary_model_full_fit_prediction_ci_lower",
            "primary_model_full_fit_prediction_ci_upper",
            "baseline_both_full_fit_prediction",
            "baseline_both_full_fit_prediction_std",
            "baseline_both_full_fit_prediction_ci_lower",
            "baseline_both_full_fit_prediction_ci_upper",
            "conservative_model_full_fit_prediction",
            "conservative_model_full_fit_prediction_std",
            "conservative_model_full_fit_prediction_ci_lower",
            "conservative_model_full_fit_prediction_ci_upper",
            "assignment_confidence_score",
            "mash_graph_novelty_score",
            "mash_graph_bridge_fraction",
            "mash_graph_external_neighbor_count",
            "amr_agreement_score",
            "mean_amr_uncertainty_score",
            "novelty_margin_vs_baseline",
            "risk_spread_probability",
            "risk_spread_severity",
            "risk_macro_region_jump_3y",
            "risk_event_within_3y",
            "risk_three_countries_within_5y",
            "operational_risk_score",
            "risk_component_std",
            "risk_uncertainty",
            "risk_abstain_flag",
            "risk_decision_tier",
            "risk_route_context",
        ]
        stability_payload = candidate_stability[
            [column for column in stability_columns if column in candidate_stability.columns]
        ].copy()
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
            who_detail[
                [
                    "backbone_id",
                    "who_mia_any_support",
                    "who_mia_any_hpecia",
                    "who_mia_mapped_fraction",
                ]
            ],
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
            mobsuite_detail[
                ["backbone_id", "mobsuite_any_literature_support", "mobsuite_any_cluster_support"]
            ],
            on="backbone_id",
        )
    if not pathogen_support.empty and "pathogen_dataset" in pathogen_support.columns:
        combined = pathogen_support.loc[
            pathogen_support["pathogen_dataset"] == "combined",
            ["backbone_id", "pd_any_support", "pd_matching_fraction"],
        ]
        dossier = coalescing_left_merge(dossier, combined, on="backbone_id")
    if not amrfinder_detail.empty:
        amrfinder_summary = amrfinder_detail.groupby("backbone_id", as_index=False).agg(
            amrfinder_any_hit=("amrfinder_any_hit", "max"),
            amrfinder_mean_gene_jaccard=("gene_jaccard", "mean"),
            amrfinder_mean_class_jaccard=("class_jaccard", "mean"),
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
    support_matrix = pd.DataFrame(
        {column: dossier[column] for column in support_columns}, index=dossier.index
    )
    dossier["support_profile_available"] = support_matrix.notna().any(axis=1)
    dossier["external_support_modalities_count"] = (
        support_matrix.fillna(False).sum(axis=1).astype(int)
    )
    primary_oof = dossier.get(
        "primary_model_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)
    ).astype(float)
    primary_full = dossier.get(
        "primary_model_full_fit_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)
    ).astype(float)
    primary_full_std = dossier.get(
        "primary_model_full_fit_prediction_std",
        pd.Series(np.nan, index=dossier.index, dtype=float),
    ).astype(float)
    baseline_oof = dossier.get(
        "baseline_both_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)
    ).astype(float)
    baseline_full = dossier.get(
        "baseline_both_full_fit_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)
    ).astype(float)
    conservative_oof = dossier.get(
        "conservative_model_oof_prediction", pd.Series(np.nan, index=dossier.index, dtype=float)
    ).astype(float)
    conservative_full = dossier.get(
        "conservative_model_full_fit_prediction",
        pd.Series(np.nan, index=dossier.index, dtype=float),
    ).astype(float)
    conservative_full_std = dossier.get(
        "conservative_model_full_fit_prediction_std",
        pd.Series(np.nan, index=dossier.index, dtype=float),
    ).astype(float)

    dossier["primary_model_candidate_score"] = primary_oof.fillna(primary_full)
    dossier["baseline_both_candidate_score"] = baseline_oof.fillna(baseline_full)
    dossier["conservative_model_candidate_score"] = conservative_oof.fillna(conservative_full)
    dossier["model_prediction_uncertainty"] = primary_full_std
    dossier["candidate_prediction_source"] = np.where(
        primary_oof.notna(),
        "oof",
        np.where(primary_full.notna(), "full_fit", "missing"),
    )
    dossier["eligible_for_oof"] = primary_oof.notna()
    dossier["primary_minus_conservative_prediction"] = dossier[
        "primary_model_candidate_score"
    ].fillna(0.0) - dossier["conservative_model_candidate_score"].fillna(0.0)
    dossier["novelty_margin_vs_baseline"] = dossier["primary_model_candidate_score"].fillna(
        0.0
    ) - dossier["baseline_both_candidate_score"].fillna(0.0)
    if "risk_uncertainty" not in dossier.columns:
        dossier["risk_uncertainty"] = primary_full_std
    else:
        dossier["risk_uncertainty"] = pd.to_numeric(
            dossier["risk_uncertainty"], errors="coerce"
        ).fillna(primary_full_std)
    combined_uncertainty = (
        primary_full_std.fillna(0.0) ** 2 + conservative_full_std.fillna(0.0) ** 2
    )
    dossier["primary_minus_conservative_uncertainty"] = np.sqrt(
        np.clip(combined_uncertainty, 0.0, None)
    )

    bootstrap = dossier.get(
        "bootstrap_top_k_frequency", pd.Series(0.0, index=dossier.index)
    ).fillna(0.0)
    variant = dossier.get("variant_top_k_frequency", pd.Series(0.0, index=dossier.index)).fillna(
        0.0
    )
    coherence = dossier.get("coherence_score", pd.Series(0.0, index=dossier.index)).fillna(0.0)
    consensus_support = (
        dossier.get("consensus_support_count", pd.Series(0, index=dossier.index))
        .fillna(0)
        .astype(int)
    )
    disagreement = dossier.get(
        "rank_disagreement_primary_vs_conservative",
        pd.Series(np.inf, index=dossier.index, dtype=float),
    ).fillna(np.inf)
    stability_available = (
        dossier.get("bootstrap_top_k_frequency", pd.Series(np.nan, index=dossier.index)).notna()
        | dossier.get(
            "variant_top_k_frequency",
            pd.Series(np.nan, index=dossier.index),
        ).notna()
    )
    dossier["candidate_confidence_tier"] = np.select(
        [
            (coherence >= 0.60)
            & (consensus_support >= 2)
            & (
                ((bootstrap >= 0.85) & (variant >= 0.75))
                | ((~stability_available) & (disagreement <= 50))
            ),
            (coherence >= 0.50)
            & (consensus_support >= 2)
            & (((bootstrap >= 0.70) & (variant >= 0.60)) | (~stability_available)),
        ],
        ["tier_a", "tier_b"],
        default="watchlist",
    )
    dossier["uncertainty_review_tier"] = _uncertainty_review_tier(dossier)
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
        ("coherence_score", "mobility_coherence_t"),
        ("assignment_confidence_score", "assignment_confidence"),
        ("mash_graph_novelty_score", "graph_novelty"),
        ("mash_graph_bridge_fraction", "graph_bridge"),
        ("amr_agreement_score", "amr_agreement"),
        ("H_external_host_range_support", "external_host_support"),
        ("replicon_architecture_norm", "replicon_complexity"),
        ("mash_novelty_norm", "graph_novelty"),
        ("backbone_purity_score", "backbone_purity"),
    ]
    available_signals: list[tuple[str, pd.Series]] = []
    for column, label in signal_specs:
        if column in working.columns:
            available_signals.append((label, pd.to_numeric(working[column], errors="coerce")))

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
    confidence_tier = (
        working.get("candidate_confidence_tier", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
    )
    false_positive_risk = (
        working.get("false_positive_risk_tier", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
    )
    uncertainty_review_tier = (
        working.get("uncertainty_review_tier", _uncertainty_review_tier(working))
        .fillna("clear")
        .astype(str)
    )
    prediction_source = (
        working.get("candidate_prediction_source", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
    )
    model_prediction_uncertainty = pd.to_numeric(
        working.get("model_prediction_uncertainty", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    assignment_confidence = pd.to_numeric(
        working.get("assignment_confidence_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    mash_graph_novelty = pd.to_numeric(
        working.get("mash_graph_novelty_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    mash_graph_bridge = pd.to_numeric(
        working.get("mash_graph_bridge_fraction", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    amr_agreement = pd.to_numeric(
        working.get("amr_agreement_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    operational_risk = pd.to_numeric(
        working.get("operational_risk_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    macro_jump_risk = pd.to_numeric(
        working.get("risk_macro_region_jump_3y", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    event_3y_risk = pd.to_numeric(
        working.get("risk_event_within_3y", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    three_country_risk = pd.to_numeric(
        working.get("risk_three_countries_within_5y", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
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
        assignment_confidence_value = assignment_confidence.loc[idx]
        amr_agreement_value = amr_agreement.loc[idx]
        mash_graph_novelty_value = mash_graph_novelty.loc[idx]
        mash_graph_bridge_value = mash_graph_bridge.loc[idx]
        if confidence_tier.loc[idx] in {"tier_a", "tier_b"}:
            monitoring_flags.append("stable_internal_signal")
        if external_support_count.loc[idx] >= 2:
            monitoring_flags.append("multi_modal_support")
        if pd.notna(assignment_confidence_value) and float(assignment_confidence_value) >= 0.85:
            monitoring_flags.append("assignment_confidence_high")
        if pd.notna(assignment_confidence_value) and float(assignment_confidence_value) <= 0.60:
            monitoring_flags.append("assignment_confidence_low")
        if novelty_margin.loc[idx] >= 0.10:
            monitoring_flags.append("outperforms_counts_baseline")
        if max(float(bootstrap_top10.loc[idx]), float(variant_top10.loc[idx])) >= 0.60:
            monitoring_flags.append("multiverse_stable")
        if pd.notna(amr_agreement_value) and float(amr_agreement_value) >= 0.75:
            monitoring_flags.append("amr_call_agreement")
        if pd.notna(mash_graph_novelty_value) and float(mash_graph_novelty_value) >= 0.75:
            monitoring_flags.append("graph_novelty_high")
        if pd.notna(mash_graph_bridge_value) and float(mash_graph_bridge_value) >= 0.50:
            monitoring_flags.append("graph_bridge_pattern")
        if false_positive_risk.loc[idx] == "low":
            monitoring_flags.append("manageable_false_positive_risk")
        if prediction_source.loc[idx] == "oof":
            monitoring_flags.append("oof_supported")
        if (
            pd.notna(model_prediction_uncertainty.loc[idx])
            and float(model_prediction_uncertainty.loc[idx]) >= 0.15
        ):
            monitoring_flags.append("posterior_uncertainty_elevated")
        if pd.notna(operational_risk.loc[idx]) and float(operational_risk.loc[idx]) >= 0.75:
            monitoring_flags.append("operationally_urgent")
        if pd.notna(event_3y_risk.loc[idx]) and float(event_3y_risk.loc[idx]) >= 0.65:
            monitoring_flags.append("short_horizon_signal")
        if pd.notna(macro_jump_risk.loc[idx]) and float(macro_jump_risk.loc[idx]) >= 0.65:
            monitoring_flags.append("macro_region_jump_signal")
        if pd.notna(three_country_risk.loc[idx]) and float(three_country_risk.loc[idx]) >= 0.65:
            monitoring_flags.append("multi_country_escalation_signal")
        if uncertainty_review_tier.loc[idx] == "review":
            monitoring_flags.append("uncertainty_review_required")
        elif uncertainty_review_tier.loc[idx] == "abstain":
            monitoring_flags.append("uncertainty_abstain_recommended")
        monitoring_text.append(
            ",".join(monitoring_flags[:4]) if monitoring_flags else "limited_monitoring_context"
        )

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
    member_count_train = working.get(
        "member_count_train", pd.Series(np.nan, index=working.index, dtype=float)
    )
    n_countries_train = working.get(
        "n_countries_train", pd.Series(np.nan, index=working.index, dtype=float)
    )
    refseq_share_train = working.get(
        "refseq_share_train", pd.Series(np.nan, index=working.index, dtype=float)
    )
    insd_share_train = working.get(
        "insd_share_train", pd.Series(np.nan, index=working.index, dtype=float)
    )
    assignment_confidence = pd.to_numeric(
        working.get("assignment_confidence_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    mash_graph_novelty = pd.to_numeric(
        working.get("mash_graph_novelty_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )
    amr_uncertainty = pd.to_numeric(
        working.get("mean_amr_uncertainty_score", pd.Series(np.nan, index=working.index)),
        errors="coerce",
    )

    working["low_coherence_risk"] = coherence.fillna(0.0) < 0.50
    working["sparse_training_support_risk"] = member_count_train.fillna(0).astype(int) <= 2
    working["narrow_geography_risk"] = n_countries_train.fillna(0).astype(int) <= 1
    dominant_source_share = (
        pd.concat([refseq_share_train, insd_share_train], axis=1).fillna(0.0).max(axis=1)
    )
    source_info_available = refseq_share_train.notna() | insd_share_train.notna()
    working["source_concentration_risk"] = source_info_available & dominant_source_share.ge(0.90)
    working["low_assignment_confidence_risk"] = (
        assignment_confidence.fillna(1.0).astype(float) < 0.60
    )
    working["graph_novelty_risk"] = mash_graph_novelty.fillna(0.0).astype(float) >= 0.75
    working["amr_uncertainty_risk"] = amr_uncertainty.fillna(1.0).astype(float) >= 0.40
    support_available = (
        working.get("support_profile_available", pd.Series(False, index=working.index, dtype=bool))
        .fillna(False)
        .astype(bool)
    )
    external_support_count = (
        working.get(
            "external_support_modalities_count",
            pd.Series(0, index=working.index, dtype=float),
        )
        .fillna(0)
        .astype(int)
    )
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
    working["stability_risk"] = stability_available & (
        (bootstrap_frequency < 0.50) | (variant_frequency < 0.50)
    )
    working["proxy_gap_risk"] = (
        working.get(
            "primary_minus_conservative_prediction",
            pd.Series(0.0, index=working.index, dtype=float),
        ).fillna(0.0)
        >= 0.15
    )
    working["uncertainty_abstain_risk"] = (
        working.get("risk_abstain_flag", pd.Series(False, index=working.index, dtype=bool))
        .fillna(False)
        .astype(bool)
    )

    risk_columns = [
        "low_coherence_risk",
        "sparse_training_support_risk",
        "narrow_geography_risk",
        "source_concentration_risk",
        "low_assignment_confidence_risk",
        "graph_novelty_risk",
        "amr_uncertainty_risk",
        "weak_external_support_risk",
        "stability_risk",
        "proxy_gap_risk",
        "uncertainty_abstain_risk",
    ]
    working["risk_flag_count"] = working[risk_columns].sum(axis=1).astype(int)
    working["false_positive_risk_tier"] = np.select(
        [working["risk_flag_count"] >= 3, working["risk_flag_count"] >= 1],
        ["high", "medium"],
        default="low",
    )
    working["uncertainty_review_tier"] = _uncertainty_review_tier(working)
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
        "assignment_confidence_score",
        "mash_graph_novelty_score",
        "mean_amr_uncertainty_score",
        "primary_minus_conservative_prediction",
        "uncertainty_review_tier",
    ] + risk_columns
    available = [column for column in columns if column in working.columns]
    sort_columns = [
        column for column in ["freeze_rank", "risk_flag_count"] if column in working.columns
    ]
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
    """Summarize shortlist precision and recall at practical candidate-list sizes."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = predictions.loc[predictions["model_name"] == model_name].copy()
        if frame.empty:
            continue
        frame = frame.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
        y = frame["spread_label"].to_numpy(dtype=int)
        prevalence = float(np.mean(y)) if len(y) else np.nan
        total_positive = max(int(y.sum()), 0)
        for top_k in top_ks:
            subset = frame.head(min(top_k, len(frame))).copy()
            selected_positive = int(subset["spread_label"].sum()) if not subset.empty else 0
            selected_negative = int(len(subset) - selected_positive)
            precision_at_k = float(selected_positive / len(subset)) if len(subset) else np.nan
            recall_at_k = (
                float(selected_positive / total_positive) if total_positive > 0 else np.nan
            )
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
                    "precision_lift_vs_prevalence": float(precision_at_k - prevalence)
                    if np.isfinite(precision_at_k) and np.isfinite(prevalence)
                    else np.nan,
                    "mean_prediction_at_k": float(subset["oof_prediction"].mean())
                    if len(subset)
                    else np.nan,
                    "min_prediction_at_k": float(subset["oof_prediction"].min())
                    if len(subset)
                    else np.nan,
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
            status_by_threshold[threshold] = (
                int(n_new >= threshold) if 1 <= train_countries <= 3 else np.nan
            )
        default_status = status_by_threshold[default_threshold]
        finite_statuses = [value for value in status_by_threshold.values() if pd.notna(value)]
        flip_count = (
            int(sum(value != default_status for value in finite_statuses)) if finite_statuses else 0
        )
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
    sort_columns = [
        column for column in ["threshold_flip_count", "priority_index"] if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False, False][: len(sort_columns)])
    return frame.reset_index(drop=True)
