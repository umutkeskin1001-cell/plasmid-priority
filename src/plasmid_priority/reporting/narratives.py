"""Narrative markdown renderers for report assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _format_interval(lower: object, upper: object, *, digits: int = 3) -> str:
    lower_value = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
    upper_value = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
    if pd.isna(lower_value) or pd.isna(upper_value):
        return "NA"
    return f"[{float(lower_value):.{digits}f}, {float(upper_value):.{digits}f}]"


def _format_pvalue(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    if float(numeric) < 0.001:
        return "<0.001"
    return f"{float(numeric):.3f}"


def _governance_watch_label() -> str:
    return "Governance watch-only"


def _pretty_report_model_label(model_name: str) -> str:
    labels = {
        "visibility_adjusted_priority": "visibility-adjusted model",
        "bio_clean_priority": "bio-clean model",
        "balanced_evidence_priority": "balanced evidence model",
        "baseline_both": "counts-only baseline",
        "natural_auc_priority": "augmented biological model",
        "phylogeny_aware_priority": "taxonomy-aware H model",
        "structured_signal_priority": "structure-aware biological model",
        "ecology_clinical_priority": "ecology-clinical biological model",
        "knownness_robust_priority": "knownness-robust biological model",
        "support_calibrated_priority": "support-calibrated biological model",
        "support_synergy_priority": "support-synergy biological model",
        "monotonic_latent_priority": "monotonic latent biological model",
        "phylo_support_fusion_priority": "phylo-support fusion model",
        "host_transfer_synergy_priority": "host-transfer synergy biological model",
        "threat_architecture_priority": "threat-architecture biological model",
        "adaptive_natural_priority": "knownness-gated natural audit",
        "adaptive_knownness_robust_priority": "knownness-gated specialist-switch audit",
        "adaptive_knownness_blend_priority": "knownness-gated blended audit",
        "adaptive_support_calibrated_blend_priority": "support-calibrated gated audit",
        "adaptive_support_synergy_blend_priority": "support-synergy gated audit",
        "adaptive_host_transfer_synergy_blend_priority": "host-transfer synergy gated audit",
        "adaptive_threat_architecture_blend_priority": "threat-architecture gated audit",
    }
    return labels.get(str(model_name), str(model_name))


def _lookup_decision_yield(
    decision_yield: pd.DataFrame, model_name: str, top_k: int
) -> pd.Series | None:
    if decision_yield.empty:
        return None
    if "model_name" not in decision_yield.columns or "top_k" not in decision_yield.columns:
        return None
    match = decision_yield.loc[
        (decision_yield["model_name"].astype(str) == str(model_name))
        & (pd.to_numeric(decision_yield["top_k"], errors="coerce") == int(top_k))
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _summarize_false_negative_audit(false_negative_audit: pd.DataFrame) -> tuple[int, str]:
    if false_negative_audit.empty:
        return 0, "none"
    top_flags: dict[str, int] = {}
    for value in (
        false_negative_audit.get("miss_driver_flags", pd.Series(dtype=str)).fillna("").astype(str)
    ):
        for token in [
            part.strip() for part in value.split(",") if part.strip() and part.strip() != "none"
        ]:
            top_flags[token] = top_flags.get(token, 0) + 1
    if not top_flags:
        return int(len(false_negative_audit)), "none"
    ordered = sorted(top_flags.items(), key=lambda item: (-item[1], item[0]))
    return int(len(false_negative_audit)), ", ".join(flag for flag, _ in ordered[:3])


def _select_confirmatory_row(
    confirmatory_cohort_summary: pd.DataFrame,
    *,
    cohort_name: str,
    model_name: str,
) -> pd.Series:
    if confirmatory_cohort_summary.empty:
        return pd.Series(dtype=object)
    match = confirmatory_cohort_summary.loc[
        (
            confirmatory_cohort_summary.get("cohort_name", pd.Series(dtype=str)).astype(str)
            == str(cohort_name)
        )
        & (
            confirmatory_cohort_summary.get("model_name", pd.Series(dtype=str)).astype(str)
            == str(model_name)
        )
        & (confirmatory_cohort_summary.get("status", pd.Series(dtype=str)).astype(str) == "ok")
    ].head(1)
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def _rolling_temporal_summary(rolling_temporal: pd.DataFrame) -> dict[str, object]:
    if rolling_temporal.empty or "status" not in rolling_temporal.columns:
        return {}
    working = rolling_temporal.loc[rolling_temporal["status"].astype(str).eq("ok")].copy()
    if working.empty:
        return {}
    roc_auc = pd.to_numeric(
        working.get("roc_auc", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    average_precision = pd.to_numeric(
        working.get("average_precision", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    brier_score = pd.to_numeric(
        working.get("brier_score", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    split_years = pd.to_numeric(
        working.get("split_year", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    horizons = pd.to_numeric(
        working.get("horizon_years", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    assignment_modes = working.get("backbone_assignment_mode", pd.Series(dtype=str)).astype(str)
    return {
        "n_rows": int(len(working)),
        "split_year_min": int(split_years.min()) if not split_years.empty else None,
        "split_year_max": int(split_years.max()) if not split_years.empty else None,
        "horizon_values": ",".join(str(int(value)) for value in sorted(horizons.unique()))
        if not horizons.empty
        else "",
        "assignment_modes": ",".join(sorted(set(mode for mode in assignment_modes if mode))),
        "roc_auc_mean": float(roc_auc.mean()) if not roc_auc.empty else np.nan,
        "roc_auc_min": float(roc_auc.min()) if not roc_auc.empty else np.nan,
        "roc_auc_max": float(roc_auc.max()) if not roc_auc.empty else np.nan,
        "average_precision_mean": float(average_precision.mean())
        if not average_precision.empty
        else np.nan,
        "average_precision_min": float(average_precision.min())
        if not average_precision.empty
        else np.nan,
        "average_precision_max": float(average_precision.max())
        if not average_precision.empty
        else np.nan,
        "brier_score_mean": float(brier_score.mean()) if not brier_score.empty else np.nan,
    }


def _primary_baseline_delta_text(model_metrics: pd.DataFrame, primary_model_name: str) -> str:
    row = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(primary_model_name)
    ].head(1)
    if row.empty:
        return "NA"
    row = row.iloc[0]
    delta = pd.to_numeric(pd.Series([row.get("delta_vs_baseline_roc_auc")]), errors="coerce").iloc[
        0
    ]
    lower = row.get("delta_vs_baseline_ci_lower")
    upper = row.get("delta_vs_baseline_ci_upper")
    if pd.isna(delta):
        return "NA"
    return f"{float(delta):.3f}, 95% CI {_format_interval(lower, upper)}"


def _blocked_holdout_summary_text(
    blocked_holdout_summary: pd.DataFrame,
    *,
    model_name: str,
) -> str:
    if blocked_holdout_summary.empty or "model_name" not in blocked_holdout_summary.columns:
        return ""
    working = blocked_holdout_summary.loc[
        blocked_holdout_summary["model_name"].astype(str).eq(str(model_name))
    ].head(1)
    if working.empty:
        return ""
    row = working.iloc[0]
    weighted_auc = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.isna(weighted_auc):
        return ""
    group_columns = str(row.get("blocked_holdout_group_columns", "") or "").strip()
    if group_columns:
        group_columns = group_columns.replace(",", " + ")
    else:
        group_columns = "blocked source/region groups"
    group_count = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_group_count")]), errors="coerce"
    ).iloc[0]
    worst_group = str(row.get("worst_blocked_holdout_group", "") or "").strip()
    worst_auc = pd.to_numeric(
        pd.Series([row.get("worst_blocked_holdout_group_roc_auc")]), errors="coerce"
    ).iloc[0]
    text = (
        f"{_pretty_report_model_label(model_name)} blocked holdout audit ({group_columns}): "
        f"weighted ROC AUC `{float(weighted_auc):.3f}`"
    )
    if pd.notna(group_count):
        text += f" across `{int(group_count)}` blocked groups"
    if worst_group and pd.notna(worst_auc):
        text += f"; hardest group `{worst_group}` at ROC AUC `{float(worst_auc):.3f}`"
    return f"{text}."


def _blocked_holdout_summary_text_tr(
    blocked_holdout_summary: pd.DataFrame,
    *,
    model_name: str,
) -> str:
    if blocked_holdout_summary.empty or "model_name" not in blocked_holdout_summary.columns:
        return ""
    working = blocked_holdout_summary.loc[
        blocked_holdout_summary["model_name"].astype(str).eq(str(model_name))
    ].head(1)
    if working.empty:
        return ""
    row = working.iloc[0]
    weighted_auc = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.isna(weighted_auc):
        return ""
    group_columns = str(row.get("blocked_holdout_group_columns", "") or "").strip()
    if group_columns:
        group_columns = group_columns.replace(",", " + ")
    else:
        group_columns = "kaynak/bölge grupları"
    group_count = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_group_count")]), errors="coerce"
    ).iloc[0]
    worst_group = str(row.get("worst_blocked_holdout_group", "") or "").strip()
    worst_auc = pd.to_numeric(
        pd.Series([row.get("worst_blocked_holdout_group_roc_auc")]), errors="coerce"
    ).iloc[0]
    text = (
        f"{_pretty_report_model_label(model_name)} için bloke edilmiş holdout denetimi ({group_columns}): "
        f"ağırlıklı ROC AUC `{float(weighted_auc):.3f}`"
    )
    if pd.notna(group_count):
        text += f" `{int(group_count)}` bloke grup üzerinde"
    if worst_group and pd.notna(worst_auc):
        text += f"; en zor grup `{worst_group}` ROC AUC `{float(worst_auc):.3f}`"
    return f"{text}."


def _country_missingness_summary_text(
    country_missingness_bounds: pd.DataFrame,
    country_missingness_sensitivity: pd.DataFrame,
    *,
    model_name: str,
) -> str:
    if country_missingness_bounds.empty:
        return ""
    working = country_missingness_bounds.loc[
        country_missingness_bounds.get("backbone_id", pd.Series(dtype=str)).astype(str).ne("")
    ].copy()
    if working.empty:
        return ""
    eligible = working.loc[
        working.get("eligible_for_country_bounds", pd.Series(False, index=working.index))
        .fillna(False)
        .astype(bool)
    ].copy()
    if eligible.empty:
        return ""

    def _label_count(column_name: str) -> tuple[int, int]:
        values = pd.to_numeric(
            eligible.get(column_name, pd.Series(np.nan, index=eligible.index)), errors="coerce"
        ).fillna(0.0)
        binary = values.astype(int)
        return int(binary.sum()), int((eligible["label_observed"].fillna(0).astype(int) != binary).sum())

    observed_pos, _ = _label_count("label_observed")
    midpoint_pos, midpoint_flips = _label_count("label_midpoint")
    optimistic_pos, optimistic_flips = _label_count("label_optimistic")
    weighted_pos, weighted_flips = _label_count("label_weighted")

    text = (
        f"{_pretty_report_model_label(model_name)} country-missingness audit "
        f"(`country_missingness_bounds.tsv`, `country_missingness_sensitivity.tsv`): "
        f"observed labels mark {observed_pos}/{len(eligible)} eligible backbones positive; "
        f"midpoint / optimistic / weighted interpretations shift {midpoint_flips}/{optimistic_flips}/{weighted_flips} labels "
        f"and yield {midpoint_pos}/{optimistic_pos}/{weighted_pos} positives."
    )
    if not country_missingness_sensitivity.empty and "model_name" in country_missingness_sensitivity.columns:
        model_rows = country_missingness_sensitivity.loc[
            country_missingness_sensitivity["model_name"].astype(str).eq(str(model_name))
        ].copy()
        if not model_rows.empty:
            if "outcome_name" in model_rows.columns:
                model_rows = model_rows.loc[
                    model_rows["outcome_name"].astype(str).isin(
                        ["label_observed", "label_midpoint", "label_optimistic", "label_weighted"]
                    )
                ].copy()
            if not model_rows.empty and {"roc_auc", "average_precision"}.issubset(model_rows.columns):
                roc_auc = pd.to_numeric(model_rows["roc_auc"], errors="coerce")
                average_precision = pd.to_numeric(model_rows["average_precision"], errors="coerce")
                if roc_auc.notna().any() and average_precision.notna().any():
                    text += (
                        f" Sensitivity across those label variants spans ROC AUC "
                        f"{float(roc_auc.min()):.3f} to {float(roc_auc.max()):.3f} and AP "
                        f"{float(average_precision.min()):.3f} to {float(average_precision.max()):.3f}."
                    )
    return f"{text}."


def _candidate_stability_summary_text(
    frame: pd.DataFrame,
    *,
    file_name: str,
    frequency_column: str,
    language: str = "en",
) -> str:
    if frame.empty or frequency_column not in frame.columns:
        return ""
    working = frame.loc[frame.get("backbone_id", pd.Series(dtype=str)).astype(str).ne("")].copy()
    if working.empty:
        return ""
    sort_columns = [frequency_column]
    if frequency_column != "bootstrap_top_10_frequency" and "bootstrap_top_10_frequency" in working.columns:
        sort_columns.append("bootstrap_top_10_frequency")
    if frequency_column != "variant_top_10_frequency" and "variant_top_10_frequency" in working.columns:
        sort_columns.append("variant_top_10_frequency")
    working = working.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    row = working.iloc[0]
    backbone_id = str(row.get("backbone_id", "") or "").strip()
    top_k = pd.to_numeric(pd.Series([row.get("top_k", np.nan)]), errors="coerce").iloc[0]
    frequency = pd.to_numeric(pd.Series([row.get(frequency_column, np.nan)]), errors="coerce").iloc[0]
    if language == "tr":
        intro = f"`{file_name}` aday sıralama kararlılığını raporlar"
        if backbone_id and pd.notna(frequency):
            if pd.notna(top_k):
                return (
                    f"{intro}; en kararlı örnek `{backbone_id}` için ilk `{int(top_k)}` içinde kalma sıklığı "
                    f"`{float(frequency):.2f}`."
                )
            return f"{intro}; en kararlı örnek `{backbone_id}` için sıklık `{float(frequency):.2f}`."
        return f"{intro}."
    intro = f"`{file_name}` records candidate rank stability"
    if frequency_column == "bootstrap_top_k_frequency":
        intro += " across bootstrap resamples"
    else:
        intro += " across model variants"
    if backbone_id and pd.notna(frequency):
        if pd.notna(top_k):
            return (
                f"{intro}; the strongest stable backbone `{backbone_id}` remains in the top-`{int(top_k)}` "
                f"set at frequency `{float(frequency):.2f}`."
            )
        return f"{intro}; the strongest stable backbone `{backbone_id}` has frequency `{float(frequency):.2f}`."
    return f"{intro}."


def build_headline_validation_summary_markdown(
    summary_table: pd.DataFrame,
    *,
    primary_model_name: str,
    governance_model_name: str,
    rolling_temporal: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
) -> str:
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = (
        variant_consistency if variant_consistency is not None else pd.DataFrame()
    )
    lines = [
        "# Headline Validation Summary",
        "",
        "This is the canonical one-page validation surface for jury review.",
        "",
        f"- Discovery primary: `{primary_model_name}`",
        f"- {_governance_watch_label()}: `{governance_model_name}`",
        "- Baseline comparator: `baseline_both`",
        "- Permutation entries below include the selection-adjusted official-model permutation audit; the fixed-score label-permutation audit is retained only as an exploratory appendix diagnostic.",
        "- The explicit leakage canary is exported separately in `future_sentinel_audit.tsv`.",
        "- The frozen acceptance audit is exported separately in `frozen_scientific_acceptance_audit.tsv`.",
        "- The nonlinear deconfounding audit is exported separately in `nonlinear_deconfounding_audit.tsv`.",
        "- Alternative endpoint audits are exported separately in `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv`.",
        "- The prospective freeze audits are exported separately in `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv`.",
        "- The graph, counterfactual, geographic-jump, and AMR-uncertainty diagnostics are exported separately in `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv`.",
        "- Frozen scientific acceptance combines matched-knownness, source holdout, spatial holdout, calibration, selection-adjusted null, and leakage review.",
        "",
        "| Surface | Model | ROC AUC | ROC AUC 95% CI | AP | AP 95% CI | Brier | Brier Skill | ECE | Max CE | Frozen Acceptance | Frozen Acceptance Reason | Selection-adjusted p | Fixed-score p | Delta vs baseline | Spatial holdout AUC | n | Positives |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_table.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("summary_label", "")),
                    str(row.get("model_name", "")),
                    f"{float(row.get('roc_auc', np.nan)):.3f}"
                    if pd.notna(row.get("roc_auc"))
                    else "NA",
                    str(row.get("roc_auc_ci", "NA")),
                    f"{float(row.get('average_precision', np.nan)):.3f}"
                    if pd.notna(row.get("average_precision"))
                    else "NA",
                    str(row.get("average_precision_ci", "NA")),
                    f"{float(row.get('brier_score', np.nan)):.3f}"
                    if pd.notna(row.get("brier_score"))
                    else "NA",
                    f"{float(row.get('brier_skill_score', np.nan)):.3f}"
                    if pd.notna(row.get("brier_skill_score"))
                    else "NA",
                    f"{float(row.get('ece', np.nan)):.3f}" if pd.notna(row.get("ece")) else "NA",
                    f"{float(row.get('max_calibration_error', np.nan)):.3f}"
                    if pd.notna(row.get("max_calibration_error"))
                    else "NA",
                    str(row.get("scientific_acceptance_status", "not_scored")),
                    str(row.get("scientific_acceptance_failed_criteria", "not_scored")),
                    _format_pvalue(row.get("selection_adjusted_empirical_p_roc_auc")),
                    _format_pvalue(row.get("permutation_p_roc_auc")),
                    (
                        f"{float(row.get('delta_vs_baseline_roc_auc', np.nan)):.3f} "
                        f"({row.get('delta_vs_baseline_ci', 'NA')})"
                    )
                    if pd.notna(row.get("delta_vs_baseline_roc_auc"))
                    else "NA",
                    f"{float(row.get('spatial_holdout_roc_auc', np.nan)):.3f}"
                    if pd.notna(row.get("spatial_holdout_roc_auc"))
                    else "NA",
                    str(int(row.get("n_backbones", 0))),
                    str(int(row.get("n_positive", 0))),
                ]
            )
            + " |"
        )
    rolling_summary = _rolling_temporal_summary(
        rolling_temporal if rolling_temporal is not None else pd.DataFrame()
    )
    if rolling_summary:
        lines.extend(
            [
                "",
                "## Rolling-Origin Validation",
                "",
                (
                    f"Nested rolling-origin validation spans outer split years {rolling_summary['split_year_min']}"
                    f" to {rolling_summary['split_year_max']} across horizons {rolling_summary['horizon_values']}"
                    f" years and assignment modes {rolling_summary['assignment_modes']}."
                ),
                (
                    f"Across {rolling_summary['n_rows']} successful outer-split rows, ROC AUC mean "
                    f"{rolling_summary['roc_auc_mean']:.3f} (range {rolling_summary['roc_auc_min']:.3f}"
                    f" to {rolling_summary['roc_auc_max']:.3f}) and AP mean {rolling_summary['average_precision_mean']:.3f}"
                    f" (range {rolling_summary['average_precision_min']:.3f} to {rolling_summary['average_precision_max']:.3f})."
                ),
                f"Mean Brier score across the successful outer splits is {rolling_summary['brier_score_mean']:.3f}.",
            ]
        )
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    if blocked_holdout_text:
        lines.extend(["", "## Blocked Holdout Audit", "", f"- {blocked_holdout_text}"])
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    if country_missingness_text:
        lines.extend(["", "## Country Missingness", "", f"- {country_missingness_text}"])
    if rank_stability_text or variant_consistency_text:
        lines.extend(["", "## Ranking Stability", ""])
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    return "\n".join(lines) + "\n"


def build_executive_summary_markdown(
    *,
    primary_model_name: str,
    governance_model_name: str,
    baseline_model_name: str,
    model_metrics: pd.DataFrame,
    confirmatory_cohort_summary: pd.DataFrame,
    false_negative_audit: pd.DataFrame,
    candidate_case_studies: pd.DataFrame,
    blocked_holdout_summary: pd.DataFrame | None = None,
    rolling_temporal: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
) -> str:
    metrics = model_metrics.set_index("model_name", drop=False)
    primary = metrics.loc[str(primary_model_name)]
    governance = (
        metrics.loc[str(governance_model_name)]
        if str(governance_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    baseline = (
        metrics.loc[str(baseline_model_name)]
        if str(baseline_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = (
        variant_consistency if variant_consistency is not None else pd.DataFrame()
    )
    false_negative_count, top_drivers = _summarize_false_negative_audit(false_negative_audit)
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    ).rstrip(".")
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    rolling_summary = _rolling_temporal_summary(
        rolling_temporal if rolling_temporal is not None else pd.DataFrame()
    )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    lines = [
        "# Executive Summary",
        "",
        "Plasmid Priority is a retrospective surveillance ranking framework for plasmid backbone classes. It does not claim causal spread prediction; it asks whether pre-2016 genomic signals are associated with post-2015 international visibility increase.",
        "",
        f"The Seer (headline model): `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"The Guard (governance watch-only): `{_pretty_report_model_label(governance_model_name)}` | ROC AUC `{float(governance.get('roc_auc', np.nan)):.3f}` | AP `{float(governance.get('average_precision', np.nan)):.3f}`.",
        f"The Baseline: `{_pretty_report_model_label(baseline_model_name)}` | ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}` | AP `{float(baseline.get('average_precision', np.nan)):.3f}`.",
        "",
        "## Method Overview",
        "",
        "```text",
        "Raw Data (PLSDB + RefSeq + Pathogen Detection)",
        "                |",
        "                v",
        "      Harmonization and Deduplication",
        "                |",
        "                v",
        "      Backbone Assignment (MOB-suite style)",
        "                |",
        "                v",
        "       Temporal Split: <=2015 | >2015",
        "                |",
        "                v",
        "        T / H / A Feature Extraction",
        "                |",
        "                v",
        "   L2-Regularized Logistic Regression (OOF)",
        "          /                           \\",
        "         v                             v",
        " Discovery Track          Governance Watch-only Track",
        "                \\               /",
        "                 v             v",
        "            Candidate Portfolio + Risk Tiers",
        "```",
        "",
        "## Validation Posture",
        "",
        "- No external validation claim is made.",
        "- Validation is framed as temporal holdout, source holdout, knownness-matched auditing, and an internal high-integrity subset audit.",
        f"- False-negative audit: `{false_negative_count}` later positives remain outside the practical shortlist; dominant miss drivers are `{top_drivers}`.",
    ]
    if rolling_summary:
        lines.append(
            f"- Rolling-origin validation: outer split years {rolling_summary['split_year_min']} to {rolling_summary['split_year_max']} across horizons {rolling_summary['horizon_values']} with assignment modes {rolling_summary['assignment_modes']}; ROC AUC mean {rolling_summary['roc_auc_mean']:.3f} (range {rolling_summary['roc_auc_min']:.3f} to {rolling_summary['roc_auc_max']:.3f})."
        )
    if not confirmatory_primary.empty:
        lines.append(
            f"- Internal high-integrity subset audit: `{int(confirmatory_primary['n_backbones'])}` backbones | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`."
        )
    if country_missingness_text:
        lines.extend(["", "## Country Missingness", "", f"- {country_missingness_text}"])
    if rank_stability_text or variant_consistency_text:
        lines.extend(["", "## Ranking Stability", ""])
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    lines.extend(
        [
            "",
            "## Release Surface",
            "",
            f"- `{int(len(candidate_case_studies))}` case studies are exported in `candidate_case_studies.tsv`.",
            "- Jury-facing narrative lives in `jury_brief.md` and `ozet_tr.md`.",
            "- `frozen_scientific_acceptance_audit.tsv` records the headline acceptance gate across matched-knownness, source holdout, spatial holdout, calibration, and leakage review.",
            f"- Blocked holdout audit is exported in `blocked_holdout_summary.tsv`{f': {blocked_holdout_text}' if blocked_holdout_text else ''}.",
            "- `nonlinear_deconfounding_audit.tsv` records the nonlinear deconfounding check used to keep knownness residualization transparent.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv` record the alternative-endpoint stress tests for ordinal, exposure-adjusted, and macro-region jump outcomes.",
            "- `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv` record the quasi-prospective freeze surface used to check whether the shortlist survives a forward-looking holdout.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv` record the leakage canary, graph audit, counterfactual shortlist comparison, geographic-jump diagnostic, and AMR-uncertainty summary.",
            "- Candidate rank stability is exported in `candidate_rank_stability.tsv` and model-variant consistency is exported in `candidate_variant_consistency.tsv`.",
            "- `calibration_threshold_summary.png` is exported as a compact calibration/threshold diagnostic when threshold-sensitivity data are available.",
            "- Figures in `reports/core_figures/` are presentation-ready.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_pitch_notes_markdown(
    *,
    primary_model_name: str,
    governance_model_name: str,
    model_metrics: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame,
    coefficient_stability_cv: pd.DataFrame,
    confirmatory_cohort_summary: pd.DataFrame,
    false_negative_audit: pd.DataFrame,
) -> str:
    metrics = model_metrics.set_index("model_name", drop=False)
    primary = (
        metrics.loc[str(primary_model_name)]
        if str(primary_model_name) in metrics.index
        else pd.Series(dtype=object)
    )
    baseline = (
        metrics.loc["baseline_both"]
        if "baseline_both" in metrics.index
        else pd.Series(dtype=object)
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, top_drivers = _summarize_false_negative_audit(false_negative_audit)
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    stable_features = "NA"
    if not coefficient_stability_cv.empty and "feature_name" in coefficient_stability_cv.columns:
        working = coefficient_stability_cv.copy()
        if "sign_stable" in working.columns:
            working = working.loc[working["sign_stable"].fillna(False).astype(bool)]
        if not working.empty:
            sort_column = "cv_of_coef" if "cv_of_coef" in working.columns else None
            if sort_column:
                working = working.sort_values(sort_column, ascending=True)
            stable_features = ",".join(working["feature_name"].astype(str).head(3).tolist())
    lines = [
        "# Pitch Notes",
        "",
        "## Olası Jüri Soruları ve Yanıtları",
        "",
        '**S: "Model sadece zaten iyi bilinen büyük backbone\'ları mı buluyor?"**',
        (
            f"C: `baseline_both` ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}` üretirken "
            f"`{primary_model_name}` ROC AUC `{float(primary.get('roc_auc', np.nan)):.3f}` üretiyor. "
            f"Delta `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`. "
            f"Eşleştirilmiş knownness/source strata audit'inde de ana model "
            f"`{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}` vs baseline `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}`."
        )
        if not matched_primary.empty and not matched_baseline.empty
        else (
            f"C: `baseline_both` ROC AUC `{float(baseline.get('roc_auc', np.nan)):.3f}`, "
            f"`{primary_model_name}` ROC AUC `{float(primary.get('roc_auc', np.nan)):.3f}`. "
            f"Delta `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`; bu, biyolojik sinyalin yalnızca popülerlik sayacı olmadığını gösterir."
        ),
        "",
        '**S: "Tüm modeller strict testi kaybediyorsa metodoloji geçerli mi?"**',
        "C: Evet. Strict matched-knownness/source-holdout testi en zor alt kohortu izole eder. Burada başarısız olmak metodolojinin çöktüğünü değil, mevcut veri yoğunluğunun bu alt dilimde sınırlı olduğunu gösterir. Bu kısıt raporda proaktif olarak açıkça belirtilir.",
        "",
        '**S: "29 özellik, 989 örnek; overfit değil mi?"**',
        f"C: Ana model L2 düzenlemeli lojistik regresyondur ve OOF tahminlerle değerlendirilir. Katsayı kararlılığı için 5-fold CV özeti ayrı verilir; en kararlı örnek sinyaller: `{stable_features}`.",
        "",
        '**S: "Bu gerçek bir tahmin sistemi mi, yoksa retrospektif analiz mi?"**',
        "C: Bu çalışma kasıtlı olarak retrospektiftir. Soru şudur: eğitim dönemindeki genomik sinyaller, sonraki dönemdeki coğrafi görünürlük artışıyla ilişkili mi? Prospektif klinik erken uyarı iddiası yapılmaz.",
        "",
        '**S: "Governance modeli neden discovery modelinden ayrı?"**',
        f"C: Discovery modeli `{_pretty_report_model_label(primary_model_name)}` ayırma gücünü optimize eder. Governance watch-only modeli `{_pretty_report_model_label(governance_model_name)}` ise kalibrasyon, belirsizlik ve abstention davranışını öne çıkarır; en yüksek AUC'u kovalamaz.",
    ]
    if not confirmatory_primary.empty:
        lines.extend(
            [
                "",
                "## Ek Not",
                "",
                f"- Confirmatory cohort sonucu: `{int(confirmatory_primary['n_backbones'])}` backbone üzerinde ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}`.",
                f"- False-negative audit: `{false_negative_count}` missed positive; baskın nedenler `{top_drivers}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_turkish_summary_markdown(
    *,
    primary_model_name: str,
    conservative_model_name: str,
    model_metrics: pd.DataFrame,
    candidate_briefs: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    decision_yield: pd.DataFrame,
    model_selection_scorecard: pd.DataFrame | None = None,
    model_selection_summary: pd.DataFrame,
    knownness_summary: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame | None = None,
    source_balance_resampling: pd.DataFrame,
    novelty_specialist_metrics: pd.DataFrame,
    adaptive_gated_metrics: pd.DataFrame,
    gate_consistency_audit: pd.DataFrame | None = None,
    secondary_outcome_performance: pd.DataFrame | None = None,
    weighted_country_outcome: pd.DataFrame | None = None,
    count_outcome_audit: pd.DataFrame | None = None,
    metadata_quality_summary: pd.DataFrame | None = None,
    operational_risk_watchlist: pd.DataFrame | None = None,
    confirmatory_cohort_summary: pd.DataFrame | None = None,
    false_negative_audit: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
    outcome_threshold: int,
) -> str:
    knownness_matched_validation = (
        knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    )
    weighted_country_outcome = (
        weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    )
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    operational_risk_watchlist = (
        operational_risk_watchlist if operational_risk_watchlist is not None else pd.DataFrame()
    )
    confirmatory_cohort_summary = (
        confirmatory_cohort_summary if confirmatory_cohort_summary is not None else pd.DataFrame()
    )
    false_negative_audit = (
        false_negative_audit if false_negative_audit is not None else pd.DataFrame()
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = (
        variant_consistency if variant_consistency is not None else pd.DataFrame()
    )
    primary = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(primary_model_name)
    ].iloc[0]
    conservative = model_metrics.loc[
        model_metrics["model_name"].astype(str) == str(conservative_model_name)
    ].iloc[0]
    baseline = model_metrics.loc[model_metrics["model_name"].astype(str) == "baseline_both"].iloc[0]
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )
    governance_model_name = str(selection_row.get("governance_primary_model", "") or "").strip()
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, false_negative_drivers = _summarize_false_negative_audit(
        false_negative_audit
    )
    blocked_holdout_text = _blocked_holdout_summary_text_tr(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    weighted_row = (
        weighted_country_outcome.loc[
            weighted_country_outcome.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in weighted_country_outcome.columns
        else pd.DataFrame()
    )
    count_row = (
        count_outcome_audit.loc[
            count_outcome_audit.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in count_outcome_audit.columns
        else pd.DataFrame()
    )
    selected_briefs = candidate_briefs.head(5) if not candidate_briefs.empty else pd.DataFrame()
    lines = [
        "# Proje Özeti",
        "",
        "## Temel Fikir",
        "",
        "Plasmid Priority, plasmid omurga sınıflarını eğitim dönemindeki genomik sinyallerle puanlayan ve bu puanların daha sonraki coğrafi görünürlük artışıyla ilişkili olup olmadığını test eden retrospektif bir analiz hattıdır.",
        "",
        f"Ana model `{primary_model_name}` için ROC AUC `{float(primary['roc_auc']):.3f}`, AP `{float(primary['average_precision']):.3f}` ve Brier Skill Score `{float(primary.get('brier_skill_score', np.nan)):.3f}` olarak raporlanır. Mevcut permütasyon denetimi sabit-skor label-permutation audit'i olduğu için model-seçim-düzeltilmiş anlamlılık iddiası olarak değil, keşifsel sinyal kontrolü olarak okunmalıdır. Sayım temelli karşılaştırma modeli `{float(baseline['roc_auc']):.3f}` ROC AUC üretir; ana modelin bu taban modele karşı kazancı `{_primary_baseline_delta_text(model_metrics, primary_model_name)}` düzeyindedir.",
        "",
        "## Model Seçimi",
        "",
        "Model seçimi tek bir metriğe göre yapılmamıştır. Birlikte okunan ölçütler şunlardır:",
        "",
        "1. Genel ROC AUC ve AP.",
        "2. Düşük bilinirlik yarısındaki performans.",
        "3. Eşleştirilmiş bilinirlik/kaynak katmanlarındaki performans.",
        "4. Kaynak dışlama denetimi ve diğer sağlamlık analizleri.",
        "5. Pratik kısa liste verimi.",
        "",
        f"Bu nedenle discovery hattında `{primary_model_name}` korunur; governance watch-only hattında ise `{governance_model_name or 'ayrı governance watch-only modeli'}` daha temkinli yorum katmanı olarak ele alınır.",
        "",
        "## Metodoloji",
        "",
        "- Ham veri PLSDB, RefSeq ve Pathogen Detection metadata kaynaklarından gelir.",
        "- Kayıtlar harmonize edilir, yinelenenler ayıklanır ve omurga sınıfı ataması yapılır.",
        "- Zaman ayrımı `<=2015` eğitim ve `>2015` sonuç penceresi olacak şekilde kuruludur.",
        "- T, H ve A eksenleri yalnızca eğitim döneminden hesaplanır.",
        f"- Yayılım etiketi, test döneminde en az `{int(outcome_threshold)}` yeni ülke görülmesiyle tanımlanır.",
        "- Değerlendirme OOF lojistik regresyon tahminleri üzerinden yapılır.",
        "",
        "## Türkiye Bağlamı",
        "",
        "WHO'nun 2025 GLASS özeti, antibiyotik direnci yükünün özellikle Güneydoğu Asya ve Doğu Akdeniz bölgelerinde yüksek olduğunu vurgular. Türkiye, AMR sürveyansı açısından bu bölgesel baskının doğrudan önemli olduğu bir ülkedir.",
        "",
        "ECDC ve WHO Europe çerçevelerinde karbapenem dirençli *Klebsiella pneumoniae* ile GSBL/ESBL üreten *Escherichia coli*, Enterobacterales kaynaklı hastane yükünün temel başlıkları arasında yer almaktadır. Bu nedenle bu çalışmanın ürettiği omurga sınıfı önceliklendirmesi, Türkiye'de genomik AMR sürveyansına doğrudan uyarlanabilecek bir kanıt-konsept çerçevesi sunar.",
        "",
        "Bu proje klinik karar desteği vermez; ancak Türkiye'de ulusal veya kurumsal genomik sürveyans akışlarında hangi omurga sınıflarının önce incelenmesi gerektiğini sistematikleştirebilir.",
        "",
        "## Ana Bulgular",
        "",
        f"- Ana model: ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"- Koruyucu model: ROC AUC `{float(conservative['roc_auc']):.3f}` | AP `{float(conservative['average_precision']):.3f}`.",
        f"- Baseline model: ROC AUC `{float(baseline['roc_auc']):.3f}` | AP `{float(baseline['average_precision']):.3f}`.",
        f"- Yanlış negatif incelemesi: kısa liste dışında kalan `{false_negative_count}` pozitif vardır; baskın nedenler `{false_negative_drivers}`.",
    ]
    if not confirmatory_primary.empty:
        lines.append(
            f"- İç yüksek-bütünlük alt-küme denetimi: `{int(confirmatory_primary['n_backbones'])}` omurga sınıfı | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`."
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Eşleştirilmiş bilinirlik/kaynak katmanları denetimi: ana model `{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}`, taban model `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}`."
        )
    if not weighted_row.empty:
        lines.append(
            f"- Ağırlıklı yeni ülke yükü ile ilişki: Spearman ρ `{float(weighted_row.iloc[0]['spearman_corr']):.3f}`."
        )
    if not count_row.empty:
        lines.append(
            f"- Ham yeni ülke sayısı ile ilişki: Spearman ρ `{float(count_row.iloc[0]['spearman_corr']):.3f}` {_format_interval(count_row.iloc[0].get('spearman_ci_lower'), count_row.iloc[0].get('spearman_ci_upper'))}."
        )
    spatial_auc = pd.to_numeric(
        pd.Series([primary.get("spatial_holdout_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.notna(spatial_auc):
        lines.append(f"- Mekânsal holdout denetimi: ağırlıklı ROC AUC `{float(spatial_auc):.3f}`.")
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="tr",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="tr",
    )
    if rank_stability_text or variant_consistency_text:
        lines.extend(["", "## Sıralama Kararlılığı", ""])
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    lines.extend(
        [
            "",
            "## Sınırlılıklar",
            "",
            "- Bu çalışma retrospektiftir; prospektif erken uyarı sistemi iddiası taşımaz.",
            "- Sonuç değişkeni doğrudan biyolojik yayılım değil, sonraki dönem ülke görünürlüğü artışıdır.",
            "- En sıkı bilinirlik eşleştirmeli kaynak dışlama testi en zor alt kohorttur; bu katmanda tüm modeller temkinli yorumlanmalıdır.",
            "- Fırsat yanlılığı tamamen giderilemez: daha erken görülen omurgaların daha uzun takip penceresi vardır.",
            "- `risk_uncertainty` bir güven aralığı değildir; risk-bileşeni uyumsuzluğu, karar sınırına yakınlık ve düşük bilinirlik cezasından türetilen operasyonel bir belirsizlik skorudur.",
            "",
            "## Örnek Adaylar",
            "",
        ]
    )
    for row in selected_briefs.itertuples(index=False):
        summary = getattr(row, "candidate_summary_tr", "")
        if summary:
            lines.append(f"- {summary}")
    lines.extend(
        [
            "",
            "## Sürüm Yüzeyi",
            "",
            "- `frozen_scientific_acceptance_audit.tsv`: doğrulayıcı kabul katmanını; eşleştirilmiş bilinirlik, kaynak dışlama, mekânsal holdout, kalibrasyon ve leakage incelemesini raporlar.",
            f"- `blocked_holdout_summary.tsv`: {blocked_holdout_text or 'iç kaynak/bölge stres testi olarak raporlanır.'}",
            "- `nonlinear_deconfounding_audit.tsv`: knownness residualization için kullanılan doğrusal olmayan karıştırma denetimini raporlar.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv` ve `macro_region_jump_outcome.tsv`: sırasal, maruziyet-düzeltilmiş ve makro-bölge sıçrama sonuçları için alternatif sonuç stres testlerini raporlar.",
            "- `prospective_candidate_freeze.tsv` ve `annual_candidate_freeze_summary.tsv`: kısa listenin ileriye dönük bir holdout üzerinde ayakta kalıp kalmadığını kontrol eden quasi-prospective freeze yüzeyini raporlar.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv` ve `amr_uncertainty_summary.tsv`: leakage canary, graph denetimi, counterfactual shortlist comparison, geographic-jump tanısı ve AMR belirsizlik özetini raporlar.",
            "- `candidate_rank_stability.tsv` ve `candidate_variant_consistency.tsv`: aday sıralama kararlılığını ve model-varyant tutarlılığını raporlar.",
            "- `calibration_threshold_summary.png`: kalibrasyon ve eşik duyarlılığı için kompakt tanı grafiğidir.",
            "- `jury_brief.md` ve `ozet_tr.md`: jüriye dönük anlatının dağıtım yüzeyleri.",
            "- `country_missingness_bounds.tsv` ve `country_missingness_sensitivity.tsv`: ülke eksikliği varsayımlarına göre etiket ve performans duyarlılığını raporlar.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_jury_brief_markdown(
    *,
    primary_model_name: str,
    conservative_model_name: str,
    model_metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    dropout_table: pd.DataFrame,
    scored: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    decision_yield: pd.DataFrame,
    model_selection_scorecard: pd.DataFrame | None = None,
    model_selection_summary: pd.DataFrame,
    knownness_summary: pd.DataFrame,
    knownness_matched_validation: pd.DataFrame | None = None,
    source_balance_resampling: pd.DataFrame,
    novelty_specialist_metrics: pd.DataFrame,
    adaptive_gated_metrics: pd.DataFrame,
    gate_consistency_audit: pd.DataFrame | None = None,
    secondary_outcome_performance: pd.DataFrame | None = None,
    weighted_country_outcome: pd.DataFrame | None = None,
    count_outcome_audit: pd.DataFrame | None = None,
    metadata_quality_summary: pd.DataFrame | None = None,
    operational_risk_watchlist: pd.DataFrame | None = None,
    confirmatory_cohort_summary: pd.DataFrame | None = None,
    false_negative_audit: pd.DataFrame | None = None,
    blocked_holdout_summary: pd.DataFrame | None = None,
    country_missingness_bounds: pd.DataFrame | None = None,
    country_missingness_sensitivity: pd.DataFrame | None = None,
    rank_stability: pd.DataFrame | None = None,
    variant_consistency: pd.DataFrame | None = None,
    outcome_threshold: int,
) -> str:
    knownness_matched_validation = (
        knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    )
    weighted_country_outcome = (
        weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    )
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    operational_risk_watchlist = (
        operational_risk_watchlist if operational_risk_watchlist is not None else pd.DataFrame()
    )
    confirmatory_cohort_summary = (
        confirmatory_cohort_summary if confirmatory_cohort_summary is not None else pd.DataFrame()
    )
    false_negative_audit = (
        false_negative_audit if false_negative_audit is not None else pd.DataFrame()
    )
    blocked_holdout_summary = (
        blocked_holdout_summary if blocked_holdout_summary is not None else pd.DataFrame()
    )
    country_missingness_bounds = (
        country_missingness_bounds if country_missingness_bounds is not None else pd.DataFrame()
    )
    country_missingness_sensitivity = (
        country_missingness_sensitivity
        if country_missingness_sensitivity is not None
        else pd.DataFrame()
    )
    rank_stability = rank_stability if rank_stability is not None else pd.DataFrame()
    variant_consistency = (
        variant_consistency if variant_consistency is not None else pd.DataFrame()
    )
    primary = model_metrics.loc[model_metrics["model_name"].astype(str) == primary_model_name].iloc[0]
    conservative = model_metrics.loc[
        model_metrics["model_name"].astype(str) == conservative_model_name
    ].iloc[0]
    baseline = model_metrics.loc[model_metrics["model_name"].astype(str) == "baseline_both"].iloc[0]
    source = model_metrics.loc[model_metrics["model_name"].astype(str) == "source_only"].head(1)
    source = source.iloc[0] if not source.empty else pd.Series(dtype=object)
    selection_row = (
        model_selection_summary.iloc[0]
        if not model_selection_summary.empty
        else pd.Series(dtype=object)
    )
    governance_model_name = str(selection_row.get("governance_primary_model", "") or "").strip()
    governance_scorecard_row = (
        model_selection_scorecard.loc[
            model_selection_scorecard.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(governance_model_name)
        ].head(1)
        if governance_model_name
        and model_selection_scorecard is not None
        and not model_selection_scorecard.empty
        else pd.DataFrame()
    )
    primary_scorecard_row = (
        model_selection_scorecard.loc[
            model_selection_scorecard.get("model_name", pd.Series(dtype=str))
            .astype(str)
            .eq(primary_model_name)
        ].head(1)
        if model_selection_scorecard is not None and not model_selection_scorecard.empty
        else pd.DataFrame()
    )
    confirmatory_primary = _select_confirmatory_row(
        confirmatory_cohort_summary,
        cohort_name="confirmatory_internal",
        model_name=primary_model_name,
    )
    false_negative_count, false_negative_drivers = _summarize_false_negative_audit(
        false_negative_audit
    )
    matched_primary = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(primary_model_name))
    ].head(1)
    matched_baseline = knownness_matched_validation.loc[
        knownness_matched_validation.get("matched_stratum", pd.Series(dtype=str))
        .astype(str)
        .eq("__weighted_overall__")
        & knownness_matched_validation.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq("baseline_both")
    ].head(1)
    weighted_row = (
        weighted_country_outcome.loc[
            weighted_country_outcome.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in weighted_country_outcome.columns
        else pd.DataFrame()
    )
    count_row = (
        count_outcome_audit.loc[
            count_outcome_audit.get("status", pd.Series(dtype=str)).astype(str) == "ok"
        ]
        .sort_values("spearman_corr", ascending=False)
        .head(1)
        if "spearman_corr" in count_outcome_audit.columns
        else pd.DataFrame()
    )
    operational_action_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("action")
            .sum()
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    operational_review_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("review")
            .sum()
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    operational_abstain_count = (
        int(
            operational_risk_watchlist.get("risk_decision_tier", pd.Series(dtype=str))
            .astype(str)
            .eq("abstain")
            .sum()
        )
        if not operational_risk_watchlist.empty
        else 0
    )
    strongest_overall = model_metrics.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    primary_vs_strongest_overlap_25 = selection_row.get(
        "primary_vs_strongest_top_25_overlap_count", np.nan
    )
    primary_vs_strongest_overlap_50 = selection_row.get(
        "primary_vs_strongest_top_50_overlap_count", np.nan
    )
    selection_rationale = str(selection_row.get("selection_rationale", "")).strip()
    governance_rationale = str(selection_row.get("governance_selection_rationale", "")).strip()
    blocked_holdout_text = _blocked_holdout_summary_text(
        blocked_holdout_summary,
        model_name=primary_model_name,
    )
    country_missingness_text = _country_missingness_summary_text(
        country_missingness_bounds,
        country_missingness_sensitivity,
        model_name=primary_model_name,
    )
    rank_stability_text = _candidate_stability_summary_text(
        rank_stability,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_k_frequency",
        language="en",
    )
    variant_consistency_text = _candidate_stability_summary_text(
        variant_consistency,
        file_name="candidate_variant_consistency.tsv",
        frequency_column="variant_top_k_frequency",
        language="en",
    )
    primary_top10 = _lookup_decision_yield(decision_yield, primary_model_name, 10)
    primary_top25 = _lookup_decision_yield(decision_yield, primary_model_name, 25)
    conservative_top10 = _lookup_decision_yield(decision_yield, conservative_model_name, 10)
    baseline_top10 = _lookup_decision_yield(decision_yield, "baseline_both", 10)
    strongest_top10 = _lookup_decision_yield(decision_yield, str(strongest_overall["model_name"]), 10)
    lines = [
        "# Jury Brief",
        "",
        "## Core Claim",
        "",
        "This framework retrospectively prioritizes plasmid backbone surveillance units using pre-2016 genomic and ecological features, then tests whether those same backbone classes later show multi-country visibility increase.",
        "",
        "## Formal Hypotheses",
        "",
        "- **H0 (null)**: A <=2015 T/H/A-derived priority signal has no discriminative association with post-2015 multi-country visibility expansion.",
        "- **H1 (alternative)**: The same priority signal is positively associated with post-2015 multi-country visibility expansion.",
        "- **Significance criterion**: empirical permutation p-value below the predeclared threshold for the current headline model.",
        "",
        "## Current Benchmark",
        "",
        f"- Discovery benchmark: `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{float(primary['roc_auc']):.3f}` | AP `{float(primary['average_precision']):.3f}`.",
        f"- Counts-only baseline: `baseline_both` | ROC AUC `{float(baseline['roc_auc']):.3f}` | AP `{float(baseline['average_precision']):.3f}`.",
        f"- Conservative benchmark: `{_pretty_report_model_label(conservative_model_name)}` | ROC AUC `{float(conservative['roc_auc']):.3f}` | AP `{float(conservative['average_precision']):.3f}`.",
        f"- Source-only control: `source_only` | ROC AUC `{float(source.get('roc_auc', np.nan)):.3f}`.",
        f"- Strongest audited metric model: `{_pretty_report_model_label(str(strongest_overall['model_name']))}` | ROC AUC `{float(strongest_overall['roc_auc']):.3f}` | AP `{float(strongest_overall['average_precision']):.3f}`.",
        (
            f"- Governance watch-only: `{_pretty_report_model_label(governance_model_name)}` | ROC AUC `{float(governance_scorecard_row.iloc[0].get('roc_auc', np.nan)):.3f}` | AP `{float(governance_scorecard_row.iloc[0].get('average_precision', np.nan)):.3f}` | strict `{'pass' if bool(governance_scorecard_row.iloc[0].get('strict_knownness_acceptance_flag', False)) else 'fail'}`."
            if not governance_scorecard_row.empty
            else "- Governance watch-only: no distinct governance candidate could be resolved from the current scorecard."
        ),
        f"- Selection-adjusted official-model permutation audit for the headline ROC AUC: `p {_format_pvalue(primary.get('selection_adjusted_empirical_p_roc_auc'))}`; the older fixed-score label-permutation entry is retained only as an exploratory appendix diagnostic.",
        f"- Delta vs counts-only baseline: `{_primary_baseline_delta_text(model_metrics, primary_model_name)}`.",
        "",
        "## Primary Model Selection Rationale",
        "",
        "Model selection was not driven by a single metric. We jointly considered ROC AUC, average precision, lower-knownness behavior, matched-knownness/source performance, source holdout robustness, and practical shortlist yield.",
        "",
        "Operationally, the headline model is preferred because it keeps the strongest balance between discrimination and shortlist usefulness.",
        "Governance track logic is kept separate from discovery-track optimization even when the shortlisted candidates partially overlap.",
        "",
        "## Strict Test Interpretation",
        "",
        "The strict matched-knownness/source-holdout test isolates the hardest low-knownness and low-support slice of the dataset. No current model fully passes this acceptance rule.",
        "",
        "This should be interpreted as a data-limited regime, not as evidence that the entire methodology collapses.",
        "",
        "## Decision Readout",
        "",
        f"- Outcome definition: later visibility in at least `{int(outcome_threshold)}` new countries.",
        f"- Operational watchlist mix: `{operational_action_count}` action + `{operational_review_count}` review + `{operational_abstain_count}` abstain rows.",
        f"- False-negative audit: `{false_negative_count}` later positives remain outside the practical shortlist; dominant miss drivers are `{false_negative_drivers}`.",
        "- `operational_risk_watchlist.tsv` is the calibrated deployment-facing table for the current shortlist.",
        f"- Current operational watchlist mix: `{operational_action_count}` action + `{operational_review_count}` review + `{operational_abstain_count}` abstain rows.",
        "- This remains a shortlist-prioritization benchmark rather than an exhaustive detector for every later positive backbone.",
    ]
    if not confirmatory_primary.empty:
        lines.append(
            f"- Internal high-integrity subset audit: `{int(confirmatory_primary['n_backbones'])}` higher-integrity backbones | ROC AUC `{float(confirmatory_primary['roc_auc']):.3f}` | AP `{float(confirmatory_primary['average_precision']):.3f}`."
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Matched knownness/source strata: primary `{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}` vs baseline `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}` weighted ROC AUC."
        )
    if not count_row.empty:
        lines.append(
            f"- Raw later new-country count alignment: Spearman ρ `{float(count_row.iloc[0]['spearman_corr']):.3f}` {_format_interval(count_row.iloc[0].get('spearman_ci_lower'), count_row.iloc[0].get('spearman_ci_upper'))}."
        )
    if not weighted_row.empty:
        lines.append(
            f"- Weighted new-country burden alignment: Spearman ρ `{float(weighted_row.iloc[0]['spearman_corr']):.3f}`."
        )
    spatial_auc = pd.to_numeric(
        pd.Series([primary.get("spatial_holdout_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.notna(spatial_auc):
        spatial_regions = pd.to_numeric(
            pd.Series([primary.get("spatial_holdout_regions")]), errors="coerce"
        ).iloc[0]
        worst_region = str(primary.get("worst_spatial_holdout_region", "") or "").strip()
        worst_region_auc = pd.to_numeric(
            pd.Series([primary.get("worst_spatial_holdout_region_roc_auc")]), errors="coerce"
        ).iloc[0]
        region_text = (
            f" across `{int(spatial_regions)}` held-out dominant regions"
            if pd.notna(spatial_regions)
            else ""
        )
        if worst_region and pd.notna(worst_region_auc):
            region_text += f"; hardest region `{worst_region}` at ROC AUC `{float(worst_region_auc):.3f}`"
        lines.append(f"- Spatial holdout audit: weighted ROC AUC `{float(spatial_auc):.3f}`{region_text}.")
    if blocked_holdout_text:
        lines.extend(
            [
                "",
                "## Blocked Holdout Audit",
                "",
                f"- {blocked_holdout_text} This is an internal source/region stress test, not external validation.",
            ]
        )
    if country_missingness_text:
        lines.extend(["", "## Country Missingness", "", f"- {country_missingness_text}"])
    if rank_stability_text or variant_consistency_text:
        lines.extend(["", "## Ranking Stability", ""])
        if rank_stability_text:
            lines.append(f"- {rank_stability_text}")
        if variant_consistency_text:
            lines.append(f"- {variant_consistency_text}")
    if pd.notna(primary_vs_strongest_overlap_25) or pd.notna(primary_vs_strongest_overlap_50):
        overlap_bits: list[str] = []
        if pd.notna(primary_vs_strongest_overlap_25):
            overlap_bits.append(f"top-25 overlap: `{int(primary_vs_strongest_overlap_25)}/25`")
        if pd.notna(primary_vs_strongest_overlap_50):
            overlap_bits.append(f"top-50 overlap: `{int(primary_vs_strongest_overlap_50)}/50`")
        lines.append(
            f"- Discovery shortlist agreement with the strongest audited metric model: {'; '.join(overlap_bits)}."
        )
    lines.extend(
        [
            "- A knownness-gated audit model (`adaptive_natural_priority`) remains useful for lower-knownness stress testing but is not the headline benchmark.",
            "- Observed host-diversity terms should be interpreted cautiously because they partly behave like sampling saturation / knownness signals.",
            "- Supportive external layers are descriptive context only; AMRFinder is optional and not required for the headline benchmark.",
            "- Only three models are official in the jury-facing narrative: discovery, governance watch-only, and baseline.",
            "",
            "## Zero-Floor Component Behavior",
            "",
            "When a backbone lacks direct evidence for a component, the normalized contribution is explicitly allowed to stay at zero rather than being imputed upward by unrelated metadata support.",
            "",
            "## OLS Residual Approach",
            "",
            "Knownness-sensitive H-support terms are residualized against knownness proxies so that the retained signal is not a disguised count effect.",
            "",
            "## Interpretation Guardrails",
            "",
            "- No external validation claim is made.",
            "- T, H and A features are computed only from `resolved_year <= 2015` rows.",
            "- The outcome is later country visibility increase, not direct biological fitness or transmission proof.",
            "- Opportunity bias is a declared limitation: backbones seen earlier have longer time-at-risk.",
            "",
            "## Release Surface",
            "",
            "- `frozen_scientific_acceptance_audit.tsv` records the headline acceptance gate across matched-knownness, source holdout, spatial holdout, calibration, and leakage review.",
            "- `blocked_holdout_summary.tsv` records the blocked source/region stress test used for the internal audit layer.",
            "- `nonlinear_deconfounding_audit.tsv` records the nonlinear deconfounding check used to keep knownness residualization transparent.",
            "- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv` record the alternative-endpoint stress tests for ordinal, exposure-adjusted, and macro-region jump outcomes.",
            "- `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv` record the quasi-prospective freeze surface used to check whether the shortlist survives a forward-looking holdout.",
            "- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv` record the leakage canary, graph audit, counterfactual shortlist comparison, geographic-jump diagnostic, and AMR-uncertainty summary.",
            "- `candidate_rank_stability.tsv` and `candidate_variant_consistency.tsv` record backbone-level ranking stability across bootstrap and model-variant audits.",
            "- `calibration_threshold_summary.png` captures the compact calibration/threshold view used in slide decks.",
            "- `reports/core_figures/` contains the rest of the presentation-ready figure pack.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_narrative_family(
    *,
    headline_validation_summary: pd.DataFrame,
    primary_model_name: str,
    governance_model_name: str,
    rolling_temporal: pd.DataFrame | None,
    blocked_holdout_summary: pd.DataFrame | None,
    country_missingness_bounds: pd.DataFrame | None,
    country_missingness_sensitivity: pd.DataFrame | None,
    rank_stability: pd.DataFrame | None,
    variant_consistency: pd.DataFrame | None,
) -> dict[str, str]:
    return {
        "headline_validation_summary.md": build_headline_validation_summary_markdown(
            headline_validation_summary,
            primary_model_name=primary_model_name,
            governance_model_name=governance_model_name,
            rolling_temporal=rolling_temporal,
            blocked_holdout_summary=blocked_holdout_summary,
            country_missingness_bounds=country_missingness_bounds,
            country_missingness_sensitivity=country_missingness_sensitivity,
            rank_stability=rank_stability,
            variant_consistency=variant_consistency,
        )
    }


def write_markdown_artifacts(outputs: Mapping[Path, str]) -> None:
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
