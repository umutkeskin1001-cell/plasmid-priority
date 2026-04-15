"""Shared narrative helpers for report assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pretty_report_model_label(model_name: str) -> str:
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


def strict_acceptance_status(row: pd.Series) -> str:
    return (
        str(row.get("scientific_acceptance_status", "not_scored") or "not_scored").strip().lower()
    )


def benchmark_scope_note(strict_acceptance_status_value: str) -> str:
    if strict_acceptance_status_value == "pass":
        return (
            "Benchmark scope note: the headline benchmark clears the frozen scientific "
            "acceptance gate, so accepted language is allowed only within the versioned "
            "benchmark contract."
        )
    return (
        "Benchmark scope note: the headline benchmark does not clear the frozen scientific "
        "acceptance gate, so the narrative remains conditional and benchmark-limited."
    )


def format_interval(lower: object, upper: object, *, digits: int = 3) -> str:
    lower_value = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
    upper_value = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
    if pd.isna(lower_value) or pd.isna(upper_value):
        return "NA"
    return f"[{float(lower_value):.{digits}f}, {float(upper_value):.{digits}f}]"


def format_pvalue(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    if float(numeric) < 0.001:
        return "<0.001"
    return f"{float(numeric):.3f}"


def governance_watch_label() -> str:
    return "Governance watch-only"


def blocked_holdout_row_text(row: pd.Series, *, model_name: str, language: str = "en") -> str:
    weighted_auc = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_roc_auc")]), errors="coerce"
    ).iloc[0]
    if pd.isna(weighted_auc):
        return ""
    group_column = str(row.get("blocked_holdout_group_columns", "") or "").strip()
    if not group_column:
        group_column = "blocked source/region group"
    elif "," in group_column:
        group_column = " + ".join(part.strip() for part in group_column.split(",") if part.strip())
    group_count = pd.to_numeric(
        pd.Series([row.get("blocked_holdout_group_count")]), errors="coerce"
    ).iloc[0]
    worst_group = str(row.get("worst_blocked_holdout_group", "") or "").strip()
    worst_auc = pd.to_numeric(
        pd.Series([row.get("worst_blocked_holdout_group_roc_auc")]), errors="coerce"
    ).iloc[0]
    if language == "tr":
        text = (
            f"{pretty_report_model_label(model_name)} için bloke edilmiş holdout denetimi "
            f"({group_column}): ağırlıklı ROC AUC `{float(weighted_auc):.3f}`"
        )
        if pd.notna(group_count):
            text += f" `{int(group_count)}` bloke grup üzerinde"
        if worst_group and pd.notna(worst_auc):
            text += f"; en zor grup `{worst_group}` ROC AUC `{float(worst_auc):.3f}`"
        return f"{text}."
    text = (
        f"{pretty_report_model_label(model_name)} blocked holdout audit ({group_column}): "
        f"weighted ROC AUC `{float(weighted_auc):.3f}`"
    )
    if pd.notna(group_count):
        text += f" across `{int(group_count)}` blocked groups"
    if worst_group and pd.notna(worst_auc):
        text += f"; hardest group `{worst_group}` at ROC AUC `{float(worst_auc):.3f}`"
    return f"{text}."


def blocked_holdout_summary_text(blocked_holdout_summary: pd.DataFrame, *, model_name: str) -> str:
    if blocked_holdout_summary.empty:
        return ""
    working = blocked_holdout_summary.loc[
        blocked_holdout_summary.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(model_name))
    ].copy()
    if working.empty:
        return ""
    texts = [
        blocked_holdout_row_text(pd.Series(row._asdict()), model_name=model_name, language="en")
        for row in working.itertuples(index=False)
    ]
    texts = [text for text in texts if text]
    return " ".join(texts)


def blocked_holdout_summary_text_tr(
    blocked_holdout_summary: pd.DataFrame, *, model_name: str
) -> str:
    if blocked_holdout_summary.empty:
        return ""
    working = blocked_holdout_summary.loc[
        blocked_holdout_summary.get("model_name", pd.Series(dtype=str))
        .astype(str)
        .eq(str(model_name))
    ].copy()
    if working.empty:
        return ""
    texts = [
        blocked_holdout_row_text(pd.Series(row._asdict()), model_name=model_name, language="tr")
        for row in working.itertuples(index=False)
    ]
    texts = [text for text in texts if text]
    return " ".join(texts)


def country_missingness_summary_text(
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
            eligible.get(column_name, pd.Series(float("nan"), index=eligible.index)),
            errors="coerce",
        ).fillna(0.0)
        binary = values.astype(int)
        return int(binary.sum()), int(
            (eligible["label_observed"].fillna(0).astype(int) != binary).sum()
        )

    observed_pos, _ = _label_count("label_observed")
    midpoint_pos, midpoint_flips = _label_count("label_midpoint")
    optimistic_pos, optimistic_flips = _label_count("label_optimistic")
    weighted_pos, weighted_flips = _label_count("label_weighted")

    text = (
        f"{pretty_report_model_label(model_name)} country-missingness audit "
        f"(`country_missingness_bounds.tsv`, `country_missingness_sensitivity.tsv`): "
        f"observed labels mark {observed_pos}/{len(eligible)} eligible backbones positive; "
        "midpoint / optimistic / weighted interpretations shift "
        f"{midpoint_flips}/{optimistic_flips}/{weighted_flips} labels and yield "
        f"{midpoint_pos}/{optimistic_pos}/{weighted_pos} positives."
    )

    if (
        not country_missingness_sensitivity.empty
        and "model_name" in country_missingness_sensitivity.columns
    ):
        model_rows = country_missingness_sensitivity.loc[
            country_missingness_sensitivity["model_name"].astype(str).eq(str(model_name))
        ].copy()
        if not model_rows.empty:
            if "outcome_name" in model_rows.columns:
                model_rows = model_rows.loc[
                    model_rows["outcome_name"]
                    .astype(str)
                    .isin(
                        ["label_observed", "label_midpoint", "label_optimistic", "label_weighted"]
                    )
                ].copy()
            if not model_rows.empty and {"roc_auc", "average_precision"}.issubset(
                model_rows.columns
            ):
                roc_auc = pd.to_numeric(model_rows["roc_auc"], errors="coerce")
                average_precision = pd.to_numeric(model_rows["average_precision"], errors="coerce")
                if roc_auc.notna().any() and average_precision.notna().any():
                    text += (
                        f" Sensitivity across those label variants spans ROC AUC "
                        f"{float(roc_auc.min()):.3f} to {float(roc_auc.max()):.3f} and AP "
                        f"{float(average_precision.min()):.3f} to "
                        f"{float(average_precision.max()):.3f}."
                    )
    return f"{text}."


def candidate_stability_summary_text(
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
    if (
        frequency_column != "bootstrap_top_10_frequency"
        and "bootstrap_top_10_frequency" in working.columns
    ):
        sort_columns.append("bootstrap_top_10_frequency")
    if (
        frequency_column != "variant_top_10_frequency"
        and "variant_top_10_frequency" in working.columns
    ):
        sort_columns.append("variant_top_10_frequency")
    working = working.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    row = working.iloc[0]
    backbone_id = str(row.get("backbone_id", "") or "").strip()
    top_k = pd.to_numeric(pd.Series([row.get("top_k", float("nan"))]), errors="coerce").iloc[0]
    frequency = pd.to_numeric(
        pd.Series([row.get(frequency_column, float("nan"))]), errors="coerce"
    ).iloc[0]
    if language == "tr":
        intro = f"`{file_name}` aday siralama kararliligini raporlar"
        if backbone_id and pd.notna(frequency):
            if pd.notna(top_k):
                return (
                    f"{intro}; en kararlı örnek `{backbone_id}` için ilk "
                    f"`{int(top_k)}` içinde kalma sıklığı `{float(frequency):.2f}`."
                )
            return (
                f"{intro}; en kararlı örnek `{backbone_id}` için sıklık `{float(frequency):.2f}`."
            )
        return f"{intro}."
    intro = f"`{file_name}` records candidate rank stability"
    if frequency_column == "bootstrap_top_k_frequency":
        intro += " across bootstrap resamples"
    else:
        intro += " across model variants"
    if backbone_id and pd.notna(frequency):
        if pd.notna(top_k):
            return (
                f"{intro}; the strongest stable backbone `{backbone_id}` remains in the "
                f"top-`{int(top_k)}` set at frequency `{float(frequency):.2f}`."
            )
        return (
            f"{intro}; the strongest stable backbone `{backbone_id}` has frequency "
            f"`{float(frequency):.2f}`."
        )
    return f"{intro}."


def summarize_false_negative_audit(false_negative_audit: pd.DataFrame) -> tuple[int, str]:
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


def select_confirmatory_row(
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


def rolling_temporal_summary(rolling_temporal: pd.DataFrame) -> dict[str, object]:
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
        "brier_score_min": float(brier_score.min()) if not brier_score.empty else np.nan,
        "brier_score_max": float(brier_score.max()) if not brier_score.empty else np.nan,
    }


def top_sign_stable_features(coefficient_stability_cv: pd.DataFrame, *, top_n: int = 3) -> str:
    if coefficient_stability_cv.empty:
        return "NA"
    working = coefficient_stability_cv.copy()
    if "sign_stable" in working.columns:
        working = working.loc[working["sign_stable"].fillna(False).astype(bool)]
    if working.empty:
        return "NA"
    sort_column = "cv_of_coef" if "cv_of_coef" in working.columns else None
    if sort_column:
        working = working.sort_values(
            [
                sort_column,
                "abs_mean_coefficient" if "abs_mean_coefficient" in working.columns else "feature",
            ]
        )
    labels = working.head(top_n)["feature"].astype(str).tolist()
    return ", ".join(labels) if labels else "NA"
