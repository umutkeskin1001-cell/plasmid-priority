#!/usr/bin/env python3
"""Assemble final summary tables and report metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD, build_context
from plasmid_priority.modeling import (
    annotate_knownness_metadata,
    fit_full_model_predictions,
    get_conservative_model_name,
    get_primary_model_name,
)
from plasmid_priority.reporting import (
    ManagedScriptRun,
    annotate_candidate_explanation_fields,
    build_amrfinder_coverage_table,
    build_backbone_identity_table,
    build_consensus_shortlist,
    build_candidate_signature_context,
    build_consensus_candidate_ranking,
    build_candidate_dossier_table,
    build_candidate_portfolio_table,
    build_candidate_risk_table,
    build_candidate_universe_table,
    build_benchmark_protocol_table,
    build_component_floor_diagnostics,
    build_decision_yield_table,
    build_false_negative_audit,
    build_h_feature_diagnostics,
    build_logistic_implementation_audit,
    build_model_family_summary,
    build_model_selection_scorecard,
    build_model_simplicity_summary,
    build_module_f_enrichment_table,
    build_module_f_top_hits,
    build_novelty_margin_summary,
    build_pathogen_group_comparison,
    build_primary_model_selection_summary,
    build_score_axis_summary,
    build_score_distribution_diagnostics,
    build_temporal_drift_summary,
    build_threshold_flip_table,
    normalize_drug_class_token,
)
from plasmid_priority.reporting.figures import generate_all_figures
from plasmid_priority.utils.dataframe import coalescing_left_merge, read_tsv
from plasmid_priority.utils.files import ensure_directory


def _metrics_to_frame(metrics_path: Path) -> pd.DataFrame:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for model_name, metrics in payload.items():
        if not isinstance(metrics, dict) or "roc_auc" not in metrics:
            continue
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_tsv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _humanize_taxon_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("_", " ")


def _humanize_source_tier(value: object) -> str:
    mapping = {
        "cross_source_supported": "cross-source supported",
        "refseq_dominant": "RefSeq-dominant",
        "insd_dominant": "INSD-dominant",
        "source_mixed": "mixed-source",
        "source_sparse": "source-sparse",
    }
    token = str(value or "").strip()
    return mapping.get(token, token.replace("_", " "))


def _humanize_evidence_tier(value: object, *, language: str = "en") -> str:
    mapping_en = {
        "tier_a": "tier A",
        "tier_b": "tier B",
        "watchlist": "watchlist",
        "novelty_watchlist": "exploratory novelty watchlist",
    }
    mapping_tr = {
        "tier_a": "A seviyesi",
        "tier_b": "B seviyesi",
        "watchlist": "izleme listesi",
        "novelty_watchlist": "kesif amacli yenilik izleme listesi",
    }
    token = str(value or "").strip()
    mapping = mapping_tr if language == "tr" else mapping_en
    return mapping.get(token, token.replace("_", " "))


def _humanize_action_tier(value: object, *, language: str = "en") -> str:
    mapping_en = {
        "core_surveillance": "core surveillance",
        "extended_watchlist": "extended watchlist",
        "low_confidence_backlog": "low-confidence review pool",
        "unassigned": "unassigned",
    }
    mapping_tr = {
        "core_surveillance": "cekirdek izlem",
        "extended_watchlist": "genisletilmis izleme listesi",
        "low_confidence_backlog": "dusuk guvenli inceleme havuzu",
        "unassigned": "atanmadi",
    }
    token = str(value or "").strip()
    mapping = mapping_tr if language == "tr" else mapping_en
    return mapping.get(token, token.replace("_", " "))


def _humanize_portfolio_track(value: object, *, language: str = "en") -> str:
    mapping_en = {
        "established_high_risk": "established high-risk shortlist",
        "novel_signal": "separate exploratory early-signal watchlist",
    }
    mapping_tr = {
        "established_high_risk": "yerlesik yuksek risk kisa listesi",
        "novel_signal": "ayri erken sinyal izleme hatti",
    }
    token = str(value or "").strip()
    mapping = mapping_tr if language == "tr" else mapping_en
    return mapping.get(token, token.replace("_", " "))


def _clean_signature_text(text: str) -> str:
    cleaned = str(text or "")
    return cleaned.replace("OR=inf", "OR>100 (seyrek payda)")


def _humanize_candidate_reason(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return ", ".join(part.replace("_", " ") for part in text.split(",") if part.strip())


def _build_outcome_robustness_grid(
    sensitivity: dict[str, dict[str, float]],
    rolling_temporal: pd.DataFrame,
    *,
    default_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not rolling_temporal.empty:
        for row in rolling_temporal.loc[rolling_temporal["status"] == "ok"].to_dict(orient="records"):
            rows.append(
                {
                    "scenario_group": "rolling_temporal",
                    "scenario_name": f"rolling_split_{row['split_year']}_{row['backbone_assignment_mode']}_{int(row.get('horizon_years', max(int(row['test_year_end']) - int(row['split_year']), 1)))}y",
                    "split_year": int(row["split_year"]),
                    "test_year_end": int(row["test_year_end"]),
                    "horizon_years": int(row.get("horizon_years", max(int(row["test_year_end"]) - int(row["split_year"]), 1))),
                    "backbone_assignment_mode": str(row["backbone_assignment_mode"]),
                    "new_country_threshold": int(default_threshold),
                    "roc_auc": float(row["roc_auc"]),
                    "average_precision": float(row["average_precision"]),
                    "positive_prevalence": float(row.get("positive_prevalence", np.nan)),
                    "average_precision_lift": float(row.get("average_precision_lift", np.nan)),
                    "average_precision_enrichment": float(row.get("average_precision_enrichment", np.nan)),
                    "brier_score": float(row["brier_score"]),
                    "n_eligible_backbones": int(row["n_eligible_backbones"]),
                    "n_positive": int(row.get("n_positive", 0)),
                }
            )

    selected_variants = {
        "alternate_split_2014": {"scenario_group": "alternate_split", "split_year": 2014, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
        "alternate_split_2016": {"scenario_group": "alternate_split", "split_year": 2016, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
        "expanded_eligibility_ge_1": {"scenario_group": "expanded_eligibility", "split_year": 2015, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
        "stable_country_outcome": {"scenario_group": "country_stability", "split_year": 2015, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
        "stable_dense_country_outcome": {"scenario_group": "country_stability", "split_year": 2015, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
        "training_only_backbone_rerun": {"scenario_group": "backbone_assignment", "split_year": 2015, "test_year_end": 2023, "backbone_assignment_mode": "training_only", "new_country_threshold": int(default_threshold)},
        "fallback_backbone_rerun": {"scenario_group": "backbone_assignment", "split_year": 2015, "test_year_end": 2023, "backbone_assignment_mode": "fallback", "new_country_threshold": int(default_threshold)},
        "source_balanced_rerun": {"scenario_group": "cohort_balance", "split_year": 2015, "test_year_end": 2023, "new_country_threshold": int(default_threshold)},
    }
    for threshold in (1, 2, 4, 5):
        if threshold == int(default_threshold):
            continue
        selected_variants[f"alternate_outcome_threshold_{threshold}"] = {
            "scenario_group": "alternate_outcome_threshold",
            "split_year": 2015,
            "test_year_end": 2023,
            "new_country_threshold": threshold,
        }
    for variant_name, metadata in selected_variants.items():
        metrics = sensitivity.get(variant_name, {})
        if not metrics or metrics.get("skipped"):
            continue
        row = {
            "scenario_group": metadata.get("scenario_group"),
            "scenario_name": variant_name,
            "split_year": metadata.get("split_year"),
            "test_year_end": metadata.get("test_year_end"),
            "backbone_assignment_mode": metadata.get("backbone_assignment_mode", "all_records"),
            "new_country_threshold": metadata.get("new_country_threshold"),
            "roc_auc": float(metrics["roc_auc"]),
            "average_precision": float(metrics["average_precision"]),
            "positive_prevalence": float(metrics.get("positive_prevalence", np.nan)),
            "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
            "average_precision_enrichment": float(metrics.get("average_precision_enrichment", np.nan)),
            "brier_score": float(metrics["brier_score"]),
            "n_eligible_backbones": int(metrics["n_eligible_backbones"]),
            "n_positive": int(metrics.get("n_positive", 0)),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["scenario_group", "split_year", "scenario_name"]).reset_index(drop=True) if rows else pd.DataFrame()


def _build_threshold_sensitivity_table(
    sensitivity: dict[str, dict[str, float]],
    *,
    default_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    default_metrics = sensitivity.get("default") or sensitivity.get("parsimonious_model") or {}
    if default_metrics and not default_metrics.get("skipped"):
        rows.append(
            {
                "new_country_threshold": int(default_threshold),
                "variant": "default",
                "roc_auc": float(default_metrics["roc_auc"]),
                "roc_auc_ci_lower": float(default_metrics.get("roc_auc_ci_lower", default_metrics["roc_auc"])),
                "roc_auc_ci_upper": float(default_metrics.get("roc_auc_ci_upper", default_metrics["roc_auc"])),
                "average_precision": float(default_metrics["average_precision"]),
                "average_precision_lift": float(default_metrics.get("average_precision_lift", np.nan)),
                "positive_prevalence": float(default_metrics.get("positive_prevalence", np.nan)),
                "n_eligible_backbones": int(default_metrics.get("n_eligible_backbones", 0)),
            }
        )
    for variant, metrics in sensitivity.items():
        if not str(variant).startswith("alternate_outcome_threshold_") or metrics.get("skipped"):
            continue
        threshold = int(str(variant).rsplit("_", 1)[-1])
        rows.append(
            {
                "new_country_threshold": threshold,
                "variant": str(variant),
                "roc_auc": float(metrics["roc_auc"]),
                "roc_auc_ci_lower": float(metrics.get("roc_auc_ci_lower", metrics["roc_auc"])),
                "roc_auc_ci_upper": float(metrics.get("roc_auc_ci_upper", metrics["roc_auc"])),
                "average_precision": float(metrics["average_precision"]),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
                "positive_prevalence": float(metrics.get("positive_prevalence", np.nan)),
                "n_eligible_backbones": int(metrics.get("n_eligible_backbones", 0)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["new_country_threshold"]).sort_values("new_country_threshold").reset_index(drop=True)


def _build_l2_sensitivity_table(sensitivity: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, metrics in sensitivity.items():
        if not str(variant).startswith("primary_l2_") or metrics.get("skipped"):
            continue
        rows.append(
            {
                "variant": str(variant),
                "l2_penalty": float(metrics.get("l2", str(variant).replace("primary_l2_", "").replace("p", "."))),
                "sample_weight_mode": str(metrics.get("sample_weight_mode", "source_balanced")),
                "roc_auc": float(metrics["roc_auc"]),
                "roc_auc_ci_lower": float(metrics.get("roc_auc_ci_lower", metrics["roc_auc"])),
                "roc_auc_ci_upper": float(metrics.get("roc_auc_ci_upper", metrics["roc_auc"])),
                "average_precision": float(metrics["average_precision"]),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
                "precision_at_top_25": float(metrics.get("precision_at_top_25", np.nan)),
                "recall_at_top_25": float(metrics.get("recall_at_top_25", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("l2_penalty").reset_index(drop=True)


def _build_weighting_sensitivity_table(sensitivity: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in ("default", "class_balanced_primary", "source_plus_class_balanced_primary", "source_balanced_rerun"):
        metrics = sensitivity.get(variant, {})
        if not metrics or metrics.get("skipped"):
            continue
        rows.append(
            {
                "variant": variant,
                "sample_weight_mode": str(metrics.get("sample_weight_mode", "source_balanced" if variant == "default" else "")),
                "roc_auc": float(metrics["roc_auc"]),
                "average_precision": float(metrics["average_precision"]),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
                "precision_at_top_25": float(metrics.get("precision_at_top_25", np.nan)),
                "recall_at_top_25": float(metrics.get("recall_at_top_25", np.nan)),
                "positive_prevalence": float(metrics.get("positive_prevalence", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _select_predefined_group_holdout_highlight(group_holdout: pd.DataFrame) -> dict[str, object] | None:
    if group_holdout.empty:
        return None
    working = group_holdout.loc[group_holdout["status"] == "ok"].copy()
    if working.empty:
        return None
    working = working.loc[
        working["group_column"].isin(["dominant_source", "dominant_region_train"])
        & working["n_test_backbones"].fillna(0).astype(int).ge(25)
    ].copy()
    if working.empty:
        return None
    return working.sort_values(["roc_auc", "n_test_backbones"], ascending=[False, False]).iloc[0].to_dict()


def _dominant_non_empty(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values.loc[values != ""]
    if values.empty:
        return ""
    return str(values.value_counts().index[0])


def _top_tokens(series: pd.Series, *, n: int = 5) -> str:
    counts: dict[str, int] = {}
    for value in series.fillna("").astype(str):
        for token in [part.strip() for part in value.split(",") if part.strip()]:
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(token for token, _ in ordered[:n])


_NON_PUBLIC_HEALTH_AMR_TERMS = (
    "MERCURY",
    "TELLURIUM",
    "CADMIUM",
    "ARSENIC",
    "COPPER",
    "SILVER",
    "NICKEL",
    "COBALT",
    "ZINC",
    "LEAD",
    "BISMUTH",
    "CHROMATE",
    "QUATERNARY AMMONIUM",
    "DISINFECTING AGENTS",
    "ANTISEPTICS",
)


def _is_public_health_amr_class(token: str) -> bool:
    upper = token.upper()
    return not any(term in upper for term in _NON_PUBLIC_HEALTH_AMR_TERMS)


def _top_public_health_amr_classes(series: pd.Series, *, n: int = 5) -> str:
    counts: dict[str, int] = {}
    for value in series.fillna("").astype(str):
        cleaned = value.replace(";", ",")
        for token in [part.strip() for part in cleaned.split(",") if part.strip()]:
            normalized = normalize_drug_class_token(token)
            if normalized and _is_public_health_amr_class(normalized):
                counts[normalized] = counts.get(normalized, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(token for token, _ in ordered[:n])


_LIKELY_AMR_GENE_PREFIXES = (
    "bla",
    "aac",
    "aad",
    "aph",
    "ant",
    "arr",
    "arm",
    "rmt",
    "qnr",
    "erm",
    "tet",
    "sul",
    "dfr",
    "mcr",
    "cat",
    "cml",
    "flo",
    "oqx",
    "qac",
    "mph",
    "fos",
    "van",
    "lnu",
    "msr",
    "mef",
    "ere",
    "sat",
    "vga",
    "lsa",
)


def _top_likely_amr_genes(series: pd.Series, *, n: int = 5) -> str:
    counts: dict[str, int] = {}
    for value in series.fillna("").astype(str):
        for token in [part.strip() for part in value.split(",") if part.strip()]:
            normalized = "".join(ch for ch in token.lower() if ch.isalnum())
            if normalized.startswith(_LIKELY_AMR_GENE_PREFIXES):
                counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(token for token, _ in ordered[:n])


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


def _build_backbone_assignment_summary(backbones: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    if backbones.empty:
        return pd.DataFrame()
    assignment = (
        backbones.groupby("backbone_assignment_rule", as_index=False)
        .agg(
            n_records=("sequence_accession", "nunique"),
            n_backbones=("backbone_id", "nunique"),
        )
    )
    scored_with_rule = scored.merge(
        backbones[["backbone_id", "backbone_assignment_rule"]].drop_duplicates(),
        on="backbone_id",
        how="left",
        validate="m:1",
    ) if not scored.empty else pd.DataFrame()
    if not scored_with_rule.empty:
        metrics = (
            scored_with_rule.groupby("backbone_assignment_rule", as_index=False)
            .agg(
                n_scored_backbones=("backbone_id", "nunique"),
                n_outcome_eligible=("spread_label", lambda values: int(pd.Series(values).notna().sum())),
                mean_coherence=("coherence_score", "mean"),
                mean_bio_priority_index=("bio_priority_index", "mean"),
                mean_evidence_support_index=("evidence_support_index", "mean"),
            )
        )
        assignment = assignment.merge(metrics, on="backbone_assignment_rule", how="left")
    return assignment.sort_values(["n_backbones", "n_records"], ascending=[False, False]).reset_index(drop=True)


def _build_candidate_brief_table(
    candidate_portfolio: pd.DataFrame,
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_portfolio.empty or backbones.empty:
        return pd.DataFrame()
    merged = backbones.merge(
        amr_consensus[["sequence_accession", "amr_gene_symbols", "amr_drug_classes"]],
        on="sequence_accession",
        how="left",
    ) if not amr_consensus.empty else backbones.copy()
    years = pd.to_numeric(merged["resolved_year"], errors="coerce").fillna(0).astype(int)
    merged = merged.assign(resolved_year_int=years)

    rows: list[dict[str, object]] = []
    candidate_ids = candidate_portfolio["backbone_id"].astype(str).tolist()
    for row in candidate_portfolio.to_dict(orient="records"):
        backbone_id = str(row["backbone_id"])
        frame = merged.loc[merged["backbone_id"].astype(str) == backbone_id].copy()
        if frame.empty:
            continue
        training = frame.loc[frame["resolved_year_int"] <= 2015].copy()
        testing = frame.loc[(frame["resolved_year_int"] > 2015) & (frame["resolved_year_int"] <= 2023)].copy()
        training_countries = sorted({value for value in training["country"].fillna("").astype(str).str.strip() if value})
        testing_countries = sorted({value for value in testing["country"].fillna("").astype(str).str.strip() if value})
        new_countries = sorted(set(testing_countries) - set(training_countries))
        summary_context = training if not training.empty else frame
        dominant_species = _dominant_non_empty(summary_context["species"])
        dominant_genus = _dominant_non_empty(summary_context["genus"])
        primary_replicon = _dominant_non_empty(summary_context["primary_replicon"])
        dominant_source = _dominant_non_empty(summary_context["record_origin"])
        top_amr_classes = _top_public_health_amr_classes(summary_context.get("amr_drug_classes", pd.Series(dtype=str)))
        top_amr_genes = _top_likely_amr_genes(summary_context.get("amr_gene_symbols", pd.Series(dtype=str)))
        source_support_tier = str(row.get("source_support_tier", ""))
        evidence_tier = str(row.get("evidence_tier", row.get("candidate_confidence_tier", "")) or "")
        action_tier = str(row.get("action_tier", row.get("recommended_monitoring_tier", "")) or "")
        consensus_membership = bool(row.get("in_consensus_top50", False))
        consensus_rank = row.get("consensus_rank", np.nan)
        enriched_signatures = str(row.get("module_f_enriched_signatures", "") or "")
        species_or_genus = _humanize_taxon_label(dominant_species or dominant_genus or backbone_id)
        summary_signatures = _clean_signature_text(" | ".join(part.strip() for part in enriched_signatures.split("|")[:2] if part.strip()))
        new_country_count = int(row.get("n_new_countries", len(new_countries)) or 0)
        primary_driver_axis = _humanize_candidate_reason(row.get("primary_driver_axis", ""))
        secondary_driver_axis = _humanize_candidate_reason(row.get("secondary_driver_axis", ""))
        mechanistic_rationale = _humanize_candidate_reason(row.get("mechanistic_rationale", ""))
        monitoring_rationale = _humanize_candidate_reason(row.get("monitoring_rationale", ""))
        track_value = str(row.get("portfolio_track", "") or "")
        track_en = _humanize_portfolio_track(track_value, language="en") or "candidate context"
        track_tr = _humanize_portfolio_track(track_value, language="tr") or "aday baglami"
        evidence_tier_en = _humanize_evidence_tier(evidence_tier, language="en") or "unspecified"
        evidence_tier_tr = _humanize_evidence_tier(evidence_tier, language="tr") or "belirtilmedi"
        action_tier_en = _humanize_action_tier(action_tier, language="en") or "unassigned"
        action_tier_tr = _humanize_action_tier(action_tier, language="tr") or "atanmadi"
        consensus_en = "yes" if consensus_membership else "no"
        consensus_tr = "evet" if consensus_membership else "hayir"
        novelty_note_en = (
            " It is not the main published-primary shortlist and should be treated as a separate exploratory watchlist."
            if track_value == "novel_signal"
            else ""
        )
        novelty_note_tr = (
            " Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir."
            if track_value == "novel_signal"
            else ""
        )
        mechanistic_note_en = (
            f" Mechanistic emphasis: {primary_driver_axis or 'mixed signal'}"
            + (f" with secondary {secondary_driver_axis}" if secondary_driver_axis else "")
            + (f"; profile tags: {mechanistic_rationale}." if mechanistic_rationale else ".")
        )
        mechanistic_note_tr = (
            f" Mekanistik agirlik: {primary_driver_axis or 'karma sinyal'}"
            + (f"; ikincil eksen {secondary_driver_axis}" if secondary_driver_axis else "")
            + (f"; profil etiketleri: {mechanistic_rationale}." if mechanistic_rationale else ".")
        )
        monitoring_note_en = f" Monitoring rationale: {monitoring_rationale}." if monitoring_rationale else ""
        monitoring_note_tr = f" Izlem gerekcesi: {monitoring_rationale}." if monitoring_rationale else ""
        summary_en = (
            f"{backbone_id} is dominated by {species_or_genus}; training-period support is {int(row.get('member_count_train', 0) or 0)} records across "
            f"{int(row.get('n_countries_train', 0) or 0)} countries, and after 2015 it appears in {new_country_count} new countries. "
            f"For review, treat it as {track_en}. Evidence tier: {evidence_tier_en}; monitoring tier: {action_tier_en}; multi-model consensus top-50: {consensus_en}."
            f"{novelty_note_en}{mechanistic_note_en} {monitoring_note_en} Main public-health AMR classes: {top_amr_classes or 'none detected'}."
        )
        summary_tr = (
            f"{backbone_id} omurgasi agirlikli olarak {species_or_genus} ile iliskilidir; egitim doneminde {int(row.get('member_count_train', 0) or 0)} kayit ve "
            f"{int(row.get('n_countries_train', 0) or 0)} ulke destegi vardir, 2015 sonrasinda ise {new_country_count} yeni ulkede gorulmustur. "
            f"Juri yorumu icin bu aday {track_tr} olarak ele alinmalidir. Kanit seviyesi {evidence_tier_tr}; izlem duzeyi {action_tier_tr}; coklu model uzlasi top-50 durumu: {consensus_tr}."
            f"{novelty_note_tr}{mechanistic_note_tr} {monitoring_note_tr} Baskin halk sagligi odakli AMR siniflari: {top_amr_classes or 'tespit edilmedi'}."
        )
        rows.append(
            {
                "portfolio_track": row.get("portfolio_track"),
                "track_rank": row.get("track_rank"),
                "backbone_id": backbone_id,
                "dominant_genus": dominant_genus,
                "dominant_species": dominant_species,
                "primary_replicon": primary_replicon,
                "dominant_record_origin": dominant_source,
                "source_support_tier": source_support_tier,
                "training_country_count": int(len(training_countries)),
                "training_country_examples": ",".join(training_countries[:5]),
                "new_country_count_post_2015": int(len(new_countries)),
                "new_country_examples_post_2015": ",".join(new_countries[:5]),
                "first_year_observed": int(frame["resolved_year_int"].loc[frame["resolved_year_int"] > 0].min())
                if (frame["resolved_year_int"] > 0).any()
                else np.nan,
                "last_year_observed": int(frame["resolved_year_int"].loc[frame["resolved_year_int"] > 0].max())
                if (frame["resolved_year_int"] > 0).any()
                else np.nan,
                "top_amr_classes": top_amr_classes,
                "top_amr_genes": top_amr_genes,
                "module_f_enriched_signatures": enriched_signatures,
                "primary_driver_axis": row.get("primary_driver_axis", ""),
                "secondary_driver_axis": row.get("secondary_driver_axis", ""),
                "mechanistic_rationale": row.get("mechanistic_rationale", ""),
                "monitoring_rationale": row.get("monitoring_rationale", ""),
                "candidate_summary_en": summary_en,
                "candidate_summary_tr": summary_tr,
            }
        )
    brief = pd.DataFrame(rows)
    if brief.empty:
        return brief
    order = pd.Categorical(brief["backbone_id"], categories=candidate_ids, ordered=True)
    return brief.assign(_order=order).sort_values(["portfolio_track", "track_rank", "_order"]).drop(columns="_order").reset_index(drop=True)


def _add_visibility_alias(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "spread_label" not in frame.columns or "visibility_expansion_label" in frame.columns:
        return frame
    aliased = frame.copy()
    aliased["visibility_expansion_label"] = aliased["spread_label"]
    return aliased


def _build_candidate_evidence_matrix(
    candidate_portfolio: pd.DataFrame,
    candidate_briefs: pd.DataFrame,
    candidate_threshold_flip: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_portfolio.empty:
        return pd.DataFrame()
    matrix = candidate_portfolio.copy()
    if not candidate_briefs.empty:
        brief_columns = [
            column
            for column in ["backbone_id", "dominant_genus", "dominant_species", "top_amr_classes", "top_amr_genes"]
            if column in candidate_briefs.columns
        ]
        if brief_columns:
            matrix = coalescing_left_merge(matrix, candidate_briefs[brief_columns], on="backbone_id")
    if not candidate_threshold_flip.empty:
        flip_columns = [
            column
            for column in [
                "backbone_id",
                "threshold_flip_count",
                "eligible_for_threshold_audit",
                "default_threshold",
                "spread_label_default",
                "label_ge_2",
                "label_ge_3",
                "label_ge_4",
            ]
            if column in candidate_threshold_flip.columns
        ]
        if flip_columns:
            matrix = coalescing_left_merge(matrix, candidate_threshold_flip[flip_columns], on="backbone_id")
    preferred_columns = [
        "portfolio_track",
        "track_rank",
        "backbone_id",
        "evidence_tier",
        "action_tier",
        "false_positive_risk_tier",
        "risk_flag_count",
        "consensus_rank",
        "consensus_support_count",
        "rank_disagreement_primary_vs_conservative",
        "primary_model_candidate_score",
        "conservative_model_candidate_score",
        "baseline_both_candidate_score",
        "novelty_margin_vs_baseline",
        "bootstrap_top_10_frequency",
        "variant_top_10_frequency",
        "external_support_modalities_count",
        "source_support_tier",
        "module_f_enriched_signature_count",
        "top_amr_classes",
        "top_amr_genes",
        "n_new_countries",
        "spread_label",
        "default_threshold",
        "spread_label_default",
        "threshold_flip_count",
        "eligible_for_threshold_audit",
        "label_ge_2",
        "label_ge_3",
        "label_ge_4",
        "dominant_species",
        "dominant_genus",
    ]
    available = [column for column in preferred_columns if column in matrix.columns]
    return matrix[available].reset_index(drop=True)


def _build_candidate_multiverse_stability(
    candidate_stability: pd.DataFrame,
    candidate_threshold_flip: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_stability.empty:
        return pd.DataFrame()
    base = candidate_stability.copy()
    threshold_cols = [
        column
        for column in [
            "backbone_id",
            "threshold_flip_count",
            "eligible_for_threshold_audit",
        ]
        if column in candidate_threshold_flip.columns
    ]
    if threshold_cols:
        base = coalescing_left_merge(
            base,
            candidate_threshold_flip[threshold_cols].drop_duplicates("backbone_id"),
            on="backbone_id",
        )
    frequency_columns = [
        "bootstrap_top_10_frequency",
        "bootstrap_top_25_frequency",
        "variant_top_10_frequency",
        "variant_top_25_frequency",
    ]
    for column in frequency_columns:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    threshold_flip = pd.to_numeric(base.get("threshold_flip_count", pd.Series(np.nan, index=base.index)), errors="coerce")
    eligible_threshold = base.get("eligible_for_threshold_audit", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    threshold_robustness = pd.Series(np.nan, index=base.index, dtype=float)
    if eligible_threshold.any():
        max_flip = max(float(threshold_flip.loc[eligible_threshold].max()), 1.0)
        threshold_robustness.loc[eligible_threshold] = (
            1.0 - threshold_flip.loc[eligible_threshold].fillna(max_flip) / max_flip
        ).clip(lower=0.0, upper=1.0)
    base["threshold_robustness_score"] = threshold_robustness
    component_scores: list[pd.Series] = []
    component_active: list[pd.Series] = []
    if "bootstrap_top_25_frequency" in base.columns:
        component_scores.append(base["bootstrap_top_25_frequency"].astype(float))
        component_active.append(pd.Series(True, index=base.index, dtype=bool))
    if "variant_top_25_frequency" in base.columns:
        component_scores.append(base["variant_top_25_frequency"].astype(float))
        component_active.append(pd.Series(True, index=base.index, dtype=bool))
    if threshold_cols:
        component_scores.append(base["threshold_robustness_score"].fillna(0.0).astype(float))
        component_active.append(eligible_threshold)
    if component_scores:
        score_sum = pd.Series(0.0, index=base.index, dtype=float)
        active_count = pd.Series(0, index=base.index, dtype=int)
        for score, active in zip(component_scores, component_active, strict=False):
            score_sum = score_sum + score.where(active, 0.0)
            active_count = active_count + active.astype(int)
        base["multiverse_component_count"] = active_count.astype(int)
        base["multiverse_stability_score"] = np.where(
            active_count > 0,
            score_sum / active_count,
            np.nan,
        )
    else:
        base["multiverse_component_count"] = 0
        base["multiverse_stability_score"] = np.nan
    base["multiverse_stability_tier"] = np.select(
        [
            base["multiverse_stability_score"].fillna(0.0) >= 0.80,
            base["multiverse_stability_score"].fillna(0.0) >= 0.55,
        ],
        ["stable", "moderately_stable"],
        default="fragile",
    )
    preferred = [
        "backbone_id",
        "multiverse_stability_score",
        "multiverse_stability_tier",
        "bootstrap_top_10_frequency",
        "bootstrap_top_25_frequency",
        "variant_top_10_frequency",
        "variant_top_25_frequency",
        "threshold_flip_count",
        "threshold_robustness_score",
        "multiverse_component_count",
        "bootstrap_rank_std",
        "variant_rank_std",
        "coherence_score",
        "knownness_score",
        "primary_model_candidate_score",
    ]
    available = [column for column in preferred if column in base.columns]
    return base[available].sort_values(
        ["multiverse_stability_score", "primary_model_candidate_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _lookup_decision_yield(decision_yield: pd.DataFrame, model_name: str, top_k: int) -> pd.Series | None:
    if decision_yield.empty:
        return None
    match = decision_yield.loc[
        (decision_yield["model_name"] == model_name) & (decision_yield["top_k"] == int(top_k))
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _prune_duplicate_table_artifacts(core_dir: Path, diag_dir: Path, core_file_names: set[str]) -> None:
    core_names = {path.name for path in core_dir.glob("*.tsv")}
    diag_names = {path.name for path in diag_dir.glob("*.tsv")}
    for name in sorted(core_names & diag_names):
        stale_path = diag_dir / name if name in core_file_names else core_dir / name
        if stale_path.exists():
            stale_path.unlink()


def _prune_shadowed_report_tables(core_dir: Path, diag_dir: Path, analysis_dir: Path) -> None:
    analysis_names = {path.name for path in analysis_dir.glob("*.tsv")}
    for directory in (core_dir, diag_dir):
        for path in directory.glob("*.tsv"):
            if path.name in analysis_names and path.exists():
                path.unlink()


def _select_summary_candidate_briefs(candidate_briefs: pd.DataFrame, *, per_track: int = 5) -> pd.DataFrame:
    if candidate_briefs.empty or "portfolio_track" not in candidate_briefs.columns:
        return candidate_briefs.head(per_track * 2).copy()
    tracks = ["established_high_risk", "novel_signal"]
    frames: list[pd.DataFrame] = []
    for track_order, track in enumerate(tracks):
        frame = candidate_briefs.loc[candidate_briefs["portfolio_track"] == track].head(per_track).copy()
        if frame.empty:
            continue
        frame["_summary_track_order"] = track_order
        frame["_summary_row_order"] = range(len(frame))
        frames.append(frame)
    if not frames:
        return candidate_briefs.head(per_track * 2).copy()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["_summary_row_order", "_summary_track_order"]).drop(
        columns=["_summary_track_order", "_summary_row_order"]
    ).reset_index(drop=True)


def _write_turkish_summary(
    output_path: Path,
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
    outcome_threshold: int,
) -> None:
    knownness_matched_validation = knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    gate_consistency_audit = gate_consistency_audit if gate_consistency_audit is not None else pd.DataFrame()
    model_selection_scorecard = model_selection_scorecard if model_selection_scorecard is not None else pd.DataFrame()
    secondary_outcome_performance = secondary_outcome_performance if secondary_outcome_performance is not None else pd.DataFrame()
    weighted_country_outcome = weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    metadata_quality_summary = metadata_quality_summary if metadata_quality_summary is not None else pd.DataFrame()
    primary = model_metrics.loc[model_metrics["model_name"] == primary_model_name].iloc[0]
    conservative = model_metrics.loc[model_metrics["model_name"] == conservative_model_name].iloc[0]
    strongest = model_metrics.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    augmented = model_metrics.loc[model_metrics["model_name"] == "natural_auc_priority"].copy()
    augmented_row = augmented.iloc[0] if not augmented.empty else pd.Series(dtype=object)
    knownness_robust = model_metrics.loc[model_metrics["model_name"] == "knownness_robust_priority"].copy()
    knownness_robust_row = knownness_robust.iloc[0] if not knownness_robust.empty else pd.Series(dtype=object)
    support_calibrated = model_metrics.loc[model_metrics["model_name"] == "support_calibrated_priority"].copy()
    support_calibrated_row = support_calibrated.iloc[0] if not support_calibrated.empty else pd.Series(dtype=object)
    support_synergy = model_metrics.loc[model_metrics["model_name"] == "support_synergy_priority"].copy()
    support_synergy_row = support_synergy.iloc[0] if not support_synergy.empty else pd.Series(dtype=object)
    host_transfer_synergy = model_metrics.loc[model_metrics["model_name"] == "host_transfer_synergy_priority"].copy()
    host_transfer_synergy_row = host_transfer_synergy.iloc[0] if not host_transfer_synergy.empty else pd.Series(dtype=object)
    threat_architecture = model_metrics.loc[model_metrics["model_name"] == "threat_architecture_priority"].copy()
    threat_architecture_row = threat_architecture.iloc[0] if not threat_architecture.empty else pd.Series(dtype=object)
    phylogeny_aware = model_metrics.loc[model_metrics["model_name"] == "phylogeny_aware_priority"].copy()
    phylogeny_aware_row = phylogeny_aware.iloc[0] if not phylogeny_aware.empty else pd.Series(dtype=object)
    structured_signal = model_metrics.loc[model_metrics["model_name"] == "structured_signal_priority"].copy()
    structured_signal_row = structured_signal.iloc[0] if not structured_signal.empty else pd.Series(dtype=object)
    selected_briefs = _select_summary_candidate_briefs(candidate_briefs, per_track=4)
    primary_top10 = _lookup_decision_yield(decision_yield, primary_model_name, 10)
    primary_top25 = _lookup_decision_yield(decision_yield, primary_model_name, 25)
    conservative_top10 = _lookup_decision_yield(decision_yield, conservative_model_name, 10)
    baseline_top10 = _lookup_decision_yield(decision_yield, "baseline_both", 10)
    established_count = int((candidate_portfolio.get("portfolio_track", pd.Series(dtype=str)) == "established_high_risk").sum()) if not candidate_portfolio.empty else 0
    novel_count = int((candidate_portfolio.get("portfolio_track", pd.Series(dtype=str)) == "novel_signal").sum()) if not candidate_portfolio.empty else 0
    selection_row = model_selection_summary.iloc[0] if not model_selection_summary.empty else pd.Series(dtype=object)
    scorecard_primary_row = model_selection_scorecard.loc[
        model_selection_scorecard.get("model_name", pd.Series(dtype=str)).astype(str).eq(primary_model_name)
    ].head(1) if not model_selection_scorecard.empty else pd.DataFrame()
    scorecard_best_row = model_selection_scorecard.head(1) if not model_selection_scorecard.empty else pd.DataFrame()
    selection_rationale = str(selection_row.get("selection_rationale", "")).strip()
    strongest_top10 = _lookup_decision_yield(decision_yield, str(strongest["model_name"]), 10)
    primary_vs_strongest_overlap = selection_row.get("primary_vs_strongest_top_10_overlap_count", np.nan)
    primary_vs_strongest_overlap_25 = selection_row.get("primary_vs_strongest_top_25_overlap_count", np.nan)
    primary_vs_strongest_overlap_50 = selection_row.get("primary_vs_strongest_top_50_overlap_count", np.nan)
    knownness_row = knownness_summary.iloc[0] if not knownness_summary.empty else pd.Series(dtype=object)
    lowest_knownness_auc = knownness_row.get("lowest_knownness_quartile_primary_roc_auc", np.nan)
    q1_supported = bool(knownness_row.get("lowest_knownness_quartile_supported", False))
    top_k_lower_half_count = knownness_row.get("top_k_lower_half_knownness_count", np.nan)
    source_balance_mean_auc = float(source_balance_resampling["roc_auc"].mean()) if not source_balance_resampling.empty else np.nan
    source_balance_std_auc = float(source_balance_resampling["roc_auc"].std()) if not source_balance_resampling.empty else np.nan
    specialist_q1 = novelty_specialist_metrics.loc[
        (novelty_specialist_metrics["cohort_name"] == "lowest_knownness_quartile")
        & (novelty_specialist_metrics["status"] == "ok")
    ].sort_values("roc_auc", ascending=False).head(1).copy() if not novelty_specialist_metrics.empty else pd.DataFrame()
    adaptive_row = adaptive_gated_metrics.loc[
        (adaptive_gated_metrics["model_name"] == "adaptive_natural_priority")
        & (adaptive_gated_metrics["status"] == "ok")
    ].copy() if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_knownness_robust_row = adaptive_gated_metrics.loc[
        (adaptive_gated_metrics["model_name"] == "adaptive_knownness_robust_priority")
        & (adaptive_gated_metrics["status"] == "ok")
    ].copy() if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_best_row = adaptive_gated_metrics.loc[
        adaptive_gated_metrics["status"] == "ok"
    ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1) if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_preferred_row = pd.DataFrame()
    if not adaptive_gated_metrics.empty and not gate_consistency_audit.empty:
        stable_models = gate_consistency_audit.loc[
            gate_consistency_audit["gate_consistency_tier"] == "stable",
            "model_name",
        ].drop_duplicates()
        adaptive_preferred_row = adaptive_gated_metrics.loc[
            (adaptive_gated_metrics["status"] == "ok")
            & adaptive_gated_metrics["model_name"].isin(stable_models)
        ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1)
    adaptive_best_gate_row = pd.DataFrame()
    if not adaptive_best_row.empty and not gate_consistency_audit.empty:
        adaptive_best_gate_row = gate_consistency_audit.loc[
            gate_consistency_audit["model_name"] == str(adaptive_best_row.iloc[0]["model_name"])
        ].sort_values("mean_abs_route_delta_near_gate", ascending=True).head(1)
    matched_primary = knownness_matched_validation.loc[
        (knownness_matched_validation["matched_stratum"] == "__weighted_overall__")
        & (knownness_matched_validation["model_name"] == primary_model_name)
    ].copy() if not knownness_matched_validation.empty else pd.DataFrame()
    matched_baseline = knownness_matched_validation.loc[
        (knownness_matched_validation["matched_stratum"] == "__weighted_overall__")
        & (knownness_matched_validation["model_name"] == "baseline_both")
    ].copy() if not knownness_matched_validation.empty else pd.DataFrame()
    macro_jump_row = secondary_outcome_performance.loc[
        (secondary_outcome_performance["outcome_name"] == "macro_region_jump_label")
        & (secondary_outcome_performance["status"] == "ok")
    ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1) if not secondary_outcome_performance.empty else pd.DataFrame()
    weighted_country_row = weighted_country_outcome.loc[
        weighted_country_outcome["status"] == "ok"
    ].sort_values("spearman_corr", ascending=False).head(1) if not weighted_country_outcome.empty else pd.DataFrame()
    count_outcome_row = count_outcome_audit.loc[
        count_outcome_audit["status"] == "ok"
    ].sort_values("spearman_corr", ascending=False).head(1) if not count_outcome_audit.empty else pd.DataFrame()
    mean_metadata_quality = float(metadata_quality_summary["metadata_quality_score"].mean()) if not metadata_quality_summary.empty and "metadata_quality_score" in metadata_quality_summary.columns else np.nan
    localized_selection_rationale = (
        "mevcut ana benchmark, marjinal farklar olsa bile headline benchmark olarak acik tutulur ve alternatif denetim gorunumleri ayrica raporlanir"
    )
    if str(primary_model_name) == str(strongest["model_name"]):
        localized_selection_rationale = "mevcut ana benchmark ile en yuksek metrikli tekil model bu veri donumunde ayni secenekte bulusmustur"
    lines = [
        "# Proje Ozeti (TR)",
        "",
        "## Temel Fikir",
        "",
        "Bu proje plasmid omurgalarini egitim donemindeki biyolojik ve ekolojik sinyallerle puanlar; daha sonra bu omurgalarin 2015 sonrasinda yeni ulkelerde gorunur olup olmadigini retrospektif olarak test eder.",
        "",
        "## Formal Hipotezler",
        "",
        "- **H0 (sifir)**: <=2015 verisinden uretilen T/H/A temelli oncelik sinyali, 2015 sonrasi cok-ulkeli gorunurluk genislemesini ayirt etmez (ROC AUC = 0.50).",
        "- **H1 (alternatif)**: Ayni oncelik sinyali, 2015 sonrasi cok-ulkeli gorunurluk genislemesi ile pozitif iliskilidir (ROC AUC > 0.50).",
        "- **Anlamlilik olcutu**: Empirik permutasyon p-degeri < 0.01; ana null audit'i, yayinlanan ana modelin ozellik seti ve ayni L2/agirliklandirma ayarlari ile kurulur.",
        "",
        "## Ana Model ve Denetim Baglami",
        "",
        f"- Mevcut ana benchmark: `{_pretty_report_model_label(primary_model_name)}` | ROC AUC `{primary['roc_auc']:.3f}` | AP `{primary['average_precision']:.3f}`.",
        f"- En yuksek metrikli denetim modeli: `{_pretty_report_model_label(str(strongest['model_name']))}` | ROC AUC `{strongest['roc_auc']:.3f}` | AP `{strongest['average_precision']:.3f}`.",
        f"- Koruyucu karsilastirma modeli: `{_pretty_report_model_label(conservative_model_name)}` | ROC AUC `{conservative['roc_auc']:.3f}` | AP `{conservative['average_precision']:.3f}`.",
        f"- Kurator odakli aday portfoyu: `{established_count}` yerlesik yuksek-risk + `{novel_count}` erken-sinyal adayi.",
    ]
    if not scorecard_primary_row.empty:
        primary_scorecard = scorecard_primary_row.iloc[0]
        best_scorecard = scorecard_best_row.iloc[0] if not scorecard_best_row.empty else primary_scorecard
        scorecard_metric_text = (
            "overall AUC, AP, lower-half/q1 knownness, matched-knownness, source holdout ve knownness-Spearman'i"
            if q1_supported
            else "overall AUC, AP, lower-half knownness, matched-knownness, source holdout ve knownness-Spearman'i"
        )
        lines.append(
            f"- Cok amacli model secim scorecard'inda ana model `{int(primary_scorecard['selection_rank'])}/{int(len(model_selection_scorecard))}` sirada; bu scorecard {scorecard_metric_text} birlikte degerlendirir."
        )
        if str(best_scorecard.get("model_name", primary_model_name)) != primary_model_name:
            lines.append(
                f"- Scorecard'da en ustte yer alan model `{best_scorecard['model_name']}` olsa da headline benchmark `{primary_model_name}` olarak korunur; secim gerekcesi yukarida ayri satirda acik verilir."
            )
    if not augmented.empty:
        lines.append(
            f"- Guclendirilmis biyolojik denetim modeli: `{_pretty_report_model_label('natural_auc_priority')}` | ROC AUC `{float(augmented_row['roc_auc']):.3f}` | AP `{float(augmented_row['average_precision']):.3f}`."
        )
    if not knownness_robust.empty:
        lines.append(
            f"- Knownness-robust biyolojik denetim modeli: `{_pretty_report_model_label('knownness_robust_priority')}` | ROC AUC `{float(knownness_robust_row['roc_auc']):.3f}` | AP `{float(knownness_robust_row['average_precision']):.3f}`."
        )
    if not support_calibrated.empty:
        lines.append(
            f"- Support-calibrated biyolojik model: `{_pretty_report_model_label('support_calibrated_priority')}` | ROC AUC `{float(support_calibrated_row['roc_auc']):.3f}` | AP `{float(support_calibrated_row['average_precision']):.3f}`."
        )
    if not support_synergy.empty:
        lines.append(
            f"- Support-synergy biyolojik model: `{_pretty_report_model_label('support_synergy_priority')}` | ROC AUC `{float(support_synergy_row['roc_auc']):.3f}` | AP `{float(support_synergy_row['average_precision']):.3f}`."
        )
    if not host_transfer_synergy.empty:
        lines.append(
            f"- Hata-odakli host-transfer sinerji modeli: `{_pretty_report_model_label('host_transfer_synergy_priority')}` | ROC AUC `{float(host_transfer_synergy_row['roc_auc']):.3f}` | AP `{float(host_transfer_synergy_row['average_precision']):.3f}`."
        )
    if not threat_architecture.empty:
        lines.append(
            f"- Tehdit-mimari denetim modeli: `{_pretty_report_model_label('threat_architecture_priority')}` | ROC AUC `{float(threat_architecture_row['roc_auc']):.3f}` | AP `{float(threat_architecture_row['average_precision']):.3f}`."
        )
    if not phylogeny_aware.empty:
        lines.append(
            f"- Taksonomi-duyarli H denetim modeli: `{_pretty_report_model_label('phylogeny_aware_priority')}` | ROC AUC `{float(phylogeny_aware_row['roc_auc']):.3f}` | AP `{float(phylogeny_aware_row['average_precision']):.3f}`."
        )
    if not structured_signal.empty:
        lines.append(
            f"- Yapisal sinyal denetim modeli: `{_pretty_report_model_label('structured_signal_priority')}` | ROC AUC `{float(structured_signal_row['roc_auc']):.3f}` | AP `{float(structured_signal_row['average_precision']):.3f}`."
        )
    if primary_top10 is not None:
        lines.append(
            f"- `{_pretty_report_model_label(primary_model_name)}` icin top-10 kesinlik `{primary_top10['precision_at_k']:.3f}`; duyarlilik `{primary_top10['recall_at_k']:.3f}`."
        )
    if strongest_top10 is not None and str(strongest["model_name"]) != primary_model_name:
        lines.append(
            f"- `{_pretty_report_model_label(str(strongest['model_name']))}` icin top-10 kesinlik `{strongest_top10['precision_at_k']:.3f}`; duyarlilik `{strongest_top10['recall_at_k']:.3f}`."
        )
    if conservative_top10 is not None and baseline_top10 is not None:
        lines.append(
            f"- Top-10 kesinlik karsilastirmasi: koruyucu model `{conservative_top10['precision_at_k']:.3f}` vs yalniz-sayim referans modeli `{baseline_top10['precision_at_k']:.3f}`."
        )
    if primary_top25 is not None:
        lines.append(
            f"- Top-25 daha gercekci karar kesitidir: `{_pretty_report_model_label(primary_model_name)}` icin kesinlik `{primary_top25['precision_at_k']:.3f}` ve duyarlilik `{primary_top25['recall_at_k']:.3f}`."
        )
    if pd.notna(primary_vs_strongest_overlap):
        overlap_text = f"- Yayinlanan ana model ile en yuksek metrikli modelin top-10 ortusmesi: `{int(primary_vs_strongest_overlap)}/10` aday"
        if pd.notna(primary_vs_strongest_overlap_25):
            overlap_text += f"; top-25 ortusmesi `{int(primary_vs_strongest_overlap_25)}/25`"
        if pd.notna(primary_vs_strongest_overlap_50):
            overlap_text += f"; top-50 ortusmesi `{int(primary_vs_strongest_overlap_50)}/50`"
        overlap_text += "."
        lines.append(overlap_text)
    if pd.notna(top_k_lower_half_count):
        lines.append(
            f"- Yayinlanan ana modelin top-25 listesinde dusuk-bilinirlik yarimindan gelen aday sayisi `{int(top_k_lower_half_count)}`'dir. Bu nedenle erken-sinyal adaylari ana kisa listeden ayri yorumlanmalidir."
        )
    if pd.notna(lowest_knownness_auc):
        lines.append(
            f"- En zor alt grup olan en dusuk bilinirlik ceyregi icin ana model ROC AUC'si `{float(lowest_knownness_auc):.3f}`; bu bolge ana modelin genel performansindan belirgin olarak daha zordur."
        )
    elif not q1_supported:
        lines.append(
            "- En dusuk bilinirlik ceyregi audit'i bu veri donumunde ayri ve tie-safe bir cohort uretmedigi icin headline savunmaya dahil edilmez; dusuk-bilinirlik stresi lower-half cohort ile raporlanir."
        )
    if not specialist_q1.empty:
        specialist_name = str(specialist_q1.iloc[0].get("model_name", "novelty_specialist_priority"))
        lines.append(
            f"- En guclu dusuk-bilinirlik uzman denetim modeli (`{specialist_name}`) en dusuk bilinirlik ceyreginde ROC AUC `{float(specialist_q1.iloc[0]['roc_auc']):.3f}` verir; novelty watchlist bu nedenle uzman erken-sinyal skoruyla birlikte okunur."
        )
    if not adaptive_row.empty:
        lines.append(
            f"- Knownness-gated audit modeli (`adaptive_natural_priority`) ust yari icin `natural_auc_priority`, alt yari icin ise dusuk-bilinirlik uzman skorunu OOF temelli olarak kullanarak genel ROC AUC `{float(adaptive_row.iloc[0]['roc_auc']):.3f}` ve AP `{float(adaptive_row.iloc[0]['average_precision']):.3f}` uretir."
        )
    if not adaptive_best_row.empty:
        best_adaptive = adaptive_best_row.iloc[0]
        base_model = str(best_adaptive.get("base_model_name", "knownness_robust_priority"))
        specialist_weight = float(best_adaptive.get("specialist_weight_lower_half", 1.0))
        lines.append(
            f"- En guclu knownness-gated audit modeli (`{best_adaptive['model_name']}`) `{_pretty_report_model_label(base_model)}` tabanini kullanir; alt bilinirlik yariminda uzman novelty skoruna `{specialist_weight:.2f}` agirlik vererek genel ROC AUC `{float(best_adaptive['roc_auc']):.3f}` ve AP `{float(best_adaptive['average_precision']):.3f}` uretir."
        )
        if not adaptive_best_gate_row.empty:
            gate_row = adaptive_best_gate_row.iloc[0]
            lines.append(
                f"- Gate consistency audit'i: `{best_adaptive['model_name']}` icin aktif kapinin en yakin `{int(gate_row['n_near_gate'])}` omurgasinda rota degisimi altinda ortalama |Δskor| `{float(gate_row['mean_abs_route_delta_near_gate']):.3f}`, p90 |Δskor| `{float(gate_row['p90_abs_route_delta_near_gate']):.3f}` ve rota-spearman `{float(gate_row['route_spearman_near_gate']):.3f}` bulundu; bu modelin gate tier'i `{str(gate_row.get('gate_consistency_tier', 'bilinmiyor'))}` olarak raporlandi."
            )
    if not adaptive_preferred_row.empty:
        preferred_adaptive = adaptive_preferred_row.iloc[0]
        if adaptive_best_row.empty or str(preferred_adaptive["model_name"]) != str(adaptive_best_row.iloc[0]["model_name"]):
            lines.append(
                f"- Kapida daha kararlı tercih edilen adaptive audit modeli `{preferred_adaptive['model_name']}` oldu; bu model `stable` gate consistency tier'ini korurken ROC AUC `{float(preferred_adaptive['roc_auc']):.3f}` ve AP `{float(preferred_adaptive['average_precision']):.3f}` uretir."
            )
    if pd.notna(source_balance_mean_auc):
        lines.append(
            f"- Kaynak-dengeli tekrarlar ortalama ROC AUC `{source_balance_mean_auc:.3f}` (sd `{source_balance_std_auc:.3f}`) verir; bu, veri kaynagi kompozisyonunun etkisinin tamamen yok olmadigini gosterir."
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Eslesik knownness/source strata audit'inde ana model agirlikli ROC AUC `{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}`, yalniz-sayim baseline ise `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}` verir."
        )
    if not macro_jump_row.empty:
        lines.append(
            f"- Ikincil outcome olarak yeni makro-bolge sicrama audit'inde en guclu model ROC AUC `{float(macro_jump_row.iloc[0]['roc_auc']):.3f}` uretir; bu, sinyalin yalnizca ulke sayisina bagli olmadigini destekler."
        )
    if not weighted_country_row.empty:
        lines.append(
            f"- Weighted yeni-ulke burden audit'inde en iyi modelin Spearman korelasyonu `{float(weighted_country_row.iloc[0]['spearman_corr']):.3f}`'tur."
        )
    if not count_outcome_row.empty:
        lines.append(
            f"- Binary esitli outcome'a ek olarak ham yeni-ulke sayisi icin en iyi count-alignment audit modeli Spearman `{float(count_outcome_row.iloc[0]['spearman_corr']):.3f}` verir."
        )
    if pd.notna(mean_metadata_quality):
        lines.append(
            f"- Backbone metadata quality ortalamasi `{mean_metadata_quality:.3f}` olarak ayri raporlanir; veri kalitesi dusuk adaylar false-negative ve risk audit'lerinde ayrica isaretlenir."
        )
    lines.extend(
        [
            "",
            "## Nasil Yorumlanmali",
            "",
            "- Outcome gercek biyolojik yayilimin birebir olcumu degil; daha cok sonraki donemde yeni ulke gorunurlugu artisidir.",
            f"- Ana outcome esigi: 2015 sonrasi en az `{int(outcome_threshold)}` yeni ulke. `candidate_threshold_flip.tsv` bu etiketin esige ne kadar hassas oldugunu gosterir.",
            "- Bu sistem tum pozitif backbone'lari yakalayan tam bir tarama araci degil; sinirli bir aday listesini onceliklendiren retrospektif bir kisa-liste aracidir.",
            "- Ana model secimi yalniz tek bir metrikten degil; genel ayiricilik, low-knownness davranisi, matched audit ve kaynak-robustlugu birlikte okunarak yapilir.",
            f"- Model secim gerekcesi: {localized_selection_rationale if selection_rationale else 'daha sade ve savunulabilir benchmark tercih edildi; daha guclu fakat daha proxy-agir modeller denetim baglaminda saklandi.'}",
            "- Gozlenen host cesitliligi ekseni dogrudan biyolojik host range olarak okunmamalidir; bu eksen kismen ornekleme doygunlugu ve bilinirlik sinyali tasir.",
            "- Guclendirilmis biyolojik denetim modelinde dis host-range sinyali, backbone safligi, atama guveni ve replikon mimarisi ek audit ozellikleri olarak ayri raporlanir; bunlar headline benchmark yerine biyolojik sinyali sikilastirma amaciyla kullanilir.",
            "- `adaptive_*` aileleri etiket veya zaman ayrimini degistirmez; yalnizca pre-2015 knownness sinyali ile dusuk-bilinirlik omurgalarda uzman novelty skorunu switch veya blend seklinde kullanir. Bu nedenle routing audit'idir, yeni bir headline benchmark degildir.",
            "- En yuksek dropout etkisinin `T_eff_norm` uzerinde olmasi, host cesitliligiyle iliskili terimlerin yonlu katsayi tasimasiyla celismez; ablation etkisi ile katsayi yorumu ayni sey degildir.",
            "- Okuma sirasi: `candidate_portfolio.tsv` -> `candidate_evidence_matrix.tsv` -> `candidate_threshold_flip.tsv`.",
            "- `novelty_watchlist` ana shortlist ile ayni sey degildir; yalniz-sayim referans modelini gecen dusuk-bilinirlik erken-sinyal adaylarini toplar.",
            "- Ana rapor dili yalnizca mevcut primary benchmark ve koruyucu benchmark icin kullanilir; diger model aileleri kesifsel denetim olarak raporlanir.",
            "- Backbone bootstrap araliklari outcome birimi olan backbone duzeyinde hesaplanir; ek group-bootstrap bu granulerlikte ayni birimi yeniden ornekler.",
            "- Spatial genelleme artik `dominant_region_train` uzerinden strict holdout olarak ayri tabloda denetlenir; bu analiz, zaman ayirimina ek ikinci bir OOD kontroludur.",
            "- Firsat yanliligi tam olarak sifirlanmis degildir: erken yillarda gozlenen backbone'larin daha uzun takip suresi vardir. Bu durum sinirlandirma olarak acikca kabul edilmelidir.",
            "- Outcome-eligibility kasitli olarak yalnizca egitimde 1-3 ulke gorunurlugune sahip backbone'lari hedefler; sistem tum backbone evreni icin degil, erken-donem izleme kisa listesi icin optimize edilmis bir aractir.",
            "- Ulke metadata kalitesi ayri `country_quality_summary.tsv` tablosunda raporlanir; eksik ulke kayitlari yayilim zayifligi gibi yorumlanmamalidir.",
            "- Permutasyon audit'leri iki ayri soruya cevap verir: ana null audit'i headline sinyalin sans ustu olup olmadigini, model-karsilastirma permutasyonlari ise modeller arasi farkin ne kadar tesadufi olabilecegini test eder.",
            "- Etik cerceve: yalnizca halka acik genom ve metadata kullanilir; bireysel hasta kimligi cikarimi veya klinik tani uretilmez.",
            "",
            "## Zero-Floor Bilesen Davranisi",
            "",
            "- T, H veya A eksenlerinden birinin ham degeri sifir ise ilgili normalize bilesen de 0.0 olarak kalir. Bu nedenle aritmetik `priority_index`, bazi backbone'larda fiilen iki aktif eksenin ortalamasi gibi davranabilir; bu bir bug degil, eksik biyolojik kanitin bilerek sifir puanlanmasidir.",
            "",
            "## OLS Residual Yaklasimi",
            "",
            "- `H_support_norm_residual`, gorunurluk ve bilinirlik proxy'lerine karsi OLS artiklasiyla hesaplanir. Amaç robust bir nedensellik modeli kurmak degil, destek sinyalinin sayim-proxy'lerinden arindirilmis deterministik bir audit ekseni elde etmektir.",
            "",
            "## Ornek Adaylar",
            "",
        ]
    )
    for row in selected_briefs.itertuples(index=False):
        lines.append(f"- {row.candidate_summary_tr}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_amrfinder_reportable(coverage: pd.DataFrame) -> bool:
    if coverage.empty or "priority_group" not in coverage.columns:
        return False
    groups = coverage.loc[coverage["priority_group"].isin(["high", "low"])].copy()
    if len(groups) < 2:
        return False
    if int(groups["n_sequences"].sum()) < 30:
        return False
    evaluable = groups["nonempty_concordance_evaluable_fraction"].fillna(0.0)
    return bool((groups["n_sequences"].fillna(0).astype(int) >= 10).all() and (evaluable >= 0.5).all())


def _write_jury_brief(
    output_path: Path,
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
    outcome_threshold: int,
) -> None:
    knownness_matched_validation = knownness_matched_validation if knownness_matched_validation is not None else pd.DataFrame()
    gate_consistency_audit = gate_consistency_audit if gate_consistency_audit is not None else pd.DataFrame()
    model_selection_scorecard = model_selection_scorecard if model_selection_scorecard is not None else pd.DataFrame()
    secondary_outcome_performance = secondary_outcome_performance if secondary_outcome_performance is not None else pd.DataFrame()
    weighted_country_outcome = weighted_country_outcome if weighted_country_outcome is not None else pd.DataFrame()
    count_outcome_audit = count_outcome_audit if count_outcome_audit is not None else pd.DataFrame()
    metadata_quality_summary = metadata_quality_summary if metadata_quality_summary is not None else pd.DataFrame()
    primary = model_metrics.loc[model_metrics["model_name"] == primary_model_name].iloc[0]
    conservative = model_metrics.loc[model_metrics["model_name"] == conservative_model_name].iloc[0]
    baseline = model_metrics.loc[model_metrics["model_name"] == "baseline_both"].iloc[0]
    source = model_metrics.loc[model_metrics["model_name"] == "source_only"].iloc[0]
    strongest_overall = model_metrics.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0]
    augmented = model_metrics.loc[model_metrics["model_name"] == "natural_auc_priority"].copy()
    augmented_row = augmented.iloc[0] if not augmented.empty else pd.Series(dtype=object)
    knownness_robust = model_metrics.loc[model_metrics["model_name"] == "knownness_robust_priority"].copy()
    knownness_robust_row = knownness_robust.iloc[0] if not knownness_robust.empty else pd.Series(dtype=object)
    support_calibrated = model_metrics.loc[model_metrics["model_name"] == "support_calibrated_priority"].copy()
    support_calibrated_row = support_calibrated.iloc[0] if not support_calibrated.empty else pd.Series(dtype=object)
    support_synergy = model_metrics.loc[model_metrics["model_name"] == "support_synergy_priority"].copy()
    support_synergy_row = support_synergy.iloc[0] if not support_synergy.empty else pd.Series(dtype=object)
    host_transfer_synergy = model_metrics.loc[model_metrics["model_name"] == "host_transfer_synergy_priority"].copy()
    host_transfer_synergy_row = host_transfer_synergy.iloc[0] if not host_transfer_synergy.empty else pd.Series(dtype=object)
    threat_architecture = model_metrics.loc[model_metrics["model_name"] == "threat_architecture_priority"].copy()
    threat_architecture_row = threat_architecture.iloc[0] if not threat_architecture.empty else pd.Series(dtype=object)
    phylogeny_aware = model_metrics.loc[model_metrics["model_name"] == "phylogeny_aware_priority"].copy()
    phylogeny_aware_row = phylogeny_aware.iloc[0] if not phylogeny_aware.empty else pd.Series(dtype=object)
    structured_signal = model_metrics.loc[model_metrics["model_name"] == "structured_signal_priority"].copy()
    structured_signal_row = structured_signal.iloc[0] if not structured_signal.empty else pd.Series(dtype=object)
    top_drop = dropout_table.loc[dropout_table["feature_name"] != "__full_model__"].sort_values(
        "roc_auc_drop_vs_full",
        ascending=False,
    ).iloc[0]
    primary_top10 = _lookup_decision_yield(decision_yield, primary_model_name, 10)
    primary_top25 = _lookup_decision_yield(decision_yield, primary_model_name, 25)
    conservative_top10 = _lookup_decision_yield(decision_yield, conservative_model_name, 10)
    baseline_top10 = _lookup_decision_yield(decision_yield, "baseline_both", 10)
    established_count = int((candidate_portfolio.get("portfolio_track", pd.Series(dtype=str)) == "established_high_risk").sum()) if not candidate_portfolio.empty else 0
    novel_count = int((candidate_portfolio.get("portfolio_track", pd.Series(dtype=str)) == "novel_signal").sum()) if not candidate_portfolio.empty else 0
    selection_row = model_selection_summary.iloc[0] if not model_selection_summary.empty else pd.Series(dtype=object)
    scorecard_primary_row = model_selection_scorecard.loc[
        model_selection_scorecard.get("model_name", pd.Series(dtype=str)).astype(str).eq(primary_model_name)
    ].head(1) if not model_selection_scorecard.empty else pd.DataFrame()
    scorecard_best_row = model_selection_scorecard.head(1) if not model_selection_scorecard.empty else pd.DataFrame()
    selection_rationale = str(selection_row.get("selection_rationale", "")).strip()
    strongest_top10 = _lookup_decision_yield(decision_yield, str(strongest_overall["model_name"]), 10)
    primary_vs_strongest_overlap = selection_row.get("primary_vs_strongest_top_10_overlap_count", np.nan)
    primary_vs_strongest_overlap_25 = selection_row.get("primary_vs_strongest_top_25_overlap_count", np.nan)
    primary_vs_strongest_overlap_50 = selection_row.get("primary_vs_strongest_top_50_overlap_count", np.nan)
    knownness_row = knownness_summary.iloc[0] if not knownness_summary.empty else pd.Series(dtype=object)
    lowest_knownness_auc = knownness_row.get("lowest_knownness_quartile_primary_roc_auc", np.nan)
    q1_supported = bool(knownness_row.get("lowest_knownness_quartile_supported", False))
    top_k_lower_half_count = knownness_row.get("top_k_lower_half_knownness_count", np.nan)
    source_balance_mean_auc = float(source_balance_resampling["roc_auc"].mean()) if not source_balance_resampling.empty else np.nan
    specialist_q1 = novelty_specialist_metrics.loc[
        (novelty_specialist_metrics["cohort_name"] == "lowest_knownness_quartile")
        & (novelty_specialist_metrics["status"] == "ok")
    ].sort_values("roc_auc", ascending=False).head(1).copy() if not novelty_specialist_metrics.empty else pd.DataFrame()
    adaptive_row = adaptive_gated_metrics.loc[
        (adaptive_gated_metrics["model_name"] == "adaptive_natural_priority")
        & (adaptive_gated_metrics["status"] == "ok")
    ].copy() if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_knownness_robust_row = adaptive_gated_metrics.loc[
        (adaptive_gated_metrics["model_name"] == "adaptive_knownness_robust_priority")
        & (adaptive_gated_metrics["status"] == "ok")
    ].copy() if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_best_row = adaptive_gated_metrics.loc[
        adaptive_gated_metrics["status"] == "ok"
    ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1) if not adaptive_gated_metrics.empty else pd.DataFrame()
    adaptive_preferred_row = pd.DataFrame()
    if not adaptive_gated_metrics.empty and not gate_consistency_audit.empty:
        stable_models = gate_consistency_audit.loc[
            gate_consistency_audit["gate_consistency_tier"] == "stable",
            "model_name",
        ].drop_duplicates()
        adaptive_preferred_row = adaptive_gated_metrics.loc[
            (adaptive_gated_metrics["status"] == "ok")
            & adaptive_gated_metrics["model_name"].isin(stable_models)
        ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1)
    adaptive_best_gate_row = pd.DataFrame()
    if not adaptive_best_row.empty and not gate_consistency_audit.empty:
        adaptive_best_gate_row = gate_consistency_audit.loc[
            gate_consistency_audit["model_name"] == str(adaptive_best_row.iloc[0]["model_name"])
        ].sort_values("mean_abs_route_delta_near_gate", ascending=True).head(1)
    matched_primary = knownness_matched_validation.loc[
        (knownness_matched_validation["matched_stratum"] == "__weighted_overall__")
        & (knownness_matched_validation["model_name"] == primary_model_name)
    ].copy() if not knownness_matched_validation.empty else pd.DataFrame()
    matched_baseline = knownness_matched_validation.loc[
        (knownness_matched_validation["matched_stratum"] == "__weighted_overall__")
        & (knownness_matched_validation["model_name"] == "baseline_both")
    ].copy() if not knownness_matched_validation.empty else pd.DataFrame()
    macro_jump_row = secondary_outcome_performance.loc[
        (secondary_outcome_performance["outcome_name"] == "macro_region_jump_label")
        & (secondary_outcome_performance["status"] == "ok")
    ].sort_values(["roc_auc", "average_precision"], ascending=False).head(1) if not secondary_outcome_performance.empty else pd.DataFrame()
    weighted_country_row = weighted_country_outcome.loc[
        weighted_country_outcome["status"] == "ok"
    ].sort_values("spearman_corr", ascending=False).head(1) if not weighted_country_outcome.empty else pd.DataFrame()
    count_outcome_row = count_outcome_audit.loc[
        count_outcome_audit["status"] == "ok"
    ].sort_values("spearman_corr", ascending=False).head(1) if not count_outcome_audit.empty else pd.DataFrame()
    mean_metadata_quality = float(metadata_quality_summary["metadata_quality_score"].mean()) if not metadata_quality_summary.empty and "metadata_quality_score" in metadata_quality_summary.columns else np.nan

    lines = [
        "# Jury Brief",
        "",
        "## Core Claim",
        "",
        "This framework retrospectively prioritizes plasmid backbone surveillance units using training-period genomic and ecological features, then asks whether those same backbones later show new-country visibility increase.",
        "",
        "## Formal Hypotheses",
        "",
        "- **H0 (null)**: A <=2015 T/H/A-derived priority signal has no discriminative association with post-2015 multi-country visibility expansion (ROC AUC = 0.50).",
        "- **H1 (alternative)**: The same priority signal is positively associated with post-2015 multi-country visibility expansion (ROC AUC > 0.50).",
        "- **Significance criterion**: empirical permutation p-value < 0.01, using the current primary model's feature set and the same L2 / weighting configuration as the headline evaluation.",
        "",
        "## Current Benchmark and Audit Context",
        "",
        f"- Current primary benchmark: `{_pretty_report_model_label(primary_model_name)}` with ROC AUC `{primary['roc_auc']:.3f}` and AP `{primary['average_precision']:.3f}`.",
        f"- Strongest audited metric model: `{_pretty_report_model_label(str(strongest_overall['model_name']))}` with ROC AUC `{strongest_overall['roc_auc']:.3f}` and AP `{strongest_overall['average_precision']:.3f}`.",
        f"- Conservative benchmark: `{_pretty_report_model_label(conservative_model_name)}` with ROC AUC `{conservative['roc_auc']:.3f}` and AP `{conservative['average_precision']:.3f}`.",
        f"- Counts-only baseline: `baseline_both` with ROC AUC `{baseline['roc_auc']:.3f}` and AP `{baseline['average_precision']:.3f}`.",
        f"- Source-only control: `source_only` with ROC AUC `{source['roc_auc']:.3f}`.",
        f"- Primary-model selection rationale: {selection_rationale or 'the current headline model is intentionally chosen for benchmark clarity rather than silent metric chasing.'}",
        "",
        "## Decision-Support Readout",
        "",
        f"- Reviewer shortlist size: `{established_count}` established high-risk + `{novel_count}` novel-signal candidates in `candidate_portfolio.tsv`.",
        "- Read candidate outputs in this order: `candidate_portfolio.tsv` -> `candidate_evidence_matrix.tsv` -> `candidate_threshold_flip.tsv`.",
        f"- Main outcome threshold: `{int(outcome_threshold)}` later new countries; use `candidate_threshold_flip.tsv` to see which candidates are definition-sensitive.",
    ]
    if not scorecard_primary_row.empty:
        primary_scorecard = scorecard_primary_row.iloc[0]
        best_scorecard = scorecard_best_row.iloc[0] if not scorecard_best_row.empty else primary_scorecard
        scorecard_metric_text = (
            "overall AUC, AP, lower-half/q1 knownness, matched-knownness, source holdout, and a knownness-dependence penalty"
            if q1_supported
            else "overall AUC, AP, lower-half knownness, matched-knownness, source holdout, and a knownness-dependence penalty"
        )
        lines.append(
            f"- In the multi-objective selection scorecard, the primary model ranks `{int(primary_scorecard['selection_rank'])}/{int(len(model_selection_scorecard))}` after combining {scorecard_metric_text}."
        )
        if str(best_scorecard.get("model_name", primary_model_name)) != primary_model_name:
            lines.append(
                f"- The scorecard leader is `{best_scorecard['model_name']}`, but the headline benchmark remains `{primary_model_name}`; the explicit selection rationale above governs that choice."
            )
    if not augmented.empty:
        lines.append(
            f"- Augmented biological audit model: `natural_auc_priority` with ROC AUC `{float(augmented_row['roc_auc']):.3f}` and AP `{float(augmented_row['average_precision']):.3f}`; this model adds external host-range, backbone purity, assignment confidence, mash-based novelty, and replicon architecture without changing the current headline benchmark."
        )
    if not knownness_robust.empty:
        lines.append(
            f"- Knownness-robust biological audit model: `knownness_robust_priority` with ROC AUC `{float(knownness_robust_row['roc_auc']):.3f}` and AP `{float(knownness_robust_row['average_precision']):.3f}`; this variant keeps the biological core but replaces external host-range with recurrent AMR structure, pMLST coherence, and eco-clinical context under class+knownness balancing."
        )
    if not support_calibrated.empty:
        lines.append(
            f"- Support-calibrated biological model: `support_calibrated_priority` with ROC AUC `{float(support_calibrated_row['roc_auc']):.3f}` and AP `{float(support_calibrated_row['average_precision']):.3f}`; this variant keeps the knownness-robust biological core but makes annotation support explicit through host-range support, pMLST presence, and AMR support depth."
        )
    if not support_synergy.empty:
        lines.append(
            f"- Support-synergy biological model: `support_synergy_priority` with ROC AUC `{float(support_synergy_row['roc_auc']):.3f}` and AP `{float(support_synergy_row['average_precision']):.3f}`; this variant keeps the support-calibrated core but adds metadata support depth, external host-range magnitude, and host-range x transfer synergy to recover sparse-support errors without adding count proxies."
        )
    if not host_transfer_synergy.empty:
        lines.append(
            f"- Error-focused host-transfer synergy model: `host_transfer_synergy_priority` with ROC AUC `{float(host_transfer_synergy_row['roc_auc']):.3f}` and AP `{float(host_transfer_synergy_row['average_precision']):.3f}`; this variant adds explicit host-range x transfer coupling to recover sparse-backbone mistakes without introducing direct knownness counts."
        )
    if not threat_architecture.empty:
        lines.append(
            f"- Threat-architecture audit model: `threat_architecture_priority` with ROC AUC `{float(threat_architecture_row['roc_auc']):.3f}` and AP `{float(threat_architecture_row['average_precision']):.3f}`; this variant keeps the host-transfer coupling but adds AMR clinical-threat burden plus replicon multiplicity to recover sparse-backbone misses with biologically interpretable structure."
        )
    if not phylogeny_aware.empty:
        lines.append(
            f"- Taxonomy-aware H audit model: `phylogeny_aware_priority` with ROC AUC `{float(phylogeny_aware_row['roc_auc']):.3f}` and AP `{float(phylogeny_aware_row['average_precision']):.3f}`; this variant preserves the augmented biological core but swaps the H axis for a lineage-aware host specialization signal."
        )
    if not structured_signal.empty:
        lines.append(
            f"- Structure-aware biological audit model: `structured_signal_priority` with ROC AUC `{float(structured_signal_row['roc_auc']):.3f}` and AP `{float(structured_signal_row['average_precision']):.3f}`; this variant keeps the taxonomy-aware H axis and adds host evenness plus recurrent AMR structure."
        )
    if primary_top10 is not None:
        lines.append(
            f"- Current primary top-10 yield: precision `{primary_top10['precision_at_k']:.3f}`, recall `{primary_top10['recall_at_k']:.3f}`."
        )
    if strongest_top10 is not None and str(strongest_overall["model_name"]) != primary_model_name:
        lines.append(
            f"- Strongest-metric top-10 yield: precision `{strongest_top10['precision_at_k']:.3f}`, recall `{strongest_top10['recall_at_k']:.3f}`."
        )
    if conservative_top10 is not None:
        lines.append(
            f"- Conservative top-10 yield: precision `{conservative_top10['precision_at_k']:.3f}`, recall `{conservative_top10['recall_at_k']:.3f}`."
        )
    if baseline_top10 is not None:
        lines.append(
            f"- Counts-only baseline top-10 yield: precision `{baseline_top10['precision_at_k']:.3f}`, recall `{baseline_top10['recall_at_k']:.3f}`."
        )
    if pd.notna(primary_vs_strongest_overlap):
        overlap_text = f"- Current-primary vs strongest-metric top-10 overlap: `{int(primary_vs_strongest_overlap)}/10` candidates"
        if pd.notna(primary_vs_strongest_overlap_25):
            overlap_text += f"; top-25 overlap: `{int(primary_vs_strongest_overlap_25)}/25`"
        if pd.notna(primary_vs_strongest_overlap_50):
            overlap_text += f"; top-50 overlap: `{int(primary_vs_strongest_overlap_50)}/50`"
        overlap_text += "."
        lines.append(overlap_text)
    if primary_top25 is not None:
        lines.append(
            f"- Top-25 is the more decision-relevant cut: current primary precision `{primary_top25['precision_at_k']:.3f}`, recall `{primary_top25['recall_at_k']:.3f}`."
        )
    if pd.notna(top_k_lower_half_count):
        lines.append(
            f"- Published-primary top-25 contains `{int(top_k_lower_half_count)}` lower-knownness candidates, so the `novel_signal` track should be read as a separate exploratory watchlist rather than as the same shortlist."
        )
    if pd.notna(lowest_knownness_auc):
        lines.append(
            f"- Lowest-knownness quartile performance is weaker (primary ROC AUC `{float(lowest_knownness_auc):.3f}`), so early-signal claims should stay conservative."
        )
    elif not q1_supported:
        lines.append(
            "- A distinct tie-safe lowest-knownness quartile could not be estimated in this data refresh, so the harshest low-visibility stress test is reported with the lower-half cohort rather than an artificial q1 split."
        )
    if not specialist_q1.empty:
        specialist_name = str(specialist_q1.iloc[0].get("model_name", "novelty_specialist_priority"))
        lines.append(
            f"- The strongest low-knownness specialist audit (`{specialist_name}`) reaches ROC AUC `{float(specialist_q1.iloc[0]['roc_auc']):.3f}` in the lowest-knownness quartile; it refines the exploratory novelty watchlist rather than replacing the headline benchmark."
        )
    if not adaptive_row.empty:
        lines.append(
            f"- A knownness-gated audit model (`adaptive_natural_priority`) keeps `natural_auc_priority` in the upper-knownness half and uses leakage-free OOF base plus OOF specialist scoring in the lower-knownness half, reaching ROC AUC `{float(adaptive_row.iloc[0]['roc_auc']):.3f}` with AP `{float(adaptive_row.iloc[0]['average_precision']):.3f}`."
        )
    if not adaptive_best_row.empty:
        best_adaptive = adaptive_best_row.iloc[0]
        base_model = str(best_adaptive.get("base_model_name", "knownness_robust_priority"))
        specialist_weight = float(best_adaptive.get("specialist_weight_lower_half", 1.0))
        lines.append(
            f"- The strongest knownness-gated audit model (`{best_adaptive['model_name']}`) uses `{_pretty_report_model_label(base_model)}` as the base score and applies specialist weight `{specialist_weight:.2f}` within the lower-knownness half, reaching ROC AUC `{float(best_adaptive['roc_auc']):.3f}` with AP `{float(best_adaptive['average_precision']):.3f}`; this remains a routing audit rather than a replacement headline benchmark."
        )
        if not adaptive_best_gate_row.empty:
            gate_row = adaptive_best_gate_row.iloc[0]
            lines.append(
                f"- Gate consistency audit: for `{best_adaptive['model_name']}`, the `{int(gate_row['n_near_gate'])}` backbones closest to the active routing boundary showed mean |Δscore| `{float(gate_row['mean_abs_route_delta_near_gate']):.3f}`, p90 |Δscore| `{float(gate_row['p90_abs_route_delta_near_gate']):.3f}`, and route Spearman `{float(gate_row['route_spearman_near_gate']):.3f}` under counterfactual route switching; this gate is currently tiered `{str(gate_row.get('gate_consistency_tier', 'unknown'))}`."
            )
    if not adaptive_preferred_row.empty:
        preferred_adaptive = adaptive_preferred_row.iloc[0]
        if adaptive_best_row.empty or str(preferred_adaptive["model_name"]) != str(adaptive_best_row.iloc[0]["model_name"]):
            lines.append(
                f"- The gate-stable preferred adaptive audit is `{preferred_adaptive['model_name']}`; it remains in the `stable` gate-consistency tier while delivering ROC AUC `{float(preferred_adaptive['roc_auc']):.3f}` with AP `{float(preferred_adaptive['average_precision']):.3f}`."
            )
    if pd.notna(source_balance_mean_auc):
        lines.append(
            f"- Source-balanced reruns average ROC AUC `{source_balance_mean_auc:.3f}`, so source composition remains a real robustness caveat rather than a fully neutral nuisance factor."
        )
    if not matched_primary.empty and not matched_baseline.empty:
        lines.append(
            f"- Within matched visibility/source strata, the current primary model still exceeds the counts-only baseline (`{float(matched_primary.iloc[0]['weighted_mean_roc_auc']):.3f}` vs `{float(matched_baseline.iloc[0]['weighted_mean_roc_auc']):.3f}` weighted ROC AUC)."
        )
    if not macro_jump_row.empty:
        lines.append(
            f"- Secondary outcome auditing now includes macro-region jumps; the strongest audited model reaches ROC AUC `{float(macro_jump_row.iloc[0]['roc_auc']):.3f}` on that harder geographic expansion endpoint."
        )
    if not weighted_country_row.empty:
        lines.append(
            f"- Ranking scores are also compared against upload-weighted new-country burden; the best audited Spearman correlation is `{float(weighted_country_row.iloc[0]['spearman_corr']):.3f}`."
        )
    if not count_outcome_row.empty:
        lines.append(
            f"- The same ranking family is now also audited against the raw later new-country count; the strongest Spearman alignment is `{float(count_outcome_row.iloc[0]['spearman_corr']):.3f}`."
        )
    if pd.notna(mean_metadata_quality):
        lines.append(
            f"- Backbone-level metadata quality is quantified directly (mean score `{mean_metadata_quality:.3f}`) and reused in the false-negative audit instead of being left as an informal caveat."
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- T, H, A, coherence, and mobility-support features are computed only from `resolved_year <= 2015` rows.",
            "- The outcome uses only later-period country visibility (`2016-2023`), not training-period feature inputs.",
            "- Supportive WHO/Pathogen Detection/CARD/MOB-suite layers are descriptive context only; AMRFinder is optional and not required for the headline benchmark or its main claims.",
            "- This is a shortlist-prioritization benchmark, not an exhaustive detection system for every later positive backbone.",
            "- Source-only performance is weak, but source-balanced reruns still matter, so source composition should be treated as a real robustness caveat.",
            "- Observed host-diversity terms should not be read as direct biological host range; in practice they partly behave like sampling saturation / knownness signals.",
            "- Additional biological audit tables now report external host-range support, backbone purity, assignment confidence, replicon architecture, and training-only mash novelty so AUC gains can be inspected mechanistically rather than treated as a black-box metric jump.",
            "- The `adaptive_*` family changes only how the pre-2015 low-knownness cohort receives the specialist score, using either a hard switch or a partial blend; it does not relax the temporal split or alter the outcome definition.",
            "- Primary-model choice is not driven by a single headline metric; overall discrimination, low-knownness behaviour, matched-strata behaviour, and source robustness are read together.",
            "- Candidate interpretation is explicitly multi-table: portfolio + evidence matrix + threshold flips + consensus/risk context, not a single raw score rank.",
            "- Only the current primary benchmark and the conservative benchmark should be treated as headline models; the rest of the model zoo is exploratory audit context, not multiple-comparison-free confirmatory evidence.",
            "- Bootstrap intervals resample the analysis unit itself (backbone rows), so the uncertainty intervals are already computed at backbone granularity rather than at raw-sequence granularity.",
            "- Spatial generalization is now audited separately via strict `dominant_region_train` holdouts, complementing the temporal split with an explicit out-of-distribution check.",
            "- Opportunity bias is not fully removable in a retrospective archive: backbones first seen earlier have longer time-at-risk for later new-country visibility. This remains a declared limitation.",
            "- Eligibility is intentionally restricted to backbones with 1-3 training countries; the system is meant for early-stage surveillance triage, not for all already-global backbones.",
            "- Country metadata completeness is reported separately in `country_quality_summary.tsv`; missing countries must not be over-interpreted as true geographic narrowness.",
            "- The project uses two permutation paradigms on purpose: the headline null asks whether the observed signal exceeds randomized labels, whereas model-comparison nulls ask whether one score family truly beats another.",
            "- Ethical scope: only public genome and country metadata are used; the framework does not infer patient identity and is not a clinical diagnostic tool.",
            "",
            "## Feature-Interpretation Note",
            "",
            f"- Highest dropout impact: `{top_drop['feature_name']}` with ROC AUC drop `{top_drop['roc_auc_drop_vs_full']:.3f}` when removed.",
            "- Highest ablation impact and strongest directional coefficient need not be the same signal; T-ablation and host-diversity interpretation answer different questions.",
            "",
            "## Boundary Conditions",
            "",
            "- This is a retrospective association framework, not a causal, mechanistic, or clinical prediction system.",
            "- The outcome is later visibility increase, not direct proof of transmission fitness or public-health impact.",
            "- Backbone definitions are operational surveillance units and should not be presented as biological truth claims.",
            "- The current primary benchmark is the official headline ranking, but adaptive and exploratory audits remain useful for stress-testing its behavior.",
            f"- Current scored backbone count: `{int(scored['backbone_id'].nunique())}`.",
            "",
            "## Zero-Floor Component Behavior",
            "",
            "- When a raw T/H/A component is genuinely zero, its normalized counterpart remains 0.0. The arithmetic headline score therefore behaves like an average across only the active evidence axes for many backbones; this is deliberate and should not be narrated as if every backbone carries three equally active signals.",
            "",
            "## OLS Residual Approach",
            "",
            "- `H_support_norm_residual` is computed with ordinary least squares against visibility/knownness proxies. The goal is deterministic proxy-subtraction for audit purposes, not a causal robust-regression claim.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    context = build_context(PROJECT_ROOT)
    scored_path = context.root / "data/scores/backbone_scored.tsv"
    backbones_path = context.root / "data/silver/plasmid_backbones.tsv"
    amr_consensus_path = context.root / "data/silver/plasmid_amr_consensus.tsv"
    metrics_path = context.root / "data/analysis/module_a_metrics.json"
    sensitivity_path = context.root / "data/analysis/sensitivity_summary.json"
    module_b_path = context.root / "data/analysis/module_b_amr_class_comparison.tsv"
    module_c_path = context.root / "data/analysis/pathogen_detection_support.tsv"
    module_c_group_path = context.root / "data/analysis/pathogen_detection_group_summary.tsv"
    module_c_clinical_path = context.root / "data/analysis/pathogen_detection_clinical_support.tsv"
    module_c_clinical_group_path = context.root / "data/analysis/pathogen_detection_clinical_group_summary.tsv"
    module_c_environmental_path = context.root / "data/analysis/pathogen_detection_environmental_support.tsv"
    module_c_environmental_group_path = context.root / "data/analysis/pathogen_detection_environmental_group_summary.tsv"
    module_c_strata_group_path = context.root / "data/analysis/pathogen_detection_strata_group_summary.tsv"
    who_detail_path = context.root / "data/analysis/who_mia_support.tsv"
    who_summary_path = context.root / "data/analysis/who_mia_group_summary.tsv"
    who_category_path = context.root / "data/analysis/who_mia_category_comparison.tsv"
    who_reference_path = context.root / "data/analysis/who_mia_reference_catalog.tsv"
    card_detail_path = context.root / "data/analysis/card_gene_support.tsv"
    card_summary_path = context.root / "data/analysis/card_group_summary.tsv"
    card_family_path = context.root / "data/analysis/card_gene_family_comparison.tsv"
    card_mechanism_path = context.root / "data/analysis/card_mechanism_comparison.tsv"
    mobsuite_detail_path = context.root / "data/analysis/mobsuite_host_range_support.tsv"
    mobsuite_summary_path = context.root / "data/analysis/mobsuite_host_range_group_summary.tsv"
    amrfinder_probe_panel_path = context.root / "data/analysis/amrfinder_probe_panel.tsv"
    amrfinder_probe_hits_path = context.root / "data/analysis/amrfinder_probe_hits.tsv"
    amrfinder_detail_path = context.root / "data/analysis/amrfinder_concordance_detail.tsv"
    amrfinder_summary_path = context.root / "data/analysis/amrfinder_concordance_summary.tsv"
    source_validation_path = context.root / "data/analysis/source_stratified_consistency.tsv"
    calibration_path = context.root / "data/analysis/calibration_table.tsv"
    family_summary_path = context.root / "data/analysis/model_family_summary.tsv"
    subgroup_performance_path = context.root / "data/analysis/model_subgroup_performance.tsv"
    comparison_path = context.root / "data/analysis/model_comparison_summary.tsv"
    calibration_metrics_path = context.root / "data/analysis/calibration_metrics.tsv"
    coefficient_path = context.root / "data/analysis/primary_model_coefficients.tsv"
    coefficient_stability_path = context.root / "data/analysis/primary_model_coefficient_stability.tsv"
    dropout_path = context.root / "data/analysis/feature_dropout_importance.tsv"
    source_balance_resampling_path = context.root / "data/analysis/source_balance_resampling.tsv"
    group_holdout_path = context.root / "data/analysis/group_holdout_performance.tsv"
    permutation_detail_path = context.root / "data/analysis/permutation_null_distribution.tsv"
    permutation_summary_path = context.root / "data/analysis/permutation_null_summary.tsv"
    negative_control_path = context.root / "data/analysis/negative_control_audit.tsv"
    logistic_impl_path = context.root / "data/analysis/logistic_implementation_audit.tsv"
    logistic_convergence_path = context.root / "data/analysis/logistic_convergence_audit.tsv"
    simplicity_path = context.root / "data/analysis/model_simplicity_summary.tsv"
    knownness_summary_path = context.root / "data/analysis/knownness_audit_summary.tsv"
    knownness_strata_path = context.root / "data/analysis/knownness_stratified_performance.tsv"
    country_quality_path = context.root / "data/analysis/country_quality_summary.tsv"
    purity_atlas_path = context.root / "data/analysis/backbone_purity_atlas.tsv"
    assignment_confidence_path = context.root / "data/analysis/assignment_confidence_summary.tsv"
    incremental_value_path = context.root / "data/analysis/incremental_value_over_baseline.tsv"
    novelty_specialist_metrics_path = context.root / "data/analysis/novelty_specialist_metrics.tsv"
    novelty_specialist_predictions_path = context.root / "data/analysis/novelty_specialist_predictions.tsv"
    adaptive_gated_metrics_path = context.root / "data/analysis/adaptive_gated_metrics.tsv"
    adaptive_gated_predictions_path = context.root / "data/analysis/adaptive_gated_predictions.tsv"
    gate_consistency_audit_path = context.root / "data/analysis/gate_consistency_audit.tsv"
    knownness_matched_validation_path = context.root / "data/analysis/knownness_matched_validation.tsv"
    nonlinear_deconfounding_path = context.root / "data/analysis/nonlinear_deconfounding_audit.tsv"
    country_upload_propensity_path = context.root / "data/analysis/country_upload_propensity.tsv"
    macro_region_jump_path = context.root / "data/analysis/macro_region_jump_outcome.tsv"
    secondary_outcome_performance_path = context.root / "data/analysis/secondary_outcome_performance.tsv"
    weighted_country_outcome_path = context.root / "data/analysis/weighted_country_outcome_audit.tsv"
    count_outcome_audit_path = context.root / "data/analysis/new_country_count_audit.tsv"
    metadata_quality_summary_path = context.root / "data/analysis/metadata_quality_summary.tsv"
    event_timing_outcomes_path = context.root / "data/analysis/event_timing_outcomes.tsv"
    exposure_adjusted_event_path = context.root / "data/analysis/exposure_adjusted_event_outcomes.tsv"
    exposure_adjusted_outcome_audit_path = context.root / "data/analysis/exposure_adjusted_outcome_audit.tsv"
    ordinal_outcome_audit_path = context.root / "data/analysis/ordinal_outcome_audit.tsv"
    country_missingness_bounds_path = context.root / "data/analysis/country_missingness_bounds.tsv"
    country_missingness_sensitivity_path = context.root / "data/analysis/country_missingness_sensitivity.tsv"
    geographic_jump_path = context.root / "data/analysis/geographic_jump_distance_outcome.tsv"
    duplicate_quality_path = context.root / "data/analysis/duplicate_completeness_change_audit.tsv"
    amr_uncertainty_path = context.root / "data/analysis/amr_uncertainty_summary.tsv"
    mash_graph_path = context.root / "data/analysis/mash_similarity_graph.tsv"
    counterfactual_shortlist_path = context.root / "data/analysis/counterfactual_shortlist_comparison.tsv"
    module_f_identity_path = context.root / "data/analysis/module_f_backbone_identity.tsv"
    module_f_enrichment_path = context.root / "data/analysis/module_f_enrichment.tsv"
    module_f_top_hits_path = context.root / "data/analysis/module_f_top_hits.tsv"
    rolling_temporal_path = context.root / "data/analysis/rolling_temporal_validation.tsv"
    rolling_assignment_diagnostic_path = context.root / "data/analysis/rolling_assignment_diagnostics.tsv"
    rank_stability_path = context.root / "data/analysis/candidate_rank_stability.tsv"
    variant_consistency_path = context.root / "data/analysis/candidate_variant_consistency.tsv"
    prospective_freeze_path = context.root / "data/analysis/prospective_candidate_freeze.tsv"
    annual_freeze_summary_path = context.root / "data/analysis/annual_candidate_freeze_summary.tsv"
    predictions_path = context.root / "data/analysis/module_a_predictions.tsv"
    analysis_dir = context.root / "data/analysis"

    class TableRouter:
        def __init__(self, core_dir, diag_dir, analysis_dir):
            self.core_dir = core_dir
            self.diag_dir = diag_dir
            self.analysis_dir = analysis_dir
            self.core_files = {
                "model_metrics.tsv",
                "model_selection_scorecard.tsv",
                "benchmark_protocol.tsv",
                "model_comparison_summary.tsv",
                "model_selection_summary.tsv",
                "decision_yield_summary.tsv",
                "candidate_evidence_matrix.tsv",
                "candidate_threshold_flip.tsv",
                "candidate_portfolio.tsv",
                "consensus_shortlist.tsv",
                "top_primary_candidates.tsv",
            }
            self.analysis_shadow_files = {
                path.name
                for path in analysis_dir.glob("*.tsv")
            }
        def __truediv__(self, name):
            if name in self.analysis_shadow_files:
                return self.analysis_dir / name
            if name in self.core_files:
                return self.core_dir / name
            return self.diag_dir / name

    core_dir = context.root / "reports/core_tables"
    diag_dir = context.root / "reports/diagnostic_tables"
    ensure_directory(core_dir)
    ensure_directory(diag_dir)
    final_tables_dir = TableRouter(core_dir, diag_dir, analysis_dir)
    
    figures_dir = context.root / "reports/_figure_router"
    jury_brief_path = context.root / "reports/jury_brief.md"
    turkish_summary_path = context.root / "reports/ozet_tr.md"
    stale_turkiye_context_path = diag_dir / "turkiye_candidate_context.tsv"

    with ManagedScriptRun(context, "24_build_reports") as run:
        for path in (
            scored_path,
            backbones_path,
            amr_consensus_path,
            metrics_path,
            sensitivity_path,
            module_b_path,
            module_c_path,
            module_c_group_path,
            module_c_clinical_path,
            module_c_clinical_group_path,
            module_c_environmental_path,
            module_c_environmental_group_path,
            module_c_strata_group_path,
            who_detail_path,
            who_summary_path,
            who_category_path,
            card_detail_path,
            card_summary_path,
            card_family_path,
            card_mechanism_path,
            mobsuite_detail_path,
            mobsuite_summary_path,
            amrfinder_probe_panel_path,
            amrfinder_probe_hits_path,
            amrfinder_detail_path,
            amrfinder_summary_path,
            source_validation_path,
            calibration_path,
            family_summary_path,
            subgroup_performance_path,
            comparison_path,
            calibration_metrics_path,
            coefficient_path,
            coefficient_stability_path,
            dropout_path,
            source_balance_resampling_path,
            group_holdout_path,
            permutation_detail_path,
            permutation_summary_path,
            negative_control_path,
            logistic_impl_path,
            logistic_convergence_path,
            simplicity_path,
            knownness_summary_path,
            knownness_strata_path,
            country_quality_path,
            purity_atlas_path,
            assignment_confidence_path,
            incremental_value_path,
            novelty_specialist_metrics_path,
            novelty_specialist_predictions_path,
            adaptive_gated_metrics_path,
            adaptive_gated_predictions_path,
            gate_consistency_audit_path,
            knownness_matched_validation_path,
            nonlinear_deconfounding_path,
            country_upload_propensity_path,
            macro_region_jump_path,
            secondary_outcome_performance_path,
            weighted_country_outcome_path,
            count_outcome_audit_path,
            metadata_quality_summary_path,
            event_timing_outcomes_path,
            exposure_adjusted_event_path,
            exposure_adjusted_outcome_audit_path,
            ordinal_outcome_audit_path,
            country_missingness_bounds_path,
            country_missingness_sensitivity_path,
            geographic_jump_path,
            duplicate_quality_path,
            amr_uncertainty_path,
            mash_graph_path,
            counterfactual_shortlist_path,
            module_f_identity_path,
            module_f_enrichment_path,
            module_f_top_hits_path,
            rolling_temporal_path,
            rolling_assignment_diagnostic_path,
            rank_stability_path,
            variant_consistency_path,
            prospective_freeze_path,
            annual_freeze_summary_path,
            predictions_path,
        ):
            if path.exists():
                run.record_input(path)
        run.record_output(jury_brief_path)
        run.record_output(turkish_summary_path)
        run.record_output(final_tables_dir / "top_primary_candidates.tsv")
        run.record_output(final_tables_dir / "consensus_shortlist.tsv")
        run.record_output(final_tables_dir / "candidate_evidence_matrix.tsv")
        run.record_output(final_tables_dir / "candidate_threshold_flip.tsv")
        run.record_output(final_tables_dir / "decision_yield_summary.tsv")
        run.record_output(final_tables_dir / "decision_budget_curve.tsv")
        run.record_output(final_tables_dir / "false_negative_audit.tsv")
        run.record_output(final_tables_dir / "model_selection_scorecard.tsv")
        run.record_output(final_tables_dir / "model_selection_summary.tsv")
        run.record_output(final_tables_dir / "benchmark_protocol.tsv")
        run.record_output(final_tables_dir / "module_b_amr_class_comparison.tsv")
        run.record_output(final_tables_dir / "model_metrics.tsv")
        run.record_output(final_tables_dir / "sensitivity_summary.tsv")
        run.record_output(final_tables_dir / "source_stratified_consistency.tsv")
        run.record_output(final_tables_dir / "calibration_table.tsv")
        run.record_output(final_tables_dir / "model_family_summary.tsv")
        run.record_output(final_tables_dir / "model_subgroup_performance.tsv")
        run.record_output(final_tables_dir / "model_comparison_summary.tsv")
        run.record_output(final_tables_dir / "calibration_metrics.tsv")
        run.record_output(final_tables_dir / "primary_model_coefficients.tsv")
        run.record_output(final_tables_dir / "primary_model_coefficient_stability.tsv")
        run.record_output(final_tables_dir / "feature_dropout_importance.tsv")
        run.record_output(final_tables_dir / "source_balance_resampling.tsv")
        run.record_output(final_tables_dir / "group_holdout_performance.tsv")
        run.record_output(final_tables_dir / "permutation_null_distribution.tsv")
        run.record_output(final_tables_dir / "permutation_null_summary.tsv")
        run.record_output(final_tables_dir / "rolling_temporal_validation.tsv")
        run.record_output(final_tables_dir / "rolling_assignment_diagnostics.tsv")
        run.record_output(final_tables_dir / "candidate_rank_stability.tsv")
        run.record_output(final_tables_dir / "candidate_variant_consistency.tsv")
        run.record_output(final_tables_dir / "candidate_multiverse_stability.tsv")
        run.record_output(final_tables_dir / "model_simplicity_summary.tsv")
        run.record_output(final_tables_dir / "knownness_audit_summary.tsv")
        run.record_output(final_tables_dir / "knownness_stratified_performance.tsv")
        run.record_output(final_tables_dir / "negative_control_audit.tsv")
        run.record_output(final_tables_dir / "logistic_implementation_audit.tsv")
        run.record_output(final_tables_dir / "logistic_convergence_audit.tsv")
        run.record_output(final_tables_dir / "outcome_robustness_grid.tsv")
        run.record_output(final_tables_dir / "threshold_sensitivity_summary.tsv")
        run.record_output(final_tables_dir / "l2_sensitivity_summary.tsv")
        run.record_output(final_tables_dir / "weighting_sensitivity_summary.tsv")
        run.record_output(final_tables_dir / "h_feature_diagnostics.tsv")
        run.record_output(final_tables_dir / "score_axis_summary.tsv")
        run.record_output(final_tables_dir / "score_distribution_diagnostics.tsv")
        run.record_output(final_tables_dir / "component_floor_diagnostics.tsv")
        run.record_output(final_tables_dir / "temporal_drift_summary.tsv")
        run.record_output(final_tables_dir / "country_quality_summary.tsv")
        run.record_output(final_tables_dir / "backbone_purity_atlas.tsv")
        run.record_output(final_tables_dir / "assignment_confidence_summary.tsv")
        run.record_output(final_tables_dir / "incremental_value_over_baseline.tsv")
        run.record_output(final_tables_dir / "novelty_specialist_metrics.tsv")
        run.record_output(final_tables_dir / "novelty_specialist_predictions.tsv")
        run.record_output(final_tables_dir / "adaptive_gated_metrics.tsv")
        run.record_output(final_tables_dir / "adaptive_gated_predictions.tsv")
        run.record_output(final_tables_dir / "gate_consistency_audit.tsv")
        run.record_output(final_tables_dir / "knownness_matched_validation.tsv")
        run.record_output(final_tables_dir / "nonlinear_deconfounding_audit.tsv")
        run.record_output(final_tables_dir / "country_upload_propensity.tsv")
        run.record_output(final_tables_dir / "macro_region_jump_outcome.tsv")
        run.record_output(final_tables_dir / "secondary_outcome_performance.tsv")
        run.record_output(final_tables_dir / "weighted_country_outcome_audit.tsv")
        run.record_output(final_tables_dir / "new_country_count_audit.tsv")
        run.record_output(final_tables_dir / "metadata_quality_summary.tsv")
        run.record_output(final_tables_dir / "event_timing_outcomes.tsv")
        run.record_output(final_tables_dir / "exposure_adjusted_event_outcomes.tsv")
        run.record_output(final_tables_dir / "exposure_adjusted_outcome_audit.tsv")
        run.record_output(final_tables_dir / "ordinal_outcome_audit.tsv")
        run.record_output(final_tables_dir / "country_missingness_bounds.tsv")
        run.record_output(final_tables_dir / "country_missingness_sensitivity.tsv")
        run.record_output(final_tables_dir / "geographic_jump_distance_outcome.tsv")
        run.record_output(final_tables_dir / "duplicate_completeness_change_audit.tsv")
        run.record_output(final_tables_dir / "amr_uncertainty_summary.tsv")
        run.record_output(final_tables_dir / "mash_similarity_graph.tsv")
        run.record_output(final_tables_dir / "counterfactual_shortlist_comparison.tsv")
        run.record_output(final_tables_dir / "annual_candidate_freeze_summary.tsv")
        run.record_output(final_tables_dir / "pathogen_detection_group_comparison.tsv")
        run.record_output(final_tables_dir / "prospective_candidate_freeze.tsv")
        run.record_output(final_tables_dir / "candidate_dossiers.tsv")
        run.record_output(final_tables_dir / "candidate_risk_flags.tsv")
        run.record_output(final_tables_dir / "novelty_watchlist.tsv")
        run.record_output(final_tables_dir / "novelty_margin_summary.tsv")
        run.record_output(final_tables_dir / "candidate_portfolio.tsv")
        run.record_output(final_tables_dir / "module_f_backbone_identity.tsv")
        run.record_output(final_tables_dir / "module_f_enrichment.tsv")
        run.record_output(final_tables_dir / "module_f_top_hits.tsv")

        scored = read_tsv(scored_path)
        backbones = read_tsv(backbones_path)
        amr_consensus = _read_if_exists(amr_consensus_path)
        model_metrics = _metrics_to_frame(metrics_path)
        predictions = read_tsv(predictions_path)
        calibration = read_tsv(calibration_path)
        source_validation = read_tsv(source_validation_path)
        primary_model_name = get_primary_model_name(model_metrics["model_name"].tolist())
        conservative_model_name = get_conservative_model_name(model_metrics["model_name"].tolist())

        def _compute_knownness(frame: pd.DataFrame) -> pd.DataFrame:
            return annotate_knownness_metadata(frame.copy())

        contextual_prediction_frames: dict[str, pd.DataFrame] = {}
        for model_name, output_column in (
            (primary_model_name, "primary_model_full_fit_prediction"),
            ("baseline_both", "baseline_both_full_fit_prediction"),
            (conservative_model_name, "conservative_model_full_fit_prediction"),
        ):
            if model_name not in set(model_metrics["model_name"].astype(str)):
                continue
            if output_column in contextual_prediction_frames:
                continue
            contextual_prediction_frames[output_column] = fit_full_model_predictions(
                scored,
                model_name=model_name,
            ).rename(columns={"prediction": output_column})
        family_summary = read_tsv(family_summary_path)
        if (
            family_summary.empty
            or "model_name" not in family_summary.columns
            or "evidence_role" not in family_summary.columns
            or primary_model_name not in set(family_summary["model_name"].astype(str))
        ):
            family_summary = build_model_family_summary(model_metrics)
        subgroup_performance = read_tsv(subgroup_performance_path)
        comparison_table = read_tsv(comparison_path)
        calibration_metrics = read_tsv(calibration_metrics_path)
        coefficient_table = read_tsv(coefficient_path)
        coefficient_stability = read_tsv(coefficient_stability_path)
        dropout_table = read_tsv(dropout_path)
        source_balance_resampling = read_tsv(source_balance_resampling_path)
        if stale_turkiye_context_path.exists():
            stale_turkiye_context_path.unlink()
        group_holdout = _read_if_exists(group_holdout_path)
        permutation_detail = _read_if_exists(permutation_detail_path)
        permutation_summary = _read_if_exists(permutation_summary_path)
        negative_control = _read_if_exists(negative_control_path)
        logistic_impl = _read_if_exists(logistic_impl_path)
        logistic_convergence = _read_if_exists(logistic_convergence_path)
        simplicity_summary = _read_if_exists(simplicity_path)
        knownness_summary = _read_if_exists(knownness_summary_path)
        knownness_strata = _read_if_exists(knownness_strata_path)
        knownness_matched_validation = _read_if_exists(knownness_matched_validation_path)
        country_quality = _read_if_exists(country_quality_path)
        country_upload_propensity = _read_if_exists(country_upload_propensity_path)
        purity_atlas = _read_if_exists(purity_atlas_path)
        assignment_confidence = _read_if_exists(assignment_confidence_path)
        incremental_value = _read_if_exists(incremental_value_path)
        novelty_specialist_metrics = _read_if_exists(novelty_specialist_metrics_path)
        novelty_specialist_predictions = _read_if_exists(novelty_specialist_predictions_path)
        adaptive_gated_metrics = _read_if_exists(adaptive_gated_metrics_path)
        adaptive_gated_predictions = _read_if_exists(adaptive_gated_predictions_path)
        gate_consistency_audit = _read_if_exists(gate_consistency_audit_path)
        nonlinear_deconfounding = _read_if_exists(nonlinear_deconfounding_path)
        macro_region_jump = _read_if_exists(macro_region_jump_path)
        secondary_outcome_performance = _read_if_exists(secondary_outcome_performance_path)
        weighted_country_outcome = _read_if_exists(weighted_country_outcome_path)
        count_outcome_audit = _read_if_exists(count_outcome_audit_path)
        metadata_quality_summary = _read_if_exists(metadata_quality_summary_path)
        event_timing_outcomes = _read_if_exists(event_timing_outcomes_path)
        exposure_adjusted_event = _read_if_exists(exposure_adjusted_event_path)
        exposure_adjusted_outcome_audit = _read_if_exists(exposure_adjusted_outcome_audit_path)
        ordinal_outcome_audit = _read_if_exists(ordinal_outcome_audit_path)
        country_missingness_bounds = _read_if_exists(country_missingness_bounds_path)
        country_missingness_sensitivity = _read_if_exists(country_missingness_sensitivity_path)
        geographic_jump = _read_if_exists(geographic_jump_path)
        duplicate_quality = _read_if_exists(duplicate_quality_path)
        amr_uncertainty = _read_if_exists(amr_uncertainty_path)
        mash_graph = _read_if_exists(mash_graph_path)
        counterfactual_shortlist = _read_if_exists(counterfactual_shortlist_path)
        module_f_identity = _read_if_exists(module_f_identity_path)
        module_f_enrichment = _read_if_exists(module_f_enrichment_path)
        module_f_top_hits = _read_if_exists(module_f_top_hits_path)
        if module_f_identity.empty and not backbones.empty and not amr_consensus.empty:
            module_f_identity = build_backbone_identity_table(scored, backbones, amr_consensus, split_year=2015)
        if module_f_enrichment.empty and not module_f_identity.empty:
            module_f_enrichment = build_module_f_enrichment_table(module_f_identity, label_column="spread_label", min_backbones=10)
        if module_f_top_hits.empty and not module_f_enrichment.empty:
            module_f_top_hits = build_module_f_top_hits(module_f_enrichment, q_threshold=0.05, max_per_group=3, max_total=20)
        rolling_temporal = _read_if_exists(rolling_temporal_path)
        rolling_assignment_diagnostics = _read_if_exists(rolling_assignment_diagnostic_path)
        rank_stability = _read_if_exists(rank_stability_path)
        variant_consistency = _read_if_exists(variant_consistency_path)
        prospective_freeze = _read_if_exists(prospective_freeze_path)
        annual_freeze_summary = _read_if_exists(annual_freeze_summary_path)
        if annual_freeze_summary.empty and not rolling_temporal.empty:
            rolling_ok = rolling_temporal.loc[rolling_temporal.get("status", pd.Series("", index=rolling_temporal.index)).astype(str) == "ok"].copy()
            if not rolling_ok.empty:
                annual_freeze_summary = rolling_ok.copy()
                if "horizon_years" not in annual_freeze_summary.columns:
                    annual_freeze_summary["horizon_years"] = (
                        annual_freeze_summary["test_year_end"].astype(int) - annual_freeze_summary["split_year"].astype(int)
                    )
                annual_freeze_summary["n_candidates"] = 25
                annual_freeze_summary["n_positive_candidates"] = np.nan
                annual_freeze_summary["precision_at_25"] = np.nan
                annual_freeze_summary["mean_n_new_countries"] = np.nan
                annual_freeze_summary["top_backbone_id"] = ""
        with sensitivity_path.open("r", encoding="utf-8") as handle:
            sensitivity = json.load(handle)
        scored["operational_priority_index"] = scored.get("operational_priority_index", scored.get("priority_index", 0.0)).fillna(
            scored.get("priority_index", 0.0)
        )
        top = scored.loc[scored["spread_label"].notna()].sort_values("operational_priority_index", ascending=False).head(100)
        top_operational_backlog = scored.loc[scored["member_count_train"].fillna(0).astype(int) > 0].sort_values(
            "operational_priority_index",
            ascending=False,
        ).head(100)
        top_bio = scored.loc[scored["spread_label"].notna()].sort_values("bio_priority_index", ascending=False).head(100)
        top_bio_backlog = scored.loc[scored["member_count_train"].fillna(0).astype(int) > 0].sort_values(
            "bio_priority_index",
            ascending=False,
        ).head(100)
        top_primary = scored.loc[scored["spread_label"].notna()].copy()
        if primary_model_name in set(predictions["model_name"].astype(str)):
            top_primary = top_primary.merge(
                predictions.loc[
                    predictions["model_name"] == primary_model_name,
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                on="backbone_id",
                how="left",
            ).sort_values("primary_model_oof_prediction", ascending=False)
        top_primary = top_primary.head(100)
        top = _add_visibility_alias(top)
        top_operational_backlog = _add_visibility_alias(top_operational_backlog)
        top_bio = _add_visibility_alias(top_bio)
        top_bio_backlog = _add_visibility_alias(top_bio_backlog)
        top_primary = _add_visibility_alias(top_primary)
        top_primary.to_csv(final_tables_dir / "top_primary_candidates.tsv", sep="\t", index=False)
        model_metrics.to_csv(final_tables_dir / "model_metrics.tsv", sep="\t", index=False)
        source_validation.to_csv(final_tables_dir / "source_stratified_consistency.tsv", sep="\t", index=False)
        calibration.to_csv(final_tables_dir / "calibration_table.tsv", sep="\t", index=False)
        family_summary.to_csv(final_tables_dir / "model_family_summary.tsv", sep="\t", index=False)
        subgroup_performance.to_csv(final_tables_dir / "model_subgroup_performance.tsv", sep="\t", index=False)
        comparison_table.to_csv(final_tables_dir / "model_comparison_summary.tsv", sep="\t", index=False)
        calibration_metrics.to_csv(final_tables_dir / "calibration_metrics.tsv", sep="\t", index=False)
        coefficient_table.to_csv(final_tables_dir / "primary_model_coefficients.tsv", sep="\t", index=False)
        coefficient_stability.to_csv(final_tables_dir / "primary_model_coefficient_stability.tsv", sep="\t", index=False)
        dropout_table.to_csv(final_tables_dir / "feature_dropout_importance.tsv", sep="\t", index=False)
        source_balance_resampling.to_csv(final_tables_dir / "source_balance_resampling.tsv", sep="\t", index=False)
        group_holdout.to_csv(final_tables_dir / "group_holdout_performance.tsv", sep="\t", index=False)
        permutation_detail.to_csv(final_tables_dir / "permutation_null_distribution.tsv", sep="\t", index=False)
        permutation_summary.to_csv(final_tables_dir / "permutation_null_summary.tsv", sep="\t", index=False)
        negative_control.to_csv(final_tables_dir / "negative_control_audit.tsv", sep="\t", index=False)
        logistic_impl.to_csv(final_tables_dir / "logistic_implementation_audit.tsv", sep="\t", index=False)
        logistic_convergence.to_csv(final_tables_dir / "logistic_convergence_audit.tsv", sep="\t", index=False)
        simplicity_summary.to_csv(final_tables_dir / "model_simplicity_summary.tsv", sep="\t", index=False)
        knownness_summary.to_csv(final_tables_dir / "knownness_audit_summary.tsv", sep="\t", index=False)
        knownness_strata.to_csv(final_tables_dir / "knownness_stratified_performance.tsv", sep="\t", index=False)
        country_quality.to_csv(final_tables_dir / "country_quality_summary.tsv", sep="\t", index=False)
        purity_atlas.to_csv(final_tables_dir / "backbone_purity_atlas.tsv", sep="\t", index=False)
        assignment_confidence.to_csv(final_tables_dir / "assignment_confidence_summary.tsv", sep="\t", index=False)
        incremental_value.to_csv(final_tables_dir / "incremental_value_over_baseline.tsv", sep="\t", index=False)
        novelty_specialist_metrics.to_csv(final_tables_dir / "novelty_specialist_metrics.tsv", sep="\t", index=False)
        novelty_specialist_predictions.to_csv(final_tables_dir / "novelty_specialist_predictions.tsv", sep="\t", index=False)
        adaptive_gated_metrics.to_csv(final_tables_dir / "adaptive_gated_metrics.tsv", sep="\t", index=False)
        adaptive_gated_predictions.to_csv(final_tables_dir / "adaptive_gated_predictions.tsv", sep="\t", index=False)
        gate_consistency_audit.to_csv(final_tables_dir / "gate_consistency_audit.tsv", sep="\t", index=False)
        knownness_matched_validation.to_csv(final_tables_dir / "knownness_matched_validation.tsv", sep="\t", index=False)
        nonlinear_deconfounding.to_csv(final_tables_dir / "nonlinear_deconfounding_audit.tsv", sep="\t", index=False)
        country_upload_propensity.to_csv(final_tables_dir / "country_upload_propensity.tsv", sep="\t", index=False)
        macro_region_jump.to_csv(final_tables_dir / "macro_region_jump_outcome.tsv", sep="\t", index=False)
        secondary_outcome_performance.to_csv(final_tables_dir / "secondary_outcome_performance.tsv", sep="\t", index=False)
        weighted_country_outcome.to_csv(final_tables_dir / "weighted_country_outcome_audit.tsv", sep="\t", index=False)
        count_outcome_audit.to_csv(final_tables_dir / "new_country_count_audit.tsv", sep="\t", index=False)
        metadata_quality_summary.to_csv(final_tables_dir / "metadata_quality_summary.tsv", sep="\t", index=False)
        event_timing_outcomes.to_csv(final_tables_dir / "event_timing_outcomes.tsv", sep="\t", index=False)
        exposure_adjusted_event.to_csv(final_tables_dir / "exposure_adjusted_event_outcomes.tsv", sep="\t", index=False)
        exposure_adjusted_outcome_audit.to_csv(final_tables_dir / "exposure_adjusted_outcome_audit.tsv", sep="\t", index=False)
        ordinal_outcome_audit.to_csv(final_tables_dir / "ordinal_outcome_audit.tsv", sep="\t", index=False)
        country_missingness_bounds.to_csv(final_tables_dir / "country_missingness_bounds.tsv", sep="\t", index=False)
        country_missingness_sensitivity.to_csv(final_tables_dir / "country_missingness_sensitivity.tsv", sep="\t", index=False)
        geographic_jump.to_csv(final_tables_dir / "geographic_jump_distance_outcome.tsv", sep="\t", index=False)
        duplicate_quality.to_csv(final_tables_dir / "duplicate_completeness_change_audit.tsv", sep="\t", index=False)
        amr_uncertainty.to_csv(final_tables_dir / "amr_uncertainty_summary.tsv", sep="\t", index=False)
        mash_graph.to_csv(final_tables_dir / "mash_similarity_graph.tsv", sep="\t", index=False)
        counterfactual_shortlist.to_csv(final_tables_dir / "counterfactual_shortlist_comparison.tsv", sep="\t", index=False)
        module_f_identity.to_csv(final_tables_dir / "module_f_backbone_identity.tsv", sep="\t", index=False)
        module_f_enrichment.to_csv(final_tables_dir / "module_f_enrichment.tsv", sep="\t", index=False)
        module_f_top_hits.to_csv(final_tables_dir / "module_f_top_hits.tsv", sep="\t", index=False)
        rolling_temporal.to_csv(final_tables_dir / "rolling_temporal_validation.tsv", sep="\t", index=False)
        rolling_assignment_diagnostics.to_csv(final_tables_dir / "rolling_assignment_diagnostics.tsv", sep="\t", index=False)
        rank_stability.to_csv(final_tables_dir / "candidate_rank_stability.tsv", sep="\t", index=False)
        variant_consistency.to_csv(final_tables_dir / "candidate_variant_consistency.tsv", sep="\t", index=False)
        prospective_freeze_export = (
            prospective_freeze.loc[prospective_freeze["spread_label"].notna()].copy()
            if not prospective_freeze.empty and "spread_label" in prospective_freeze.columns
            else prospective_freeze.copy()
        )
        prospective_freeze_export = _add_visibility_alias(prospective_freeze_export)
        prospective_freeze_export.to_csv(final_tables_dir / "prospective_candidate_freeze.tsv", sep="\t", index=False)
        annual_freeze_summary.to_csv(final_tables_dir / "annual_candidate_freeze_summary.tsv", sep="\t", index=False)

        sensitivity_rows = []
        for variant, metrics in sensitivity.items():
            row = {"variant": variant}
            row.update(metrics)
            sensitivity_rows.append(row)
        pd.DataFrame(sensitivity_rows).to_csv(final_tables_dir / "sensitivity_summary.tsv", sep="\t", index=False)
        threshold_sensitivity = _build_threshold_sensitivity_table(
            sensitivity,
            default_threshold=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        )
        threshold_sensitivity.to_csv(final_tables_dir / "threshold_sensitivity_summary.tsv", sep="\t", index=False)
        l2_sensitivity = _build_l2_sensitivity_table(sensitivity)
        l2_sensitivity.to_csv(final_tables_dir / "l2_sensitivity_summary.tsv", sep="\t", index=False)
        weighting_sensitivity = _build_weighting_sensitivity_table(sensitivity)
        weighting_sensitivity.to_csv(final_tables_dir / "weighting_sensitivity_summary.tsv", sep="\t", index=False)

        if module_b_path.exists():
            module_b = _read_if_exists(module_b_path)
            module_b.to_csv(final_tables_dir / "module_b_amr_class_comparison.tsv", sep="\t", index=False)
        else:
            module_b = pd.DataFrame()

        module_c = _read_if_exists(module_c_path)
        module_c_group = _read_if_exists(module_c_group_path)
        module_c_clinical = _read_if_exists(module_c_clinical_path)
        module_c_clinical_group = _read_if_exists(module_c_clinical_group_path)
        module_c_environmental = _read_if_exists(module_c_environmental_path)
        module_c_environmental_group = _read_if_exists(module_c_environmental_group_path)
        module_c_strata_group = _read_if_exists(module_c_strata_group_path)
        if not module_c.empty:
            module_c.to_csv(final_tables_dir / "pathogen_detection_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_support.tsv")
        if not module_c_group.empty:
            module_c_group.to_csv(final_tables_dir / "pathogen_detection_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_group_summary.tsv")
        if not module_c_clinical.empty:
            module_c_clinical.to_csv(final_tables_dir / "pathogen_detection_clinical_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_clinical_support.tsv")
        if not module_c_clinical_group.empty:
            module_c_clinical_group.to_csv(final_tables_dir / "pathogen_detection_clinical_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_clinical_group_summary.tsv")
        if not module_c_environmental.empty:
            module_c_environmental.to_csv(final_tables_dir / "pathogen_detection_environmental_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_environmental_support.tsv")
        if not module_c_environmental_group.empty:
            module_c_environmental_group.to_csv(final_tables_dir / "pathogen_detection_environmental_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_environmental_group_summary.tsv")
        if not module_c_strata_group.empty:
            module_c_strata_group.to_csv(final_tables_dir / "pathogen_detection_strata_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "pathogen_detection_strata_group_summary.tsv")

        who_detail = _read_if_exists(who_detail_path)
        who_summary = _read_if_exists(who_summary_path)
        who_category = _read_if_exists(who_category_path)
        who_reference = _read_if_exists(who_reference_path)
        card_detail = _read_if_exists(card_detail_path)
        card_summary = _read_if_exists(card_summary_path)
        card_family = _read_if_exists(card_family_path)
        card_mechanism = _read_if_exists(card_mechanism_path)
        mobsuite_detail = _read_if_exists(mobsuite_detail_path)
        mobsuite_summary = _read_if_exists(mobsuite_summary_path)
        amrfinder_probe_panel = _read_if_exists(amrfinder_probe_panel_path)
        amrfinder_probe_hits = _read_if_exists(amrfinder_probe_hits_path)
        amrfinder_detail = _read_if_exists(amrfinder_detail_path)
        amrfinder_summary = _read_if_exists(amrfinder_summary_path)
        amrfinder_coverage = build_amrfinder_coverage_table(amrfinder_summary)
        amrfinder_reportable = _is_amrfinder_reportable(amrfinder_coverage)
        amrfinder_sequences_total = int(amrfinder_coverage["n_sequences"].sum()) if not amrfinder_coverage.empty else 0
        if amrfinder_sequences_total == 0:
            amrfinder_reason = "AMRFinder executable/report unavailable; appendix-only probe skipped"
        elif not amrfinder_reportable:
            amrfinder_reason = "probe panel too small or imbalanced; keep as appendix-only sanity check"
        else:
            amrfinder_reason = "coverage acceptable for descriptive appendix reporting"
        amrfinder_assessment = pd.DataFrame(
            [
                {
                    "reportable_in_main_report": bool(amrfinder_reportable),
                    "reason": amrfinder_reason,
                    "n_sequences_total": amrfinder_sequences_total,
                    "high_group_sequences": int(amrfinder_coverage.loc[amrfinder_coverage["priority_group"] == "high", "n_sequences"].sum())
                    if not amrfinder_coverage.empty
                    else 0,
                    "low_group_sequences": int(amrfinder_coverage.loc[amrfinder_coverage["priority_group"] == "low", "n_sequences"].sum())
                    if not amrfinder_coverage.empty
                    else 0,
                }
            ]
        )
        if not who_detail.empty:
            who_detail.to_csv(final_tables_dir / "who_mia_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "who_mia_support.tsv")
        if not who_summary.empty:
            who_summary.to_csv(final_tables_dir / "who_mia_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "who_mia_group_summary.tsv")
        if not who_category.empty:
            who_category.to_csv(final_tables_dir / "who_mia_category_comparison.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "who_mia_category_comparison.tsv")
        if not who_reference.empty:
            who_reference.to_csv(final_tables_dir / "who_mia_reference_catalog.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "who_mia_reference_catalog.tsv")
        if not card_detail.empty:
            card_detail.to_csv(final_tables_dir / "card_gene_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "card_gene_support.tsv")
        if not card_summary.empty:
            card_summary.to_csv(final_tables_dir / "card_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "card_group_summary.tsv")
        if not card_family.empty:
            card_family.to_csv(final_tables_dir / "card_gene_family_comparison.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "card_gene_family_comparison.tsv")
        if not card_mechanism.empty:
            card_mechanism.to_csv(final_tables_dir / "card_mechanism_comparison.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "card_mechanism_comparison.tsv")
        if not mobsuite_detail.empty:
            mobsuite_detail.to_csv(final_tables_dir / "mobsuite_host_range_support.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "mobsuite_host_range_support.tsv")
        if not mobsuite_summary.empty:
            mobsuite_summary.to_csv(final_tables_dir / "mobsuite_host_range_group_summary.tsv", sep="\t", index=False)
            run.record_output(final_tables_dir / "mobsuite_host_range_group_summary.tsv")
        if amrfinder_reportable:
            if not amrfinder_probe_panel.empty:
                amrfinder_probe_panel.to_csv(final_tables_dir / "amrfinder_probe_panel.tsv", sep="\t", index=False)
                run.record_output(final_tables_dir / "amrfinder_probe_panel.tsv")
            if not amrfinder_probe_hits.empty:
                amrfinder_probe_hits.to_csv(final_tables_dir / "amrfinder_probe_hits.tsv", sep="\t", index=False)
                run.record_output(final_tables_dir / "amrfinder_probe_hits.tsv")
            if not amrfinder_detail.empty:
                amrfinder_detail.to_csv(final_tables_dir / "amrfinder_concordance_detail.tsv", sep="\t", index=False)
                run.record_output(final_tables_dir / "amrfinder_concordance_detail.tsv")
            if not amrfinder_summary.empty:
                amrfinder_summary.to_csv(final_tables_dir / "amrfinder_concordance_summary.tsv", sep="\t", index=False)
                run.record_output(final_tables_dir / "amrfinder_concordance_summary.tsv")
        else:
            for stale_name in (
                "amrfinder_probe_panel.tsv",
                "amrfinder_probe_hits.tsv",
                "amrfinder_concordance_detail.tsv",
                "amrfinder_concordance_summary.tsv",
            ):
                stale_path = final_tables_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()

        pathogen_detail_frames = [frame for frame in (module_c, module_c_clinical, module_c_environmental) if not frame.empty]
        pathogen_group_comparison = build_pathogen_group_comparison(
            pd.concat(pathogen_detail_frames, ignore_index=True) if pathogen_detail_frames else pd.DataFrame()
        )
        pathogen_group_comparison.to_csv(final_tables_dir / "pathogen_detection_group_comparison.tsv", sep="\t", index=False)

        candidate_context_frames = [top.copy(), top_bio.copy(), top_primary.copy()]
        if not prospective_freeze.empty and "backbone_id" in prospective_freeze.columns:
            freeze_ids = prospective_freeze["backbone_id"].astype(str).unique().tolist()
            freeze_rows = scored.loc[scored["backbone_id"].astype(str).isin(freeze_ids)].copy()
            if not freeze_rows.empty:
                candidate_context_frames.append(freeze_rows)
        candidate_context = (
            pd.concat(candidate_context_frames, ignore_index=True)
            .drop_duplicates(subset=["backbone_id"], keep="first")
            .sort_values("priority_index", ascending=False)
            .reset_index(drop=True)
        )
        candidate_context = candidate_context.loc[candidate_context["spread_label"].notna()].copy()
        candidate_stability = candidate_context.copy()
        if not rank_stability.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                rank_stability.drop(columns=["base_priority_index"], errors="ignore"),
                on="backbone_id",
            )
        if not variant_consistency.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                variant_consistency.drop(columns=["base_priority_index"], errors="ignore"),
                on="backbone_id",
            )
        if not who_detail.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                who_detail[["backbone_id", "who_mia_any_support", "who_mia_any_hpecia", "who_mia_mapped_fraction"]],
                on="backbone_id",
            )
        if not card_detail.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                card_detail[["backbone_id", "card_any_support", "card_match_fraction"]],
                on="backbone_id",
            )
        if not mobsuite_detail.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                mobsuite_detail[["backbone_id", "mobsuite_any_literature_support", "mobsuite_any_cluster_support"]],
                on="backbone_id",
            )
        if not module_c.empty:
            pd_combined = module_c.loc[module_c["pathogen_dataset"] == "combined", ["backbone_id", "pd_any_support", "pd_matching_fraction"]]
            candidate_stability = coalescing_left_merge(candidate_stability, pd_combined, on="backbone_id")
        if primary_model_name in set(predictions["model_name"].astype(str)):
            primary_predictions = predictions.loc[predictions["model_name"] == primary_model_name, ["backbone_id", "oof_prediction"]]
            primary_predictions = primary_predictions.rename(columns={"oof_prediction": "primary_model_oof_prediction"})
            candidate_stability = coalescing_left_merge(candidate_stability, primary_predictions, on="backbone_id")
        if "baseline_both" in set(predictions["model_name"].astype(str)):
            baseline_predictions = predictions.loc[predictions["model_name"] == "baseline_both", ["backbone_id", "oof_prediction"]]
            baseline_predictions = baseline_predictions.rename(columns={"oof_prediction": "baseline_both_oof_prediction"})
            candidate_stability = coalescing_left_merge(candidate_stability, baseline_predictions, on="backbone_id")
            primary_oof = candidate_stability.get(
                "primary_model_oof_prediction",
                pd.Series(np.nan, index=candidate_stability.index, dtype=float),
            ).astype(float)
            baseline_oof = candidate_stability.get(
                "baseline_both_oof_prediction",
                pd.Series(np.nan, index=candidate_stability.index, dtype=float),
            ).astype(float)
            candidate_stability["novelty_margin_vs_baseline"] = (
                primary_oof.fillna(0.0)
                - baseline_oof.fillna(0.0)
            )
        for prediction_frame in contextual_prediction_frames.values():
            if prediction_frame.empty:
                continue
            candidate_stability = coalescing_left_merge(candidate_stability, prediction_frame, on="backbone_id")
        knownness_meta_all = pd.DataFrame()
        knownness_meta_eligible = pd.DataFrame()
        if not knownness_summary.empty:
            knownness_meta_all = _compute_knownness(
                scored[["backbone_id", "log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"]].copy()
            )
            eligible_backbones = predictions.loc[
                predictions["model_name"] == primary_model_name,
                ["backbone_id"],
            ].drop_duplicates()
            knownness_meta_eligible = eligible_backbones.merge(
                scored[["backbone_id", "log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"]],
                on="backbone_id",
                how="left",
            )
            knownness_meta_eligible = _compute_knownness(knownness_meta_eligible)
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                knownness_meta_all[["backbone_id", "knownness_score", "knownness_half"]],
                on="backbone_id",
            )
        if not metadata_quality_summary.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                metadata_quality_summary[
                    [
                        column
                        for column in [
                            "backbone_id",
                            "metadata_quality_score",
                            "metadata_quality_tier",
                            "country_coverage_fraction",
                            "duplicate_fraction",
                        ]
                        if column in metadata_quality_summary.columns
                    ]
                ],
                on="backbone_id",
            )
        if not macro_region_jump.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                macro_region_jump[
                    [
                        column
                        for column in [
                            "backbone_id",
                            "n_new_macro_regions",
                            "macro_region_jump_label",
                            "n_new_host_families",
                            "host_family_jump_label",
                            "n_new_host_orders",
                            "host_order_jump_label",
                            "weighted_new_country_burden",
                            "rarity_weighted_new_country_burden",
                        ]
                        if column in macro_region_jump.columns
                    ]
                ],
                on="backbone_id",
            )

        primary_oof = candidate_stability.get(
            "primary_model_oof_prediction",
            pd.Series(np.nan, index=candidate_stability.index, dtype=float),
        ).astype(float)
        primary_full = candidate_stability.get(
            "primary_model_full_fit_prediction",
            pd.Series(np.nan, index=candidate_stability.index, dtype=float),
        ).astype(float)
        conservative_oof = candidate_stability.get(
            "conservative_model_oof_prediction",
            pd.Series(np.nan, index=candidate_stability.index, dtype=float),
        ).astype(float)
        conservative_full = candidate_stability.get(
            "conservative_model_full_fit_prediction",
            pd.Series(np.nan, index=candidate_stability.index, dtype=float),
        ).astype(float)
        candidate_stability["primary_model_candidate_score"] = primary_oof.fillna(primary_full)
        candidate_stability["conservative_model_candidate_score"] = conservative_oof.fillna(conservative_full)

        candidate_stability["high_confidence_candidate"] = (
            candidate_stability["priority_index"].notna()
            & candidate_stability["coherence_score"].fillna(0.0).ge(0.5)
            & candidate_stability["bootstrap_top_k_frequency"].fillna(0.0).ge(0.7)
            & candidate_stability["variant_top_k_frequency"].fillna(0.0).ge(0.6)
        )
        consensus_candidates = build_consensus_candidate_ranking(
            candidate_stability,
            primary_score_column="primary_model_candidate_score",
            conservative_score_column="conservative_model_candidate_score",
            top_k=50,
        )

        scored_for_h = scored.copy()
        if primary_model_name in set(predictions["model_name"].astype(str)):
            scored_for_h = scored_for_h.merge(
                predictions.loc[
                    predictions["model_name"] == primary_model_name,
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                on="backbone_id",
                how="left",
            )
        h_feature_diagnostics = build_h_feature_diagnostics(
            scored_for_h,
            model_metrics=model_metrics,
            coefficient_table=coefficient_table,
            dropout_table=dropout_table,
            mobsuite_detail=mobsuite_detail,
            score_column="primary_model_oof_prediction",
        )
        h_feature_diagnostics.to_csv(final_tables_dir / "h_feature_diagnostics.tsv", sep="\t", index=False)

        score_axis_summary = build_score_axis_summary(
            scored,
            predictions,
            primary_model_name=primary_model_name,
            baseline_model_name="baseline_both",
        )
        score_axis_summary.to_csv(final_tables_dir / "score_axis_summary.tsv", sep="\t", index=False)

        score_distribution_diagnostics = build_score_distribution_diagnostics(scored)
        score_distribution_diagnostics.to_csv(final_tables_dir / "score_distribution_diagnostics.tsv", sep="\t", index=False)

        component_floor_diagnostics = build_component_floor_diagnostics(scored)
        component_floor_diagnostics.to_csv(final_tables_dir / "component_floor_diagnostics.tsv", sep="\t", index=False)

        temporal_drift_summary = build_temporal_drift_summary(backbones)
        temporal_drift_summary.to_csv(final_tables_dir / "temporal_drift_summary.tsv", sep="\t", index=False)

        if simplicity_summary.empty:
            simplicity_summary = build_model_simplicity_summary(
                model_metrics,
                predictions,
                primary_model_name=primary_model_name,
                conservative_model_name=conservative_model_name,
            )
            simplicity_summary.to_csv(final_tables_dir / "model_simplicity_summary.tsv", sep="\t", index=False)

        outcome_robustness = _build_outcome_robustness_grid(
            sensitivity,
            rolling_temporal,
            default_threshold=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        )
        outcome_robustness.to_csv(final_tables_dir / "outcome_robustness_grid.tsv", sep="\t", index=False)

        dossier_base = consensus_candidates.head(25).copy() if not consensus_candidates.empty else (
            prospective_freeze.head(25).copy() if not prospective_freeze.empty else candidate_stability.head(25).copy()
        )
        candidate_dossiers = build_candidate_dossier_table(
            dossier_base,
            candidate_stability=candidate_stability,
            predictions=predictions,
            primary_model_name=primary_model_name,
            conservative_model_name=conservative_model_name,
            who_detail=who_detail,
            card_detail=card_detail,
            mobsuite_detail=mobsuite_detail,
            pathogen_support=module_c,
            amrfinder_detail=amrfinder_detail,
        )
        candidate_risk = build_candidate_risk_table(candidate_dossiers)
        if not candidate_risk.empty:
            candidate_dossiers = coalescing_left_merge(
                candidate_dossiers,
                candidate_risk[["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]],
                on="backbone_id",
            )
        candidate_dossiers.to_csv(final_tables_dir / "candidate_dossiers.tsv", sep="\t", index=False)
        candidate_risk.to_csv(final_tables_dir / "candidate_risk_flags.tsv", sep="\t", index=False)
        novelty_watchlist = scored.copy()
        if primary_model_name in set(predictions["model_name"].astype(str)):
            novelty_watchlist = novelty_watchlist.merge(
                predictions.loc[
                    predictions["model_name"] == primary_model_name,
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                on="backbone_id",
                how="left",
            )
        if "baseline_both" in set(predictions["model_name"].astype(str)):
            novelty_watchlist = novelty_watchlist.merge(
                predictions.loc[
                    predictions["model_name"] == "baseline_both",
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "baseline_both_oof_prediction"}),
                on="backbone_id",
                how="left",
            )
        if not knownness_meta_eligible.empty:
            novelty_watchlist = novelty_watchlist.merge(
                knownness_meta_eligible[["backbone_id", "knownness_score", "knownness_half"]].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
        if not novelty_specialist_predictions.empty:
            novelty_watchlist = novelty_watchlist.merge(
                novelty_specialist_predictions[
                    [
                        column
                        for column in [
                            "backbone_id",
                            "novelty_specialist_prediction",
                            "knownness_quartile",
                            "training_cohort",
                        ]
                        if column in novelty_specialist_predictions.columns
                    ]
                ].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
        if not who_detail.empty:
            novelty_watchlist = novelty_watchlist.merge(
                who_detail[["backbone_id", "who_mia_any_support"]],
                on="backbone_id",
                how="left",
            )
        if not card_detail.empty:
            novelty_watchlist = novelty_watchlist.merge(
                card_detail[["backbone_id", "card_any_support"]],
                on="backbone_id",
                how="left",
            )
        if not mobsuite_detail.empty:
            novelty_watchlist = novelty_watchlist.merge(
                mobsuite_detail[["backbone_id", "mobsuite_any_literature_support"]],
                on="backbone_id",
                how="left",
            )
        if not module_c.empty:
            novelty_watchlist = novelty_watchlist.merge(
                module_c.loc[module_c["pathogen_dataset"] == "combined", ["backbone_id", "pd_any_support"]],
                on="backbone_id",
                how="left",
            )
        for column in (
            "who_mia_any_support",
            "card_any_support",
            "mobsuite_any_literature_support",
            "pd_any_support",
        ):
            if column in novelty_watchlist.columns:
                novelty_watchlist[column] = novelty_watchlist[column].fillna(False).astype(bool)
        novelty_watchlist["external_support_modalities_count"] = novelty_watchlist[
            [column for column in ("who_mia_any_support", "card_any_support", "mobsuite_any_literature_support", "pd_any_support") if column in novelty_watchlist.columns]
        ].sum(axis=1).astype(int)
        novelty_watchlist["primary_model_candidate_score"] = novelty_watchlist["primary_model_oof_prediction"].astype(float)
        novelty_watchlist["baseline_both_candidate_score"] = novelty_watchlist["baseline_both_oof_prediction"].astype(float)
        novelty_watchlist["candidate_prediction_source"] = "oof"
        novelty_watchlist["eligible_for_oof"] = True
        novelty_watchlist["novelty_margin_vs_baseline"] = np.nan
        novelty_watchlist_margin_mask = (
            novelty_watchlist["primary_model_oof_prediction"].notna()
            & novelty_watchlist["baseline_both_oof_prediction"].notna()
        )
        novelty_watchlist.loc[novelty_watchlist_margin_mask, "novelty_margin_vs_baseline"] = (
            novelty_watchlist.loc[novelty_watchlist_margin_mask, "primary_model_oof_prediction"].astype(float)
            - novelty_watchlist.loc[novelty_watchlist_margin_mask, "baseline_both_oof_prediction"].astype(float)
        )
        if "novelty_specialist_prediction" not in novelty_watchlist.columns:
            novelty_watchlist["novelty_specialist_prediction"] = 0.0
        for prediction_frame in contextual_prediction_frames.values():
            if prediction_frame.empty:
                continue
            novelty_watchlist = novelty_watchlist.merge(prediction_frame, on="backbone_id", how="left")
        conservative_oof_column = None
        if conservative_model_name in set(predictions["model_name"].astype(str)):
            conservative_oof_column = "conservative_model_oof_prediction"
            novelty_watchlist = novelty_watchlist.merge(
                predictions.loc[
                    predictions["model_name"] == conservative_model_name,
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": conservative_oof_column}),
                on="backbone_id",
                how="left",
            )
        novelty_watchlist = novelty_watchlist.loc[
            novelty_watchlist["spread_label"].notna()
            & novelty_watchlist["member_count_train"].fillna(0).astype(int).gt(0)
            & novelty_watchlist["knownness_half"].fillna("").eq("lower_half")
        ].copy()
        novelty_watchlist = novelty_watchlist.loc[
            novelty_watchlist["novelty_margin_vs_baseline"].fillna(0.0).gt(0)
            & (
                novelty_watchlist["member_count_train"].fillna(0).astype(int).ge(2)
                | novelty_watchlist["external_support_modalities_count"].fillna(0).astype(int).gt(0)
            )
        ].copy()
        novelty_watchlist = novelty_watchlist.sort_values(
            ["novelty_specialist_prediction", "novelty_margin_vs_baseline", "primary_model_oof_prediction", "priority_index"],
            ascending=[False, False, False, False],
        ).head(25)
        if conservative_oof_column is not None or "conservative_model_full_fit_prediction" in novelty_watchlist.columns:
            conservative_oof = novelty_watchlist.get(
                conservative_oof_column,
                pd.Series(np.nan, index=novelty_watchlist.index, dtype=float),
            ).astype(float)
            conservative_full = novelty_watchlist.get(
                "conservative_model_full_fit_prediction",
                pd.Series(np.nan, index=novelty_watchlist.index, dtype=float),
            ).astype(float)
            novelty_watchlist["conservative_model_candidate_score"] = conservative_oof.fillna(conservative_full)
        novelty_watchlist = _add_visibility_alias(novelty_watchlist)
        novelty_watchlist.to_csv(final_tables_dir / "novelty_watchlist.tsv", sep="\t", index=False)

        novelty_frontier = scored.loc[scored["spread_label"].notna()].copy()
        if primary_model_name in set(predictions["model_name"].astype(str)):
            novelty_frontier = novelty_frontier.merge(
                predictions.loc[
                    predictions["model_name"] == primary_model_name,
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "primary_model_oof_prediction"}),
                on="backbone_id",
                how="left",
            )
        if "baseline_both" in set(predictions["model_name"].astype(str)):
            novelty_frontier = novelty_frontier.merge(
                predictions.loc[
                    predictions["model_name"] == "baseline_both",
                    ["backbone_id", "oof_prediction"],
                ].rename(columns={"oof_prediction": "baseline_both_oof_prediction"}),
                on="backbone_id",
                how="left",
            )
        if not knownness_meta_eligible.empty:
            novelty_frontier = novelty_frontier.merge(
                knownness_meta_eligible[["backbone_id", "knownness_score", "knownness_half"]].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
        if not novelty_specialist_predictions.empty:
            novelty_frontier = novelty_frontier.merge(
                novelty_specialist_predictions[
                    [
                        column
                        for column in [
                            "backbone_id",
                            "novelty_specialist_prediction",
                            "knownness_quartile",
                        ]
                        if column in novelty_specialist_predictions.columns
                    ]
                ].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
        novelty_frontier["novelty_margin_vs_baseline"] = np.nan
        novelty_frontier_margin_mask = (
            novelty_frontier["primary_model_oof_prediction"].notna()
            & novelty_frontier["baseline_both_oof_prediction"].notna()
        )
        novelty_frontier.loc[novelty_frontier_margin_mask, "novelty_margin_vs_baseline"] = (
            novelty_frontier.loc[novelty_frontier_margin_mask, "primary_model_oof_prediction"].astype(float)
            - novelty_frontier.loc[novelty_frontier_margin_mask, "baseline_both_oof_prediction"].astype(float)
        )
        if "novelty_specialist_prediction" not in novelty_frontier.columns:
            novelty_frontier["novelty_specialist_prediction"] = 0.0
        novelty_frontier = _add_visibility_alias(novelty_frontier)

        novelty_margin_summary = build_novelty_margin_summary(
            predictions,
            scored,
            primary_model_name=primary_model_name,
            baseline_model_name="baseline_both",
        )
        novelty_margin_summary.to_csv(final_tables_dir / "novelty_margin_summary.tsv", sep="\t", index=False)

        candidate_portfolio = build_candidate_portfolio_table(
            candidate_dossiers,
            novelty_watchlist,
            established_n=10,
            novel_n=10,
        )
        if not candidate_portfolio.empty:
            if not consensus_candidates.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    consensus_candidates[
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
                    ],
                    on="backbone_id",
                )
            if not candidate_stability.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    candidate_stability[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "coherence_score",
                                "bootstrap_top_k_frequency",
                                "bootstrap_top_10_frequency",
                                "bootstrap_top_25_frequency",
                            ]
                            if column in candidate_stability.columns
                        ]
                    ],
                    on="backbone_id",
                )
            if not variant_consistency.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    variant_consistency[
                        [
                            column
                            for column in [
                                "backbone_id",
                                "variant_top_k_frequency",
                                "variant_top_10_frequency",
                                "variant_top_25_frequency",
                            ]
                            if column in variant_consistency.columns
                        ]
                    ],
                    on="backbone_id",
                )
            portfolio_risk = build_candidate_risk_table(candidate_portfolio)
            if not portfolio_risk.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    portfolio_risk[["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]],
                    on="backbone_id",
                )
            candidate_portfolio["in_consensus_top50"] = candidate_portfolio.get(
                "consensus_rank",
                pd.Series(np.nan, index=candidate_portfolio.index),
            ).notna()
            bootstrap_top10 = candidate_portfolio.get(
                "bootstrap_top_10_frequency",
                pd.Series(0.0, index=candidate_portfolio.index, dtype=float),
            ).fillna(0.0)
            risk_tier = candidate_portfolio.get(
                "false_positive_risk_tier",
                pd.Series("unknown", index=candidate_portfolio.index, dtype=object),
            ).fillna("unknown").astype(str)
            candidate_portfolio["recommended_monitoring_tier"] = np.select(
                [
                    (bootstrap_top10 >= 0.80) & risk_tier.isin(["low", "medium"]),
                    (bootstrap_top10 >= 0.50) & risk_tier.ne("high"),
                ],
                ["core_surveillance", "extended_watchlist"],
                default="low_confidence_backlog",
            )
            candidate_portfolio["evidence_tier"] = candidate_portfolio.get(
                "candidate_confidence_tier",
                pd.Series("unknown", index=candidate_portfolio.index),
            ).fillna("unknown")
            candidate_portfolio["action_tier"] = candidate_portfolio["recommended_monitoring_tier"]
        candidate_signature_context = build_candidate_signature_context(
            candidate_portfolio,
            module_f_identity,
            module_f_enrichment,
            q_threshold=0.05,
            max_signatures_per_candidate=5,
        )
        if not candidate_signature_context.empty:
            candidate_portfolio = coalescing_left_merge(candidate_portfolio, candidate_signature_context, on="backbone_id")
            candidate_dossiers = coalescing_left_merge(candidate_dossiers, candidate_signature_context, on="backbone_id")
        candidate_portfolio = _add_visibility_alias(candidate_portfolio)
        high_confidence_export = candidate_stability.loc[
            candidate_stability["high_confidence_candidate"].fillna(False)
        ].copy()
        if "spread_label" in high_confidence_export.columns:
            high_confidence_export = high_confidence_export.loc[
                high_confidence_export["spread_label"].fillna(0).astype(float) >= 1.0
            ].copy()
        if not candidate_risk.empty:
            high_confidence_export = coalescing_left_merge(
                high_confidence_export,
                candidate_risk[["backbone_id", "false_positive_risk_tier", "risk_flag_count", "risk_flags"]],
                on="backbone_id",
            )
            high_confidence_export = high_confidence_export.loc[
                high_confidence_export["false_positive_risk_tier"].fillna("high").ne("high")
            ].copy()
        high_confidence_export = _add_visibility_alias(high_confidence_export)

        candidate_universe = build_candidate_universe_table(
            scored=scored,
            consensus_candidates=consensus_candidates,
            candidate_dossiers=candidate_dossiers,
            candidate_portfolio=candidate_portfolio,
            novelty_watchlist=novelty_watchlist,
            prospective_freeze=prospective_freeze_export,
            high_confidence_candidates=high_confidence_export,
            candidate_risk=candidate_risk,
        )
        candidate_universe = _add_visibility_alias(candidate_universe)

        candidate_threshold_flip = build_threshold_flip_table(
            scored,
            candidate_ids=candidate_universe["backbone_id"].astype(str).tolist() if not candidate_universe.empty else None,
            default_threshold=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        )
        candidate_threshold_flip.to_csv(final_tables_dir / "candidate_threshold_flip.tsv", sep="\t", index=False)
        candidate_multiverse_stability = _build_candidate_multiverse_stability(
            candidate_stability,
            candidate_threshold_flip,
        )
        candidate_multiverse_stability.to_csv(final_tables_dir / "candidate_multiverse_stability.tsv", sep="\t", index=False)
        if not candidate_multiverse_stability.empty:
            merge_columns = [
                column
                for column in [
                    "backbone_id",
                    "multiverse_stability_score",
                    "multiverse_stability_tier",
                    "threshold_robustness_score",
                ]
                if column in candidate_multiverse_stability.columns
            ]
            if not candidate_portfolio.empty:
                candidate_portfolio = coalescing_left_merge(
                    candidate_portfolio,
                    candidate_multiverse_stability[merge_columns],
                    on="backbone_id",
                )
            if not candidate_dossiers.empty:
                candidate_dossiers = coalescing_left_merge(
                    candidate_dossiers,
                    candidate_multiverse_stability[merge_columns],
                    on="backbone_id",
                )
        exposure_merge_columns = [
            column
            for column in [
                "backbone_id",
                "new_country_rate_per_year",
                "weighted_new_country_rate_per_year",
                "rarity_weighted_new_country_rate_per_year",
                "first_event_speed_score",
                "third_event_speed_score",
                "fast_or_broad_expansion_label",
            ]
            if column in exposure_adjusted_event.columns
        ]
        if len(exposure_merge_columns) > 1:
            exposure_payload = exposure_adjusted_event[exposure_merge_columns].drop_duplicates("backbone_id")
            if not candidate_portfolio.empty:
                candidate_portfolio = coalescing_left_merge(candidate_portfolio, exposure_payload, on="backbone_id")
            if not candidate_dossiers.empty:
                candidate_dossiers = coalescing_left_merge(candidate_dossiers, exposure_payload, on="backbone_id")
        candidate_dossiers = annotate_candidate_explanation_fields(candidate_dossiers)
        candidate_portfolio = annotate_candidate_explanation_fields(candidate_portfolio)
        consensus_shortlist = build_consensus_shortlist(
            consensus_candidates,
            candidate_portfolio,
            candidate_multiverse_stability,
            top_k=25,
        )
        consensus_shortlist = _add_visibility_alias(consensus_shortlist)
        consensus_shortlist.to_csv(final_tables_dir / "consensus_shortlist.tsv", sep="\t", index=False)

        decision_yield = build_decision_yield_table(
            predictions,
            model_names=list(
                dict.fromkeys(
                    [
                        primary_model_name,
                        conservative_model_name,
                        "knownness_robust_priority",
                        "host_transfer_synergy_priority",
                        "threat_architecture_priority",
                        "natural_auc_priority",
                        "baseline_both",
                        "evidence_aware_priority",
                    ]
                )
            ),
        )
        decision_yield.to_csv(final_tables_dir / "decision_yield_summary.tsv", sep="\t", index=False)
        stable_adaptive_model_names = (
            gate_consistency_audit.loc[
                gate_consistency_audit.get("gate_consistency_tier", pd.Series(dtype=str)).astype(str).eq("stable"),
                "model_name",
            ].drop_duplicates()
            if not gate_consistency_audit.empty and "gate_consistency_tier" in gate_consistency_audit.columns
            else pd.Series(dtype=str)
        )
        preferred_adaptive_metrics = adaptive_gated_metrics.loc[
            adaptive_gated_metrics["model_name"].isin(stable_adaptive_model_names)
        ].sort_values("roc_auc", ascending=False)
        if not preferred_adaptive_metrics.empty:
            best_adaptive_model_name = str(preferred_adaptive_metrics.iloc[0]["model_name"])
        else:
            best_adaptive_model_name = (
                str(adaptive_gated_metrics.sort_values("roc_auc", ascending=False).iloc[0]["model_name"])
                if not adaptive_gated_metrics.empty
                else "adaptive_natural_priority"
            )
        decision_budget_curve = build_decision_yield_table(
            predictions,
            model_names=list(
                dict.fromkeys(
                    [
                        primary_model_name,
                        conservative_model_name,
                        "baseline_both",
                        "natural_auc_priority",
                        "threat_architecture_priority",
                        "contextual_bio_priority",
                        best_adaptive_model_name,
                    ]
                )
            ),
            top_ks=tuple(range(5, 105, 5)),
        )
        decision_budget_curve.to_csv(final_tables_dir / "decision_budget_curve.tsv", sep="\t", index=False)
        false_negative_audit = build_false_negative_audit(
            scored,
            predictions,
            primary_model_name=primary_model_name,
            metadata_quality=metadata_quality_summary,
            candidate_threshold_flip=candidate_threshold_flip,
            shortlist_cutoffs=(25, 50),
            top_n=50,
        )
        false_negative_audit.to_csv(final_tables_dir / "false_negative_audit.tsv", sep="\t", index=False)

        model_selection_scorecard = build_model_selection_scorecard(
            model_metrics,
            predictions,
            scored,
            knownness_matched_validation=knownness_matched_validation,
            group_holdout=group_holdout,
            model_names=model_metrics["model_name"].astype(str).tolist(),
        )
        model_selection_scorecard.to_csv(final_tables_dir / "model_selection_scorecard.tsv", sep="\t", index=False)

        model_selection_summary = build_primary_model_selection_summary(
            model_metrics,
            primary_model_name=primary_model_name,
            conservative_model_name=conservative_model_name,
            predictions=predictions,
            family_summary=family_summary,
            simplicity_summary=simplicity_summary,
        )
        model_selection_summary.to_csv(final_tables_dir / "model_selection_summary.tsv", sep="\t", index=False)
        benchmark_protocol = build_benchmark_protocol_table(
            model_metrics,
            model_selection_summary,
            adaptive_gated_metrics=adaptive_gated_metrics,
            gate_consistency_audit=gate_consistency_audit,
        )
        benchmark_protocol.to_csv(final_tables_dir / "benchmark_protocol.tsv", sep="\t", index=False)

        candidate_dossiers.to_csv(final_tables_dir / "candidate_dossiers.tsv", sep="\t", index=False)
        candidate_portfolio.to_csv(final_tables_dir / "candidate_portfolio.tsv", sep="\t", index=False)
        candidate_briefs = _build_candidate_brief_table(candidate_portfolio, backbones, amr_consensus)
        candidate_evidence_matrix = _build_candidate_evidence_matrix(
            candidate_portfolio,
            candidate_briefs,
            candidate_threshold_flip,
        )
        candidate_evidence_matrix.to_csv(final_tables_dir / "candidate_evidence_matrix.tsv", sep="\t", index=False)
        if not candidate_briefs.empty:
            candidate_stability = coalescing_left_merge(
                candidate_stability,
                candidate_briefs[["backbone_id", "dominant_genus", "dominant_species"]],
                on="backbone_id",
            )
        _prune_duplicate_table_artifacts(core_dir, diag_dir, final_tables_dir.core_files)
        _prune_shadowed_report_tables(core_dir, diag_dir, analysis_dir)
        _write_turkish_summary(
            turkish_summary_path,
            primary_model_name=primary_model_name,
            conservative_model_name=conservative_model_name,
            model_metrics=model_metrics,
            candidate_briefs=candidate_briefs,
            candidate_portfolio=candidate_portfolio,
            decision_yield=decision_yield,
            model_selection_scorecard=model_selection_scorecard,
            model_selection_summary=model_selection_summary,
            knownness_summary=knownness_summary,
            knownness_matched_validation=knownness_matched_validation,
            source_balance_resampling=source_balance_resampling,
            novelty_specialist_metrics=novelty_specialist_metrics,
            adaptive_gated_metrics=adaptive_gated_metrics,
            gate_consistency_audit=gate_consistency_audit,
            secondary_outcome_performance=secondary_outcome_performance,
            weighted_country_outcome=weighted_country_outcome,
            count_outcome_audit=count_outcome_audit,
            metadata_quality_summary=metadata_quality_summary,
            outcome_threshold=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        )
        legacy_detailed_summary = context.root / "reports/tubitak_detayli_proje_ozeti_tr.txt"
        if legacy_detailed_summary.exists():
            legacy_detailed_summary.unlink()

        figure_paths = generate_all_figures(
            scored=scored,
            predictions=predictions,
            calibration=calibration,
            model_metrics=model_metrics,
            coefficient_table=coefficient_table,
            coefficient_stability=coefficient_stability,
            dropout_table=dropout_table,
            candidate_stability=candidate_stability,
            figures_dir=figures_dir,
            primary_model_name=primary_model_name,
        )
        for figure_path in figure_paths:
            run.record_output(Path(figure_path))
        if not amrfinder_reportable:
            stale_figure = figures_dir / "amrfinder_concordance.png"
            if stale_figure.exists():
                stale_figure.unlink()

        _write_jury_brief(
            jury_brief_path,
            primary_model_name=primary_model_name,
            conservative_model_name=conservative_model_name,
            model_metrics=model_metrics,
            family_summary=family_summary,
            dropout_table=dropout_table,
            scored=scored,
            candidate_portfolio=candidate_portfolio,
            decision_yield=decision_yield,
            model_selection_scorecard=model_selection_scorecard,
            model_selection_summary=model_selection_summary,
            knownness_summary=knownness_summary,
            knownness_matched_validation=knownness_matched_validation,
            source_balance_resampling=source_balance_resampling,
            novelty_specialist_metrics=novelty_specialist_metrics,
            adaptive_gated_metrics=adaptive_gated_metrics,
            gate_consistency_audit=gate_consistency_audit,
            secondary_outcome_performance=secondary_outcome_performance,
            weighted_country_outcome=weighted_country_outcome,
            count_outcome_audit=count_outcome_audit,
            metadata_quality_summary=metadata_quality_summary,
            outcome_threshold=DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD,
        )
        run.set_rows_out("top_backbones_rows", int(len(top)))
        run.set_metric("figure_count", len(figure_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
