"""Candidate and robustness helper functions extracted from scripts/24_build_reports.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from plasmid_priority.reporting.external_support import normalize_drug_class_token


def humanize_taxon_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("_", " ")


def humanize_source_tier(value: object) -> str:
    mapping = {
        "cross_source_supported": "cross-source supported",
        "refseq_dominant": "RefSeq-dominant",
        "insd_dominant": "INSD-dominant",
        "source_mixed": "mixed-source",
        "source_sparse": "source-sparse",
    }
    token = str(value or "").strip()
    return mapping.get(token, token.replace("_", " "))


def humanize_evidence_tier(value: object, *, language: str = "en") -> str:
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


def humanize_action_tier(value: object, *, language: str = "en") -> str:
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


def humanize_portfolio_track(value: object, *, language: str = "en") -> str:
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


def clean_signature_text(text: str) -> str:
    cleaned = str(text or "")
    return cleaned.replace("OR=inf", "OR>100 (seyrek payda)")


def humanize_candidate_reason(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    labels = {
        "mobility_coherence_t": "mobility coherence (T-axis support)",
    }
    return ", ".join(
        labels.get(part.strip(), part.strip().replace("_", " "))
        for part in text.split(",")
        if part.strip()
    )


def build_outcome_robustness_grid(
    sensitivity: dict[str, dict[str, float]],
    rolling_temporal: pd.DataFrame,
    *,
    default_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not rolling_temporal.empty:
        rows_for_status = rolling_temporal.loc[rolling_temporal["status"] == "ok"].to_dict(
            orient="records"
        )
        for rolling_row in rows_for_status:
            rows.append(
                {
                    "scenario_group": "rolling_temporal",
                    "scenario_name": _rolling_scenario_name(rolling_row),
                    "split_year": int(rolling_row["split_year"]),
                    "test_year_end": int(rolling_row["test_year_end"]),
                    "horizon_years": int(
                        rolling_row.get(
                            "horizon_years",
                            max(
                                int(rolling_row["test_year_end"]) - int(rolling_row["split_year"]),
                                1,
                            ),
                        )
                    ),
                    "backbone_assignment_mode": str(rolling_row["backbone_assignment_mode"]),
                    "new_country_threshold": int(default_threshold),
                    "roc_auc": float(rolling_row["roc_auc"]),
                    "average_precision": float(rolling_row["average_precision"]),
                    "positive_prevalence": float(rolling_row.get("positive_prevalence", np.nan)),
                    "average_precision_lift": float(
                        rolling_row.get("average_precision_lift", np.nan)
                    ),
                    "average_precision_enrichment": float(
                        rolling_row.get("average_precision_enrichment", np.nan)
                    ),
                    "brier_score": float(rolling_row["brier_score"]),
                    "n_eligible_backbones": int(rolling_row["n_eligible_backbones"]),
                    "n_positive": int(rolling_row.get("n_positive", 0)),
                }
            )

    selected_variants = {
        "alternate_split_2014": {
            "scenario_group": "alternate_split",
            "split_year": 2014,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
        "alternate_split_2016": {
            "scenario_group": "alternate_split",
            "split_year": 2016,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
        "expanded_eligibility_ge_1": {
            "scenario_group": "expanded_eligibility",
            "split_year": 2015,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
        "stable_country_outcome": {
            "scenario_group": "country_stability",
            "split_year": 2015,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
        "stable_dense_country_outcome": {
            "scenario_group": "country_stability",
            "split_year": 2015,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
        "training_only_backbone_rerun": {
            "scenario_group": "backbone_assignment",
            "split_year": 2015,
            "test_year_end": 2023,
            "backbone_assignment_mode": "training_only",
            "new_country_threshold": int(default_threshold),
        },
        "fallback_backbone_rerun": {
            "scenario_group": "backbone_assignment",
            "split_year": 2015,
            "test_year_end": 2023,
            "backbone_assignment_mode": "fallback",
            "new_country_threshold": int(default_threshold),
        },
        "source_balanced_rerun": {
            "scenario_group": "cohort_balance",
            "split_year": 2015,
            "test_year_end": 2023,
            "new_country_threshold": int(default_threshold),
        },
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
        variant_row: dict[str, object] = {
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
            "average_precision_enrichment": float(
                metrics.get("average_precision_enrichment", np.nan)
            ),
            "brier_score": float(metrics["brier_score"]),
            "n_eligible_backbones": int(metrics["n_eligible_backbones"]),
            "n_positive": int(metrics.get("n_positive", 0)),
        }
        rows.append(variant_row)

    return (
        pd.DataFrame(rows)
        .sort_values(["scenario_group", "split_year", "scenario_name"])
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def _mapping_int(row: Mapping[Any, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mapping_str(row: Mapping[Any, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(value if value is not None else default)


def _rolling_scenario_name(row: Mapping[Any, Any]) -> str:
    split_year = _mapping_int(row, "split_year", 0)
    test_year_end = _mapping_int(row, "test_year_end", split_year + 1)
    assignment_mode = _mapping_str(row, "backbone_assignment_mode", "all_records")
    fallback_horizon = max(test_year_end - split_year, 1)
    horizon_years = _mapping_int(row, "horizon_years", fallback_horizon)
    return f"rolling_split_{split_year}_{assignment_mode}_{horizon_years}y"


def build_threshold_sensitivity_table(
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
                "roc_auc_ci_lower": float(
                    default_metrics.get("roc_auc_ci_lower", default_metrics["roc_auc"])
                ),
                "roc_auc_ci_upper": float(
                    default_metrics.get("roc_auc_ci_upper", default_metrics["roc_auc"])
                ),
                "average_precision": float(default_metrics["average_precision"]),
                "average_precision_lift": float(
                    default_metrics.get("average_precision_lift", np.nan)
                ),
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
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["new_country_threshold"])
        .sort_values("new_country_threshold")
        .reset_index(drop=True)
    )


def build_l2_sensitivity_table(sensitivity: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, metrics in sensitivity.items():
        if not str(variant).startswith("primary_l2_") or metrics.get("skipped"):
            continue
        rows.append(
            {
                "variant": str(variant),
                "l2_penalty": float(
                    metrics.get("l2", str(variant).replace("primary_l2_", "").replace("p", "."))
                ),
                "sample_weight_mode": str(metrics.get("sample_weight_mode", "")),
                "roc_auc": float(metrics["roc_auc"]),
                "roc_auc_ci_lower": float(metrics.get("roc_auc_ci_lower", metrics["roc_auc"])),
                "roc_auc_ci_upper": float(metrics.get("roc_auc_ci_upper", metrics["roc_auc"])),
                "average_precision": float(metrics["average_precision"]),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
                "brier_score": float(metrics.get("brier_score", np.nan)),
                "precision_at_top_25": float(metrics.get("precision_at_top_25", np.nan)),
                "recall_at_top_25": float(metrics.get("recall_at_top_25", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("l2_penalty").reset_index(drop=True)


def build_weighting_sensitivity_table(sensitivity: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in (
        "default",
        "class_plus_knownness_balanced_primary",
        "knownness_balanced_primary",
        "source_plus_class_balanced_primary",
        "source_balanced_rerun",
    ):
        metrics = sensitivity.get(variant, {})
        if not metrics or metrics.get("skipped"):
            continue
        rows.append(
            {
                "variant": variant,
                "sample_weight_mode": str(metrics.get("sample_weight_mode", "")),
                "roc_auc": float(metrics["roc_auc"]),
                "average_precision": float(metrics["average_precision"]),
                "average_precision_lift": float(metrics.get("average_precision_lift", np.nan)),
                "brier_score": float(metrics.get("brier_score", np.nan)),
                "precision_at_top_25": float(metrics.get("precision_at_top_25", np.nan)),
                "recall_at_top_25": float(metrics.get("recall_at_top_25", np.nan)),
                "positive_prevalence": float(metrics.get("positive_prevalence", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def select_predefined_group_holdout_highlight(
    group_holdout: pd.DataFrame,
) -> dict[str, object] | None:
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
    top_row = working.sort_values(["roc_auc", "n_test_backbones"], ascending=[False, False]).iloc[0]
    return {str(key): value for key, value in top_row.to_dict().items()}


def dominant_non_empty(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values.loc[values != ""]
    if values.empty:
        return ""
    value_counts = values.value_counts()
    if value_counts.empty:
        return ""
    return str(value_counts.index[0])


def top_tokens(series: pd.Series, *, n: int = 5) -> str:
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


def top_public_health_amr_classes(series: pd.Series, *, n: int = 5) -> str:
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


def top_likely_amr_genes(series: pd.Series, *, n: int = 5) -> str:
    counts: dict[str, int] = {}
    for value in series.fillna("").astype(str):
        for token in [part.strip() for part in value.split(",") if part.strip()]:
            normalized = "".join(ch for ch in token.lower() if ch.isalnum())
            if normalized.startswith(_LIKELY_AMR_GENE_PREFIXES):
                counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(token for token, _ in ordered[:n])


def build_backbone_assignment_summary(
    backbones: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    if backbones.empty:
        return pd.DataFrame()
    assignment = backbones.groupby("backbone_assignment_rule", as_index=False).agg(
        n_records=("sequence_accession", "nunique"),
        n_backbones=("backbone_id", "nunique"),
    )
    scored_with_rule = (
        scored.merge(
            backbones[["backbone_id", "backbone_assignment_rule"]].drop_duplicates(),
            on="backbone_id",
            how="left",
            validate="m:1",
        )
        if not scored.empty
        else pd.DataFrame()
    )
    if not scored_with_rule.empty:
        metrics = scored_with_rule.groupby("backbone_assignment_rule", as_index=False).agg(
            n_scored_backbones=("backbone_id", "nunique"),
            n_outcome_eligible=(
                "spread_label",
                lambda values: int(pd.Series(values).notna().sum()),
            ),
            mean_coherence=("coherence_score", "mean"),
            mean_bio_priority_index=("bio_priority_index", "mean"),
            mean_evidence_support_index=("evidence_support_index", "mean"),
        )
        assignment = assignment.merge(metrics, on="backbone_assignment_rule", how="left")
    return assignment.sort_values(
        ["n_backbones", "n_records"], ascending=[False, False]
    ).reset_index(drop=True)
