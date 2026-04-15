"""Feature taxonomy and builders for the clinical hazard branch."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from plasmid_priority.shared.contracts import validate_branch_feature_set

CLINICAL_HAZARD_FEATURE_CATEGORIES: dict[str, str] = {
    "A_eff_norm": "amr",
    "A_raw_norm": "amr",
    "A_recurrence_norm": "amr",
    "amr_gene_count_norm": "amr",
    "amr_class_count_norm": "amr",
    "amr_mechanism_diversity_norm": "amr",
    "clinical_context_fraction_norm": "context",
    "pathogenic_context_fraction_norm": "context",
    "mdr_proxy_fraction_norm": "escalation",
    "xdr_proxy_fraction_norm": "escalation",
    "pd_clinical_support_norm": "context",
    "vfdb_cargo_burden_norm": "cargo",
    "stress_response_burden_norm": "cargo",
    "last_resort_convergence_norm": "escalation",
    "amr_clinical_escalation_norm": "escalation",
    "backbone_purity_norm": "quality",
    "assignment_confidence_norm": "quality",
    "mash_neighbor_distance_train_norm": "novelty",
    "amr_support_norm": "support",
    "amr_support_norm_residual": "support",
    "metadata_support_depth_norm": "support",
    "context_support_guard_norm": "support",
    "H_external_host_range_norm": "host",
    "A_clinical_context_synergy_norm": "interaction",
    "A_host_range_synergy_norm": "interaction",
    "A_novelty_synergy_norm": "interaction",
    "amr_load_density_norm": "amr",
    "amr_mdr_proxy_norm": "escalation",
    "amr_xdr_proxy_norm": "escalation",
    "silent_carrier_risk_norm": "risk",
    "evolutionary_jump_score_norm": "risk",
    "coherence_score": "architecture",
    "orit_support": "support",
}

CLINICAL_HAZARD_ALLOWED_FEATURES = frozenset(CLINICAL_HAZARD_FEATURE_CATEGORIES)


def classify_clinical_hazard_feature(feature_name: str) -> str:
    normalized = str(feature_name).strip()
    if not normalized:
        raise ValueError("feature_name cannot be empty")
    try:
        return CLINICAL_HAZARD_FEATURE_CATEGORIES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unsupported clinical hazard feature: {normalized}") from exc


def validate_clinical_hazard_feature_set(
    features: Sequence[str], *, label: str = "clinical hazard feature set"
) -> None:
    validate_branch_feature_set(features, label=label)
    normalized = [str(feature).strip() for feature in features if str(feature).strip()]
    unsupported = sorted(
        feature for feature in normalized if feature not in CLINICAL_HAZARD_ALLOWED_FEATURES
    )
    if unsupported:
        joined = ", ".join(f"`{feature}`" for feature in unsupported)
        raise ValueError(f"{label} contains unsupported features: {joined}")


def _numeric_series(
    frame: pd.DataFrame, candidates: Sequence[str], default: float = 0.0
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def build_clinical_hazard_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build split-safe features for the clinical hazard branch."""
    working = frame.copy()
    working["amr_gene_count_norm"] = _numeric_series(
        working, ("amr_gene_count_norm", "amr_gene_count", "amr_hit_count")
    )
    working["amr_class_count_norm"] = _numeric_series(
        working, ("amr_class_count_norm", "amr_class_count")
    )
    working["amr_mechanism_diversity_norm"] = _numeric_series(
        working,
        ("amr_mechanism_diversity_norm", "amr_mechanism_diversity", "mechanism_diversity_norm"),
    )
    working["clinical_context_fraction_norm"] = _numeric_series(
        working,
        ("clinical_context_fraction_norm", "clinical_context_fraction"),
    )
    working["pathogenic_context_fraction_norm"] = _numeric_series(
        working,
        ("pathogenic_context_fraction_norm", "pathogenic_context_fraction"),
    )
    working["mdr_proxy_fraction_norm"] = _numeric_series(
        working,
        ("mdr_proxy_fraction_norm", "mdr_proxy_fraction"),
    )
    working["xdr_proxy_fraction_norm"] = _numeric_series(
        working,
        ("xdr_proxy_fraction_norm", "xdr_proxy_fraction"),
    )
    working["pd_clinical_support_norm"] = _numeric_series(
        working,
        ("pd_clinical_support_norm", "pd_clinical_support"),
    )
    working["vfdb_cargo_burden_norm"] = _numeric_series(
        working,
        ("vfdb_cargo_burden_norm", "vfdb_cargo_burden", "virulence_burden_norm"),
    )
    working["stress_response_burden_norm"] = _numeric_series(
        working,
        ("stress_response_burden_norm", "stress_response_burden"),
    )
    working["last_resort_convergence_norm"] = _numeric_series(
        working,
        ("last_resort_convergence_norm", "last_resort_convergence"),
    )
    working["amr_clinical_escalation_norm"] = _numeric_series(
        working,
        ("amr_clinical_escalation_norm", "amr_clinical_escalation"),
    )
    working["amr_mdr_proxy_norm"] = _numeric_series(
        working,
        ("amr_mdr_proxy_norm", "mdr_proxy_fraction_norm", "mdr_proxy_fraction"),
    )
    working["amr_xdr_proxy_norm"] = _numeric_series(
        working,
        ("amr_xdr_proxy_norm", "xdr_proxy_fraction_norm", "xdr_proxy_fraction"),
    )
    working["amr_load_density_norm"] = _numeric_series(
        working,
        ("amr_load_density_norm", "amr_load_density"),
    )
    working["silent_carrier_risk_norm"] = _numeric_series(
        working,
        ("silent_carrier_risk_norm", "silent_carrier_risk"),
    )
    working["evolutionary_jump_score_norm"] = _numeric_series(
        working,
        ("evolutionary_jump_score_norm", "evolutionary_jump_score"),
    )
    working["A_clinical_context_synergy_norm"] = _numeric_series(
        working, ("A_eff_norm",), default=0.0
    ).fillna(0.0) * working["clinical_context_fraction_norm"].fillna(0.0)
    working["A_host_range_synergy_norm"] = _numeric_series(
        working, ("A_eff_norm",), default=0.0
    ).fillna(0.0) * _numeric_series(working, ("H_external_host_range_norm",), default=0.0).fillna(
        0.0
    )
    working["A_novelty_synergy_norm"] = _numeric_series(
        working, ("A_eff_norm",), default=0.0
    ).fillna(0.0) * _numeric_series(
        working, ("mash_neighbor_distance_train_norm",), default=0.0
    ).fillna(0.0)
    working["backbone_purity_norm"] = _numeric_series(
        working, ("backbone_purity_norm",), default=0.0
    )
    working["assignment_confidence_norm"] = _numeric_series(
        working, ("assignment_confidence_norm",), default=0.0
    )
    working["mash_neighbor_distance_train_norm"] = _numeric_series(
        working,
        ("mash_neighbor_distance_train_norm",),
    )
    working["amr_support_norm"] = _numeric_series(working, ("amr_support_norm",), default=0.0)
    working["amr_support_norm_residual"] = _numeric_series(
        working, ("amr_support_norm_residual",), default=0.0
    )
    working["metadata_support_depth_norm"] = _numeric_series(
        working, ("metadata_support_depth_norm",), default=0.0
    )
    working["context_support_guard_norm"] = _numeric_series(
        working, ("context_support_guard_norm",), default=0.0
    )
    working["H_external_host_range_norm"] = _numeric_series(
        working, ("H_external_host_range_norm",), default=0.0
    )
    working["coherence_score"] = _numeric_series(working, ("coherence_score",), default=0.0)
    working["orit_support"] = _numeric_series(working, ("orit_support",), default=0.0)
    for column in CLINICAL_HAZARD_ALLOWED_FEATURES:
        if column not in working.columns:
            working[column] = 0.0
    return working
