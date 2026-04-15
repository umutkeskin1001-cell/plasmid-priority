"""Feature taxonomy and admissibility helpers for the geo spread branch."""

from __future__ import annotations

from collections.abc import Sequence

GEO_SPREAD_FEATURE_CATEGORIES: dict[str, str] = {
    "log1p_member_count_train": "sampling_proxy",
    "log1p_n_countries_train": "sampling_proxy",
    "refseq_share_train": "sampling_proxy",
    "geo_country_record_count_train": "sampling_proxy",
    "geo_country_entropy_train": "geography_context",
    "geo_macro_region_entropy_train": "geography_context",
    "geo_dominant_region_share_train": "geography_context",
    "T_eff_norm": "intrinsic",
    "H_obs_specialization_norm": "intrinsic",
    "H_phylogenetic_specialization_norm": "intrinsic",
    "A_eff_norm": "intrinsic",
    "coherence_score": "intrinsic",
    "host_phylogenetic_dispersion_norm": "ecological",
    "host_taxon_evenness_norm": "ecological",
    "ecology_context_diversity_norm": "ecological",
    "backbone_purity_norm": "quality_proxy",
    "assignment_confidence_norm": "quality_proxy",
    "mash_neighbor_distance_train_norm": "quality_proxy",
    "orit_support": "support_proxy",
    "H_external_host_range_norm": "ecological",
    "clinical_context_fraction_norm": "context_proxy",
    "pathogenic_context_fraction_norm": "context_proxy",
    "support_shrinkage_norm": "support_proxy",
    "mash_graph_novelty_score": "novelty_proxy",
    "mash_graph_bridge_fraction": "novelty_proxy",
}

GEO_SPREAD_ALLOWED_FEATURES = frozenset(GEO_SPREAD_FEATURE_CATEGORIES)


def classify_geo_spread_feature(feature_name: str) -> str:
    """Return the admissibility class for a geo spread feature."""
    normalized = str(feature_name).strip()
    if not normalized:
        raise ValueError("feature_name cannot be empty")
    try:
        return GEO_SPREAD_FEATURE_CATEGORIES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unsupported geo spread feature: {normalized}") from exc


def validate_geo_spread_feature_set(
    features: Sequence[str], *, label: str = "geo spread feature set"
) -> None:
    """Reject geo feature sets that contain unsupported or empty features."""
    normalized = [str(feature).strip() for feature in features if str(feature).strip()]
    if not normalized:
        raise ValueError(f"{label} must contain at least one feature")
    unsupported = sorted(
        feature for feature in normalized if feature not in GEO_SPREAD_ALLOWED_FEATURES
    )
    if unsupported:
        joined = ", ".join(f"`{feature}`" for feature in unsupported)
        raise ValueError(f"{label} contains unsupported features: {joined}")
