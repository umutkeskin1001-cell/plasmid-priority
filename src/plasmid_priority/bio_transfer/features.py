"""Feature taxonomy and builders for the bio transfer branch."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from plasmid_priority.shared.contracts import validate_branch_feature_set

BIO_TRANSFER_FEATURE_CATEGORIES: dict[str, str] = {
    "log1p_member_count_train": "baseline",
    "log1p_n_countries_train": "baseline",
    "orit_support": "mobility",
    "mobility_support_norm": "mobility",
    "mob_suite_support_norm": "mobility",
    "relaxase_support_norm": "mobility",
    "conjugation_support_norm": "mobility",
    "replicon_complexity_norm": "architecture",
    "backbone_purity_norm": "architecture",
    "assignment_confidence_norm": "architecture",
    "T_eff_norm": "transfer",
    "H_obs_specialization_norm": "host",
    "H_phylogenetic_specialization_norm": "host",
    "A_eff_norm": "cargo",
    "coherence_score": "transfer",
    "host_phylogenetic_dispersion_norm": "host",
    "host_taxon_evenness_norm": "host",
    "ecology_context_diversity_norm": "context",
    "H_external_host_range_norm": "host",
    "mash_neighbor_distance_train_norm": "novelty",
    "mash_graph_novelty_score": "novelty",
    "mash_graph_bridge_fraction": "novelty",
    "host_range_breadth_norm": "host",
    "host_breadth_mobility_synergy_norm": "interaction",
    "mobility_host_synergy_norm": "interaction",
    "T_H_obs_synergy_norm": "interaction",
    "T_A_synergy_norm": "interaction",
}

BIO_TRANSFER_ALLOWED_FEATURES = frozenset(BIO_TRANSFER_FEATURE_CATEGORIES)


def classify_bio_transfer_feature(feature_name: str) -> str:
    normalized = str(feature_name).strip()
    if not normalized:
        raise ValueError("feature_name cannot be empty")
    try:
        return BIO_TRANSFER_FEATURE_CATEGORIES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unsupported bio transfer feature: {normalized}") from exc


def validate_bio_transfer_feature_set(
    features: Sequence[str], *, label: str = "bio transfer feature set"
) -> None:
    validate_branch_feature_set(features, label=label)
    normalized = [str(feature).strip() for feature in features if str(feature).strip()]
    unsupported = sorted(
        feature for feature in normalized if feature not in BIO_TRANSFER_ALLOWED_FEATURES
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


def _presence_series(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            series = frame[column]
            if pd.api.types.is_bool_dtype(series):
                return series.fillna(False).astype(bool)
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                return numeric.fillna(0.0) > 0.0
            text = series.fillna("").astype(str).str.strip().str.lower()
            return text.isin({"true", "t", "1", "yes", "y"})
    return pd.Series(False, index=frame.index, dtype=bool)


def _synthesise_support(frame: pd.DataFrame, target: str, sources: Sequence[str]) -> pd.Series:
    if target in frame.columns:
        return pd.to_numeric(frame[target], errors="coerce").fillna(0.0)
    return _numeric_series(frame, sources, default=0.0)


def build_bio_transfer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build split-safe features for the bio transfer branch."""
    working = frame.copy()
    working["mobility_support_norm"] = _synthesise_support(
        working,
        "mobility_support_norm",
        ("orit_support", "mobility_support", "mobility_score"),
    )
    working["mob_suite_support_norm"] = _synthesise_support(
        working,
        "mob_suite_support_norm",
        ("mobility_support_norm", "mob_suite_support", "mob_suite_score"),
    )
    working["relaxase_support_norm"] = _synthesise_support(
        working,
        "relaxase_support_norm",
        ("mobility_support_norm", "relaxase_support", "relaxase_score"),
    )
    working["conjugation_support_norm"] = _synthesise_support(
        working,
        "conjugation_support_norm",
        ("mobility_support_norm", "conjugation_support", "conjugation_score"),
    )
    working["replicon_complexity_norm"] = _synthesise_support(
        working,
        "replicon_complexity_norm",
        (
            "mean_n_replicon_types_norm",
            "plasmidfinder_complexity_norm",
            "replicon_architecture_norm",
        ),
    )
    working["host_range_breadth_norm"] = _synthesise_support(
        working,
        "host_range_breadth_norm",
        (
            "H_external_host_range_norm",
            "H_external_host_range_support",
            "host_phylogenetic_dispersion_norm",
        ),
    )
    working["host_breadth_mobility_synergy_norm"] = working["host_range_breadth_norm"].fillna(
        0.0
    ) * working["mobility_support_norm"].fillna(0.0)
    working["mobility_host_synergy_norm"] = _numeric_series(
        working, ("T_eff_norm",), default=0.0
    ).fillna(0.0) * working["host_range_breadth_norm"].fillna(0.0)
    working["T_H_obs_synergy_norm"] = _synthesise_support(
        working,
        "T_H_obs_synergy_norm",
        ("T_eff_norm", "T_H_obs_synergy_norm"),
    )
    working["T_A_synergy_norm"] = _synthesise_support(
        working,
        "T_A_synergy_norm",
        ("T_A_synergy_norm", "T_eff_norm"),
    )
    for column in BIO_TRANSFER_ALLOWED_FEATURES:
        if column not in working.columns:
            working[column] = 0.0
    return working
