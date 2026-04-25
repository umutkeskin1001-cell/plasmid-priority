"""Host (H) feature computation.

This module provides functions for computing host diversity-related features
for plasmid backbone analysis.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.config import PipelineSettings, build_context, find_project_root
from plasmid_priority.utils.dataframe import clean_text_series as _clean_text_series

HOST_TAXONOMY_LEVELS = [
    ("TAXONOMY_phylum", "phylum"),
    ("TAXONOMY_class", "class"),
    ("TAXONOMY_order", "order"),
    ("TAXONOMY_family", "family"),
    ("TAXONOMY_genus", "genus"),
]

_PAIRWISE_HOST_DISTANCE_BY_SHARED_LEVEL = {
    0: 0.0,  # shared phylum
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: 0.40,
    5: 1.0,  # no shared level
}

_MAX_HOST_SIGNATURES_FOR_PAIRWISE_DISTANCE = 100


def _pipeline_settings(settings: PipelineSettings | None = None) -> PipelineSettings:
    """Get pipeline settings, with optional injection for testing."""
    if settings is not None:
        return settings
    return build_context(find_project_root(Path(__file__).resolve())).pipeline_settings


def _series_or_default(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


@lru_cache(maxsize=256)
def _taxonomy_rank_lookup() -> pd.DataFrame:
    taxonomy_path = (
        find_project_root(Path(__file__).resolve())
        / "data"
        / "raw"
        / "plsdb_meta_tables"
        / "taxonomy.csv"
    )
    if not taxonomy_path.exists():
        return pd.DataFrame(
            columns=["taxonomy_uid", *[column for column, _ in HOST_TAXONOMY_LEVELS]],
        ).set_index("taxonomy_uid")
    lookup = pd.read_csv(
        taxonomy_path,
        usecols=["TAXONOMY_UID", *[column for column, _ in HOST_TAXONOMY_LEVELS]],
        low_memory=False,
    ).rename(columns={"TAXONOMY_UID": "taxonomy_uid"})
    lookup["taxonomy_uid"] = pd.to_numeric(lookup["taxonomy_uid"], errors="coerce").astype("Int64")
    lookup = lookup.dropna(subset=["taxonomy_uid"]).drop_duplicates(subset=["taxonomy_uid"])
    return lookup.set_index("taxonomy_uid")


def _normalized_taxon_identifier(values: pd.Series, fallback: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = pd.Series("", index=values.index, dtype=object)
    valid_numeric = numeric.notna() & (numeric > 0)
    if bool(valid_numeric.any()):
        normalized.loc[valid_numeric] = numeric.loc[valid_numeric].round().astype(int).astype(str)
    fallback_clean = _clean_text_series(fallback)
    fallback_mask = normalized.eq("") & fallback_clean.ne("")
    if bool(fallback_mask.any()):
        normalized.loc[fallback_mask] = "name:" + fallback_clean.loc[fallback_mask]
    return normalized


def _host_taxonomy_signature_series(records: pd.DataFrame) -> pd.Series:
    taxonomy_lookup = _taxonomy_rank_lookup()
    taxonomy_uid = pd.to_numeric(
        _series_or_default(records, "taxonomy_uid", np.nan),
        errors="coerce",
    ).astype("Int64")
    signatures: list[pd.Series] = []
    for id_column, fallback_column in HOST_TAXONOMY_LEVELS:
        direct_values = _series_or_default(records, id_column, np.nan)
        if id_column in records.columns:
            mapped_values = direct_values
        elif taxonomy_lookup.empty or taxonomy_uid.isna().all():
            mapped_values = pd.Series(np.nan, index=records.index, dtype=float)
        else:
            mapped_values = taxonomy_uid.map(taxonomy_lookup[id_column])
        fallback_values = _series_or_default(records, fallback_column, "")
        signatures.append(_normalized_taxon_identifier(mapped_values, fallback_values))
    tuples = list(zip(*(series.to_list() for series in signatures), strict=False))
    return pd.Series(tuples, index=records.index, dtype=object)


def _host_signature_is_nonempty(signature: tuple[str, ...]) -> bool:
    return any(str(token).strip() for token in signature)


def _pairwise_host_taxonomy_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not _host_signature_is_nonempty(left) or not _host_signature_is_nonempty(right):
        return 0.0
    for level_index in range(len(left) - 1, -1, -1):
        left_value = str(left[level_index]).strip()
        right_value = str(right[level_index]).strip()
        if left_value and right_value and left_value == right_value:
            return float(_PAIRWISE_HOST_DISTANCE_BY_SHARED_LEVEL[level_index])
    return float(_PAIRWISE_HOST_DISTANCE_BY_SHARED_LEVEL[-1])


def _subsample_signatures_for_pairwise_distance(
    signatures: list[tuple[str, ...]],
    *,
    max_signatures: int = _MAX_HOST_SIGNATURES_FOR_PAIRWISE_DISTANCE,
) -> list[tuple[str, ...]]:
    if len(signatures) <= max_signatures:
        return signatures
    indices = np.linspace(0, len(signatures) - 1, num=max_signatures, dtype=int)
    return [signatures[int(index)] for index in indices]


def _mean_pairwise_host_taxonomy_distance(signatures: list[tuple[str, ...]]) -> float:
    unique_signatures = [
        signature
        for signature in dict.fromkeys(signatures)
        if _host_signature_is_nonempty(signature)
    ]
    if len(unique_signatures) < 2:
        return 0.0
    subsampled = _subsample_signatures_for_pairwise_distance(unique_signatures)
    distances = []
    for i in range(len(subsampled)):
        for j in range(i + 1, len(subsampled)):
            distances.append(_pairwise_host_taxonomy_distance(subsampled[i], subsampled[j]))
    return float(sum(distances) / len(distances)) if distances else 0.0


def _normalized_shannon_evenness(values: list[object]) -> float:
    if not values:
        return 0.0
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return 0.0
    unique = dict.fromkeys(cleaned)
    if len(unique) <= 1:
        return 0.0
    counts = [cleaned.count(item) for item in unique]
    total = sum(counts)
    if total <= 0:
        return 0.0
    proportions = [count / total for count in counts if count > 0]
    log_proportions = [math.log(p) for p in proportions if p > 0]
    shannon = -sum(p * lp for p, lp in zip(proportions, log_proportions))
    log_richness = math.log(len(unique))
    if log_richness <= 0:
        return 0.0
    return float(shannon / log_richness)


def _rank_score_series(values: pd.Series) -> pd.Series:
    HOST_RANGE_RANK_SCORES = {
        "strain": 0.05,
        "species": 0.10,
        "genus": 0.20,
        "family": 0.40,
        "order": 0.60,
        "class": 0.80,
        "phylum": 1.00,
        "multi-phylla": 1.00,
    }
    normalized = _clean_text_series(values).str.lower()
    scores = normalized.map(lambda x: HOST_RANGE_RANK_SCORES.get(x, 0.0))
    return scores.astype(float)


def _global_max_normalized_richness(unique_counts: pd.Series) -> pd.Series:
    if unique_counts.empty:
        return unique_counts
    max_count = float(pd.to_numeric(unique_counts, errors="coerce").max())
    if max_count <= 0.0:
        return unique_counts.astype(float)
    return unique_counts.astype(float) / max_count


def _menhinick_normalized_richness(
    unique_counts: pd.Series,
    observation_counts: pd.Series,
) -> pd.Series:
    result: pd.Series = unique_counts.astype(float) / np.sqrt(observation_counts.astype(float))
    max_result: float = float(pd.to_numeric(result, errors="coerce").max())
    if max_result > 0.0:
        result = result / max_result
    return result.fillna(0.0)


def _training_period_records(
    records: pd.DataFrame,
    *,
    split_year: int,
    label: str,
) -> pd.DataFrame:
    year_values = pd.to_numeric(
        records.get("year", pd.Series(0, index=records.index)),
        errors="coerce",
    )
    return records.loc[year_values < split_year].copy()


def compute_feature_h(
    records: pd.DataFrame,
    *,
    split_year: int | None = None,
    host_evenness_bias_power: float | None = None,
) -> pd.DataFrame:
    """Compute observed host diversity from training-period observations.

    Note: This is a placeholder. The full implementation is in features/core.py.
    Import from core to use the complete implementation.
    """
    raise NotImplementedError(
        "compute_feature_h is implemented in features/core.py. "
        "Use: from plasmid_priority.features import compute_feature_h"
    )
