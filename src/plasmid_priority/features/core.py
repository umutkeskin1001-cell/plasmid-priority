"""Backbone-level feature engineering for T, H, and A components."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.config import find_project_root
from plasmid_priority.utils.math import geometric_mean as _geometric_mean


def _support_factor(n: int, pseudocount: float = 3.0) -> float:
    return float(n / (n + pseudocount)) if n > 0 else 0.0


def _split_values(cell: object) -> set[str]:
    if pd.isna(cell):
        return set()
    text = str(cell).strip()
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


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

CLINICAL_CONTEXT_TERMS = (
    "clinical",
    "patient",
    "hospital",
    "infection",
    "sepsis",
    "urine",
    "blood",
    "pathogen",
)
ENVIRONMENTAL_CONTEXT_TERMS = (
    "environment",
    "wastewater",
    "waste water",
    "river",
    "lake",
    "soil",
    "sewage",
    "sediment",
    "effluent",
    "surface water",
)
HOST_ASSOCIATED_CONTEXT_TERMS = (
    "host-associated",
    "host associated",
    "human",
    "animal",
    "feces",
    "faeces",
    "gut",
    "intestinal",
    "rectal",
    "nasal",
)
FOOD_CONTEXT_TERMS = (
    "food",
    "meat",
    "milk",
    "cheese",
    "beef",
    "chicken",
    "fish",
    "seafood",
    "vegetable",
    "produce",
)

HOST_TAXONOMY_LEVELS: tuple[tuple[str, str], ...] = (
    ("TAXONOMY_phylum_id", "TAXONOMY_phylum"),
    ("TAXONOMY_class_id", "TAXONOMY_class"),
    ("TAXONOMY_order_id", "TAXONOMY_order"),
    ("TAXONOMY_family_id", "TAXONOMY_family"),
    ("TAXONOMY_genus_id", "genus"),
    ("TAXONOMY_species_id", "species"),
)

_PAIRWISE_HOST_DISTANCE_BY_SHARED_LEVEL = {
    5: 0.0,
    4: 0.20,
    3: 0.40,
    2: 0.60,
    1: 0.80,
    0: 0.90,
    -1: 1.0,
}


def _series_or_default(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


@lru_cache(maxsize=1)
def _taxonomy_rank_lookup() -> pd.DataFrame:
    taxonomy_path = find_project_root(Path(__file__).resolve()) / "data" / "raw" / "plsdb_meta_tables" / "taxonomy.csv"
    if not taxonomy_path.exists():
        return pd.DataFrame(columns=["taxonomy_uid", *[column for column, _ in HOST_TAXONOMY_LEVELS]]).set_index("taxonomy_uid")
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
    taxonomy_uid = pd.to_numeric(_series_or_default(records, "taxonomy_uid", np.nan), errors="coerce").astype("Int64")
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


def _mean_pairwise_host_taxonomy_distance(signatures: list[tuple[str, ...]]) -> float:
    unique_signatures = [signature for signature in dict.fromkeys(signatures) if _host_signature_is_nonempty(signature)]
    if len(unique_signatures) < 2:
        return 0.0
    distances: list[float] = []
    for left_index in range(len(unique_signatures) - 1):
        for right_index in range(left_index + 1, len(unique_signatures)):
            distances.append(
                _pairwise_host_taxonomy_distance(
                    unique_signatures[left_index],
                    unique_signatures[right_index],
                )
            )
    if not distances:
        return 0.0
    return float(np.mean(distances))


def _normalized_shannon_evenness(values: list[object]) -> float:
    filtered = [value for value in values if str(value).strip()]
    if len(filtered) < 2:
        return 0.0
    counts: dict[object, int] = {}
    for value in filtered:
        counts[value] = counts.get(value, 0) + 1
    n_unique = len(counts)
    if n_unique < 2:
        return 0.0
    total = float(sum(counts.values()))
    proportions = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = float(-(proportions * np.log(proportions)).sum())
    max_entropy = float(np.log(n_unique))
    if not np.isfinite(max_entropy) or max_entropy <= 0.0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _rank_score_series(values: pd.Series) -> pd.Series:
    cleaned = _clean_text_series(values).str.lower()
    return cleaned.map(HOST_RANGE_RANK_SCORES).fillna(0.0).astype(float)


def _pmid_count(cell: object) -> int:
    values = _split_values(cell)
    return int(sum(token.isdigit() for token in values))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def _context_label_from_row(row: pd.Series) -> str:
    package = str(row.get("BIOSAMPLE_package", "") or "")
    title = str(row.get("BIOSAMPLE_title", "") or "")
    ecosystem = str(row.get("ECOSYSTEM_tags", "") or "")
    disease = str(row.get("DISEASE_tags", "") or "")
    pathogenicity = str(row.get("BIOSAMPLE_pathogenicity", "") or "")
    combined = " | ".join([package, title, ecosystem, disease, pathogenicity]).strip()
    if not combined:
        return "unknown"
    lowered = combined.lower()
    if _contains_any(lowered, CLINICAL_CONTEXT_TERMS) or pathogenicity.strip():
        return "clinical"
    if _contains_any(lowered, FOOD_CONTEXT_TERMS):
        return "food"
    if _contains_any(lowered, ENVIRONMENTAL_CONTEXT_TERMS):
        return "environmental"
    if _contains_any(lowered, HOST_ASSOCIATED_CONTEXT_TERMS):
        return "host_associated"
    return "other"


def _dominant_share(series: pd.Series) -> float:
    values = _clean_text_series(series)
    values = values.loc[values.ne("")]
    if values.empty:
        return 0.0
    return float(values.value_counts(normalize=True).iloc[0])


def _grouped_dominant_share(backbone_ids: pd.Series, values: pd.Series) -> pd.Series:
    cleaned = _clean_text_series(values)
    mask = cleaned.ne("")
    if not bool(mask.any()):
        return pd.Series(dtype=float)
    counts = (
        pd.DataFrame(
            {
                "backbone_id": backbone_ids.loc[mask].astype(str).to_numpy(),
                "value": cleaned.loc[mask].to_numpy(),
            }
        )
        .groupby(["backbone_id", "value"], sort=False)
        .size()
    )
    totals = counts.groupby(level=0, sort=False).sum()
    maxima = counts.groupby(level=0, sort=False).max()
    return (maxima / totals).astype(float)


def _lowered_series_contains_any(lowered: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(False, index=lowered.index, dtype=bool)
    for term in terms:
        mask = mask | lowered.str.contains(term, regex=False)
    return mask


def _series_contains_any(values: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    return _lowered_series_contains_any(_clean_text_series(values).str.lower(), terms)


def _vectorized_context_labels(frame: pd.DataFrame) -> pd.Series:
    package = _clean_text_series(_series_or_default(frame, "BIOSAMPLE_package"))
    title = _clean_text_series(_series_or_default(frame, "BIOSAMPLE_title"))
    ecosystem = _clean_text_series(_series_or_default(frame, "ECOSYSTEM_tags"))
    disease = _clean_text_series(_series_or_default(frame, "DISEASE_tags"))
    pathogenicity = _clean_text_series(_series_or_default(frame, "BIOSAMPLE_pathogenicity"))
    combined = package.str.cat(title, sep=" | ").str.cat(ecosystem, sep=" | ").str.cat(disease, sep=" | ").str.cat(pathogenicity, sep=" | ")
    combined_lower = combined.str.lower()
    empty_mask = combined.str.strip().eq("")
    labels = pd.Series("other", index=frame.index, dtype=object)
    labels.loc[empty_mask] = "unknown"
    remaining = ~empty_mask
    clinical_mask = remaining & (pathogenicity.ne("") | _lowered_series_contains_any(combined_lower, CLINICAL_CONTEXT_TERMS))
    labels.loc[clinical_mask] = "clinical"
    remaining = remaining & ~clinical_mask
    food_mask = remaining & _lowered_series_contains_any(combined_lower, FOOD_CONTEXT_TERMS)
    labels.loc[food_mask] = "food"
    remaining = remaining & ~food_mask
    environmental_mask = remaining & _lowered_series_contains_any(combined_lower, ENVIRONMENTAL_CONTEXT_TERMS)
    labels.loc[environmental_mask] = "environmental"
    remaining = remaining & ~environmental_mask
    host_associated_mask = remaining & _lowered_series_contains_any(combined_lower, HOST_ASSOCIATED_CONTEXT_TERMS)
    labels.loc[host_associated_mask] = "host_associated"
    return labels


def _global_max_normalized_richness(unique_counts: pd.Series) -> pd.Series:
    unique = unique_counts.astype(float).fillna(0.0).clip(lower=0.0)
    max_count = float(unique.max())
    if max_count <= 0.0:
        return pd.Series(0.0, index=unique.index, dtype=float)
    denominator = math.log1p(max_count)
    return (np.log1p(unique) / denominator).clip(lower=0.0, upper=1.0)


def _menhinick_normalized_richness(unique_counts: pd.Series, observation_counts: pd.Series) -> pd.Series:
    unique = unique_counts.astype(float).fillna(0.0).clip(lower=0.0)
    observations = observation_counts.astype(float).fillna(0.0).clip(lower=0.0)
    denominator = np.sqrt(observations.clip(lower=1.0))
    menhinick = np.divide(
        unique,
        denominator,
        out=np.zeros(len(unique), dtype=float),
        where=observations > 0.0,
    )
    max_value = float(np.nanmax(menhinick)) if len(menhinick) else 0.0
    if not np.isfinite(max_value) or max_value <= 0.0:
        return pd.Series(0.0, index=unique.index, dtype=float)
    return pd.Series(np.clip(menhinick / max_value, 0.0, 1.0), index=unique.index, dtype=float)


def build_training_canonical_table(records: pd.DataFrame, amr_consensus: pd.DataFrame, *, split_year: int = 2015) -> pd.DataFrame:
    """Collapse training-period records to canonical representatives for T/A aggregation."""
    training = records.loc[records["resolved_year"].fillna(0).astype(int) <= split_year].copy()
    merged = training.merge(amr_consensus, on="sequence_accession", how="left", validate="m:1")

    merged["amr_gene_count"] = merged["amr_gene_count"].fillna(0).astype(int)
    merged["amr_class_count"] = merged["amr_class_count"].fillna(0).astype(int)
    merged["amr_hit_count"] = merged["amr_hit_count"].fillna(0).astype(int)
    merged["amr_drug_classes"] = merged["amr_drug_classes"].fillna("")

    grouped = merged.groupby(["backbone_id", "canonical_id"], sort=False)
    rows: list[dict[str, object]] = []
    for (backbone_id, canonical_id), frame in grouped:
        class_union = sorted({drug_class for cell in frame["amr_drug_classes"] for drug_class in _split_values(cell)})
        rows.append(
            {
                "backbone_id": backbone_id,
                "canonical_id": canonical_id,
                "has_relaxase": bool(frame["has_relaxase"].any()),
                "has_mpf": bool(frame["has_mpf"].any()),
                "has_orit": bool(frame["has_orit"].any()),
                "is_mobilizable": bool(frame["is_mobilizable"].any()),
                "amr_gene_count": int(frame["amr_gene_count"].max()),
                "amr_class_count": int(frame["amr_class_count"].max()),
                "amr_hit_count": int(frame["amr_hit_count"].max()),
                "amr_drug_classes": ",".join(class_union),
            }
        )
    return pd.DataFrame(rows)


def compute_feature_t(training_canonical: pd.DataFrame) -> pd.DataFrame:
    """Compute the mobility component at backbone level."""
    grouped = training_canonical.groupby("backbone_id", sort=False)
    rows: list[dict[str, object]] = []
    for backbone_id, frame in grouped:
        n = len(frame)
        relaxase_support = float(frame["has_relaxase"].mean())
        mpf_support = float(frame["has_mpf"].mean())
        orit_support = float(frame["has_orit"].mean())
        mobilizable_support = float(frame["is_mobilizable"].mean())
        t_raw = (1.0 * relaxase_support + 1.0 * mpf_support + 0.75 * orit_support + 0.50 * mobilizable_support) / 3.25
        support_shrinkage = _support_factor(n)
        rows.append(
            {
                "backbone_id": backbone_id,
                "member_count_train": n,
                "relaxase_support": relaxase_support,
                "mpf_support": mpf_support,
                "orit_support": orit_support,
                "mobilizable_support": mobilizable_support,
                "support_shrinkage": support_shrinkage,
                "T_raw": t_raw,
                "T_eff": t_raw * support_shrinkage,
            }
        )
    return pd.DataFrame(rows)


def compute_feature_h(records: pd.DataFrame, *, split_year: int = 2015) -> pd.DataFrame:
    """Compute observed host diversity from training-period observations."""
    training = records.loc[records["resolved_year"].fillna(0).astype(int) <= split_year].copy()
    output_columns = [
        "backbone_id",
        "host_observation_count",
        "host_taxon_signature_count",
        "genus_unique_count",
        "H_genus_richness_norm",
        "family_unique_count",
        "order_unique_count",
        "class_unique_count",
        "phylum_unique_count",
        "H_genus_norm",
        "phylo_breadth_score",
        "phylo_pairwise_dispersion_score",
        "phylo_breadth_augmented_score",
        "host_taxon_evenness_score",
        "host_support_factor",
        "predicted_host_range_score",
        "reported_host_range_score",
        "H_external_host_range_score",
        "H_external_host_range_support",
        "H_raw",
        "H_phylogenetic_raw",
        "H_augmented_raw",
        "H_phylogenetic_augmented_raw",
        "H_eff",
        "H_phylogenetic_eff",
        "H_augmented_eff",
        "H_phylogenetic_augmented_eff",
    ]
    if training.empty:
        return pd.DataFrame(columns=output_columns)

    backbone_order = training["backbone_id"].drop_duplicates().astype(str)
    genus_values = _clean_text_series(_series_or_default(training, "genus"))
    genus_nonempty = genus_values.ne("")
    genus_frame = pd.DataFrame(
        {
            "backbone_id": training.loc[genus_nonempty, "backbone_id"].astype(str).to_numpy(),
            "value": genus_values.loc[genus_nonempty].to_numpy(),
        }
    )

    host_observation_count = genus_frame.groupby("backbone_id", sort=False).size().reindex(backbone_order, fill_value=0).astype(int)
    genus_unique_count = genus_frame.groupby("backbone_id", sort=False)["value"].nunique().reindex(backbone_order, fill_value=0).astype(int)
    genus_evenness = (
        genus_frame.groupby("backbone_id", sort=False)["value"]
        .agg(lambda values: _normalized_shannon_evenness(list(values)))
        .reindex(backbone_order, fill_value=0.0)
        .astype(float)
    )
    genus_richness_norm = _global_max_normalized_richness(genus_unique_count)
    genus_norm = _menhinick_normalized_richness(genus_unique_count, host_observation_count)

    rank_to_output = {
        "TAXONOMY_family": "family_unique_count",
        "TAXONOMY_order": "order_unique_count",
        "TAXONOMY_class": "class_unique_count",
        "TAXONOMY_phylum": "phylum_unique_count",
    }
    breadth_components: dict[str, pd.Series] = {}
    rank_unique_counts: dict[str, pd.Series] = {}
    for rank, output_name in rank_to_output.items():
        rank_values = _clean_text_series(_series_or_default(training, rank))
        rank_nonempty = rank_values.ne("")
        rank_frame = pd.DataFrame(
            {
                "backbone_id": training.loc[rank_nonempty, "backbone_id"].astype(str).to_numpy(),
                "value": rank_values.loc[rank_nonempty].to_numpy(),
            }
        )
        unique_counts = rank_frame.groupby("backbone_id", sort=False)["value"].nunique().reindex(backbone_order, fill_value=0).astype(int)
        rank_unique_counts[output_name] = unique_counts
        component = _global_max_normalized_richness(unique_counts)
        component = component.where(unique_counts > 0)
        breadth_components[rank] = component

    breadth_frame = pd.DataFrame(breadth_components, index=backbone_order)
    breadth_array = breadth_frame.to_numpy(dtype=float)
    positive_mask = np.isfinite(breadth_array) & (breadth_array > 0.0)
    log_values = np.full_like(breadth_array, np.nan, dtype=float)
    log_values[positive_mask] = np.log(breadth_array[positive_mask])
    positive_count = positive_mask.sum(axis=1)
    log_sum = np.nansum(log_values, axis=1)
    breadth_logs = np.divide(
        log_sum,
        positive_count,
        out=np.zeros(len(backbone_order), dtype=float),
        where=positive_count > 0,
    )
    phylo_breadth = np.exp(breadth_logs)
    phylo_breadth = pd.Series(np.where(positive_count > 0, phylo_breadth, 0.0), index=backbone_order, dtype=float)

    host_signatures = _host_taxonomy_signature_series(training)
    signature_nonempty = host_signatures.map(_host_signature_is_nonempty)
    if bool(signature_nonempty.any()):
        signature_frame = pd.DataFrame(
            {
                "backbone_id": training.loc[signature_nonempty, "backbone_id"].astype(str).to_numpy(),
                "signature": host_signatures.loc[signature_nonempty].to_list(),
            }
        )
        host_taxon_signature_count = (
            signature_frame.groupby("backbone_id", sort=False)["signature"]
            .agg(lambda values: len(dict.fromkeys(values)))
            .reindex(backbone_order, fill_value=0)
            .astype(int)
        )
        host_taxon_evenness = (
            signature_frame.groupby("backbone_id", sort=False)["signature"]
            .agg(lambda values: _normalized_shannon_evenness(list(values)))
            .reindex(backbone_order, fill_value=0.0)
            .astype(float)
        )
        phylo_pairwise_dispersion = (
            signature_frame.groupby("backbone_id", sort=False)["signature"]
            .agg(lambda values: _mean_pairwise_host_taxonomy_distance(list(values)))
            .reindex(backbone_order, fill_value=0.0)
            .astype(float)
        )
    else:
        host_taxon_signature_count = pd.Series(0, index=backbone_order, dtype=int)
        host_taxon_evenness = pd.Series(0.0, index=backbone_order, dtype=float)
        phylo_pairwise_dispersion = pd.Series(0.0, index=backbone_order, dtype=float)
    phylo_breadth_augmented = pd.Series(
        np.clip(
            (0.65 * phylo_breadth.to_numpy(dtype=float))
            + (0.35 * phylo_pairwise_dispersion.to_numpy(dtype=float)),
            0.0,
            1.0,
        ),
        index=backbone_order,
        dtype=float,
    )

    # Country-adjusted observation count to mitigate sequencing propensity bias
    # Plasmids observed only in countries with massive sequencing operations should be discounted
    country_series = _clean_text_series(_series_or_default(training, "country"))
    country_counts = country_series[country_series.ne("")].value_counts()
    if not country_counts.empty:
        # Normalize to a weight where the median country has weight 1.0, and massive countries have weight < 1.0
        median_count = country_counts.median()
        country_weights = np.clip(np.sqrt(median_count / country_counts), 0.1, 2.0)
        
        # Apply weights to the training records
        record_weights = country_series.map(country_weights).fillna(1.0)
        
        # Weighted observation count per backbone
        weighted_obs = pd.Series(record_weights.to_numpy(), index=training["backbone_id"]).groupby(level=0, sort=False).sum()
        host_support = weighted_obs.reindex(backbone_order, fill_value=0.0).astype(float)
        host_support = host_support / (host_support + 5.0)
    else:
        host_support = host_observation_count.astype(float) / (host_observation_count.astype(float) + 5.0)

    # Mitigate sampling bias (e.g. from highly sequenced E. coli) using Shannon Evenness.
    # We apply a square root to evenness to penalize heavily biased samples without completely zeroing them.
    evenness_array = genus_evenness.to_numpy(dtype=float)
    bias_correction = np.clip(np.sqrt(evenness_array), 0.1, 1.0)

    # Keep the directly observed richness term for diagnostics, but align the
    # main H definition with the project plan: corrected genus richness combined
    # with higher-rank phylogenetic breadth via geometric mean, adjusted for sampling bias.
    h_raw = np.sqrt(
        genus_norm.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        * phylo_breadth.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    ) * bias_correction
    h_raw = pd.Series(np.clip(h_raw, 0.0, 1.0), index=backbone_order, dtype=float)
    
    h_phylogenetic_raw = np.sqrt(
        genus_norm.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        * phylo_breadth_augmented.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    ) * bias_correction
    h_phylogenetic_raw = pd.Series(np.clip(h_phylogenetic_raw, 0.0, 1.0), index=backbone_order, dtype=float)

    predicted_rank_values = _rank_score_series(_series_or_default(training, "predicted_host_range_overall_rank"))
    reported_rank_values = _rank_score_series(_series_or_default(training, "reported_host_range_lit_rank"))
    predicted_support_mask = predicted_rank_values > 0.0
    reported_support_mask = reported_rank_values > 0.0
    predicted_score = (
        predicted_rank_values.groupby(training["backbone_id"].astype(str), sort=False)
        .mean()
        .reindex(backbone_order, fill_value=0.0)
        .astype(float)
    )
    reported_score = (
        reported_rank_values.groupby(training["backbone_id"].astype(str), sort=False)
        .mean()
        .reindex(backbone_order, fill_value=0.0)
        .astype(float)
    )
    predicted_support = (
        predicted_support_mask.groupby(training["backbone_id"].astype(str), sort=False)
        .mean()
        .reindex(backbone_order, fill_value=0.0)
        .astype(float)
    )
    reported_support = (
        reported_support_mask.groupby(training["backbone_id"].astype(str), sort=False)
        .mean()
        .reindex(backbone_order, fill_value=0.0)
        .astype(float)
    )
    external_score = pd.concat(
        [
            predicted_score.rename("predicted"),
            reported_score.rename("reported"),
        ],
        axis=1,
    ).replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    external_support = np.maximum(predicted_support.to_numpy(dtype=float), reported_support.to_numpy(dtype=float))
    external_score_array = external_score.to_numpy(dtype=float)
    h_augmented_raw_array = np.where(
        external_score_array > 0.0,
        np.sqrt(
            np.clip(h_raw.to_numpy(dtype=float), 0.0, None)
            * np.clip(external_score_array, 0.0, None)
        ),
        h_raw.to_numpy(dtype=float),
    )
    h_augmented_raw = pd.Series(np.clip(h_augmented_raw_array, 0.0, 1.0), index=backbone_order, dtype=float)
    h_phylogenetic_augmented_raw_array = np.where(
        external_score_array > 0.0,
        np.sqrt(
            np.clip(h_phylogenetic_raw.to_numpy(dtype=float), 0.0, None)
            * np.clip(external_score_array, 0.0, None)
        ),
        h_phylogenetic_raw.to_numpy(dtype=float),
    )
    h_phylogenetic_augmented_raw = pd.Series(
        np.clip(h_phylogenetic_augmented_raw_array, 0.0, 1.0),
        index=backbone_order,
        dtype=float,
    )

    result = pd.DataFrame(
        {
            "backbone_id": backbone_order.to_list(),
            "host_observation_count": host_observation_count.to_numpy(dtype=int),
            "host_taxon_signature_count": host_taxon_signature_count.to_numpy(dtype=int),
            "genus_unique_count": genus_unique_count.to_numpy(dtype=int),
            "H_genus_richness_norm": genus_richness_norm.to_numpy(dtype=float),
            "family_unique_count": rank_unique_counts["family_unique_count"].to_numpy(dtype=int),
            "order_unique_count": rank_unique_counts["order_unique_count"].to_numpy(dtype=int),
            "class_unique_count": rank_unique_counts["class_unique_count"].to_numpy(dtype=int),
            "phylum_unique_count": rank_unique_counts["phylum_unique_count"].to_numpy(dtype=int),
            "H_genus_norm": genus_norm.to_numpy(dtype=float),
            "phylo_breadth_score": phylo_breadth.to_numpy(dtype=float),
            "phylo_pairwise_dispersion_score": phylo_pairwise_dispersion.to_numpy(dtype=float),
            "phylo_breadth_augmented_score": phylo_breadth_augmented.to_numpy(dtype=float),
            "host_taxon_evenness_score": host_taxon_evenness.to_numpy(dtype=float),
            "host_support_factor": host_support.to_numpy(dtype=float),
            "predicted_host_range_score": predicted_score.to_numpy(dtype=float),
            "reported_host_range_score": reported_score.to_numpy(dtype=float),
            "H_external_host_range_score": external_score.to_numpy(dtype=float),
            "H_external_host_range_support": external_support.astype(float),
            "H_raw": h_raw.to_numpy(dtype=float),
            "H_phylogenetic_raw": h_phylogenetic_raw.to_numpy(dtype=float),
            "H_augmented_raw": h_augmented_raw.to_numpy(dtype=float),
            "H_phylogenetic_augmented_raw": h_phylogenetic_augmented_raw.to_numpy(dtype=float),
            "H_eff": (h_raw * host_support).to_numpy(dtype=float),
            "H_phylogenetic_eff": (h_phylogenetic_raw * host_support).to_numpy(dtype=float),
            "H_augmented_eff": (h_augmented_raw * host_support).to_numpy(dtype=float),
            "H_phylogenetic_augmented_eff": (h_phylogenetic_augmented_raw * host_support).to_numpy(dtype=float),
        }
    )
    return result


def _mean_prevalence_from_sets(class_sets: list[set[str]]) -> float:
    if not class_sets:
        return 0.0
    universe = sorted({item for class_set in class_sets for item in class_set})
    if not universe:
        return 0.0
    prevalences = []
    total = len(class_sets)
    for item in universe:
        prevalences.append(sum(item in class_set for class_set in class_sets) / total)
    return float(sum(prevalences) / len(prevalences))


def _mean_recurrent_prevalence_from_sets(class_sets: list[set[str]]) -> float:
    if not class_sets:
        return 0.0
    counts: dict[str, int] = {}
    total = len(class_sets)
    for class_set in class_sets:
        for item in class_set:
            counts[item] = counts.get(item, 0) + 1
    recurrent = [count / total for count in counts.values() if count >= 2]
    if not recurrent:
        return 0.0
    return float(sum(recurrent) / len(recurrent))


def compute_feature_a(training_canonical: pd.DataFrame) -> pd.DataFrame:
    """Compute AMR burden and consistency summaries at backbone level.

    A_consistency is the mean prevalence of individual AMR drug classes
    across backbone members. Higher values mean the same drug resistance
    classes repeatedly appear in different plasmids of the same backbone.
    It is NOT a traditional inter-rater reliability metric despite the name.
    """
    from plasmid_priority.schemas.who_mia import WHO_MIA_CLASS_MAP
    
    who_weights = {
        "HPCIA": 3.0,
        "CIA": 2.0,
        "HIA": 1.0,
        "IA": 1.0,
    }

    grouped = training_canonical.groupby("backbone_id", sort=False)
    rows: list[dict[str, object]] = []
    for backbone_id, frame in grouped:
        n = len(frame)
        class_sets = [_split_values(value) for value in frame["amr_drug_classes"]]
        
        threat_scores = []
        for cset in class_sets:
            score = 0.0
            for c in cset:
                mapping = WHO_MIA_CLASS_MAP.get(c)
                if mapping:
                    score += who_weights.get(mapping["who_mia_category"], 1.0)
                else:
                    score += 1.0
            threat_scores.append(score)
        
        mean_threat_score = sum(threat_scores) / max(1, len(threat_scores))

        consistency = _mean_prevalence_from_sets(class_sets)
        recurrence = _mean_recurrent_prevalence_from_sets(class_sets)
        amr_support = _support_factor(int((frame["amr_hit_count"] > 0).sum()))
        rows.append(
            {
                "backbone_id": backbone_id,
                "mean_amr_class_count": float(frame["amr_class_count"].mean()),
                "mean_amr_gene_count": float(frame["amr_gene_count"].mean()),
                "mean_amr_clinical_threat_score": float(mean_threat_score),
                "A_consistency": consistency,
                "A_recurrence": recurrence,
                "amr_support_factor": amr_support,
                "canonical_member_count_train": n,
            }
        )
    return pd.DataFrame(rows)


def build_backbone_table(
    records: pd.DataFrame,
    coherence: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
    new_country_threshold: int | None = None,
) -> pd.DataFrame:
    """Build the backbone-level table used for scoring and retrospective outcome evaluation."""
    if records.empty:
        return pd.DataFrame()

    if new_country_threshold is None:
        from plasmid_priority.config import DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD
        new_country_threshold = DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD

    working = records.copy()
    working["backbone_id"] = working["backbone_id"].astype(str)
    years = pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    training = working.loc[years <= split_year].copy()
    testing = working.loc[(years > split_year) & (years <= test_year_end)].copy()

    backbone_order = working["backbone_id"].drop_duplicates()
    backbone_table = pd.DataFrame({"backbone_id": backbone_order.to_list()})

    member_count_total = working.groupby("backbone_id", sort=False)["canonical_id"].nunique()
    member_count_train = training.groupby("backbone_id", sort=False)["canonical_id"].nunique()

    training_pairs = training.assign(country_clean=_clean_text_series(training["country"]))
    training_pairs = training_pairs.loc[training_pairs["country_clean"].ne(""), ["backbone_id", "country_clean"]].drop_duplicates()
    testing_pairs = testing.assign(country_clean=_clean_text_series(testing["country"]))
    testing_pairs = testing_pairs.loc[testing_pairs["country_clean"].ne(""), ["backbone_id", "country_clean"]].drop_duplicates()

    n_countries_train = training_pairs.groupby("backbone_id", sort=False).size()
    n_countries_test = testing_pairs.groupby("backbone_id", sort=False).size()
    if training_pairs.empty or testing_pairs.empty:
        n_new_countries = pd.Series(dtype=int)
    else:
        new_pairs = testing_pairs.merge(training_pairs, on=["backbone_id", "country_clean"], how="left", indicator=True)
        new_pairs = new_pairs.loc[new_pairs["_merge"] == "left_only"]
        n_new_countries = new_pairs.groupby("backbone_id", sort=False).size()

    refseq_share_train = training.assign(_is_refseq=training["record_origin"].eq("refseq").astype(float)).groupby("backbone_id", sort=False)["_is_refseq"].mean()
    insd_share_train = training.assign(_is_insd=training["record_origin"].eq("insd").astype(float)).groupby("backbone_id", sort=False)["_is_insd"].mean()

    purity_table = pd.DataFrame({"backbone_id": backbone_order.astype(str).to_list()})
    if not training.empty:
        training_backbone_ids = training["backbone_id"].astype(str)
        genus_purity = _grouped_dominant_share(training_backbone_ids, _series_or_default(training, "genus"))
        family_purity = _grouped_dominant_share(training_backbone_ids, _series_or_default(training, "TAXONOMY_family"))
        mobility_purity = _grouped_dominant_share(training_backbone_ids, _series_or_default(training, "predicted_mobility"))
        replicon_purity = _grouped_dominant_share(training_backbone_ids, _series_or_default(training, "primary_replicon"))

        pmlst_scheme_series = _clean_text_series(_series_or_default(training, "PMLST_scheme"))
        pmlst_st_series = _clean_text_series(_series_or_default(training, "PMLST_sequence_type"))
        pmlst_allele_series = _clean_text_series(_series_or_default(training, "PMLST_alleles"))
        pmlst_presence_fraction = (
            (pmlst_scheme_series.ne("") | pmlst_st_series.ne("") | pmlst_allele_series.ne(""))
            .groupby(training_backbone_ids, sort=False)
            .mean()
            .astype(float)
        )
        pmlst_scheme_purity = _grouped_dominant_share(training_backbone_ids, pmlst_scheme_series)
        pmlst_st_purity = _grouped_dominant_share(training_backbone_ids, pmlst_st_series)
        pmlst_allele_purity = _grouped_dominant_share(training_backbone_ids, pmlst_allele_series)
        pmlst_components = pd.concat(
            [
                pmlst_scheme_purity.rename("scheme"),
                pmlst_st_purity.rename("st"),
                pmlst_allele_purity.rename("allele"),
            ],
            axis=1,
        ).fillna(0.0)
        positive_component_count = pmlst_components.gt(0.0).sum(axis=1)
        pmlst_internal_coherence = pd.Series(0.0, index=pmlst_components.index, dtype=float)
        positive_mask = positive_component_count > 0
        if bool(positive_mask.any()):
            pmlst_internal_coherence.loc[positive_mask] = (
                pmlst_components.loc[positive_mask].sum(axis=1) / positive_component_count.loc[positive_mask]
            ).astype(float)
        pmlst_coherence_score = (
            pmlst_presence_fraction.reindex(pmlst_internal_coherence.index, fill_value=0.0) * pmlst_internal_coherence
        ).astype(float)

        pmid_counts = _series_or_default(training, "associated_pmid(s)").map(_pmid_count).astype(int)
        mean_pmid_count = pmid_counts.groupby(training_backbone_ids, sort=False).mean().astype(float)
        max_pmid_count = pmid_counts.groupby(training_backbone_ids, sort=False).max().astype(int)

        context_labels = _vectorized_context_labels(training)
        context_table = pd.DataFrame(
            {
                "backbone_id": training_backbone_ids.to_numpy(),
                "context_label": context_labels.to_numpy(),
            }
        )
        context_nonempty = context_table.loc[context_table["context_label"].ne("unknown")].copy()
        context_unique = (
            context_nonempty.groupby("backbone_id", sort=False)["context_label"].nunique()
            if not context_nonempty.empty
            else pd.Series(dtype=float)
        )
        context_diversity = (context_unique.astype(float) / 4.0).clip(lower=0.0, upper=1.0)
        clinical_fraction = context_table["context_label"].eq("clinical").groupby(context_table["backbone_id"], sort=False).mean().astype(float)
        environmental_fraction = context_table["context_label"].eq("environmental").groupby(context_table["backbone_id"], sort=False).mean().astype(float)
        host_associated_fraction = context_table["context_label"].eq("host_associated").groupby(context_table["backbone_id"], sort=False).mean().astype(float)
        food_fraction = context_table["context_label"].eq("food").groupby(context_table["backbone_id"], sort=False).mean().astype(float)
        pathogenic_fraction = (
            _clean_text_series(_series_or_default(training, "BIOSAMPLE_pathogenicity"))
            .ne("")
            .groupby(training_backbone_ids, sort=False)
            .mean()
            .astype(float)
        )

        n_replicon_types = pd.to_numeric(_series_or_default(training, "n_replicon_types", 0.0), errors="coerce").fillna(0.0)
        mean_n_replicon_types = n_replicon_types.groupby(training_backbone_ids, sort=False).mean().astype(float)
        multi_replicon_fraction = (n_replicon_types > 1.0).groupby(training_backbone_ids, sort=False).mean().astype(float)
        assignment_primary_fraction = (
            _clean_text_series(_series_or_default(training, "backbone_assignment_rule"))
            .eq("primary_cluster_id")
            .groupby(training_backbone_ids, sort=False)
            .mean()
            .astype(float)
        )
        mash_neighbor_distance_train_mean = (
            pd.to_numeric(_series_or_default(training, "mash_neighbor_distance", 0.0), errors="coerce")
            .fillna(0.0)
            .groupby(training_backbone_ids, sort=False)
            .mean()
            .astype(float)
        )

        purity_table["genus_purity_train"] = purity_table["backbone_id"].map(genus_purity).fillna(0.0).astype(float)
        purity_table["family_purity_train"] = purity_table["backbone_id"].map(family_purity).fillna(0.0).astype(float)
        purity_table["mobility_purity_train"] = purity_table["backbone_id"].map(mobility_purity).fillna(0.0).astype(float)
        purity_table["replicon_purity_train"] = purity_table["backbone_id"].map(replicon_purity).fillna(0.0).astype(float)
        purity_table["backbone_purity_score"] = purity_table[
            ["genus_purity_train", "family_purity_train", "mobility_purity_train", "replicon_purity_train"]
        ].mean(axis=1)
        purity_table["mean_n_replicon_types_train"] = purity_table["backbone_id"].map(mean_n_replicon_types).fillna(0.0).astype(float)
        purity_table["multi_replicon_fraction_train"] = purity_table["backbone_id"].map(multi_replicon_fraction).fillna(0.0).astype(float)
        purity_table["primary_replicon_diversity_train"] = np.where(
            purity_table["replicon_purity_train"] > 0.0,
            1.0 - purity_table["replicon_purity_train"],
            0.0,
        )
        purity_table["assignment_primary_fraction"] = purity_table["backbone_id"].map(assignment_primary_fraction).fillna(0.0).astype(float)
        purity_table["assignment_confidence_score"] = 0.5 + (0.5 * purity_table["assignment_primary_fraction"])
        purity_table["pmlst_presence_fraction_train"] = purity_table["backbone_id"].map(pmlst_presence_fraction).fillna(0.0).astype(float)
        purity_table["pmlst_scheme_purity_train"] = purity_table["backbone_id"].map(pmlst_scheme_purity).fillna(0.0).astype(float)
        purity_table["pmlst_st_purity_train"] = purity_table["backbone_id"].map(pmlst_st_purity).fillna(0.0).astype(float)
        purity_table["pmlst_allele_purity_train"] = purity_table["backbone_id"].map(pmlst_allele_purity).fillna(0.0).astype(float)
        purity_table["pmlst_coherence_score"] = purity_table["backbone_id"].map(pmlst_coherence_score).fillna(0.0).astype(float)
        purity_table["mean_pmid_count_train"] = purity_table["backbone_id"].map(mean_pmid_count).fillna(0.0).astype(float)
        purity_table["max_pmid_count_train"] = purity_table["backbone_id"].map(max_pmid_count).fillna(0).astype(int)
        purity_table["clinical_context_fraction_train"] = purity_table["backbone_id"].map(clinical_fraction).fillna(0.0).astype(float)
        purity_table["environmental_context_fraction_train"] = purity_table["backbone_id"].map(environmental_fraction).fillna(0.0).astype(float)
        purity_table["host_associated_context_fraction_train"] = purity_table["backbone_id"].map(host_associated_fraction).fillna(0.0).astype(float)
        purity_table["food_context_fraction_train"] = purity_table["backbone_id"].map(food_fraction).fillna(0.0).astype(float)
        purity_table["pathogenic_context_fraction_train"] = purity_table["backbone_id"].map(pathogenic_fraction).fillna(0.0).astype(float)
        purity_table["ecology_context_diversity_train"] = purity_table["backbone_id"].map(context_diversity).fillna(0.0).astype(float)
        purity_table["ecology_context_score"] = np.clip(
            0.35 * purity_table["ecology_context_diversity_train"]
            + 0.25 * purity_table["clinical_context_fraction_train"]
            + 0.15 * purity_table["environmental_context_fraction_train"]
            + 0.10 * purity_table["host_associated_context_fraction_train"]
            + 0.05 * purity_table["food_context_fraction_train"]
            + 0.10 * purity_table["pathogenic_context_fraction_train"],
            0.0,
            1.0,
        ).astype(float)
        purity_table["mash_neighbor_distance_train_mean"] = purity_table["backbone_id"].map(mash_neighbor_distance_train_mean).fillna(0.0).astype(float)

    backbone_table["member_count_train"] = backbone_table["backbone_id"].map(member_count_train).fillna(0).astype(int)
    backbone_table["member_count_total"] = backbone_table["backbone_id"].map(member_count_total).fillna(0).astype(int)
    backbone_table["n_countries_train"] = backbone_table["backbone_id"].map(n_countries_train).fillna(0).astype(int)
    backbone_table["n_countries_test"] = backbone_table["backbone_id"].map(n_countries_test).fillna(0).astype(int)
    backbone_table["n_new_countries"] = backbone_table["backbone_id"].map(n_new_countries).fillna(0).astype(int)
    backbone_table["refseq_share_train"] = backbone_table["backbone_id"].map(refseq_share_train).fillna(0.0).astype(float)
    backbone_table["insd_share_train"] = backbone_table["backbone_id"].map(insd_share_train).fillna(0.0).astype(float)
    backbone_table["split_year"] = int(split_year)
    backbone_table["test_year_end"] = int(test_year_end)

    spread_label = pd.Series(np.nan, index=backbone_table.index, dtype=float)
    eligible = backbone_table["n_countries_train"].between(1, 3, inclusive="both")
    spread_label.loc[eligible] = backbone_table.loc[eligible, "n_new_countries"].ge(new_country_threshold).astype(int)
    backbone_table["spread_label"] = spread_label
    backbone_table["visibility_expansion_label"] = spread_label

    if "backbone_seen_in_training" in working.columns:
        testing_seen = testing.assign(_seen=testing["backbone_seen_in_training"].fillna(False).astype(bool).astype(float)).groupby("backbone_id", sort=False)["_seen"].agg(["sum", "mean"])
        backbone_table["n_test_records_seen_in_training"] = backbone_table["backbone_id"].map(testing_seen["sum"]).fillna(0).astype(int)
        backbone_table["test_seen_in_training_fraction"] = backbone_table["backbone_id"].map(testing_seen["mean"]).fillna(0.0).astype(float)

    if not purity_table.empty:
        backbone_table = backbone_table.merge(purity_table, on="backbone_id", how="left", validate="1:1")

    if not coherence.empty:
        backbone_table = backbone_table.merge(coherence, on="backbone_id", how="left", validate="1:1")
    return backbone_table
