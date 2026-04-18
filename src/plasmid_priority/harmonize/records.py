"""Record-level harmonization for canonical PLSDB tables."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd

from plasmid_priority.utils.country_constants import COUNTRY_ALIAS_GROUPS
from plasmid_priority.utils.dataframe import clean_text_series as _clean_text_series_impl
from plasmid_priority.utils.dataframe import dominant_share, read_tsv

LOCATION_SEGMENT_SPLIT = re.compile(r"[,;:/|()\[\]]+")


def clean_text_series(values: pd.Series) -> pd.Series:
    """Backward-compatible export for shared text cleaning utility."""
    return _clean_text_series_impl(values)


@lru_cache(maxsize=16384)
def _normalize_location_key(value: object) -> str:
    if pd.isna(cast(Any, value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_country_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COUNTRY_ALIAS_GROUPS.items():
        for alias in aliases | {canonical.upper()}:
            lookup[_normalize_location_key(alias)] = canonical
    return lookup


COUNTRY_LOOKUP = _build_country_lookup()
COUNTRY_MAX_TOKEN_LENGTH = max(len(alias.split()) for alias in COUNTRY_LOOKUP)


@lru_cache(maxsize=16384)
def _resolve_country_from_segment(segment: str) -> str:
    normalized = _normalize_location_key(segment)
    if not normalized:
        return ""
    direct = COUNTRY_LOOKUP.get(normalized)
    if direct is not None:
        return direct
    tokens = normalized.split()
    max_length = min(COUNTRY_MAX_TOKEN_LENGTH, len(tokens))
    for length in range(max_length, 0, -1):
        for start in range(len(tokens) - length, -1, -1):
            candidate = " ".join(tokens[start : start + length])
            resolved = COUNTRY_LOOKUP.get(candidate)
            if resolved is not None:
                return resolved
    return ""


def _clean_marker_list(value: object) -> str:
    if pd.isna(cast(Any, value)):
        return ""
    items = sorted({item.strip() for item in str(value).split(",") if item.strip()})
    return ",".join(items)


def _marker_count(value: object) -> int:
    cleaned = _clean_marker_list(value)
    return 0 if not cleaned else len(cleaned.split(","))


def _sorted_unique_markers(values: pd.Series) -> str:
    cleaned = sorted(
        {str(value).strip() for value in values if pd.notna(value) and str(value).strip()}
    )
    return ",".join(cleaned)


def _dominant_non_empty(values: pd.Series) -> str:
    cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
    if not cleaned:
        return ""
    return str(pd.Series(cleaned).value_counts().idxmax())


def _summarize_plasmidfinder_hits(plasmidfinder_path: Path) -> pd.DataFrame:
    plasmidfinder = pd.read_csv(
        plasmidfinder_path,
        usecols=["NUCCORE_ACC", "typing", "identity", "coverage"],
    ).rename(columns={"NUCCORE_ACC": "sequence_accession"})
    if plasmidfinder.empty:
        return pd.DataFrame(
            columns=[
                "sequence_accession",
                "plasmidfinder_types",
                "plasmidfinder_dominant_type",
                "plasmidfinder_hit_count",
                "plasmidfinder_type_count",
                "plasmidfinder_dominant_type_share",
                "plasmidfinder_max_identity",
                "plasmidfinder_mean_identity",
                "plasmidfinder_mean_coverage",
            ]
        )

    plasmidfinder["typing"] = plasmidfinder["typing"].fillna("").astype(str).str.strip()
    plasmidfinder["identity"] = pd.to_numeric(plasmidfinder["identity"], errors="coerce").fillna(
        0.0
    )
    plasmidfinder["coverage"] = pd.to_numeric(plasmidfinder["coverage"], errors="coerce").fillna(
        0.0
    )
    plasmidfinder = plasmidfinder.loc[plasmidfinder["typing"].ne("")].copy()
    if plasmidfinder.empty:
        return pd.DataFrame(
            columns=[
                "sequence_accession",
                "plasmidfinder_types",
                "plasmidfinder_dominant_type",
                "plasmidfinder_hit_count",
                "plasmidfinder_type_count",
                "plasmidfinder_dominant_type_share",
                "plasmidfinder_max_identity",
                "plasmidfinder_mean_identity",
                "plasmidfinder_mean_coverage",
            ]
        )

    grouped = plasmidfinder.groupby("sequence_accession", sort=False)
    return grouped.agg(
        plasmidfinder_types=("typing", _sorted_unique_markers),
        plasmidfinder_dominant_type=("typing", _dominant_non_empty),
        plasmidfinder_hit_count=("typing", "size"),
        plasmidfinder_type_count=("typing", "nunique"),
        plasmidfinder_dominant_type_share=("typing", dominant_share),
        plasmidfinder_max_identity=("identity", "max"),
        plasmidfinder_mean_identity=("identity", "mean"),
        plasmidfinder_mean_coverage=("coverage", "mean"),
    ).reset_index()


def normalize_country(value: object) -> str:
    """Extract a canonical country token from a biosample location field.

    The parser is intentionally conservative: it only returns a country when a
    curated country/territory alias is found somewhere in the location text.
    Free-text addresses, institutions, road names, and coordinates therefore
    resolve to the empty string instead of contaminating the country field.
    """
    if pd.isna(cast(Any, value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    segments = [
        segment.strip() for segment in LOCATION_SEGMENT_SPLIT.split(text) if segment.strip()
    ]
    if not segments:
        segments = [text]

    for segment in reversed(segments):
        resolved = _resolve_country_from_segment(segment)
        if resolved:
            return resolved
    return ""


def build_harmonized_plasmid_table(
    inventory_path: Path,
    typing_path: Path,
    biosample_path: Path,
    plasmidfinder_path: Path | None = None,
) -> pd.DataFrame:
    """Merge inventory metadata, typing annotations, and biosample location fields."""
    canonical = read_tsv(inventory_path)

    typing = pd.read_csv(
        typing_path,
        usecols=[
            "NUCCORE_ACC",
            "gc",
            "size",
            "num_contigs",
            "rep_type(s)",
            "relaxase_type(s)",
            "mpf_type",
            "orit_type(s)",
            "predicted_mobility",
            "mash_neighbor_distance",
            "predicted_host_range_overall_rank",
            "predicted_host_range_overall_name",
            "reported_host_range_lit_rank",
            "reported_host_range_lit_name",
            "associated_pmid(s)",
            "primary_cluster_id",
            "secondary_cluster_id",
            "observed_host_range_ncbi_rank",
            "observed_host_range_ncbi_name",
            "PMLST_scheme",
            "PMLST_sequence_type",
            "PMLST_alleles",
        ],
    ).rename(columns={"NUCCORE_ACC": "sequence_accession"})

    biosample = pd.read_csv(
        biosample_path,
        usecols=[
            "BIOSAMPLE_UID",
            "LOCATION_name",
            "LOCATION_query",
            "BIOSAMPLE_title",
            "BIOSAMPLE_package",
            "BIOSAMPLE_pathogenicity",
            "ECOSYSTEM_tags",
            "DISEASE_tags",
        ],
    ).rename(columns={"BIOSAMPLE_UID": "biosample_uid"})

    harmonized = canonical.merge(typing, on="sequence_accession", how="left", validate="m:1")
    harmonized = harmonized.merge(biosample, on="biosample_uid", how="left", validate="m:1")
    if plasmidfinder_path is not None:
        plasmidfinder = _summarize_plasmidfinder_hits(plasmidfinder_path)
        harmonized = harmonized.merge(
            plasmidfinder, on="sequence_accession", how="left", validate="m:1"
        )

    harmonized["country"] = (
        harmonized["LOCATION_name"].fillna(harmonized["LOCATION_query"]).map(normalize_country)
    )
    harmonized["replicon_types"] = harmonized["rep_type(s)"].map(_clean_marker_list)
    harmonized["relaxase_types"] = harmonized["relaxase_type(s)"].map(_clean_marker_list)
    harmonized["orit_types"] = harmonized["orit_type(s)"].map(_clean_marker_list)
    harmonized["n_replicon_types"] = harmonized["replicon_types"].map(_marker_count)
    harmonized["n_relaxase_types"] = harmonized["relaxase_types"].map(_marker_count)
    harmonized["n_orit_types"] = harmonized["orit_types"].map(_marker_count)
    harmonized["primary_replicon"] = harmonized["replicon_types"].str.split(",").str[0].fillna("")
    harmonized["predicted_mobility"] = (
        harmonized["predicted_mobility"].fillna("unknown").astype(str)
    )
    harmonized["primary_cluster_id"] = harmonized["primary_cluster_id"].fillna("").astype(str)
    harmonized["mash_neighbor_distance"] = pd.to_numeric(
        harmonized.get("mash_neighbor_distance", pd.Series(0.0, index=harmonized.index)),
        errors="coerce",
    ).fillna(0.0)
    harmonized["plasmidfinder_types"] = (
        harmonized.get("plasmidfinder_types", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["plasmidfinder_dominant_type"] = (
        harmonized.get("plasmidfinder_dominant_type", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    for column in (
        "plasmidfinder_hit_count",
        "plasmidfinder_type_count",
        "plasmidfinder_dominant_type_share",
        "plasmidfinder_max_identity",
        "plasmidfinder_mean_identity",
        "plasmidfinder_mean_coverage",
    ):
        harmonized[column] = pd.to_numeric(
            harmonized.get(column, pd.Series(0.0, index=harmonized.index)),
            errors="coerce",
        ).fillna(0.0)
    harmonized["predicted_host_range_overall_rank"] = (
        harmonized.get("predicted_host_range_overall_rank", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    harmonized["predicted_host_range_overall_name"] = (
        harmonized.get("predicted_host_range_overall_name", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["reported_host_range_lit_rank"] = (
        harmonized.get("reported_host_range_lit_rank", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    harmonized["reported_host_range_lit_name"] = (
        harmonized.get("reported_host_range_lit_name", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["associated_pmid(s)"] = (
        harmonized.get("associated_pmid(s)", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_scheme"] = (
        harmonized.get("PMLST_scheme", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_sequence_type"] = (
        harmonized.get("PMLST_sequence_type", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["PMLST_alleles"] = (
        harmonized.get("PMLST_alleles", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["BIOSAMPLE_pathogenicity"] = (
        harmonized.get("BIOSAMPLE_pathogenicity", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["ECOSYSTEM_tags"] = (
        harmonized.get("ECOSYSTEM_tags", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["DISEASE_tags"] = (
        harmonized.get("DISEASE_tags", pd.Series("", index=harmonized.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    harmonized["typing_gc"] = pd.to_numeric(
        harmonized.get("gc", pd.Series(0.0, index=harmonized.index)),
        errors="coerce",
    ).fillna(0.0)
    harmonized["typing_size"] = pd.to_numeric(
        harmonized.get("size", pd.Series(0.0, index=harmonized.index)),
        errors="coerce",
    ).fillna(0.0)
    harmonized["typing_num_contigs"] = pd.to_numeric(
        harmonized.get("num_contigs", pd.Series(0.0, index=harmonized.index)),
        errors="coerce",
    ).fillna(0.0)

    harmonized["has_country"] = harmonized["country"].astype(str).str.len() > 0
    harmonized["has_relaxase"] = harmonized["n_relaxase_types"] > 0
    harmonized["has_mpf"] = harmonized["mpf_type"].fillna("").astype(str).str.len() > 0
    harmonized["has_orit"] = harmonized["n_orit_types"] > 0
    harmonized["is_mobilizable"] = harmonized["predicted_mobility"].isin(
        ["mobilizable", "conjugative"]
    )
    harmonized["is_conjugative"] = harmonized["predicted_mobility"].eq("conjugative")

    return harmonized
