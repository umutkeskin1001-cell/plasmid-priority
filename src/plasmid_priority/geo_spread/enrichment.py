"""Leakage-safe feature enrichment for the geo spread branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.utils.geography import country_to_macro_region


def _normalized_entropy(values: pd.Series) -> float:
    counts = values.astype(str).str.strip().replace("", np.nan).dropna().value_counts()
    if len(counts) <= 1:
        return 0.0
    probs = counts / float(counts.sum())
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / np.log(len(counts)))


def _dominant_share(values: pd.Series) -> float:
    counts = (
        values.astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts(normalize=True)
    )
    if counts.empty:
        return 0.0
    return float(counts.iloc[0])


def build_geo_spread_context_features(
    records: pd.DataFrame,
    *,
    split_year: int,
) -> pd.DataFrame:
    """Derive branch-safe geographic context features from training-period records."""
    if records.empty:
        return pd.DataFrame(columns=["backbone_id"])
    working = records.copy()
    if "backbone_id" not in working.columns:
        return pd.DataFrame(columns=["backbone_id"])
    working["backbone_id"] = working["backbone_id"].astype(str)
    working["resolved_year"] = pd.to_numeric(
        working.get("resolved_year"), errors="coerce"
    ).fillna(0).astype(int)
    working["country_clean"] = (
        working.get("country", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    working = working.loc[
        working["resolved_year"].between(1, int(split_year))
        & working["country_clean"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["backbone_id"])
    working["macro_region"] = working["country_clean"].map(country_to_macro_region).fillna("")
    summary = (
        working.groupby("backbone_id", as_index=False)
        .agg(
            geo_country_record_count_train=("country_clean", "size"),
            geo_country_entropy_train=("country_clean", _normalized_entropy),
            geo_macro_region_entropy_train=("macro_region", _normalized_entropy),
            geo_dominant_region_share_train=("macro_region", _dominant_share),
        )
        .reset_index(drop=True)
    )
    for column in summary.columns:
        if column == "backbone_id":
            continue
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0).astype(float)
    return summary


def default_geo_spread_records_path() -> Path:
    """Return the default plasmid backbone records path for geo spread enrichment."""
    return build_context().data_dir / "silver" / "plasmid_backbones.tsv"


def load_geo_spread_records(path: str | Path | None = None) -> pd.DataFrame:
    """Load the minimal record-level surface required for geo spread enrichment."""
    candidate = Path(path) if path is not None else default_geo_spread_records_path()
    if not candidate.exists():
        return pd.DataFrame(columns=["backbone_id", "country", "resolved_year"])
    return pd.read_csv(
        candidate,
        sep="\t",
        usecols=["backbone_id", "country", "resolved_year"],
    )


def enrich_geo_spread_scored_table(
    scored: pd.DataFrame,
    *,
    split_year: int,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach geo-context features to the scored backbone table."""
    if scored.empty:
        return scored.copy()
    if all(
        column in scored.columns
        for column in (
            "geo_country_record_count_train",
            "geo_country_entropy_train",
            "geo_macro_region_entropy_train",
            "geo_dominant_region_share_train",
        )
    ):
        enriched = scored.copy()
        for column in (
            "geo_country_record_count_train",
            "geo_country_entropy_train",
            "geo_macro_region_entropy_train",
            "geo_dominant_region_share_train",
        ):
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
        return enriched
    record_frame = records if records is not None else load_geo_spread_records()
    if record_frame.empty:
        enriched = scored.copy()
        for column in (
            "geo_country_record_count_train",
            "geo_country_entropy_train",
            "geo_macro_region_entropy_train",
            "geo_dominant_region_share_train",
        ):
            if column not in enriched.columns:
                enriched[column] = 0.0
        return enriched
    features = build_geo_spread_context_features(record_frame, split_year=split_year)
    enriched = scored.merge(features, on="backbone_id", how="left")
    for column in (
        "geo_country_record_count_train",
        "geo_country_entropy_train",
        "geo_macro_region_entropy_train",
        "geo_dominant_region_share_train",
    ):
        enriched[column] = pd.to_numeric(enriched.get(column), errors="coerce").fillna(0.0)
    return enriched
