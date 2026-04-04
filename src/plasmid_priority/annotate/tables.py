"""Accession-level mobility and AMR aggregation tables."""

from __future__ import annotations

import pandas as pd


def _sorted_unique(values: pd.Series) -> str:
    cleaned = sorted(
        {str(value).strip() for value in values if pd.notna(value) and str(value).strip()}
    )
    return ",".join(cleaned)


def build_mobility_table(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Extract accession-level mobility annotations from the harmonized table."""
    columns = [
        "sequence_accession",
        "predicted_mobility",
        "primary_cluster_id",
        "secondary_cluster_id",
        "replicon_types",
        "primary_replicon",
        "relaxase_types",
        "mpf_type",
        "orit_types",
        "has_relaxase",
        "has_mpf",
        "has_orit",
        "is_mobilizable",
        "is_conjugative",
    ]
    mobility = harmonized[columns].drop_duplicates(subset=["sequence_accession"]).copy()
    mobility["mobility_support_score"] = (
        mobility["has_relaxase"].astype(int)
        + mobility["has_mpf"].astype(int)
        + mobility["has_orit"].astype(int)
        + mobility["is_mobilizable"].astype(int)
    ) / 4.0
    return mobility


def build_amr_hits_table(amr_path: str) -> pd.DataFrame:
    """Load and lightly normalize accession-level AMR hit rows."""
    amr = pd.read_csv(amr_path, sep="\t")
    amr["gene_symbol"] = (
        amr["gene_symbol"].fillna(amr["gene_symbol2"]).fillna("").astype(str).str.strip()
    )
    amr["drug_class"] = amr["drug_class"].fillna("").astype(str).str.strip().str.upper()
    amr["confidence_level"] = (
        amr["predicted_phenotype_confidence_level"]
        .fillna("unspecified")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return amr[
        [
            "NUCCORE_ACC",
            "gene_symbol",
            "drug_class",
            "sequence_identity",
            "coverage_percentage",
            "confidence_level",
        ]
    ].rename(columns={"NUCCORE_ACC": "sequence_accession"})


def build_amr_consensus(amr_hits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AMR hits into accession-level consensus summaries."""
    if amr_hits.empty:
        return pd.DataFrame(
            columns=[
                "sequence_accession",
                "amr_gene_symbols",
                "amr_drug_classes",
                "amr_gene_count",
                "amr_class_count",
                "amr_hit_count",
                "amr_any",
            ]
        )

    grouped = amr_hits.groupby("sequence_accession", sort=False)
    consensus = grouped.agg(
        amr_gene_symbols=("gene_symbol", _sorted_unique),
        amr_drug_classes=("drug_class", _sorted_unique),
        amr_hit_count=("gene_symbol", "size"),
    ).reset_index()
    consensus["amr_gene_count"] = consensus["amr_gene_symbols"].map(
        lambda value: 0 if not value else len(value.split(","))
    )
    consensus["amr_class_count"] = consensus["amr_drug_classes"].map(
        lambda value: 0 if not value else len(value.split(","))
    )
    consensus["amr_any"] = consensus["amr_hit_count"] > 0
    return consensus
