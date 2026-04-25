#!/usr/bin/env python3
"""Normalize accession-level AMR hits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import SequenceAnnotationCache
from plasmid_priority.cache import stable_hash
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def _normalize_amr_hits(amr: pd.DataFrame) -> pd.DataFrame:
    if amr.empty:
        return pd.DataFrame(
            columns=[
                "sequence_accession",
                "gene_symbol",
                "drug_class",
                "sequence_identity",
                "coverage_percentage",
                "confidence_level",
            ],
        )
    normalized = amr.copy()
    normalized["gene_symbol"] = (
        normalized["gene_symbol"]
        .fillna(normalized.get("gene_symbol2", ""))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    normalized["drug_class"] = (
        normalized["drug_class"].fillna("").astype(str).str.strip().str.upper()
    )
    normalized["confidence_level"] = (
        normalized["predicted_phenotype_confidence_level"]
        .fillna("unspecified")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return normalized[
        [
            "NUCCORE_ACC",
            "gene_symbol",
            "drug_class",
            "sequence_identity",
            "coverage_percentage",
            "confidence_level",
        ]
    ].rename(columns={"NUCCORE_ACC": "sequence_accession"})


def _accession_payload_manifest(amr_raw: pd.DataFrame) -> pd.DataFrame:
    if amr_raw.empty:
        return pd.DataFrame(columns=["sequence_accession", "raw_payload"])
    working = amr_raw.copy()
    working["NUCCORE_ACC"] = working["NUCCORE_ACC"].astype(str)
    grouped = (
        working.sort_values(["NUCCORE_ACC"])
        .groupby("NUCCORE_ACC", sort=False)
        .apply(lambda frame: frame.to_json(orient="records", date_format="iso"))
        .reset_index(name="raw_payload")
        .rename(columns={"NUCCORE_ACC": "sequence_accession"})
    )
    return grouped  # type: ignore


def main() -> int:
    context = build_context(PROJECT_ROOT)
    amr_path = context.asset_path("plsdb_meta_tables_dir") / "amr.tsv"
    output_path = context.data_dir / "silver/plasmid_amr_hits.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "07_annotate_amr") as run:
        run.record_input(amr_path)
        run.record_output(output_path)
        amr_raw = read_tsv(amr_path)
        manifest_source = _accession_payload_manifest(amr_raw)
        cache = SequenceAnnotationCache(context.data_dir / "tmp", "amr_hits")
        parameter_hash = stable_hash(
            {
                "script": "07_annotate_amr",
                "normalized_columns": [
                    "gene_symbol",
                    "drug_class",
                    "sequence_identity",
                    "coverage_percentage",
                    "confidence_level",
                ],
            },
        )
        manifest = cache.build_manifest(
            manifest_source,
            sequence_column="sequence_accession",
            hash_columns=["raw_payload"],
            tool_version="internal_v1",
            db_version="amr.tsv",
            parameter_hash=parameter_hash,
        )
        hit_manifest, miss_manifest = cache.split_cached(manifest)
        cached_rows = cache.materialize_hits(
            hit_manifest,
            columns=[
                "sequence_accession",
                "gene_symbol",
                "drug_class",
                "sequence_identity",
                "coverage_percentage",
                "confidence_level",
            ],
        )
        if miss_manifest.empty:
            recomputed_rows = cached_rows.iloc[0:0].copy()
        else:
            miss_accessions = set(miss_manifest["sequence_accession"].astype(str))
            raw_subset = amr_raw.loc[
                amr_raw["NUCCORE_ACC"].astype(str).isin(miss_accessions)
            ].copy()
            recomputed_rows = _normalize_amr_hits(raw_subset)
            if not recomputed_rows.empty:
                cache_payload = miss_manifest[["sequence_accession", "cache_key"]].merge(
                    recomputed_rows,
                    on="sequence_accession",
                    how="inner",
                )
                cache.upsert(cache_payload)
        amr_hits = (
            pd.concat([cached_rows, recomputed_rows], ignore_index=True, sort=False)
            .sort_values(["sequence_accession", "gene_symbol", "drug_class"])
            .reset_index(drop=True)
        )
        amr_hits.to_csv(output_path, sep="\t", index=False)
        run.set_metric("cache_hits", int(len(hit_manifest)))
        run.set_metric("cache_misses", int(len(miss_manifest)))
        run.set_rows_out("plasmid_amr_hits_rows", int(len(amr_hits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
