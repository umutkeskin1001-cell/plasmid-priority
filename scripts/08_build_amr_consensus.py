#!/usr/bin/env python3
"""Aggregate AMR hits into accession-level consensus summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import SequenceAnnotationCache, build_amr_consensus
from plasmid_priority.cache import stable_hash
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def _hits_manifest_source(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame(columns=["sequence_accession", "hits_payload"])
    working = hits.copy()
    working["sequence_accession"] = working["sequence_accession"].astype(str)
    grouped = (
        working.sort_values(["sequence_accession"])
        .groupby("sequence_accession", sort=False)
        .apply(lambda frame: frame.to_json(orient="records", date_format="iso"))
        .reset_index(name="hits_payload")
    )
    return grouped  # type: ignore


def main() -> int:
    context = build_context(PROJECT_ROOT)
    hits_path = context.data_dir / "silver/plasmid_amr_hits.tsv"
    output_path = context.data_dir / "silver/plasmid_amr_consensus.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "08_build_amr_consensus") as run:
        run.record_input(hits_path)
        run.record_output(output_path)
        hits = read_tsv(hits_path)
        cache = SequenceAnnotationCache(context.data_dir / "tmp", "amr_consensus")
        parameter_hash = stable_hash({"script": "08_build_amr_consensus"})
        manifest_source = _hits_manifest_source(hits)
        manifest = cache.build_manifest(
            manifest_source,
            sequence_column="sequence_accession",
            hash_columns=["hits_payload"],
            tool_version="internal_v1",
            db_version="amr_hits_v1",
            parameter_hash=parameter_hash,
        )
        hit_manifest, miss_manifest = cache.split_cached(manifest)
        cached_rows = cache.materialize_hits(
            hit_manifest,
            columns=[
                "sequence_accession",
                "amr_gene_symbols",
                "amr_drug_classes",
                "amr_gene_count",
                "amr_class_count",
                "amr_hit_count",
                "amr_any",
            ],
        )
        if miss_manifest.empty:
            recomputed_rows = cached_rows.iloc[0:0].copy()
        else:
            miss_accessions = set(miss_manifest["sequence_accession"].astype(str))
            recomputed_rows = build_amr_consensus(
                hits.loc[hits["sequence_accession"].astype(str).isin(miss_accessions)].copy(),
            )
            if not recomputed_rows.empty:
                cache_payload = miss_manifest[["sequence_accession", "cache_key"]].merge(
                    recomputed_rows,
                    on="sequence_accession",
                    how="inner",
                )
                cache.upsert(cache_payload)
        consensus = (
            pd.concat([cached_rows, recomputed_rows], ignore_index=True, sort=False)
            .drop_duplicates(subset=["sequence_accession"], keep="last")
            .sort_values("sequence_accession")
            .reset_index(drop=True)
        )
        consensus.to_csv(output_path, sep="\t", index=False)
        run.set_metric("cache_hits", int(len(hit_manifest)))
        run.set_metric("cache_misses", int(len(miss_manifest)))
        run.set_rows_out("plasmid_amr_consensus_rows", int(len(consensus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
