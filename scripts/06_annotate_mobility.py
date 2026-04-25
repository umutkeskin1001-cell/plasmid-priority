#!/usr/bin/env python3
"""Write accession-level mobility annotations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.annotate import SequenceAnnotationCache, build_mobility_table
from plasmid_priority.cache import stable_hash
from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def main() -> int:
    context = build_context(PROJECT_ROOT)
    dedup_path = context.data_dir / "silver/plasmid_deduplicated.tsv"
    output_path = context.data_dir / "silver/plasmid_mobility.tsv"
    ensure_directory(output_path.parent)

    with ManagedScriptRun(context, "06_annotate_mobility") as run:
        run.record_input(dedup_path)
        run.record_output(output_path)
        records = read_tsv(dedup_path)
        cache = SequenceAnnotationCache(context.data_dir / "tmp", "mobility")
        parameter_hash = stable_hash(
            {
                "script": "06_annotate_mobility",
                "selected_columns": [
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
                ],
            },
        )
        manifest = cache.build_manifest(
            records,
            sequence_column="sequence_accession",
            hash_columns=[
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
            ],
            tool_version="internal_v1",
            db_version="n/a",
            parameter_hash=parameter_hash,
        )
        hit_manifest, miss_manifest = cache.split_cached(manifest)
        cached_rows = cache.materialize_hits(
            hit_manifest,
            columns=[
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
                "mobility_support_score",
            ],
        )
        if miss_manifest.empty:
            recomputed_rows = cached_rows.iloc[0:0].copy()
        else:
            miss_keys = set(miss_manifest["sequence_accession"].astype(str))
            to_recompute = records.loc[
                records["sequence_accession"].astype(str).isin(miss_keys)
            ].copy()
            recomputed_rows = build_mobility_table(to_recompute)
            if not recomputed_rows.empty:
                cache_payload = miss_manifest[["sequence_accession", "cache_key"]].merge(
                    recomputed_rows,
                    on="sequence_accession",
                    how="left",
                )
                cache.upsert(cache_payload)
        mobility = (
            pd.concat([cached_rows, recomputed_rows], ignore_index=True, sort=False)
            .drop_duplicates(subset=["sequence_accession"], keep="last")
            .sort_values("sequence_accession")
            .reset_index(drop=True)
        )
        mobility.to_csv(output_path, sep="\t", index=False)
        run.set_metric("cache_hits", int(len(hit_manifest)))
        run.set_metric("cache_misses", int(len(miss_manifest)))
        run.set_rows_out("plasmid_mobility_rows", int(len(mobility)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
