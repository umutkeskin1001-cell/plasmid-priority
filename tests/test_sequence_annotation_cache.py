from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from plasmid_priority.annotate.sequence_cache import SequenceAnnotationCache


def test_sequence_annotation_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = SequenceAnnotationCache(Path(tmp_dir), "mobility")
        source = pd.DataFrame(
            [
                {"sequence_accession": "A", "predicted_mobility": "mobilizable"},
                {"sequence_accession": "B", "predicted_mobility": "conjugative"},
            ],
        )
        manifest = cache.build_manifest(
            source,
            sequence_column="sequence_accession",
            hash_columns=["predicted_mobility"],
            tool_version="internal_v1",
            db_version="n/a",
            parameter_hash="p1",
        )
        hit_manifest, miss_manifest = cache.split_cached(manifest)
        assert hit_manifest.empty
        assert len(miss_manifest) == 2

        rows = miss_manifest[["sequence_accession", "cache_key"]].merge(
            pd.DataFrame(
                [
                    {"sequence_accession": "A", "score": 0.1},
                    {"sequence_accession": "B", "score": 0.2},
                ],
            ),
            on="sequence_accession",
            how="inner",
        )
        cache.upsert(rows)

        hit_manifest2, miss_manifest2 = cache.split_cached(manifest)
        assert len(hit_manifest2) == 2
        assert miss_manifest2.empty
        materialized = cache.materialize_hits(
            hit_manifest2, columns=["sequence_accession", "score"]
        )
        assert len(materialized) == 2
        assert set(materialized["sequence_accession"]) == {"A", "B"}


def test_sequence_annotation_cache_invalidates_on_input_change() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = SequenceAnnotationCache(Path(tmp_dir), "amr_hits")
        base = pd.DataFrame([{"sequence_accession": "A", "raw_payload": "x"}])
        manifest_1 = cache.build_manifest(
            base,
            sequence_column="sequence_accession",
            hash_columns=["raw_payload"],
            tool_version="internal_v1",
            db_version="amr.tsv",
            parameter_hash="p1",
        )
        cache.upsert(
            manifest_1[["sequence_accession", "cache_key"]].assign(gene_symbol="blaTEM"),
        )
        hit_manifest, miss_manifest = cache.split_cached(manifest_1)
        assert len(hit_manifest) == 1
        assert miss_manifest.empty

        changed = pd.DataFrame([{"sequence_accession": "A", "raw_payload": "x_changed"}])
        manifest_2 = cache.build_manifest(
            changed,
            sequence_column="sequence_accession",
            hash_columns=["raw_payload"],
            tool_version="internal_v1",
            db_version="amr.tsv",
            parameter_hash="p1",
        )
        hit_manifest2, miss_manifest2 = cache.split_cached(manifest_2)
        assert hit_manifest2.empty
        assert len(miss_manifest2) == 1
