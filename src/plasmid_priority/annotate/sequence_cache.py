"""Sequence-level persistent cache for annotation scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plasmid_priority.cache import stable_hash
from plasmid_priority.utils.files import ensure_directory

LOGGER = logging.getLogger(__name__)


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _hash_series_row(row: pd.Series) -> str:
    payload = {
        str(k): (_stringify(v) if not isinstance(v, (int, float, bool)) else v)
        for k, v in row.items()
    }
    return stable_hash(payload)


@dataclass
class SequenceAnnotationCache:
    """Per-step cache for sequence/accession-level annotation outputs."""

    cache_root: Path
    step_name: str

    def __post_init__(self) -> None:
        base = ensure_directory(self.cache_root / "annotation_sequence_cache")
        self.table_path = base / f"{self.step_name}.parquet"

    def build_key(
        self,
        *,
        sequence_accession: str,
        input_hash: str,
        tool_version: str,
        db_version: str,
        parameter_hash: str,
    ) -> str:
        return stable_hash(
            {
                "sequence_accession": _stringify(sequence_accession),
                "input_hash": _stringify(input_hash),
                "tool_version": _stringify(tool_version),
                "db_version": _stringify(db_version),
                "parameter_hash": _stringify(parameter_hash),
            },
        )

    def build_manifest(
        self,
        frame: pd.DataFrame,
        *,
        sequence_column: str,
        hash_columns: list[str] | None = None,
        tool_version: str,
        db_version: str,
        parameter_hash: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["sequence_accession", "input_hash", "cache_key"])
        working = frame.pipe(pd.DataFrame.copy)
        working[sequence_column] = working[sequence_column].astype(str)
        if hash_columns is None:
            hash_columns = [col for col in working.columns if col != sequence_column]
        subset = working[[sequence_column, *hash_columns]].copy()
        subset = subset.sort_values([sequence_column]).reset_index(drop=True)
        subset["input_hash"] = subset[hash_columns].apply(_hash_series_row, axis=1)
        subset = subset.rename(columns={sequence_column: "sequence_accession"})
        subset["cache_key"] = subset.apply(
            lambda row: self.build_key(
                sequence_accession=str(row["sequence_accession"]),
                input_hash=str(row["input_hash"]),
                tool_version=tool_version,
                db_version=db_version,
                parameter_hash=parameter_hash,
            ),
            axis=1,
        )
        return subset[["sequence_accession", "input_hash", "cache_key"]]

    def load(self) -> pd.DataFrame:
        if not self.table_path.exists():
            return pd.DataFrame()
        try:
            cached = pd.read_parquet(self.table_path)
        except Exception as exc:
            LOGGER.warning(
                "Caught suppressed exception: %s",
                exc,
                exc_info=True,
            )
            return pd.DataFrame()
        return cached if isinstance(cached, pd.DataFrame) else pd.DataFrame()

    def split_cached(self, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if manifest.empty:
            return manifest.copy(), manifest.copy()
        cached = self.load()
        if cached.empty or "cache_key" not in cached.columns:
            return manifest.iloc[0:0].copy(), manifest.copy()
        cached_keys = set(cached["cache_key"].astype(str))
        hits = manifest.loc[manifest["cache_key"].astype(str).isin(cached_keys)].copy()
        misses = manifest.loc[~manifest["cache_key"].astype(str).isin(cached_keys)].copy()
        return hits, misses

    def materialize_hits(
        self,
        hit_manifest: pd.DataFrame,
        *,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if hit_manifest.empty:
            return pd.DataFrame(columns=columns or [])
        cached = self.load()
        if cached.empty:
            return pd.DataFrame(columns=columns or [])
        merged = hit_manifest[["cache_key"]].merge(cached, on="cache_key", how="inner")
        if columns:
            available = [col for col in columns if col in merged.columns]
            return merged[available].copy()
        return merged.copy()

    def upsert(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        existing = self.load()
        merged = pd.concat([existing, rows], ignore_index=True, sort=False)
        if "cache_key" in merged.columns:
            merged = merged.drop_duplicates(subset=["cache_key"], keep="last")
        ensure_directory(self.table_path.parent)
        merged.to_parquet(self.table_path, index=False)
