"""Bronze-layer metadata preparation and raw inventory generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import pandas as pd

from plasmid_priority.io.fasta import iter_fasta_summaries
from plasmid_priority.utils.files import ensure_directory

TAXONOMY_COLUMNS = [
    "TAXONOMY_UID",
    "TAXONOMY_superkingdom",
    "TAXONOMY_phylum",
    "TAXONOMY_class",
    "TAXONOMY_order",
    "TAXONOMY_family",
    "TAXONOMY_genus",
    "TAXONOMY_species",
]

BRONZE_INVENTORY_COLUMNS = [
    "source_dataset",
    "sequence_accession",
    "nuccore_uid",
    "assembly_uid",
    "biosample_uid",
    "record_origin",
    "resolved_year",
    "fasta_description",
    "sequence_length",
    "taxonomy_uid",
    "TAXONOMY_phylum",
    "TAXONOMY_class",
    "TAXONOMY_order",
    "TAXONOMY_family",
    "genus",
    "species",
    "topology",
    "status",
    "metadata_status",
]

_TEXT_NULLS = {"", "nan", "none", "null"}


def _normalize_text_series(series: pd.Series, *, lower: bool = False) -> pd.Series:
    """Normalize text fields to stripped strings with sentinel nulls removed."""
    cleaned = series.astype("string").fillna("").str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(_TEXT_NULLS), "")
    if lower:
        cleaned = cleaned.str.lower()
    return cleaned


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _build_plsdb_canonical_metadata_duckdb(
    plsdb_metadata_path: Path,
    taxonomy_path: Path,
) -> pd.DataFrame | None:
    try:
        import duckdb
    except ImportError:
        return None

    query = f"""
        WITH metadata AS (
            SELECT
                NUCCORE_ACC,
                NUCCORE_UID,
                ASSEMBLY_UID,
                BIOSAMPLE_UID,
                NUCCORE_Source,
                NUCCORE_CreateDate,
                NUCCORE_Description,
                NUCCORE_Length,
                CAST(TAXONOMY_UID AS BIGINT) AS TAXONOMY_UID,
                NUCCORE_Topology,
                STATUS
            FROM read_csv_auto({_sql_literal(plsdb_metadata_path)}, delim='\\t', header=True)
        ),
        taxonomy AS (
            SELECT DISTINCT
                CAST(TAXONOMY_UID AS BIGINT) AS TAXONOMY_UID,
                TAXONOMY_phylum,
                TAXONOMY_class,
                TAXONOMY_order,
                TAXONOMY_family,
                TAXONOMY_genus,
                TAXONOMY_species
            FROM read_csv_auto({_sql_literal(taxonomy_path)}, delim='\\t', header=True)
        )
        SELECT
            'plsdb' AS source_dataset,
            metadata.NUCCORE_ACC,
            metadata.NUCCORE_UID,
            metadata.ASSEMBLY_UID,
            metadata.BIOSAMPLE_UID,
            metadata.NUCCORE_Source,
            metadata.NUCCORE_CreateDate,
            metadata.NUCCORE_Description,
            metadata.NUCCORE_Length,
            metadata.TAXONOMY_UID,
            taxonomy.TAXONOMY_phylum,
            taxonomy.TAXONOMY_class,
            taxonomy.TAXONOMY_order,
            taxonomy.TAXONOMY_family,
            taxonomy.TAXONOMY_genus,
            taxonomy.TAXONOMY_species,
            metadata.NUCCORE_Topology,
            metadata.STATUS
        FROM metadata
        LEFT JOIN taxonomy USING (TAXONOMY_UID)
    """
    with duckdb.connect(database=":memory:") as connection:
        return connection.execute(query).fetchdf()


def build_plsdb_canonical_metadata(
    plsdb_metadata_path: Path,
    taxonomy_path: Path,
    *,
    use_duckdb: bool = False,
) -> pd.DataFrame:
    """Build a normalized PLSDB metadata table with joined taxonomy."""
    joined = pd.DataFrame()
    if use_duckdb:
        duckdb_joined = _build_plsdb_canonical_metadata_duckdb(plsdb_metadata_path, taxonomy_path)
        if duckdb_joined is not None and not duckdb_joined.empty:
            joined = duckdb_joined
    if joined.empty:
        metadata = pd.read_csv(plsdb_metadata_path, sep="\t")
        taxonomy = pd.read_csv(taxonomy_path, sep="\t", usecols=TAXONOMY_COLUMNS)
        taxonomy = taxonomy.drop_duplicates(subset=["TAXONOMY_UID"])
        joined = metadata.merge(taxonomy, on="TAXONOMY_UID", how="left", validate="m:1")

    resolved_date = pd.to_datetime(joined["NUCCORE_CreateDate"], errors="coerce")
    canonical = pd.DataFrame(
        {
            "source_dataset": "plsdb",
            "sequence_accession": _normalize_text_series(joined["NUCCORE_ACC"]),
            "nuccore_uid": joined["NUCCORE_UID"].astype("Int64"),
            "assembly_uid": joined["ASSEMBLY_UID"].astype("Int64"),
            "biosample_uid": joined["BIOSAMPLE_UID"].astype("Int64"),
            "record_origin": _normalize_text_series(joined["NUCCORE_Source"], lower=True),
            "resolved_year": resolved_date.dt.year.astype("Int64"),
            "fasta_description": _normalize_text_series(joined["NUCCORE_Description"]),
            "sequence_length": joined["NUCCORE_Length"].astype("Int64"),
            "taxonomy_uid": joined["TAXONOMY_UID"].astype("Int64"),
            "TAXONOMY_phylum": _normalize_text_series(joined["TAXONOMY_phylum"]),
            "TAXONOMY_class": _normalize_text_series(joined["TAXONOMY_class"]),
            "TAXONOMY_order": _normalize_text_series(joined["TAXONOMY_order"]),
            "TAXONOMY_family": _normalize_text_series(joined["TAXONOMY_family"]),
            "genus": _normalize_text_series(joined["TAXONOMY_genus"]),
            "species": _normalize_text_series(joined["TAXONOMY_species"]),
            "topology": _normalize_text_series(joined["NUCCORE_Topology"], lower=True),
            "status": _normalize_text_series(joined["STATUS"]),
            "metadata_status": "canonical_plsdb",
        }
    )

    return canonical


def write_plsdb_canonical_metadata(
    plsdb_metadata_path: Path,
    taxonomy_path: Path,
    output_path: Path,
) -> int:
    """Write normalized PLSDB metadata to TSV and return the row count."""
    frame = build_plsdb_canonical_metadata(plsdb_metadata_path, taxonomy_path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, sep="\t", index=False)
    return int(len(frame))


def iter_refseq_inventory_rows(
    refseq_fasta_path: Path, *, limit: int | None = None
) -> Iterator[dict[str, object]]:
    """Yield raw RefSeq FASTA inventory rows for the bronze layer."""
    for index, record in enumerate(iter_fasta_summaries(refseq_fasta_path), start=1):
        yield {
            "source_dataset": "refseq_raw",
            "sequence_accession": record.accession,
            "nuccore_uid": "",
            "assembly_uid": "",
            "biosample_uid": "",
            "record_origin": "refseq",
            "resolved_year": "",
            "fasta_description": record.description,
            "sequence_length": record.sequence_length,
            "taxonomy_uid": "",
            "TAXONOMY_phylum": "",
            "TAXONOMY_class": "",
            "TAXONOMY_order": "",
            "TAXONOMY_family": "",
            "genus": "",
            "species": "",
            "topology": "",
            "status": "",
            "metadata_status": "header_only_raw_inventory",
        }
        if limit is not None and index >= limit:
            break


def write_bronze_inventory(
    plsdb_frame: pd.DataFrame,
    refseq_rows: Iterator[dict[str, object]],
    output_path: Path,
) -> int:
    """Write the heterogeneous bronze inventory table as TSV."""
    aligned_plsdb = plsdb_frame.loc[:, BRONZE_INVENTORY_COLUMNS]
    ensure_directory(output_path.parent)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(BRONZE_INVENTORY_COLUMNS)

        for row in aligned_plsdb.itertuples(index=False, name=None):
            writer.writerow(row)
            row_count += 1

        for row in refseq_rows:
            writer.writerow([row[column] for column in BRONZE_INVENTORY_COLUMNS])
            row_count += 1

    return row_count
