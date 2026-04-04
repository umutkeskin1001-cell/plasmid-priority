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


def build_plsdb_canonical_metadata(
    plsdb_metadata_path: Path,
    taxonomy_path: Path,
) -> pd.DataFrame:
    """Build a normalized PLSDB metadata table with joined taxonomy."""
    metadata = pd.read_csv(plsdb_metadata_path, sep="\t")
    taxonomy = pd.read_csv(taxonomy_path, usecols=TAXONOMY_COLUMNS)
    taxonomy = taxonomy.drop_duplicates(subset=["TAXONOMY_UID"])

    joined = metadata.merge(taxonomy, on="TAXONOMY_UID", how="left", validate="m:1")

    resolved_date = pd.to_datetime(joined["NUCCORE_CreateDate"], errors="coerce")
    canonical = pd.DataFrame(
        {
            "source_dataset": "plsdb",
            "sequence_accession": joined["NUCCORE_ACC"].astype(str).str.strip(),
            "nuccore_uid": joined["NUCCORE_UID"].astype("Int64"),
            "assembly_uid": joined["ASSEMBLY_UID"].astype("Int64"),
            "biosample_uid": joined["BIOSAMPLE_UID"].astype("Int64"),
            "record_origin": joined["NUCCORE_Source"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower(),
            "resolved_year": resolved_date.dt.year.astype("Int64"),
            "fasta_description": joined["NUCCORE_Description"].fillna("").astype(str).str.strip(),
            "sequence_length": joined["NUCCORE_Length"].astype("Int64"),
            "taxonomy_uid": joined["TAXONOMY_UID"].astype("Int64"),
            "TAXONOMY_phylum": joined["TAXONOMY_phylum"].fillna("").astype(str).str.strip(),
            "TAXONOMY_class": joined["TAXONOMY_class"].fillna("").astype(str).str.strip(),
            "TAXONOMY_order": joined["TAXONOMY_order"].fillna("").astype(str).str.strip(),
            "TAXONOMY_family": joined["TAXONOMY_family"].fillna("").astype(str).str.strip(),
            "genus": joined["TAXONOMY_genus"].fillna("").astype(str).str.strip(),
            "species": joined["TAXONOMY_species"].fillna("").astype(str).str.strip(),
            "topology": joined["NUCCORE_Topology"].fillna("").astype(str).str.strip().str.lower(),
            "status": joined["STATUS"].fillna("").astype(str).str.strip(),
            "metadata_status": "canonical_plsdb",
        }
    )

    canonical["sequence_accession"] = canonical["sequence_accession"].replace({"nan": ""})
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
    ensure_directory(output_path.parent)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BRONZE_INVENTORY_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in plsdb_frame.to_dict(orient="records"):
            writer.writerow(row)
            row_count += 1

        for row in refseq_rows:
            writer.writerow(row)
            row_count += 1

    return row_count
