"""Streaming FASTA readers and writers."""

from __future__ import annotations

import gzip
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO, cast

from plasmid_priority.utils.files import ensure_directory


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


@dataclass(frozen=True)
class FastaRecordSummary:
    """Minimal FASTA record inventory used for bronze-layer bookkeeping."""

    accession: str
    header: str
    description: str
    sequence_length: int


def extract_accession(header_line: str) -> str:
    """Extract the accession token from a FASTA header."""
    header = header_line[1:] if header_line.startswith(">") else header_line
    return header.split(maxsplit=1)[0].strip()


def peek_first_header(path: Path) -> str:
    """Return the first FASTA header line without trailing newline."""
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                return line.rstrip("\n")
    raise ValueError(f"No FASTA header found in {path}")


def iter_fasta_summaries(path: Path) -> Iterator[FastaRecordSummary]:
    """Yield accession, header, description, and sequence length for each record."""
    current_header: str | None = None
    sequence_length = 0

    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    accession = extract_accession(current_header)
                    description = current_header[1 + len(accession) :].strip()
                    yield FastaRecordSummary(
                        accession=accession,
                        header=current_header[1:],
                        description=description,
                        sequence_length=sequence_length,
                    )
                current_header = line
                sequence_length = 0
                continue
            sequence_length += len(line)

    if current_header is not None:
        accession = extract_accession(current_header)
        description = current_header[1 + len(accession) :].strip()
        yield FastaRecordSummary(
            accession=accession,
            header=current_header[1:],
            description=description,
            sequence_length=sequence_length,
        )


def concatenate_fastas(
    input_paths: list[Path],
    output_path: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Concatenate FASTA files in deterministic order using streaming I/O."""
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Refusing to overwrite existing FASTA: {output_path}. Use overwrite=True."
        )

    ensure_directory(output_path.parent)
    per_file: list[dict[str, object]] = []
    total_records = 0
    total_bases = 0
    temp_path: Path | None = None

    writer: BinaryIO | None = None
    if not dry_run:
        handle = tempfile.NamedTemporaryFile(
            "wb",
            dir=output_path.parent,
            delete=False,
        )
        writer = cast(BinaryIO, handle)
        temp_path = Path(handle.name)

    try:
        for input_path in input_paths:
            record_count = 0
            base_count = 0
            last_line_had_newline = True

            with _open_binary(input_path) as reader:
                for line in reader:
                    if line.startswith(b">"):
                        record_count += 1
                    else:
                        stripped = line.strip()
                        if stripped:
                            base_count += len(stripped)

                    if writer is not None:
                        writer.write(line)

                    last_line_had_newline = line.endswith(b"\n")

            if writer is not None and not last_line_had_newline:
                writer.write(b"\n")

            per_file.append(
                {
                    "path": str(input_path),
                    "record_count": record_count,
                    "base_count": base_count,
                }
            )
            total_records += record_count
            total_bases += base_count

        if writer is not None:
            writer.flush()
            writer.close()
            assert temp_path is not None
            os.replace(temp_path, output_path)

        return {
            "output_path": str(output_path),
            "record_count": total_records,
            "base_count": total_bases,
            "input_files": per_file,
            "dry_run": dry_run,
        }
    except Exception:
        if writer is not None and not writer.closed:
            writer.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
