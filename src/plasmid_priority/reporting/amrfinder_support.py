"""Optional AMRFinder probe support for concordance checking."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd

from plasmid_priority.reporting.external_support import (
    normalize_drug_class_token,
    normalize_gene_symbol,
    select_priority_groups,
    split_field_tokens,
)
from plasmid_priority.utils.files import ensure_directory


def _fasta_index_path(input_path: Path) -> Path:
    return input_path.with_suffix(input_path.suffix + ".idx.tsv")


def _build_fasta_record_index(input_path: Path, index_path: Path) -> dict[str, tuple[int, int]]:
    ensure_directory(index_path.parent)
    rows: list[tuple[str, int, int]] = []
    current_accession = ""
    current_start = 0
    current_offset = 0
    with input_path.open("rb") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            next_offset = reader.tell()
            if line.startswith(b">"):
                if current_accession:
                    rows.append((current_accession, current_start, current_offset))
                current_accession = (
                    line[1:].split(maxsplit=1)[0].decode("utf-8", errors="ignore").strip()
                )
                current_start = current_offset
            current_offset = next_offset
    if current_accession:
        rows.append((current_accession, current_start, current_offset))

    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=index_path.parent, delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            for accession, start, end in rows:
                handle.write(f"{accession}\t{start}\t{end}\n")
        temp_path.replace(index_path)
    except (OSError, IOError):
        temp_path.unlink(missing_ok=True)
        raise
    return {accession: (start, end) for accession, start, end in rows}


def _load_fasta_record_index(input_path: Path) -> dict[str, tuple[int, int]]:
    index_path = _fasta_index_path(input_path)
    if (not index_path.exists()) or index_path.stat().st_mtime < input_path.stat().st_mtime:
        return _build_fasta_record_index(input_path, index_path)
    index: dict[str, tuple[int, int]] = {}
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            accession, start, end = line.rstrip("\n").split("\t")
            index[accession] = (int(start), int(end))
    return index


def select_amrfinder_probe_panel(
    scored: pd.DataFrame,
    backbones: pd.DataFrame,
    *,
    n_per_group: int = 6,
    score_column: str = "priority_index",
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Select a small high/low priority panel for AMRFinder concordance checks."""
    selected = select_priority_groups(
        scored,
        n_per_group=n_per_group,
        score_column=score_column,
        eligible_only=eligible_only,
    )
    if selected.empty:
        return pd.DataFrame()

    columns = ["backbone_id", "sequence_accession"]
    for optional in ("is_canonical_representative", "species", "genus"):
        if optional in backbones.columns:
            columns.append(optional)
    merged = backbones[columns].merge(
        selected[["backbone_id", "priority_group", "priority_index", "selection_score"]],
        on="backbone_id",
        how="inner",
    )
    if "is_canonical_representative" in merged.columns:
        merged["is_canonical_representative"] = (
            merged["is_canonical_representative"].fillna(False).astype(bool)
        )
        merged = merged.sort_values(
            [
                "priority_group",
                "selection_score",
                "is_canonical_representative",
                "sequence_accession",
            ],
            ascending=[True, False, False, True],
        )
    else:
        merged = merged.sort_values(
            ["priority_group", "selection_score", "sequence_accession"],
            ascending=[True, False, True],
        )
    panel = merged.groupby("backbone_id", sort=False).head(1).reset_index(drop=True)
    return panel


def write_selected_fasta_records(
    input_path: Path, accessions: list[str], output_path: Path
) -> dict[str, object]:
    """Write a small FASTA containing only the requested accessions."""
    target = set(accessions)
    found: set[str] = set()
    ensure_directory(output_path.parent)

    if input_path.suffix != ".gz":
        index = _load_fasta_record_index(input_path)
        selected = [(accession, *index[accession]) for accession in target if accession in index]
        selected.sort(key=lambda item: item[1])
        with input_path.open("rb") as reader, output_path.open("wb") as writer:
            for accession, start, end in selected:
                reader.seek(start)
                remaining = max(end - start, 0)
                while remaining > 0:
                    chunk = reader.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    writer.write(chunk)
                    remaining -= len(chunk)
                found.add(accession)
    else:
        import gzip

        capture = False
        with (
            gzip.open(input_path, "rt", encoding="utf-8") as reader,
            output_path.open("w", encoding="utf-8") as writer,
        ):
            for raw_line in reader:
                if raw_line.startswith(">"):
                    if found == target and not capture:
                        break
                    accession = raw_line[1:].split(maxsplit=1)[0].strip()
                    capture = accession in target and accession not in found
                    if capture:
                        writer.write(raw_line)
                        found.add(accession)
                    continue
                if capture:
                    writer.write(raw_line)

    missing = sorted(target - found)
    return {"requested": len(target), "found": len(found), "missing": missing}


def latest_amrfinder_release(amrfinder_db_root: Path) -> Path:
    """Return the newest AMRFinder DB release directory."""
    releases = sorted(path for path in amrfinder_db_root.iterdir() if path.is_dir())
    if not releases:
        raise FileNotFoundError(f"No AMRFinder release directory found under {amrfinder_db_root}")
    return releases[-1]


def run_amrfinder_probe(
    fasta_path: Path,
    output_path: Path,
    *,
    amrfinder_db_root: Path,
    threads: int = 1,
) -> dict[str, object]:
    """Run AMRFinder on a small nucleotide panel."""
    ensure_directory(output_path.parent)
    database_dir = latest_amrfinder_release(amrfinder_db_root)
    command = [
        "amrfinder",
        "--nucleotide",
        str(fasta_path),
        "--database",
        str(database_dir),
        "--plus",
        "--threads",
        str(threads),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return {
        "database_dir": str(database_dir),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def parse_amrfinder_probe_report(report_path: Path) -> pd.DataFrame:
    """Load and normalize the small AMRFinder probe report."""
    if not report_path.exists() or report_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=[
                "sequence_accession",
                "amrfinder_gene_symbols",
                "amrfinder_class_tokens",
                "amrfinder_hit_count",
            ]
        )

    hits = pd.read_csv(report_path, sep="\t")
    hits["Element symbol"] = hits["Element symbol"].fillna("").astype(str).str.strip()
    hits["Class"] = hits["Class"].fillna("").map(normalize_drug_class_token)
    grouped = hits.groupby("Contig id", sort=False)
    frame = grouped.agg(
        amrfinder_gene_symbols=(
            "Element symbol",
            lambda values: ",".join(sorted({value for value in values if value})),
        ),
        amrfinder_class_tokens=(
            "Class",
            lambda values: ",".join(sorted({value for value in values if value})),
        ),
        amrfinder_hit_count=("Element symbol", "size"),
    ).reset_index()
    return frame.rename(columns={"Contig id": "sequence_accession"})


def _token_set(value: object, *, normalize_fn: Callable[[object], str]) -> set[str]:
    tokens = split_field_tokens(value, separators=(",", ";"))
    return {normalize_fn(token) for token in tokens if normalize_fn(token)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def build_amrfinder_concordance_tables(
    panel: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    amrfinder_probe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare local AMRFinder probe results with the provided accession-level AMR consensus."""
    detail_columns = [
        "backbone_id",
        "priority_group",
        "priority_index",
        "sequence_accession",
        "existing_gene_count",
        "amrfinder_gene_count",
        "shared_gene_count",
        "gene_jaccard",
        "existing_class_count",
        "amrfinder_class_count",
        "shared_class_count",
        "class_jaccard",
        "amrfinder_hit_count",
        "amrfinder_any_hit",
        "any_amr_evidence",
    ]
    summary_columns = [
        "priority_group",
        "n_sequences",
        "n_with_amrfinder_hits",
        "n_with_any_amr_evidence",
        "mean_gene_jaccard",
        "median_gene_jaccard",
        "mean_gene_jaccard_nonempty",
        "median_gene_jaccard_nonempty",
        "mean_class_jaccard",
        "median_class_jaccard",
        "mean_class_jaccard_nonempty",
        "median_class_jaccard_nonempty",
        "mean_shared_gene_count",
        "mean_shared_class_count",
    ]

    if panel.empty:
        return _empty_frame(detail_columns), _empty_frame(summary_columns)

    merged = panel.merge(
        amr_consensus[["sequence_accession", "amr_gene_symbols", "amr_drug_classes"]],
        on="sequence_accession",
        how="left",
    ).merge(
        amrfinder_probe,
        on="sequence_accession",
        how="left",
    )

    detail_rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        existing_genes = _token_set(
            row.get("amr_gene_symbols", ""), normalize_fn=normalize_gene_symbol
        )
        amrfinder_genes = _token_set(
            row.get("amrfinder_gene_symbols", ""), normalize_fn=normalize_gene_symbol
        )
        existing_classes = _token_set(
            row.get("amr_drug_classes", ""), normalize_fn=normalize_drug_class_token
        )
        amrfinder_classes = _token_set(
            row.get("amrfinder_class_tokens", ""), normalize_fn=normalize_drug_class_token
        )
        amrfinder_hit_count = row.get("amrfinder_hit_count", 0)
        amrfinder_hit_count = 0 if pd.isna(amrfinder_hit_count) else int(amrfinder_hit_count)
        any_amr_evidence = bool(
            existing_genes or amrfinder_genes or existing_classes or amrfinder_classes
        )

        shared_genes = existing_genes & amrfinder_genes
        shared_classes = existing_classes & amrfinder_classes
        detail_rows.append(
            {
                "backbone_id": row["backbone_id"],
                "priority_group": row["priority_group"],
                "priority_index": row["priority_index"],
                "sequence_accession": row["sequence_accession"],
                "existing_gene_count": len(existing_genes),
                "amrfinder_gene_count": len(amrfinder_genes),
                "shared_gene_count": len(shared_genes),
                "gene_jaccard": _jaccard(existing_genes, amrfinder_genes),
                "existing_class_count": len(existing_classes),
                "amrfinder_class_count": len(amrfinder_classes),
                "shared_class_count": len(shared_classes),
                "class_jaccard": _jaccard(existing_classes, amrfinder_classes),
                "amrfinder_hit_count": amrfinder_hit_count,
                "amrfinder_any_hit": amrfinder_hit_count > 0,
                "any_amr_evidence": any_amr_evidence,
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["priority_group", "priority_index"],
        ascending=[True, False],
    )
    summary_rows: list[dict[str, object]] = []
    for priority_group, frame in detail.groupby("priority_group", sort=False):
        nonempty = frame.loc[frame["any_amr_evidence"]]
        summary_rows.append(
            {
                "priority_group": priority_group,
                "n_sequences": int(frame["sequence_accession"].nunique()),
                "n_with_amrfinder_hits": int(frame["amrfinder_any_hit"].sum()),
                "n_with_any_amr_evidence": int(frame["any_amr_evidence"].sum()),
                "mean_gene_jaccard": float(frame["gene_jaccard"].mean())
                if not nonempty.empty
                else float("nan"),
                "median_gene_jaccard": float(frame["gene_jaccard"].median())
                if not nonempty.empty
                else float("nan"),
                "mean_gene_jaccard_nonempty": float(nonempty["gene_jaccard"].mean())
                if not nonempty.empty
                else float("nan"),
                "median_gene_jaccard_nonempty": float(nonempty["gene_jaccard"].median())
                if not nonempty.empty
                else float("nan"),
                "mean_class_jaccard": float(frame["class_jaccard"].mean())
                if not nonempty.empty
                else float("nan"),
                "median_class_jaccard": float(frame["class_jaccard"].median())
                if not nonempty.empty
                else float("nan"),
                "mean_class_jaccard_nonempty": float(nonempty["class_jaccard"].mean())
                if not nonempty.empty
                else float("nan"),
                "median_class_jaccard_nonempty": float(nonempty["class_jaccard"].median())
                if not nonempty.empty
                else float("nan"),
                "mean_shared_gene_count": float(frame["shared_gene_count"].mean()),
                "mean_shared_class_count": float(frame["shared_class_count"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=summary_columns)

    overall_nonempty = detail.loc[detail["any_amr_evidence"]]
    overall = pd.DataFrame(
        [
            {
                "priority_group": "overall",
                "n_sequences": int(detail["sequence_accession"].nunique()),
                "n_with_amrfinder_hits": int(detail["amrfinder_any_hit"].sum()),
                "n_with_any_amr_evidence": int(detail["any_amr_evidence"].sum()),
                "mean_gene_jaccard": float(detail["gene_jaccard"].mean()),
                "median_gene_jaccard": float(detail["gene_jaccard"].median()),
                "mean_gene_jaccard_nonempty": float(overall_nonempty["gene_jaccard"].mean())
                if not overall_nonempty.empty
                else float("nan"),
                "median_gene_jaccard_nonempty": float(overall_nonempty["gene_jaccard"].median())
                if not overall_nonempty.empty
                else float("nan"),
                "mean_class_jaccard": float(detail["class_jaccard"].mean()),
                "median_class_jaccard": float(detail["class_jaccard"].median()),
                "mean_class_jaccard_nonempty": float(overall_nonempty["class_jaccard"].mean())
                if not overall_nonempty.empty
                else float("nan"),
                "median_class_jaccard_nonempty": float(overall_nonempty["class_jaccard"].median())
                if not overall_nonempty.empty
                else float("nan"),
                "mean_shared_gene_count": float(detail["shared_gene_count"].mean()),
                "mean_shared_class_count": float(detail["shared_class_count"].mean()),
            }
        ],
        columns=summary_columns,
    )
    return detail, pd.concat([summary, overall], ignore_index=True)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
