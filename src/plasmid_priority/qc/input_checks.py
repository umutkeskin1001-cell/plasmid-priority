"""Manifest-driven validation for required repository inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from plasmid_priority.config import ProjectContext
from plasmid_priority.io.fasta import peek_first_header
from plasmid_priority.io.tabular import peek_table_columns, read_ncbi_assembly_summary_columns
from plasmid_priority.schemas import DataAssetSpec

BLAST_INDEX_SUFFIXES = (".nhr", ".nin", ".nsq")


@dataclass
class AssetCheckResult:
    key: str
    path: str
    status: str
    required: bool
    stage: str
    details: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    results: list[AssetCheckResult]
    contract_notes: list[str]

    @property
    def ok(self) -> bool:
        return not any(result.status == "error" for result in self.results)

    @property
    def errors(self) -> list[AssetCheckResult]:
        return [result for result in self.results if result.status == "error"]

    @property
    def warnings(self) -> list[AssetCheckResult]:
        return [result for result in self.results if result.status == "warning"]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "contract_notes": self.contract_notes,
            "results": [
                {
                    "key": result.key,
                    "path": result.path,
                    "status": result.status,
                    "required": result.required,
                    "stage": result.stage,
                    "details": result.details,
                }
                for result in self.results
            ],
        }

    def raise_for_errors(self) -> None:
        if not self.ok:
            keys = ", ".join(sorted(result.key for result in self.errors))
            raise RuntimeError(f"Input validation failed for: {keys}")


def _non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _status_for_missing(asset: DataAssetSpec) -> str:
    if asset.required:
        return "error"
    if asset.stage.value == "derived":
        return "missing_derived"
    return "missing_optional"


def _check_expected_columns(asset: DataAssetSpec, path: Path) -> list[str]:
    if not asset.expected_columns:
        return []
    if path.name == "assembly_summary_refseq.txt":
        columns = read_ncbi_assembly_summary_columns(path)
    else:
        delimiter = "," if path.suffix == ".csv" else "\t"
        columns = peek_table_columns(path, delimiter=delimiter)
    missing = sorted(set(asset.expected_columns) - set(columns))
    if not missing:
        return [f"Columns verified ({len(asset.expected_columns)} required fields present)."]
    return [f"Missing expected columns: {', '.join(missing)}"]


def _check_expected_children(asset: DataAssetSpec, path: Path) -> list[str]:
    missing = [name for name in asset.expected_children if not (path / name).exists()]
    if not missing:
        return [f"Expected children verified ({len(asset.expected_children)} entries)."]
    return [f"Missing expected child paths: {', '.join(missing)}"]


def _check_glob(asset: DataAssetSpec, path: Path) -> list[str]:
    if not asset.expected_glob:
        return []
    matches = sorted(path.glob(asset.expected_glob))
    if len(matches) >= (asset.min_matches or 1):
        return [f"Glob matched {len(matches)} files for pattern {asset.expected_glob}."]
    return [f"Glob matched only {len(matches)} files for pattern {asset.expected_glob}."]


def _check_amrfinder_release(path: Path) -> list[str]:
    release_dirs = sorted(child for child in path.iterdir() if child.is_dir())
    if not release_dirs:
        return ["No AMRFinder release directory found."]
    release_dir = release_dirs[-1]
    missing = [name for name in ("version.txt", "AMRProt.fa", "AMR_CDS.fa") if not (release_dir / name).exists()]
    if missing:
        return [f"Latest AMRFinder release {release_dir.name} is missing: {', '.join(missing)}."]
    return [f"Using AMRFinder release {release_dir.name}."]


def _check_blast_index_siblings(path: Path) -> list[str]:
    missing = [
        suffix
        for suffix in BLAST_INDEX_SUFFIXES
        if not list(path.parent.glob(f"*{suffix}"))
    ]
    if missing:
        return [f"Missing BLAST index suffixes in directory: {', '.join(missing)}"]
    return ["BLAST index siblings verified (.nhr, .nin, .nsq)."]


def _check_fasta_header(path: Path) -> list[str]:
    header = peek_first_header(path)
    return [f"First FASTA header: {header[:120]}"]


def _run_asset_specific_checks(asset: DataAssetSpec, path: Path) -> tuple[str, list[str]]:
    details: list[str] = []
    status = "ok"

    if asset.kind.value == "file" and not _non_empty_file(path):
        return "error", ["File is empty."]

    if asset.kind.value == "directory" and not path.is_dir():
        return "error", ["Expected a directory."]

    if asset.expected_columns:
        column_details = _check_expected_columns(asset, path)
        details.extend(column_details)
        if any(detail.startswith("Missing expected columns") for detail in column_details):
            status = "error"

    if asset.expected_children:
        child_details = _check_expected_children(asset, path)
        details.extend(child_details)
        if any(detail.startswith("Missing expected child paths") for detail in child_details):
            status = "error"

    if asset.expected_glob:
        glob_details = _check_glob(asset, path)
        details.extend(glob_details)
        if any(detail.startswith("Glob matched only") for detail in glob_details):
            status = "error"

    if asset.key == "amrfinder_db_dir":
        amr_details = _check_amrfinder_release(path)
        details.extend(amr_details)
        if any(detail.startswith("No AMRFinder") or "missing:" in detail for detail in amr_details):
            status = "error"

    if asset.key in {"resfinder_fasta", "plasmidfinder_fasta"}:
        blast_details = _check_blast_index_siblings(path)
        details.extend(blast_details)
        if any(detail.startswith("Missing BLAST") for detail in blast_details):
            status = "error"

    if asset.key in {"plsdb_sequences_fasta", "refseq_plasmids_fasta", "bronze_all_plasmids_fasta"} and path.exists():
        details.extend(_check_fasta_header(path))

    if asset.key == "plsdb_meta_tables_dir":
        child_specs = {
            "nuccore.csv": ["NUCCORE_ACC", "NUCCORE_CreateDate", "NUCCORE_Source"],
            "taxonomy.csv": ["TAXONOMY_UID", "TAXONOMY_genus", "TAXONOMY_species"],
            "plasmidfinder.csv": ["NUCCORE_ACC", "typing", "identity"],
            "amr.tsv": ["NUCCORE_ACC", "gene_symbol", "drug_class"],
        }
        for child_name, expected in child_specs.items():
            child_path = path / child_name
            delimiter = "," if child_path.suffix == ".csv" else "\t"
            columns = peek_table_columns(child_path, delimiter=delimiter)
            missing = sorted(set(expected) - set(columns))
            if missing:
                details.append(f"{child_name} is missing columns: {', '.join(missing)}")
                status = "error"
            else:
                details.append(f"{child_name} header verified.")

    return status, details


def run_input_checks(context: ProjectContext) -> ValidationReport:
    """Validate all declared assets against the local repository state."""
    results: list[AssetCheckResult] = []

    for asset in context.contract.assets:
        path = asset.resolved_path(context.root)

        if not path.exists():
            results.append(
                AssetCheckResult(
                    key=asset.key,
                    path=str(path),
                    status=_status_for_missing(asset),
                    required=asset.required,
                    stage=asset.stage.value,
                    details=[asset.description],
                )
            )
            continue

        status, details = _run_asset_specific_checks(asset, path)
        if asset.notes:
            details.extend(asset.notes)

        results.append(
            AssetCheckResult(
                key=asset.key,
                path=str(path),
                status=status,
                required=asset.required,
                stage=asset.stage.value,
                details=details,
            )
        )

    return ValidationReport(results=results, contract_notes=context.contract.notes)
