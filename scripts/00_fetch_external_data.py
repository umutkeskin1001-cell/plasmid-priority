#!/usr/bin/env python3
"""Fetch the external data assets required by the repository contract."""

from __future__ import annotations

import argparse
import json
import gzip
import io
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.utils.files import ensure_directory, atomic_write_json

AMRFINDER_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/"
CARD_ARCHIVE_URL = "https://card.mcmaster.ca/latest/data"
WHO_MIA_PDF_URL = "https://cdn.who.int/media/docs/default-source/gcp/who-mia-list-2024-lv.pdf"
PATHOGEN_RESULTS_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/"
RESFINDER_DOWNLOADS_URL = "https://bitbucket.org/genomicepidemiology/resfinder_db/downloads/?iframe=true&spa=0&tab=tags"
PLASMIDFINDER_DB_DOWNLOAD_URL = "https://bitbucket.org/genomicepidemiology/plasmidfinder_db/get/2.1.tar.gz"
MOBSUITE_ARCHIVE_URL = "https://zenodo.org/records/10304948/files/data.tar.gz?download=1"
VFDB_ANNOTATIONS_URL = "https://www.mgc.ac.cn/VFs/Down/VFs.xls.gz"
CONJSCAN_REPO_ZIP_URL = "https://codeload.github.com/macsy-models/CONJScan/zip/refs/heads/main"
BACMET_EXP_MAPPING_URL = "http://bacmet.biomedicine.gu.se/download/BacMet2_EXP.753.mapping.txt"
BACMET_PRE_MAPPING_URL = "http://bacmet.biomedicine.gu.se/download/BacMet2_PRE.155512.mapping.txt"
KEGG_REST_BASE_URL = "https://rest.kegg.jp/"

DEFAULT_PATHOGEN_GROUPS = (
    "Acinetobacter",
    "Campylobacter",
    "Clostridioides_difficile",
    "Enterococcus_faecalis",
    "Enterococcus_faecium",
    "Escherichia_coli_Shigella",
    "Klebsiella",
    "Pseudomonas_aeruginosa",
    "Salmonella",
    "Staphylococcus_aureus",
    "Streptococcus_pneumoniae",
)

PATHOGEN_OUTPUT_COLUMNS = (
    "#Organism group",
    "Scientific name",
    "Location",
    "Source type",
    "Assembly",
    "AMR genotypes",
    "Host",
    "asm_acc",
    "geo_loc_name",
    "scientific_name",
    "source_type",
    "host",
    "species_taxid",
    "taxid",
)


def _download(url: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    request = Request(url, headers={"User-Agent": "plasmid-priority-data-fetch/1.0"})
    with urlopen(request) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "plasmid-priority-data-fetch/1.0"})
    with urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_if_needed(url: str, output_path: Path, *, force: bool = False) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return False
    _download(url, output_path)
    return True


def _temporary_path(*, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.close()
    return Path(handle.name)


def _bitbucket_latest_download_url(downloads_url: str, *, extension: str = "tar.bz2") -> str:
    html = _download_text(downloads_url)
    match = re.search(rf'href="([^"]+\.{re.escape(extension)})"', html)
    if not match:
        raise FileNotFoundError(f"Could not locate a {extension} archive in {downloads_url}")
    return urljoin(downloads_url, match.group(1))


def _tar_contains_member(archive_path: Path, member_name: str) -> bool:
    if not archive_path.exists() or archive_path.stat().st_size <= 0:
        return False
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                normalized = member.name.lstrip("./")
                if normalized == member_name or normalized.endswith(f"/{member_name}"):
                    return True
    except tarfile.TarError:
        return False
    return False


def _extract_first_tar_member(
    archive_path: Path,
    *,
    member_name: str,
    output_path: Path,
) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            normalized = member.name.lstrip("./")
            if normalized == member_name or normalized.endswith(f"/{member_name}"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"Member {member_name} was not readable in {archive_path}")
                with output_path.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                return
    raise FileNotFoundError(f"Member {member_name} not found in {archive_path}")


def _extract_concatenated_fasta_members(
    archive_path: Path,
    *,
    output_path: Path,
    include_suffix: str = ".fsa",
) -> list[str]:
    member_names: list[str] = []
    with tarfile.open(archive_path, "r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.lower().endswith(include_suffix)
            and "/test/" not in member.name.lower()
            and "/tests/" not in member.name.lower()
        ]
        members.sort(key=lambda member: member.name)
        with output_path.open("wb") as handle:
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                if not data:
                    continue
                handle.write(data)
                if not data.endswith(b"\n"):
                    handle.write(b"\n")
                member_names.append(member.name)
    return member_names


def _latest_amrfinder_version() -> str:
    version = _download_text(urljoin(AMRFINDER_BASE_URL, "version.txt")).strip()
    if not version:
        raise RuntimeError("AMRFinder version.txt was empty.")
    return version


def fetch_amrfinder_database(external_root: Path, *, force: bool = False) -> dict[str, object]:
    amrfinder_root = ensure_directory(external_root / "amrfinder_db")
    version = _latest_amrfinder_version()
    release_dir = ensure_directory(amrfinder_root / version)
    files = ("version.txt", "AMRProt.fa", "AMR_CDS.fa", "database_format_version.txt")
    for filename in files:
        _download_if_needed(urljoin(AMRFINDER_BASE_URL, filename), release_dir / filename, force=force)
    return {
        "version": version,
        "release_dir": str(release_dir),
        "files": [str(release_dir / filename) for filename in files],
    }


def fetch_card_archive(external_root: Path, *, force: bool = False) -> dict[str, object]:
    card_root = ensure_directory(external_root / "card")
    archive_path = card_root / "card-data.tar.bz2"
    if not force and _tar_contains_member(archive_path, "card.json"):
        return {"path": str(archive_path), "downloaded": False}
    changed = _download_if_needed(CARD_ARCHIVE_URL, archive_path, force=True)
    return {"path": str(archive_path), "downloaded": changed}


def fetch_resfinder_database(external_root: Path, *, force: bool = False) -> dict[str, object]:
    resfinder_root = ensure_directory(external_root / "resfinder_db")
    fasta_path = resfinder_root / "all.fsa"
    if not force and fasta_path.exists() and fasta_path.stat().st_size > 0:
        return {"path": str(fasta_path), "downloaded": False}

    archive_url = _bitbucket_latest_download_url(RESFINDER_DOWNLOADS_URL)
    temp_archive = _temporary_path(suffix=".tar.bz2")
    try:
        _download(archive_url, temp_archive)
        _extract_first_tar_member(temp_archive, member_name="all.fsa", output_path=fasta_path)
        return {
            "path": str(fasta_path),
            "downloaded": True,
            "source_archive": archive_url,
        }
    finally:
        temp_archive.unlink(missing_ok=True)


def fetch_plasmidfinder_database(external_root: Path, *, force: bool = False) -> dict[str, object]:
    plasmid_root = ensure_directory(external_root / "plasmidfinder_db")
    fasta_path = plasmid_root / "plasmidfinder_all.fsa"
    if not force and fasta_path.exists() and fasta_path.stat().st_size > 0:
        return {"path": str(fasta_path), "downloaded": False}

    temp_archive = _temporary_path(suffix=".tar.gz")
    try:
        _download(PLASMIDFINDER_DB_DOWNLOAD_URL, temp_archive)
        source_members = _extract_concatenated_fasta_members(temp_archive, output_path=fasta_path)
        return {
            "path": str(fasta_path),
            "downloaded": True,
            "source_archive": PLASMIDFINDER_DB_DOWNLOAD_URL,
            "source_members": source_members,
        }
    finally:
        temp_archive.unlink(missing_ok=True)


def fetch_mobsuite_database(external_root: Path, *, force: bool = False) -> dict[str, object]:
    mobsuite_root = ensure_directory(external_root / "mobsuite_db")
    tar_path = mobsuite_root / "data.tar"
    if not force and _tar_contains_member(tar_path, "host_range_literature_plasmidDB.txt") and _tar_contains_member(
        tar_path, "clusters.txt"
    ):
        return {"path": str(tar_path), "downloaded": False}

    temp_archive = _temporary_path(suffix=".tar.gz")
    temp_tar = _temporary_path(suffix=".tar")
    try:
        _download(MOBSUITE_ARCHIVE_URL, temp_archive)
        with tarfile.open(temp_archive, "r:gz") as source_archive, tarfile.open(temp_tar, "w") as output_archive:
            for target_name in ("host_range_literature_plasmidDB.txt", "clusters.txt"):
                matched = None
                for member in source_archive.getmembers():
                    normalized = member.name.lstrip("./")
                    if normalized == target_name or normalized.endswith(f"/{target_name}"):
                        matched = member
                        break
                if matched is None:
                    raise FileNotFoundError(f"{target_name} not found in MOB-suite archive.")
                extracted = source_archive.extractfile(matched)
                if extracted is None:
                    raise FileNotFoundError(f"{target_name} could not be extracted from MOB-suite archive.")
                payload = extracted.read()
                info = tarfile.TarInfo(name=target_name)
                info.size = len(payload)
                output_archive.addfile(info, io.BytesIO(payload))
        shutil.move(str(temp_tar), str(tar_path))
        return {
            "path": str(tar_path),
            "downloaded": True,
            "source_archive": MOBSUITE_ARCHIVE_URL,
        }
    finally:
        temp_archive.unlink(missing_ok=True)
        temp_tar.unlink(missing_ok=True)


def fetch_vfdb_annotations(external_root: Path, *, force: bool = False) -> dict[str, object]:
    vfdb_root = ensure_directory(external_root / "vfdb")
    output_path = vfdb_root / "vfdb_annotations.tsv"
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        return {"path": str(output_path), "downloaded": False}

    import xlrd  # type: ignore[import-not-found]

    temp_path = _temporary_path(suffix=".xls.gz")
    try:
        _download(VFDB_ANNOTATIONS_URL, temp_path)
        with gzip.open(temp_path, "rb") as handle:
            raw = handle.read()
        workbook = xlrd.open_workbook(file_contents=raw)
        sheet = workbook.sheet_by_index(0)
        headers = [
            str(value).strip() if str(value).strip() else f"col_{index}"
            for index, value in enumerate(sheet.row_values(1))
        ]
        rows: list[dict[str, object]] = []
        for row_index in range(2, sheet.nrows):
            values = sheet.row_values(row_index)
            if not any(str(value).strip() for value in values):
                continue
            row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
        return {"path": str(output_path), "downloaded": True}
    finally:
        temp_path.unlink(missing_ok=True)


def fetch_bacmet_annotations(external_root: Path, *, force: bool = False) -> dict[str, object]:
    bacmet_root = ensure_directory(external_root / "bacmet")
    output_path = bacmet_root / "bacmet_annotations.tsv"
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        return {"path": str(output_path), "downloaded": False}

    def _load_tsv(url: str) -> pd.DataFrame:
        request = Request(url, headers={"User-Agent": "plasmid-priority-data-fetch/1.0"})
        with urlopen(request) as response:
            return pd.read_csv(response, sep="\t", dtype=str, low_memory=False)

    exp = _load_tsv(BACMET_EXP_MAPPING_URL)
    exp = exp.rename(
        columns={
            "BacMet_ID": "bacmet_id",
            "Gene_name": "gene_name",
            "Accession": "accession",
            "Organism": "organism",
            "Location": "location",
            "Compound": "compound",
        }
    )
    exp["dataset"] = "BacMet2_EXP"
    exp["gi_number"] = ""
    exp["genbank_id"] = ""
    exp["ncbi_annotation"] = ""

    pred = _load_tsv(BACMET_PRE_MAPPING_URL)
    pred = pred.rename(
        columns={
            "GI_number": "gi_number",
            "GenBank_ID": "genbank_id",
            "Gene_name": "gene_name",
            "Organism": "organism",
            "Compound": "compound",
            "NCBI_annotation": "ncbi_annotation",
        }
    )
    pred["dataset"] = "BacMet2_PRE"
    pred["bacmet_id"] = ""
    pred["accession"] = pred.get("genbank_id", "")
    pred["location"] = ""

    columns = [
        "dataset",
        "bacmet_id",
        "gi_number",
        "genbank_id",
        "accession",
        "gene_name",
        "organism",
        "location",
        "compound",
        "ncbi_annotation",
    ]
    combined = pd.concat([exp.reindex(columns=columns), pred.reindex(columns=columns)], ignore_index=True)
    combined.to_csv(output_path, sep="\t", index=False)
    return {
        "path": str(output_path),
        "downloaded": True,
        "rows": int(len(combined)),
    }


def fetch_kegg_tables(external_root: Path, *, force: bool = False) -> dict[str, object]:
    kegg_root = ensure_directory(external_root / "kegg")
    pathways_path = kegg_root / "pathways.tsv"
    ko_pathway_path = kegg_root / "ko_pathway.tsv"
    ko_list_path = kegg_root / "ko_list.tsv"
    if (
        not force
        and pathways_path.exists()
        and pathways_path.stat().st_size > 0
        and ko_pathway_path.exists()
        and ko_pathway_path.stat().st_size > 0
        and ko_list_path.exists()
        and ko_list_path.stat().st_size > 0
    ):
        return {
            "pathways_path": str(pathways_path),
            "ko_pathway_path": str(ko_pathway_path),
            "ko_list_path": str(ko_list_path),
            "downloaded": False,
        }

    pathways = _download_text(urljoin(KEGG_REST_BASE_URL, "list/pathway"))
    ko_list = _download_text(urljoin(KEGG_REST_BASE_URL, "list/ko"))
    ko_pathway = _download_text(urljoin(KEGG_REST_BASE_URL, "link/pathway/ko"))

    def _two_column_tsv(text: str, output_path: Path, first_name: str, second_name: str) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            left, right = line.split("\t", 1)
            rows.append({first_name: left.replace("path:", "").replace("ko:", ""), second_name: right})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_path, sep="\t", index=False)
        return frame

    pathways_frame = _two_column_tsv(pathways, pathways_path, "pathway_id", "pathway_name")
    ko_list_frame = _two_column_tsv(ko_list, ko_list_path, "ko_id", "ko_description")
    ko_pathway_rows: list[dict[str, str]] = []
    for line in ko_pathway.splitlines():
        line = line.strip()
        if not line:
            continue
        left, right = line.split("\t", 1)
        ko_pathway_rows.append(
            {
                "ko_id": left.replace("ko:", ""),
                "pathway_id": right.replace("path:", ""),
            }
        )
    ko_pathway_frame = pd.DataFrame(ko_pathway_rows)
    ko_pathway_frame.to_csv(ko_pathway_path, sep="\t", index=False)
    return {
        "pathways_path": str(pathways_path),
        "ko_pathway_path": str(ko_pathway_path),
        "ko_list_path": str(ko_list_path),
        "downloaded": True,
        "pathway_rows": int(len(pathways_frame)),
        "ko_rows": int(len(ko_list_frame)),
        "ko_pathway_rows": int(len(ko_pathway_frame)),
    }


def fetch_conjscan_annotations(external_root: Path, *, force: bool = False) -> dict[str, object]:
    conjscan_root = ensure_directory(external_root / "conjscan")
    output_path = conjscan_root / "conjscan_annotations.tsv"
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        return {"path": str(output_path), "downloaded": False}

    temp_zip = _temporary_path(suffix=".zip")
    try:
        _download(CONJSCAN_REPO_ZIP_URL, temp_zip)
        with zipfile.ZipFile(temp_zip) as archive:
            member_names = archive.namelist()
            definition_members = sorted(
                name for name in member_names if "/definitions/" in name and name.endswith(".xml")
            )
            profile_members = sorted(
                name for name in member_names if "/profiles/" in name and name.endswith(".hmm")
            )
            rows: list[dict[str, object]] = []
            for definition_name in definition_members:
                scope = Path(definition_name).parts[-2]
                system_name = Path(definition_name).stem
                if system_name == "MOB":
                    profile_prefix = "T4SS_MOB"
                elif system_name.startswith("dCONJ_"):
                    profile_prefix = "T4SS_" + system_name.removeprefix("dCONJ_type")
                elif system_name.startswith("T4SS_type"):
                    profile_prefix = "T4SS_" + system_name.removeprefix("T4SS_type")
                else:
                    profile_prefix = system_name
                matched_profiles = [name for name in profile_members if Path(name).stem.startswith(profile_prefix)]
                rows.append(
                    {
                        "package": "CONJScan",
                        "scope": scope,
                        "system_name": system_name,
                        "definition_file": definition_name,
                        "profile_prefix": profile_prefix,
                        "profile_count": len(matched_profiles),
                        "profile_files": ";".join(matched_profiles),
                        "source_repository": "https://github.com/macsy-models/CONJScan",
                    }
                )
        pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
        return {"path": str(output_path), "downloaded": True, "rows": len(rows)}
    finally:
        temp_zip.unlink(missing_ok=True)


def fetch_iceberg_annotations(external_root: Path, *, force: bool = False) -> dict[str, object]:
    iceberg_root = ensure_directory(external_root / "iceberg")
    output_path = iceberg_root / "iceberg_annotations.tsv"
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        return {"path": str(output_path), "downloaded": False}

    def _read_json_table(url: str, *, source_table: str, source_key: str) -> pd.DataFrame:
        payload = _download_text(url)
        frame = pd.read_json(io.StringIO(payload))
        frame["source_table"] = source_table
        frame["source_key"] = source_key
        return frame

    frames = [
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getdata1.php?key=ICE"), source_table="getdata1", source_key="ICE"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getdata1.php?key=IME"), source_table="getdata1", source_key="IME"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getdata1.php?key=CIME"), source_table="getdata1", source_key="CIME"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=or"), source_table="getmeta", source_key="or"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=sk"), source_table="getmeta", source_key="sk"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=ga"), source_table="getmeta", source_key="ga"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=ur"), source_table="getmeta", source_key="ur"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=wa"), source_table="getmeta", source_key="wa"),
        _read_json_table(urljoin("https://tool2-mml.sjtu.edu.cn/ICEberg3/", "getmeta.php?key=na"), source_table="getmeta", source_key="na"),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output_path, sep="\t", index=False)
    return {"path": str(output_path), "downloaded": True, "rows": int(len(combined))}


def _parse_latest_metadata_filename(directory_url: str) -> str:
    html = _download_text(directory_url)
    matches = re.findall(r'href="([^"]+\.metadata\.tsv)"', html)
    if not matches:
        raise FileNotFoundError(f"Could not locate a metadata TSV in {directory_url}")
    return matches[0]


def _normalize_pathogen_frame(frame: pd.DataFrame, *, group_name: str) -> pd.DataFrame:
    working = frame.copy()
    working["#Organism group"] = group_name
    if "scientific_name" not in working.columns and "#label" in working.columns:
        working["scientific_name"] = working["#label"].astype(str).str.split("|").str[3].fillna("")
    if "scientific_name" not in working.columns:
        working["scientific_name"] = ""
    if "asm_acc" not in working.columns and "Assembly" in working.columns:
        working["asm_acc"] = working["Assembly"]
    if "asm_acc" not in working.columns:
        working["asm_acc"] = ""
    if "geo_loc_name" not in working.columns and "Location" in working.columns:
        working["geo_loc_name"] = working["Location"]
    if "geo_loc_name" not in working.columns and "source_type" in working.columns:
        working["geo_loc_name"] = working["source_type"]
    if "geo_loc_name" not in working.columns:
        working["geo_loc_name"] = ""
    if "source_type" not in working.columns:
        working["source_type"] = ""
    if "host" not in working.columns:
        working["host"] = ""
    if "AMR_genotypes" not in working.columns:
        working["AMR_genotypes"] = ""
    if "species_taxid" not in working.columns:
        working["species_taxid"] = ""
    if "taxid" not in working.columns:
        working["taxid"] = ""
    working["Scientific name"] = working["scientific_name"].fillna("").astype(str)
    working["Location"] = working["geo_loc_name"].fillna("").astype(str)
    working["Source type"] = working["source_type"].fillna("").astype(str)
    working["Assembly"] = working["asm_acc"].fillna("").astype(str)
    working["AMR genotypes"] = working["AMR_genotypes"].fillna("").astype(str)
    working["Host"] = working["host"].fillna("").astype(str)
    for column in PATHOGEN_OUTPUT_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working.loc[:, PATHOGEN_OUTPUT_COLUMNS]


def _pathogen_metadata_urls(groups: Iterable[str]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    for group in groups:
        directory_url = urljoin(PATHOGEN_RESULTS_BASE_URL, f"{group}/latest_kmer/Metadata/")
        filename = _parse_latest_metadata_filename(directory_url)
        urls.append((group, urljoin(directory_url, filename)))
    return urls


def fetch_pathogen_detection_metadata(
    external_root: Path,
    *,
    groups: Iterable[str],
    force: bool = False,
) -> dict[str, object]:
    pathogen_root = ensure_directory(external_root / "pathogen_detection")
    combined_path = pathogen_root / "combined_metadata.tsv"
    clinical_path = pathogen_root / "clinical.tsv"
    environmental_path = pathogen_root / "environmental.tsv"
    manifest_path = pathogen_root / "manifest.json"

    if (
        combined_path.exists()
        and combined_path.stat().st_size > 0
        and clinical_path.exists()
        and clinical_path.stat().st_size > 0
        and environmental_path.exists()
        and environmental_path.stat().st_size > 0
        and not force
    ):
        return {
            "combined_path": str(combined_path),
            "clinical_path": str(clinical_path),
            "environmental_path": str(environmental_path),
            "downloaded": False,
        }
    for path in (combined_path, clinical_path, environmental_path, manifest_path):
        path.unlink(missing_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="pathogen-detection-", dir=str(pathogen_root)))
    source_rows: list[dict[str, object]] = []
    combined_handle = combined_path.open("w", encoding="utf-8")
    clinical_handle = clinical_path.open("w", encoding="utf-8")
    environmental_handle = environmental_path.open("w", encoding="utf-8")
    try:
        combined_header_written = False
        clinical_header_written = False
        environmental_header_written = False
        for group, url in _pathogen_metadata_urls(groups):
            local_path = temp_dir / f"{group}.metadata.tsv"
            _download(url, local_path)
            row_count = 0
            clinical_count = 0
            environmental_count = 0
            for chunk in pd.read_csv(local_path, sep="\t", dtype=str, low_memory=False, chunksize=50000):
                normalized = _normalize_pathogen_frame(chunk, group_name=group)
                normalized.to_csv(combined_handle, sep="\t", index=False, header=not combined_header_written)
                combined_header_written = True
                source_type_text = normalized["Source type"].fillna("").astype(str).str.lower()
                location_text = normalized["Location"].fillna("").astype(str).str.lower()
                clinical_mask = source_type_text.str.contains("clinical|hospital|patient|human", regex=True) | location_text.str.contains(
                    "clinical|hospital|patient|human", regex=True
                )
                environmental_mask = source_type_text.str.contains(
                    "environmental|wastewater|soil|water|river|food", regex=True
                ) | location_text.str.contains("environmental|wastewater|soil|water|river|food", regex=True)
                if clinical_mask.any():
                    normalized.loc[clinical_mask].to_csv(clinical_handle, sep="\t", index=False, header=not clinical_header_written)
                    clinical_header_written = True
                if environmental_mask.any():
                    normalized.loc[environmental_mask].to_csv(environmental_handle, sep="\t", index=False, header=not environmental_header_written)
                    environmental_header_written = True
                row_count += int(len(normalized))
                clinical_count += int(clinical_mask.sum())
                environmental_count += int(environmental_mask.sum())
            source_rows.append(
                {
                    "group": group,
                    "url": url,
                    "rows": row_count,
                    "clinical_rows": clinical_count,
                    "environmental_rows": environmental_count,
                }
            )
        atomic_write_json(
            manifest_path,
            {
                "groups": list(groups),
                "sources": source_rows,
                "combined_rows": int(sum(entry["rows"] for entry in source_rows)),
                "clinical_rows": int(sum(entry["clinical_rows"] for entry in source_rows)),
                "environmental_rows": int(sum(entry["environmental_rows"] for entry in source_rows)),
            },
        )
        return {
            "combined_path": str(combined_path),
            "clinical_path": str(clinical_path),
            "environmental_path": str(environmental_path),
            "downloaded": True,
        }
    finally:
        combined_handle.close()
        clinical_handle.close()
        environmental_handle.close()
        if not combined_header_written:
            combined_path.unlink(missing_ok=True)
        if not clinical_header_written:
            clinical_path.unlink(missing_ok=True)
        if not environmental_header_written:
            environmental_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def fetch_who_mia_pdf(external_root: Path, *, force: bool = False) -> dict[str, object]:
    who_root = ensure_directory(external_root / "who_mia")
    pdf_path = who_root / "who-mia-list-2024.pdf"
    if not force and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return {"path": str(pdf_path), "downloaded": False}
    _download_if_needed(WHO_MIA_PDF_URL, pdf_path, force=force)
    return {"path": str(pdf_path), "downloaded": True}


def fetch_clinical_context_lookup(external_root: Path, *, force: bool = False) -> dict[str, object]:
    lookup_path = ensure_directory(external_root).joinpath("clinical_context_lookup.tsv")
    if lookup_path.exists() and lookup_path.stat().st_size > 0 and not force:
        return {"path": str(lookup_path), "downloaded": False}
    frame = pd.DataFrame(
        [
            {"context_token": "clinical", "clinical_prior": 1.0, "environmental_prior": 0.0},
            {"context_token": "hospital", "clinical_prior": 1.0, "environmental_prior": 0.0},
            {"context_token": "patient", "clinical_prior": 1.0, "environmental_prior": 0.0},
            {"context_token": "human", "clinical_prior": 1.0, "environmental_prior": 0.0},
            {"context_token": "environmental", "clinical_prior": 0.0, "environmental_prior": 1.0},
            {"context_token": "wastewater", "clinical_prior": 0.0, "environmental_prior": 1.0},
            {"context_token": "soil", "clinical_prior": 0.0, "environmental_prior": 1.0},
            {"context_token": "water", "clinical_prior": 0.0, "environmental_prior": 1.0},
            {"context_token": "food", "clinical_prior": 0.0, "environmental_prior": 1.0},
        ]
    )
    frame.to_csv(lookup_path, sep="\t", index=False)
    return {"path": str(lookup_path), "downloaded": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download assets even if they already exist.")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also download the larger optional archives such as MOB-suite.",
    )
    parser.add_argument(
        "--pathogen-group",
        dest="pathogen_groups",
        action="append",
        help="Add a Pathogen Detection organism group to download. May be specified multiple times.",
    )
    args = parser.parse_args()

    context = build_context(Path(__file__).resolve().parents[1])
    external_root = ensure_directory(context.data_dir / "external")
    groups = tuple(args.pathogen_groups or DEFAULT_PATHOGEN_GROUPS)

    manifest = {
        "amrfinder_db": fetch_amrfinder_database(external_root, force=args.force),
        "card_archive": fetch_card_archive(external_root, force=args.force),
        "resfinder_db": fetch_resfinder_database(external_root, force=args.force),
        "plasmidfinder_db": fetch_plasmidfinder_database(external_root, force=args.force),
        "vfdb_annotations": fetch_vfdb_annotations(external_root, force=args.force),
        "bacmet_annotations": fetch_bacmet_annotations(external_root, force=args.force),
        "kegg_tables": fetch_kegg_tables(external_root, force=args.force),
        "conjscan_annotations": fetch_conjscan_annotations(external_root, force=args.force),
        "iceberg_annotations": fetch_iceberg_annotations(external_root, force=args.force),
        "pathogen_detection": fetch_pathogen_detection_metadata(
            external_root,
            groups=groups,
            force=args.force,
        ),
        "who_mia_pdf": fetch_who_mia_pdf(external_root, force=args.force),
        "clinical_context_lookup": fetch_clinical_context_lookup(external_root, force=args.force),
        "groups": list(groups),
        "skipped_large_assets": ["uniprot_sprot_dat"],
    }
    if args.include_large:
        manifest["mobsuite_db"] = fetch_mobsuite_database(external_root, force=args.force)
    else:
        manifest["skipped_large_assets"].append("mobsuite_db")
    atomic_write_json(external_root / "fetch_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
