"""Helpers for maintaining small local snapshot-style data subsets."""

from __future__ import annotations

import shutil
from pathlib import Path

from plasmid_priority.utils.files import ensure_directory

SNAPSHOT_PROFILES: dict[str, tuple[str, ...]] = {
    "report-pack": (
        "analysis",
        "scores/backbone_scored.tsv",
        "scores/backbone_scored.parquet",
        "silver/plasmid_backbones.tsv",
        "silver/plasmid_amr_consensus.tsv",
    ),
}


def _collect_profile_paths(data_root: Path, profile: str) -> list[Path]:
    try:
        entries = SNAPSHOT_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported snapshot profile: {profile}") from exc
    paths: list[Path] = []
    for entry in entries:
        candidate = data_root / entry
        if candidate.exists():
            paths.append(candidate)
    return paths


def profile_targets(profile: str) -> tuple[str, ...]:
    try:
        return SNAPSHOT_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported snapshot profile: {profile}") from exc


def profile_has_content(data_root: Path, profile: str) -> bool:
    return bool(_collect_profile_paths(data_root, profile))


def clear_profile_outputs(data_root: Path, profile: str) -> None:
    for entry in profile_targets(profile):
        target = data_root / entry
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def sync_profile_outputs(
    source_root: Path,
    destination_root: Path,
    profile: str,
    *,
    clean_first: bool = True,
) -> list[Path]:
    selected_paths = _collect_profile_paths(source_root, profile)
    if not selected_paths:
        raise FileNotFoundError(
            f"No files found for snapshot profile '{profile}' under {source_root}."
        )
    ensure_directory(destination_root)
    if clean_first:
        clear_profile_outputs(destination_root, profile)
    copied: list[Path] = []
    for source in selected_paths:
        relative = source.relative_to(source_root)
        target = destination_root / relative
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            ensure_directory(target.parent)
            shutil.copy2(source, target)
        copied.append(target)
    return copied
