"""Branch data inventory reporting helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.utils.files import file_sha256


def _path_info(path: Path, *, used: bool, kind: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.resolve()),
        "exists": bool(exists),
        "used": bool(used and exists),
        "kind": kind,
        "size": int(path.stat().st_size) if exists and path.is_file() else None,
        "sha256": file_sha256(path) if exists and path.is_file() else None,
    }


def build_branch_inventory(
    project_root: Path,
    *,
    used_paths: Iterable[Path | str],
    unused_paths: Iterable[Path | str] | None = None,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build used and unused inventory tables for a branch run."""
    context = build_context(project_root, data_root=data_root)
    used_resolved = {Path(path).resolve() for path in used_paths}
    unused_resolved = {Path(path).resolve() for path in (unused_paths or [])}
    rows: list[dict[str, Any]] = []
    for asset in context.contract.assets:
        asset_path = asset.resolved_path(context.root, context.data_dir)
        rows.append(_path_info(asset_path, used=asset_path in used_resolved, kind=asset.kind.value))
    for path in sorted(used_resolved | unused_resolved):
        if any(row["path"] == str(path) for row in rows):
            continue
        rows.append(_path_info(path, used=path in used_resolved, kind="file" if path.is_file() else "directory"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), {"used_count": 0, "unused_count": 0, "missing_count": 0}
    used_frame = frame.loc[frame["used"].fillna(False)].copy().reset_index(drop=True)
    unused_frame = frame.loc[~frame["used"].fillna(False)].copy().reset_index(drop=True)
    summary = {
        "used_count": int(len(used_frame)),
        "unused_count": int(len(unused_frame)),
        "missing_count": int((~frame["exists"].fillna(False)).sum()),
    }
    return used_frame, unused_frame, summary
