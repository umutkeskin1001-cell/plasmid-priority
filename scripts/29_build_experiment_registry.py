#!/usr/bin/env python3
"""Isolate exploratory search artifacts and build an experiments registry."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd



from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import ensure_directory, relative_path_str


LEGACY_EXPERIMENT_FILENAMES = (
    "adaptive_blend_search.tsv",
    "adaptive_knownness_blend_search.tsv",
    "adaptive_knownness_oof_blend_search.tsv",
    "knownness_deep_search.tsv",
    "knownness_greedy_search.tsv",
    "knownness_natural_model_search.tsv",
    "tmp_model_search.tsv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_legacy_experiment_outputs(context_root: Path, experiments_dir: Path) -> list[Path]:
    moved: list[Path] = []
    analysis_dir = context_root / "data" / "analysis"
    for filename in LEGACY_EXPERIMENT_FILENAMES:
        source = analysis_dir / filename
        target = experiments_dir / filename
        if not source.exists():
            continue
        ensure_directory(target.parent)
        if target.exists():
            source_mtime = source.stat().st_mtime
            target_mtime = target.stat().st_mtime
            if source_mtime > target_mtime or source.stat().st_size != target.stat().st_size:
                target.unlink()
                source.replace(target)
                moved.append(target)
            else:
                source.unlink()
        else:
            source.replace(target)
            moved.append(target)
    return moved


def _registry_rows(experiments_dir: Path, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(experiments_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"README.md", "registry.tsv"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "artifact_name": path.name,
                "relative_path": relative_path_str(path, root),
                "suffix": path.suffix,
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                "sha256": _sha256(path),
            }
        )
    return rows


def main() -> int:
    context = build_context()
    experiments_dir = ensure_directory(context.experiments_dir)
    registry_path = experiments_dir / "registry.tsv"

    with ManagedScriptRun(context, "29_build_experiment_registry") as run:
        moved = _move_legacy_experiment_outputs(context.root, experiments_dir)
        rows = _registry_rows(experiments_dir, context.root)
        registry = pd.DataFrame(rows).sort_values(["artifact_name", "relative_path"]).reset_index(drop=True)
        registry.to_csv(registry_path, sep="\t", index=False)

        for path in moved:
            run.note(f"moved legacy experiment artifact to {relative_path_str(path, context.root)}")
        run.record_output(registry_path)
        run.set_rows_out("registry_rows", int(len(registry)))
        run.set_metric("n_moved_legacy_artifacts", int(len(moved)))
        run.set_metric("n_registered_artifacts", int(len(registry)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
