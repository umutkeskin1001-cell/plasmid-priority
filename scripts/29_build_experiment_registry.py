#!/usr/bin/env python3
"""Isolate exploratory search artifacts and build an experiments registry."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.protocol import ScientificProtocol, build_protocol_hash
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory, file_sha256, relative_path_str

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


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _reference_data_path(context) -> Path:
    gold_table = context.data_dir / "gold/official_modeling_table.tsv"
    if gold_table.exists():
        return gold_table
    return context.data_dir / "scores/backbone_scored.tsv"


def _composite_experiment_id(
    *,
    protocol_hash: str,
    data_hash: str,
    model_surface_hash: str,
    git_commit_short: str,
) -> str:
    payload = "|".join([protocol_hash, data_hash, model_surface_hash, git_commit_short])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    context = build_context(root)
    protocol = ScientificProtocol.from_config(context.config)
    protocol_hash = build_protocol_hash(protocol)
    data_hash = file_sha256(_reference_data_path(context))
    git_commit_short = _git_commit(root)[:12]
    for path in sorted(experiments_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"README.md", "registry.tsv", "registry.json"}:
            continue
        stat = path.stat()
        model_surface_hash = _sha256(path)
        rows.append(
            {
                "artifact_name": path.name,
                "relative_path": relative_path_str(path, root),
                "suffix": path.suffix,
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "sha256": model_surface_hash,
                "protocol_hash": protocol_hash,
                "data_hash": data_hash,
                "model_surface_hash": model_surface_hash,
                "git_commit_short": git_commit_short,
                "experiment_id": _composite_experiment_id(
                    protocol_hash=protocol_hash,
                    data_hash=data_hash,
                    model_surface_hash=model_surface_hash,
                    git_commit_short=git_commit_short,
                ),
            }
        )
    return rows


def main() -> int:
    context = build_context()
    experiments_dir = ensure_directory(context.experiments_dir)
    registry_path = experiments_dir / "registry.tsv"
    registry_json_path = experiments_dir / "registry.json"

    with ManagedScriptRun(context, "29_build_experiment_registry") as run:
        moved = _move_legacy_experiment_outputs(context.root, experiments_dir)
        rows = _registry_rows(experiments_dir, context.root)
        registry = (
            pd.DataFrame(rows)
            .sort_values(["artifact_name", "relative_path"])
            .reset_index(drop=True)
        )
        registry.to_csv(registry_path, sep="\t", index=False)
        atomic_write_json(
            registry_json_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "project_root": str(context.root),
                "rows": registry.to_dict(orient="records"),
            },
        )

        for path in moved:
            run.note(f"moved legacy experiment artifact to {relative_path_str(path, context.root)}")
        run.record_output(registry_path)
        run.record_output(registry_json_path)
        run.set_rows_out("registry_rows", int(len(registry)))
        run.set_metric("n_moved_legacy_artifacts", int(len(moved)))
        run.set_metric("n_registered_artifacts", int(len(registry)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
