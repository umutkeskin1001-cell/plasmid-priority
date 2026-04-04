"""Filesystem helpers used across pipeline scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to disk."""
    ensure_directory(path.parent)
    serializable_payload = _json_safe(payload)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(serializable_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        tmp_path = Path(handle.name)

    os.replace(tmp_path, path)


def path_signature(path: Path) -> dict[str, object]:
    """Return a cheap filesystem signature for cache invalidation."""
    resolved = path.resolve()
    if resolved.is_dir():
        digest = hashlib.sha256()
        entry_count = 0
        for current_root, dirnames, filenames in os.walk(resolved):
            dirnames.sort()
            filenames.sort()
            root_path = Path(current_root)
            for dirname in dirnames:
                child = root_path / dirname
                child_stat = child.stat()
                relative = child.relative_to(resolved)
                digest.update(
                    f"D\t{relative}\t{child_stat.st_size}\t{child_stat.st_mtime_ns}\n".encode(
                        "utf-8"
                    )
                )
                entry_count += 1
            for filename in filenames:
                child = root_path / filename
                child_stat = child.stat()
                relative = child.relative_to(resolved)
                digest.update(
                    f"F\t{relative}\t{child_stat.st_size}\t{child_stat.st_mtime_ns}\n".encode(
                        "utf-8"
                    )
                )
                entry_count += 1
        return {
            "path": str(resolved),
            "digest": digest.hexdigest(),
            "entry_count": entry_count,
            "kind": "directory",
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def materialize_recorded_paths(root: Path, recorded_paths: Iterable[str]) -> list[Path]:
    """Resolve ManagedScriptRun path strings back to absolute paths."""
    materialized: list[Path] = []
    for value in recorded_paths:
        path = Path(str(value))
        materialized.append(path if path.is_absolute() else root / path)
    return _dedupe_paths(materialized)


def project_python_source_paths(
    project_root: Path,
    *,
    script_path: Path | None = None,
) -> list[Path]:
    """Collect python sources that should invalidate script-level caches."""
    paths: list[Path] = []
    if script_path is not None and script_path.exists():
        paths.append(script_path)
    package_root = project_root / "src" / "plasmid_priority"
    if package_root.exists():
        paths.extend(sorted(package_root.rglob("*.py")))
    return _dedupe_paths(paths)


def load_signature_manifest(
    manifest_path: Path,
    *,
    input_paths: Iterable[Path],
    source_paths: Iterable[Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Return True when cached outputs still match current inputs, code, and metadata."""
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    resolved_inputs = _dedupe_paths(input_paths)
    if any(not path.exists() for path in resolved_inputs):
        return False
    expected_inputs = [path_signature(path) for path in resolved_inputs]
    if payload.get("input_signatures") != expected_inputs:
        return False

    resolved_sources = _dedupe_paths(source_paths or [])
    if any(not path.exists() for path in resolved_sources):
        return False
    expected_sources = [path_signature(path) for path in resolved_sources]
    if payload.get("source_signatures") != expected_sources:
        return False

    if payload.get("metadata") != _json_safe(metadata or {}):
        return False

    cached_outputs = payload.get("output_signatures")
    if not isinstance(cached_outputs, list) or not cached_outputs:
        return False
    actual_outputs: list[dict[str, object]] = []
    for entry in cached_outputs:
        if not isinstance(entry, dict):
            return False
        path_value = entry.get("path")
        if not path_value:
            return False
        output_path = Path(str(path_value))
        if not output_path.exists():
            return False
        actual_outputs.append(path_signature(output_path))
    return cached_outputs == actual_outputs


def write_signature_manifest(
    manifest_path: Path,
    *,
    input_paths: Iterable[Path],
    output_paths: Iterable[Path],
    source_paths: Iterable[Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the current input/output/code signatures for future cache checks."""
    resolved_inputs = [path for path in _dedupe_paths(input_paths) if path.exists()]
    resolved_outputs = [path for path in _dedupe_paths(output_paths) if path.exists()]
    resolved_sources = [path for path in _dedupe_paths(source_paths or []) if path.exists()]
    atomic_write_json(
        manifest_path,
        {
            "metadata": _json_safe(metadata or {}),
            "input_signatures": [path_signature(path) for path in resolved_inputs],
            "output_signatures": [path_signature(path) for path in resolved_outputs],
            "source_signatures": [path_signature(path) for path in resolved_sources],
        },
    )


def relative_path_str(path: Path, root: Path) -> str:
    """Format a path relative to project root when possible."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
