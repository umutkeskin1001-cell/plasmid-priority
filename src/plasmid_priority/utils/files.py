"""Filesystem helpers used across pipeline scripts."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
import os
import tempfile
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


def relative_path_str(path: Path, root: Path) -> str:
    """Format a path relative to project root when possible."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
