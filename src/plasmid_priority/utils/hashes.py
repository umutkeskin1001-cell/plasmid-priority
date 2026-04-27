"""Hashing utilities for data integrity and caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plasmid_priority.utils.files import path_signature


def stable_hash(payload: object) -> str:
    """Generate a stable SHA-256 hash for a JSON-serializable object."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()


def cache_key_path(path: Path) -> Path:
    """Derive the cache key file path from a target data path."""
    return path.with_name(path.name + ".cache_key")


def cache_key_payload(
    *,
    protocol_hash: str,
    input_paths: list[Path],
    source_paths: list[Path],
) -> dict[str, object]:
    """Build a payload that uniquely identifies a script execution state."""
    return {
        "protocol_hash": protocol_hash,
        "input_manifest": [path_signature(p) for p in input_paths if p.exists()],
        "source_manifest": [path_signature(p) for p in source_paths if p.exists()],
    }
