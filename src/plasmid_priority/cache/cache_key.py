"""Cache key helpers for content-addressed workflow artifacts."""


import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def software_fingerprint() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def build_step_cache_key(
    *,
    step_name: str,
    source_hash: str,
    input_manifest_hash: str,
    args: Sequence[str],
    env: Mapping[str, str] | Sequence[tuple[str, str]],
    config_hash: str,
    protocol_hash: str,
    software: Mapping[str, Any] | None = None,
    external_fingerprints: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    env_mapping = dict(env) if not isinstance(env, Mapping) else dict(env.items())
    payload: dict[str, Any] = {
        "step_name": step_name,
        "source_hash": source_hash,
        "input_manifest_hash": input_manifest_hash,
        "args_hash": stable_hash(list(args)),
        "env_hash": stable_hash(env_mapping),
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "software_fingerprint": dict(software or software_fingerprint()),
        "external_fingerprints": dict(external_fingerprints or {}),
        "schema_version": "artifact-key-v1",
    }
    return stable_hash(payload), payload
