"""Manifest schema and validation for cached workflow artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasmid_priority.utils.files import path_signature_with_hash


def _signature_equal(expected: dict[str, object], current: dict[str, object]) -> bool:
    for key in ("path", "size", "mtime_ns", "kind", "sha256", "digest", "entry_count"):
        if expected.get(key) != current.get(key):
            return False
    return True


@dataclass(frozen=True)
class CachedOutputArtifact:
    target_path: str
    blob_name: str
    signature: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheManifest:
    step_name: str
    cache_key: str
    created_at: str
    cache_key_payload: dict[str, Any]
    input_manifest: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    outputs: list[CachedOutputArtifact]
    schema_version: str = "artifact-cache-manifest-v1"

    @classmethod
    def create(
        cls,
        *,
        step_name: str,
        cache_key: str,
        cache_key_payload: dict[str, Any],
        input_manifest: dict[str, dict[str, Any]],
        summary: dict[str, Any],
        outputs: list[CachedOutputArtifact],
    ) -> "CacheManifest":
        return cls(
            step_name=step_name,
            cache_key=cache_key,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            cache_key_payload=cache_key_payload,
            input_manifest=input_manifest,
            summary=summary,
            outputs=outputs,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CacheManifest:
        outputs = [
            CachedOutputArtifact(
                target_path=str(entry.get("target_path", "")),
                blob_name=str(entry.get("blob_name", "")),
                signature=dict(entry.get("signature", {})),
            )
            for entry in payload.get("outputs", [])
            if isinstance(entry, dict)
        ]
        return cls(
            step_name=str(payload.get("step_name", "")),
            cache_key=str(payload.get("cache_key", "")),
            created_at=str(payload.get("created_at", "")),
            cache_key_payload=dict(payload.get("cache_key_payload", {})),
            input_manifest={
                str(key): dict(value)
                for key, value in dict(payload.get("input_manifest", {})).items()
                if isinstance(value, dict)
            },
            summary=dict(payload.get("summary", {})),
            outputs=outputs,
            schema_version=str(payload.get("schema_version", "artifact-cache-manifest-v1")),
        )

    def input_manifest_is_current(self) -> bool:
        for expected in self.input_manifest.values():
            path_value = expected.get("path")
            if not path_value:
                return False
            path = Path(str(path_value))
            if not path.exists():
                return False
            try:
                current = path_signature_with_hash(path, include_file_hash=True)
            except (OSError, ValueError):
                return False
            if not _signature_equal(expected, current):
                return False
        return True
