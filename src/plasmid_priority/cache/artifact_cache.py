"""Persistent content-addressed artifact cache."""


import json
import shutil
from pathlib import Path
from typing import Any

from plasmid_priority.cache.cache_manifest import CachedOutputArtifact, CacheManifest
from plasmid_priority.utils.files import (
    atomic_write_json,
    ensure_directory,
    file_sha256,
    path_signature_with_hash,
)


class ArtifactCache:
    """Filesystem-backed artifact cache keyed by content hash."""

    def __init__(self, root: Path) -> None:
        self.root = root
        ensure_directory(self.root)

    def _entry_dir(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / cache_key

    def _manifest_path(self, cache_key: str) -> Path:
        return self._entry_dir(cache_key) / "manifest.json"

    def _blobs_dir(self, cache_key: str) -> Path:
        return self._entry_dir(cache_key) / "blobs"

    def load(self, cache_key: str) -> CacheManifest | None:
        manifest_path = self._manifest_path(cache_key)
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        manifest = CacheManifest.from_dict(payload)
        if manifest.cache_key != cache_key:
            return None
        return manifest

    def publish(
        self,
        *,
        step_name: str,
        cache_key: str,
        cache_key_payload: dict[str, Any],
        summary: dict[str, Any],
        output_paths: list[Path],
    ) -> CacheManifest:
        self._entry_dir(cache_key)
        blobs_dir = self._blobs_dir(cache_key)
        ensure_directory(blobs_dir)

        outputs: list[CachedOutputArtifact] = []
        for path in output_paths:
            resolved = path.resolve()
            if not resolved.exists() or not resolved.is_file():
                continue
            signature = path_signature_with_hash(resolved, include_file_hash=True)
            digest = signature.get("sha256")
            if not isinstance(digest, str) or not digest:
                digest = file_sha256(resolved)
            blob_name = f"{digest}{resolved.suffix}"
            blob_path = blobs_dir / blob_name
            if not blob_path.exists():
                shutil.copy2(resolved, blob_path)
            outputs.append(
                CachedOutputArtifact(
                    target_path=str(resolved),
                    blob_name=blob_name,
                    signature=dict(signature),
                ),
            )

        manifest = CacheManifest.create(
            step_name=step_name,
            cache_key=cache_key,
            cache_key_payload=cache_key_payload,
            input_manifest={
                str(key): dict(value)
                for key, value in dict(summary.get("input_manifest", {})).items()
                if isinstance(value, dict)
            },
            summary=dict(summary),
            outputs=outputs,
        )
        atomic_write_json(self._manifest_path(cache_key), manifest.to_dict())
        return manifest

    def restore(
        self,
        manifest: CacheManifest,
        *,
        summary_path: Path,
    ) -> bool:
        if not manifest.input_manifest_is_current():
            return False

        blobs_dir = self._blobs_dir(manifest.cache_key)
        for output in manifest.outputs:
            blob_path = blobs_dir / output.blob_name
            if not blob_path.exists():
                return False
            target_path = Path(output.target_path)
            ensure_directory(target_path.parent)
            shutil.copy2(blob_path, target_path)

        atomic_write_json(summary_path, dict(manifest.summary))
        return True
