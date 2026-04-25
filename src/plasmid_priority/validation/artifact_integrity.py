"""Release artifact integrity and provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_release_artifact_integrity(project_root: Path) -> dict[str, object]:
    """Validate integrity of reproducibility and release artifacts."""
    root = Path(project_root).resolve()
    errors: list[str] = []
    manifest_path = root / "docs" / "reproducibility_manifest.json"
    reproducibility_manifest = _read_json(manifest_path)
    if reproducibility_manifest is None:
        errors.append(f"invalid or missing JSON: {manifest_path}")
    else:
        for key in ("protocol_hash", "data_contract_sha", "benchmarks_hash"):
            value = str(reproducibility_manifest.get(key, "")).strip()
            if not value:
                errors.append(f"reproducibility manifest missing key: {key}")
        dossiers = reproducibility_manifest.get("candidate_evidence_dossiers", [])
        if not isinstance(dossiers, list):
            errors.append("candidate_evidence_dossiers must be a list")
        else:
            for rel in dossiers:
                path = root / str(rel)
                if not path.exists():
                    errors.append(f"missing dossier referenced by manifest: {path}")

    reviewer_pack_dir = root / "reports" / "reviewer_pack"
    runner = reviewer_pack_dir / "run_reproducibility.sh"
    if not runner.exists():
        errors.append(f"missing reproducibility runner: {runner}")
    elif not runner.read_text(encoding="utf-8").strip():
        errors.append(f"empty reproducibility runner: {runner}")

    release_manifest_path = root / "reports" / "release" / "plasmid_priority_release_manifest.json"
    if release_manifest_path.exists():
        release_manifest = _read_json(release_manifest_path)
        if release_manifest is None:
            errors.append(f"invalid release manifest JSON: {release_manifest_path}")
        else:
            files = release_manifest.get("files", [])
            if not isinstance(files, list):
                errors.append("release manifest field 'files' must be a list")
            else:
                for row in files:
                    if not isinstance(row, dict):
                        continue
                    rel = str(row.get("relative_path", "")).strip()
                    expected = str(row.get("sha256", "")).strip()
                    if not rel or not expected:
                        continue
                    bundle_target = root / "reports" / "release" / "bundle" / rel
                    source_target = root / rel
                    target = bundle_target if bundle_target.exists() else source_target
                    if not target.exists():
                        errors.append(f"release manifest references missing file: {target}")
                        continue
                    actual = _sha256(target)
                    if actual != expected:
                        errors.append(f"sha256 mismatch for {target}: {actual} != {expected}")

    return {"status": "pass" if not errors else "fail", "errors": errors}
