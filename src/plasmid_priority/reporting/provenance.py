"""Canonical provenance helpers for reporting and pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from plasmid_priority.protocol import ScientificProtocol
from plasmid_priority.reporting.artifact_contracts import validate_provenance_record
from plasmid_priority.utils.files import atomic_write_json, path_signature_with_hash


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_payload(payload: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def build_protocol_snapshot(protocol: ScientificProtocol) -> dict[str, Any]:
    """Return the canonical protocol payload used for provenance hashes."""
    return {
        "acceptance_thresholds": protocol.acceptance_thresholds,
        "ablation_model_names": list(protocol.ablation_model_names),
        "conservative_model_name": protocol.conservative_model_name,
        "core_model_names": list(protocol.core_model_names),
        "governance_model_fallback": protocol.governance_model_fallback,
        "governance_model_name": protocol.governance_model_name,
        "governance_model_policy": protocol.governance_model_policy,
        "min_new_countries_for_spread": int(protocol.min_new_countries_for_spread),
        "official_model_names": list(protocol.official_model_names),
        "outcome_definition": protocol.outcome_definition,
        "primary_model_fallback": protocol.primary_model_fallback,
        "primary_model_name": protocol.primary_model_name,
        "research_model_names": list(protocol.research_model_names),
        "selection_adjusted_p_max": float(protocol.selection_adjusted_p_max),
        "split_year": int(protocol.split_year),
    }


def build_protocol_id(protocol: ScientificProtocol) -> str:
    """Build a human-readable stable identifier for the scientific protocol."""
    pmax = str(protocol.selection_adjusted_p_max).replace(".", "p")
    return (
        f"split{int(protocol.split_year)}"
        f"-min{int(protocol.min_new_countries_for_spread)}"
        f"-primary-{protocol.primary_model_name}"
        f"-governance-{protocol.governance_model_name}"
        f"-pmax{pmax}"
    )


def build_protocol_hash(protocol: ScientificProtocol) -> str:
    return _hash_payload(build_protocol_snapshot(protocol))


def build_signature_hash(paths: Iterable[Path]) -> tuple[str, list[dict[str, Any]]]:
    """Hash path signatures using the same filesystem metadata used by manifests."""
    signatures: list[dict[str, Any]] = []
    for path in _dedupe_paths(paths):
        signatures.append(path_signature_with_hash(path))
    return _hash_payload(signatures), signatures


def build_artifact_provenance(
    *,
    protocol: ScientificProtocol,
    artifact_family: str,
    input_paths: Iterable[Path],
    source_paths: Iterable[Path],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a canonical provenance record for a report or pipeline artifact."""
    input_hash, input_signatures = build_signature_hash(input_paths)
    code_hash, code_signatures = build_signature_hash(source_paths)
    provenance: dict[str, Any] = {
        "artifact_family": str(artifact_family),
        "generated_at": generated_at or _utc_now(),
        "protocol_id": build_protocol_id(protocol),
        "protocol_hash": build_protocol_hash(protocol),
        "code_hash": code_hash,
        "input_data_hash": input_hash,
        "protocol_snapshot": build_protocol_snapshot(protocol),
        "input_signatures": input_signatures,
        "code_signatures": code_signatures,
    }
    validate_provenance_record(provenance, artifact_name=str(artifact_family))
    return provenance


def provenance_matches_current(
    record: dict[str, Any],
    *,
    protocol: ScientificProtocol,
    input_paths: Iterable[Path],
    source_paths: Iterable[Path],
) -> bool:
    """Return True when the provided provenance still matches the current surface."""
    try:
        validate_provenance_record(record, artifact_name="provenance")
    except ValueError:
        return False
    try:
        expected = build_artifact_provenance(
            protocol=protocol,
            artifact_family=str(record.get("artifact_family", "")),
            input_paths=input_paths,
            source_paths=source_paths,
            generated_at=str(record.get("generated_at", "")),
        )
    except (OSError, ValueError):
        return False
    keys = ("artifact_family", "protocol_id", "protocol_hash", "code_hash", "input_data_hash")
    return all(str(record.get(key, "")) == str(expected.get(key, "")) for key in keys)


def write_provenance_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a provenance payload to disk."""
    atomic_write_json(path, payload)


def load_provenance_json(path: Path) -> dict[str, Any]:
    """Load a provenance payload from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Provenance file must contain a JSON object: {path}")
    return payload
