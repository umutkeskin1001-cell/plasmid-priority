"""Canonical scientific metadata builders used by docs and reviewer surfaces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from plasmid_priority.config import ProjectContext, context_config_paths
from plasmid_priority.protocol import (
    ScientificProtocol,
    build_protocol_hash,
    build_protocol_snapshot,
)
from plasmid_priority.utils.files import file_sha256


def _sha256_payload(payload: Any) -> str:
    encoded = repr(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_benchmarks_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.exists() or not path.is_file():
            continue
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


def build_canonical_metadata(context: ProjectContext) -> dict[str, Any]:
    """Build deterministic canonical metadata from protocol + benchmark config + data contract."""
    protocol = ScientificProtocol.from_config(context.config)
    protocol_snapshot = build_protocol_snapshot(protocol)
    protocol_hash = build_protocol_hash(protocol)
    benchmarks_path = context.root / "config" / "benchmarks.yaml"
    benchmarks_payload = _load_benchmarks_yaml(benchmarks_path)
    config_paths = [Path(path) for path in context_config_paths(context)]
    data_contract_path = context.root / "data" / "manifests" / "data_contract.json"

    return {
        "authority": {
            "benchmark_truth": [
                "config/benchmarks.yaml",
                "src/plasmid_priority/protocol.py",
                "docs/benchmark_contract.md",
            ],
            "legacy_surface": "config.yaml#models",
            "legacy_surface_status": "legacy_research_only",
        },
        "protocol": protocol_snapshot,
        "protocol_hash": protocol_hash,
        "benchmarks": benchmarks_payload,
        "benchmarks_hash": _sha256_payload(benchmarks_payload),
        "config_sha": _hash_paths(config_paths),
        "data_contract_sha": file_sha256(data_contract_path) if data_contract_path.exists() else "",
        "metadata_hash": _sha256_payload(
            {
                "protocol_hash": protocol_hash,
                "benchmarks": benchmarks_payload,
                "config_sha": _hash_paths(config_paths),
                "data_contract_sha": (
                    file_sha256(data_contract_path) if data_contract_path.exists() else ""
                ),
            }
        ),
    }
