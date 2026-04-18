from __future__ import annotations

from pathlib import Path

from plasmid_priority.config import build_context
from plasmid_priority.governance import build_canonical_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_metadata_declares_authority_surface() -> None:
    context = build_context(PROJECT_ROOT)
    metadata = build_canonical_metadata(context)

    authority = metadata["authority"]
    assert authority["benchmark_truth"] == [
        "config/benchmarks.yaml",
        "src/plasmid_priority/protocol.py",
        "docs/benchmark_contract.md",
    ]
    assert authority["legacy_surface_status"] == "legacy_research_only"
    assert metadata["protocol_hash"]
    assert metadata["data_contract_sha"]
    assert metadata["config_sha"]
