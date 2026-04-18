#!/usr/bin/env python3
"""Generate canonical scientific contract docs from benchmark + protocol metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasmid_priority.config import build_context
from plasmid_priority.governance import build_canonical_metadata
from plasmid_priority.utils.files import ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _render_benchmark_contract(metadata: dict) -> str:
    protocol = metadata.get("protocol", {})
    acceptance = protocol.get("acceptance_thresholds", {})
    benchmarks = metadata.get("benchmarks", {})
    lines = [
        "# Benchmark Contract",
        "",
        "This document is generated from canonical scientific metadata.",
        "",
        "## Version",
        "",
        f"- protocol_hash: `{metadata.get('protocol_hash', '')}`",
        f"- benchmarks_hash: `{metadata.get('benchmarks_hash', '')}`",
        f"- data_contract_sha: `{metadata.get('data_contract_sha', '')}`",
        f"- config_sha: `{metadata.get('config_sha', '')}`",
        "",
        "## Canonical Authority",
        "",
        "- primary_source: `config/benchmarks.yaml`",
        "- protocol_resolver: `src/plasmid_priority/protocol.py`",
        "- generated_doc: `docs/benchmark_contract.md`",
        "",
        "## Global Protocol Scope",
        "",
        f"- split_year: `{protocol.get('split_year', '')}`",
        f"- horizon_years: `{protocol.get('horizon_years', '')}`",
        f"- min_new_countries_for_spread: `{protocol.get('min_new_countries_for_spread', '')}`",
        f"- min_new_host_genera_for_transfer: `{protocol.get('min_new_host_genera_for_transfer', '')}`",
        f"- min_new_host_families_for_transfer: `{protocol.get('min_new_host_families_for_transfer', '')}`",
        f"- primary_model_name: `{protocol.get('primary_model_name', '')}`",
        f"- governance_model_name: `{protocol.get('governance_model_name', '')}`",
        f"- conservative_model_name: `{protocol.get('conservative_model_name', '')}`",
        "",
        "## Branch Benchmarks",
        "",
    ]
    for name in sorted(benchmarks):
        benchmark = benchmarks.get(name, {}).get("benchmark", {})
        lines.extend(
            [
                f"### `{name}`",
                f"- name: `{benchmark.get('name', '')}`",
                f"- split_year: `{benchmark.get('split_year', '')}`",
                f"- horizon_years: `{benchmark.get('horizon_years', '')}`",
                f"- assignment_mode: `{benchmark.get('assignment_mode', '')}`",
                f"- label_column: `{benchmark.get('label_column', '')}`",
                f"- outcome_column: `{benchmark.get('outcome_column', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Acceptance Thresholds",
            "",
            f"- matched_knownness_gap_min: `{acceptance.get('matched_knownness_gap_min', '')}`",
            f"- source_holdout_gap_min: `{acceptance.get('source_holdout_gap_min', '')}`",
            f"- spatial_holdout_gap_min: `{acceptance.get('spatial_holdout_gap_min', '')}`",
            f"- ece_max: `{acceptance.get('ece_max', '')}`",
            f"- selection_adjusted_p_max: `{acceptance.get('selection_adjusted_p_max', '')}`",
            "",
            "## Claim Guardrails",
            "",
            "- If strict acceptance fails, claims must remain conditional and benchmark-limited.",
            "- Missing branch predictions must remain explicit and cannot be silently imputed.",
            "- Uncertainty and instability must be surfaced in the official consensus output.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_scientific_protocol(metadata: dict) -> str:
    return "\n".join(
        [
            "# Scientific Protocol",
            "",
            "This protocol is generated from the canonical authority surface:",
            "`config/benchmarks.yaml` + `src/plasmid_priority/protocol.py`.",
            "",
            "```json",
            json.dumps(metadata.get("protocol", {}), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _render_model_card(metadata: dict) -> str:
    protocol = metadata.get("protocol", {})
    return "\n".join(
        [
            "# Model Card",
            "",
            "## Official Surface",
            "",
            f"- official_model_names: `{protocol.get('official_model_names', [])}`",
            f"- core_model_names: `{protocol.get('core_model_names', [])}`",
            f"- research_model_names: `{protocol.get('research_model_names', [])}`",
            f"- ablation_model_names: `{protocol.get('ablation_model_names', [])}`",
            "",
            "## Governance",
            "",
            f"- primary_model_name: `{protocol.get('primary_model_name', '')}`",
            f"- governance_model_name: `{protocol.get('governance_model_name', '')}`",
            f"- conservative_model_name: `{protocol.get('conservative_model_name', '')}`",
            "",
            "## Legacy Status",
            "",
            "- `config.yaml#models` is legacy/research surface and not official benchmark truth.",
            "",
        ]
    )


def _render_data_card(metadata: dict) -> str:
    benchmarks = metadata.get("benchmarks", {})
    lines = [
        "# Data Card",
        "",
        "## Branch Data Contracts",
        "",
    ]
    for name in sorted(benchmarks):
        benchmark = benchmarks.get(name, {}).get("benchmark", {})
        required_columns = benchmark.get("required_columns", [])
        lines.extend(
            [
                f"### `{name}`",
                f"- benchmark_name: `{benchmark.get('name', '')}`",
                f"- label_column: `{benchmark.get('label_column', '')}`",
                f"- outcome_column: `{benchmark.get('outcome_column', '')}`",
                f"- required_columns: `{required_columns}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance Pins",
            "",
            f"- data_contract_sha: `{metadata.get('data_contract_sha', '')}`",
            f"- config_sha: `{metadata.get('config_sha', '')}`",
            f"- protocol_hash: `{metadata.get('protocol_hash', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_reviewer_pack(metadata: dict) -> str:
    return "\n".join(
        [
            "# Reviewer Pack",
            "",
            "This folder is generated from canonical metadata.",
            "",
            "## Verify",
            "",
            "1. `make generate-scientific-contracts`",
            "2. `make protocol-freshness`",
            "3. `make quality`",
            "",
            "## Evidence Pins",
            "",
            f"- protocol_hash: `{metadata.get('protocol_hash', '')}`",
            f"- data_contract_sha: `{metadata.get('data_contract_sha', '')}`",
            f"- config_sha: `{metadata.get('config_sha', '')}`",
            "",
            "## Constraints",
            "",
            "- Claims remain retrospective surveillance claims.",
            "- Scientific boundary text is mandatory in jury-facing reports.",
            "",
        ]
    )


def _write(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    context = build_context(args.project_root)
    metadata = build_canonical_metadata(context)

    _write(
        args.project_root / "docs" / "benchmark_contract.md", _render_benchmark_contract(metadata)
    )
    _write(
        args.project_root / "docs" / "scientific_protocol.md", _render_scientific_protocol(metadata)
    )
    _write(args.project_root / "docs" / "model_card.md", _render_model_card(metadata))
    _write(args.project_root / "docs" / "data_card.md", _render_data_card(metadata))
    _write(
        args.project_root / "reports" / "reviewer_pack" / "README.md",
        _render_reviewer_pack(metadata),
    )
    _write(
        args.project_root / "reports" / "reviewer_pack" / "canonical_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
