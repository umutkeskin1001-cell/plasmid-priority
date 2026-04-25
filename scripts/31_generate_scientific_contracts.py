#!/usr/bin/env python3
"""Generate canonical scientific contract docs from benchmark + protocol metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.evidence import default_claim_levels, derive_claim_level
from plasmid_priority.governance import build_canonical_metadata
from plasmid_priority.io.table_io import read_table
from plasmid_priority.reporting.literature_validation import (
    generate_literature_validation_artifacts,
)
from plasmid_priority.utils.files import ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _render_benchmark_contract(metadata: dict) -> str:  # type: ignore
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
            ],
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
        ],
    )
    return "\n".join(lines)


def _render_scientific_protocol(metadata: dict) -> str:  # type: ignore
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
        ],
    )


def _render_model_card(metadata: dict) -> str:  # type: ignore
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
        ],
    )


def _render_data_card(metadata: dict) -> str:  # type: ignore
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
            ],
        )
    lines.extend(
        [
            "## Provenance Pins",
            "",
            f"- data_contract_sha: `{metadata.get('data_contract_sha', '')}`",
            f"- config_sha: `{metadata.get('config_sha', '')}`",
            f"- protocol_hash: `{metadata.get('protocol_hash', '')}`",
            "",
        ],
    )
    return "\n".join(lines)


def _render_reviewer_pack(metadata: dict) -> str:  # type: ignore
    return "\n".join(
        [
            "# Reviewer Pack",
            "",
            "This folder is generated from canonical metadata.",
            "",
            "## Verify",
            "",
            "1. `make reviewer-package`",
            "2. `make protocol-freshness`",
            "3. `make scientific-contract-gate`",
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
            "- Candidate-level evidence dossiers are generated under `candidate_evidence_dossiers/`.",
            "",
        ],
    )


def _default_label_cards() -> list[dict[str, str]]:
    return [
        {
            "label_name": "spread_label",
            "description": "Binary spread target over the configured horizon and split protocol.",
            "caveats": "Retrospective proxy of expansion; not a prospective intervention endpoint.",
        },
        {
            "label_name": "visibility_expansion_label",
            "description": "Mirror of spread label used in visibility and surveillance analyses.",
            "caveats": "Depends on metadata completeness and surveillance density.",
        },
    ]


def _render_label_card_bundle(metadata: dict) -> str:  # type: ignore
    claim_levels = default_claim_levels()
    label_cards = _default_label_cards()
    lines = [
        "# Label Card Bundle",
        "",
        "## Claim Levels",
        "",
        *[f"- `{level}`" for level in claim_levels],
        "",
        "## Label Cards",
        "",
    ]
    for card in label_cards:
        lines.extend(
            [
                f"### `{card['label_name']}`",
                f"- description: {card['description']}",
                f"- caveats: {card['caveats']}",
                "",
            ],
        )
    lines.extend(
        [
            "## Protocol Pins",
            "",
            f"- protocol_hash: `{metadata.get('protocol_hash', '')}`",
            f"- data_contract_sha: `{metadata.get('data_contract_sha', '')}`",
            "",
        ],
    )
    return "\n".join(lines)


def _build_reproducibility_manifest(metadata: dict) -> dict[str, object]:  # type: ignore
    claim_levels = default_claim_levels()
    return {
        "protocol_hash": metadata.get("protocol_hash", ""),
        "benchmarks_hash": metadata.get("benchmarks_hash", ""),
        "data_contract_sha": metadata.get("data_contract_sha", ""),
        "config_sha": metadata.get("config_sha", ""),
        "claim_levels": claim_levels,
        "label_cards": _default_label_cards(),
        "reviewer_package": {
            "entrypoint": "make reviewer-package",
            "dossiers_dir": "reports/reviewer_pack/candidate_evidence_dossiers",
        },
    }


def _write(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return _sha256(path)


def _read_candidate_sources(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_path = project_root / "reports" / "core_tables" / "candidate_portfolio.tsv"
    evidence_path = project_root / "reports" / "core_tables" / "candidate_evidence_matrix.tsv"
    if not portfolio_path.exists() or not evidence_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        portfolio = read_table(portfolio_path)
        evidence = read_table(evidence_path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return portfolio, evidence


def _generate_candidate_evidence_dossiers(project_root: Path, *, top_k: int = 25) -> list[str]:
    reviewer_pack_dir = ensure_directory(project_root / "reports" / "reviewer_pack")
    dossier_dir = ensure_directory(reviewer_pack_dir / "candidate_evidence_dossiers")
    for stale in dossier_dir.glob("*.md"):
        stale.unlink(missing_ok=True)

    portfolio, evidence = _read_candidate_sources(project_root)
    if portfolio.empty or evidence.empty:
        index_path = dossier_dir / "index.md"
        _write(
            index_path,
            "\n".join(
                [
                    "# Candidate Evidence Dossiers",
                    "",
                    "No dossier generated because candidate source tables were not available.",
                ],
            ),
        )
        return [str(index_path.relative_to(project_root))]

    if "backbone_id" not in portfolio.columns or "backbone_id" not in evidence.columns:
        index_path = dossier_dir / "index.md"
        _write(index_path, "# Candidate Evidence Dossiers\n\nNo valid backbone_id columns found.\n")
        return [str(index_path.relative_to(project_root))]

    merged = portfolio.merge(evidence, on="backbone_id", how="left", suffixes=("", "_evidence"))
    if "priority_index" in merged.columns:
        merged = merged.sort_values("priority_index", ascending=False)
    merged = merged.drop_duplicates(subset=["backbone_id"]).head(top_k)

    created: list[str] = []
    index_lines = [
        "# Candidate Evidence Dossiers",
        "",
        f"Generated dossiers for top `{len(merged)}` candidates.",
        "",
    ]
    for row in merged.to_dict(orient="records"):
        backbone_id = str(row.get("backbone_id", "")).strip()
        if not backbone_id:
            continue
        priority_index = float(
            pd.to_numeric(pd.Series([row.get("priority_index")]), errors="coerce").iloc[0] or 0.0
        )
        evidence_support = float(
            pd.to_numeric(pd.Series([row.get("evidence_support_index")]), errors="coerce").iloc[0]
            or 0.0
        )
        claim_level = derive_claim_level(
            observed_signal=priority_index > 0.0,
            proxy_only=True,
            literature_support=evidence_support >= 0.5,
            external_validation=False,
        )
        dossier_name = f"{backbone_id}.md".replace("/", "_")
        dossier_path = dossier_dir / dossier_name
        _write(
            dossier_path,
            "\n".join(
                [
                    f"# Evidence Dossier: {backbone_id}",
                    "",
                    "## Summary",
                    "",
                    f"- backbone_id: `{backbone_id}`",
                    f"- claim_level: `{claim_level}`",
                    f"- priority_index: `{priority_index:.4f}`",
                    f"- evidence_support_index: `{evidence_support:.4f}`",
                    "",
                    "## Caveats",
                    "",
                    "- Retrospective evidence only.",
                    "- No wet-lab prospective validation included in this dossier.",
                    "",
                    "## Raw Fields",
                    "",
                    "```json",
                    json.dumps(row, indent=2, sort_keys=True, default=str),
                    "```",
                    "",
                ],
            ),
        )
        rel = str(dossier_path.relative_to(project_root))
        created.append(rel)
        index_lines.append(f"- [{backbone_id}](./{dossier_name})")

    index_path = dossier_dir / "index.md"
    _write(index_path, "\n".join(index_lines))
    created.insert(0, str(index_path.relative_to(project_root)))
    return created


def _write_reproducibility_runner(project_root: Path) -> str:
    script_path = project_root / "reports" / "reviewer_pack" / "run_reproducibility.sh"
    _write(
        script_path,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"',
                'cd "$ROOT_DIR"',
                "make reviewer-package",
                "make scientific-contract-gate",
                "make artifact-integrity",
            ],
        ),
    )
    script_path.chmod(0o755)
    return str(script_path.relative_to(project_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    context = build_context(args.project_root)
    metadata = build_canonical_metadata(context)
    try:
        literature_artifacts = generate_literature_validation_artifacts(
            args.project_root, top_k=100
        )
    except Exception:
        core_dir = ensure_directory(args.project_root / "reports" / "core_tables")
        diag_dir = ensure_directory(args.project_root / "reports" / "diagnostic_tables")
        matrix_path = core_dir / "literature_validation_matrix.tsv"
        inventory_path = diag_dir / "literature_evidence_inventory.tsv"
        matrix_path.write_text(
            "backbone_id\tpriority_index\tpattern\trisk_category\tliterature_match_count\texample_pmids\tclaim_level\n",
            encoding="utf-8",
        )
        inventory_path.write_text("pub_year\tn_records\n", encoding="utf-8")
        literature_artifacts = {
            "matrix_path": str(matrix_path),
            "inventory_path": str(inventory_path),
            "status": "fallback_empty",
        }

    _write(
        args.project_root / "docs" / "benchmark_contract.md",
        _render_benchmark_contract(metadata),
    )
    _write(
        args.project_root / "docs" / "scientific_protocol.md",
        _render_scientific_protocol(metadata),
    )
    _write(args.project_root / "docs" / "model_card.md", _render_model_card(metadata))
    _write(args.project_root / "docs" / "data_card.md", _render_data_card(metadata))
    _write(
        args.project_root / "docs" / "label_card_bundle.md",
        _render_label_card_bundle(metadata),
    )
    _write(
        args.project_root / "docs" / "reproducibility_manifest.json",
        json.dumps(
            {
                **_build_reproducibility_manifest(metadata),
                "validation_matrix": {
                    "temporal_holdout": "reports/diagnostic_tables/rolling_temporal_validation.tsv",
                    "source_holdout": "reports/core_tables/blocked_holdout_summary.tsv",
                    "country_holdout": "reports/core_tables/spatial_holdout_summary.tsv",
                    "calibration": "data/analysis/calibration_metrics.tsv",
                    "negative_control": "reports/diagnostic_tables/negative_control_audit.tsv",
                    "literature_validation": literature_artifacts.get("matrix_path", ""),
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _write(
        args.project_root / "reports" / "reviewer_pack" / "README.md",
        _render_reviewer_pack(metadata),
    )
    _write(
        args.project_root / "reports" / "reviewer_pack" / "canonical_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True),
    )
    dossiers = _generate_candidate_evidence_dossiers(args.project_root)
    reproducibility_runner = _write_reproducibility_runner(args.project_root)
    manifest_path = args.project_root / "docs" / "reproducibility_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, dict):
        manifest_payload["candidate_evidence_dossiers"] = dossiers
        manifest_payload["reproducibility_runner"] = reproducibility_runner
        manifest_payload["artifacts"] = {
            "model_version": str(metadata.get("protocol_hash", "")),
            "feature_schema_hash": _optional_sha256(
                args.project_root / "reports" / "core_tables" / "model_metrics.tsv",
            ),
            "protocol_hash": str(metadata.get("protocol_hash", "")),
            "training_data_hash": _optional_sha256(
                args.project_root / "data" / "scores" / "backbone_scored.tsv",
            ),
            "calibration_artifact_hash": _optional_sha256(
                args.project_root / "data" / "analysis" / "calibration_metrics.tsv",
            ),
        }
        _write(manifest_path, json.dumps(manifest_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
