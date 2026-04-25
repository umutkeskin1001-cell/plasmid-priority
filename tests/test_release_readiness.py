from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plasmid_priority.validation.release_readiness import evaluate_release_readiness


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_evaluate_release_readiness_passes_for_minimal_surface() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(
            root / "docs/reproducibility_manifest.json",
            json.dumps(
                {
                    "protocol_hash": "p1",
                    "data_contract_sha": "d1",
                    "benchmarks_hash": "b1",
                    "claim_levels": [
                        "observed",
                        "proxy",
                        "literature_supported",
                        "externally_validated",
                    ],
                    "label_cards": [
                        {"label_name": "spread_label", "description": "d", "caveats": "c"},
                    ],
                    "candidate_evidence_dossiers": [
                        "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
                    ],
                },
            ),
        )
        for rel in (
            "docs/model_card.md",
            "docs/data_card.md",
            "docs/benchmark_contract.md",
            "docs/scientific_protocol.md",
            "docs/pre_registration.md",
            "reports/reviewer_pack/canonical_metadata.json",
            "reports/reviewer_pack/run_reproducibility.sh",
            "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
            "reports/release/jury_dashboard.html",
            "reports/diagnostic_tables/rolling_temporal_validation.tsv",
            "reports/core_tables/blocked_holdout_summary.tsv",
            "reports/core_tables/spatial_holdout_summary.tsv",
            "data/analysis/calibration_metrics.tsv",
            "reports/diagnostic_tables/negative_control_audit.tsv",
            "reports/core_tables/literature_validation_matrix.tsv",
            "src/plasmid_priority/sensitivity/variant_cache.py",
            "src/plasmid_priority/reporting/cache.py",
            "src/plasmid_priority/api/sdk.py",
            "scripts/45_generate_independent_audit_packet.py",
            "config/rag_corpus.yaml",
        ):
            _write(root / rel, "x\n")
        _write(
            root / "docs/label_card_bundle.md",
            "# Label Card Bundle\n\n## Claim Levels\n\n## Label Cards\n\n",
        )
        _write(
            root / "src/plasmid_priority/api/app.py",
            "from plasmid_priority.api.artifact_registry import ArtifactRegistry, ArtifactUnavailableError\n"
            '@app.post("/score/backbones")\n'
            '@app.post("/score/backbones/batch")\n'
            '@app.post("/graphql")\n',
        )
        _write(
            root / "config/performance_budgets.yaml",
            "\n".join(
                [
                    "modes:",
                    "  smoke-local: {budget_seconds: 1}",
                    "  dev-refresh: {budget_seconds: 1}",
                    "  model-refresh: {budget_seconds: 1}",
                    "  report-refresh: {budget_seconds: 1}",
                    "  release-full: {budget_seconds: 1}",
                ],
            ),
        )
        result = evaluate_release_readiness(root)
        assert result["status"] == "pass"
