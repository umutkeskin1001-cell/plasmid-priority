from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plasmid_priority.validation.scientific_contract import validate_release_scientific_contract


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_release_scientific_contract_passes_with_required_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(root / "docs/model_card.md", "# model card\n")
        _write(root / "docs/data_card.md", "# data card\n")
        _write(root / "docs/benchmark_contract.md", "# benchmark\n")
        _write(root / "docs/scientific_protocol.md", "# protocol\n")
        _write(
            root / "docs/label_card_bundle.md",
            "# Label Card Bundle\n\n## Claim Levels\n\n## Label Cards\n\n",
        )
        _write(
            root / "docs/reproducibility_manifest.json",
            json.dumps(
                {
                    "claim_levels": [
                        "observed",
                        "proxy",
                        "literature_supported",
                        "externally_validated",
                    ],
                    "label_cards": [
                        {
                            "label_name": "spread_label",
                            "description": "desc",
                            "caveats": "caveat",
                        },
                    ],
                    "candidate_evidence_dossiers": [
                        "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
                    ],
                },
            ),
        )
        _write(root / "reports/reviewer_pack/canonical_metadata.json", "{}")
        _write(root / "reports/reviewer_pack/run_reproducibility.sh", "#!/usr/bin/env bash\n")
        _write(
            root / "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
            "# dossiers\n",
        )
        _write(root / "reports/diagnostic_tables/rolling_temporal_validation.tsv", "x\n1\n")
        _write(root / "reports/core_tables/blocked_holdout_summary.tsv", "x\n1\n")
        _write(root / "reports/core_tables/spatial_holdout_summary.tsv", "x\n1\n")
        _write(root / "data/analysis/calibration_metrics.tsv", "x\n1\n")
        _write(root / "reports/diagnostic_tables/negative_control_audit.tsv", "x\n1\n")
        _write(root / "reports/core_tables/literature_validation_matrix.tsv", "x\n1\n")

        result = validate_release_scientific_contract(root)
        assert result["status"] == "pass"
        assert result["errors"] == []


def test_validate_release_scientific_contract_fails_when_claim_levels_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(root / "docs/model_card.md", "# model card\n")
        _write(root / "docs/data_card.md", "# data card\n")
        _write(root / "docs/benchmark_contract.md", "# benchmark\n")
        _write(root / "docs/scientific_protocol.md", "# protocol\n")
        _write(root / "docs/label_card_bundle.md", "# Label Card Bundle\n\n## Label Cards\n\n")
        _write(
            root / "docs/reproducibility_manifest.json",
            json.dumps(
                {
                    "claim_levels": ["observed"],
                    "label_cards": [],
                    "candidate_evidence_dossiers": [],
                },
            ),
        )
        _write(root / "reports/reviewer_pack/canonical_metadata.json", "{}")
        _write(root / "reports/reviewer_pack/run_reproducibility.sh", "#!/usr/bin/env bash\n")
        _write(
            root / "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
            "# dossiers\n",
        )
        _write(root / "reports/diagnostic_tables/rolling_temporal_validation.tsv", "x\n1\n")
        _write(root / "reports/core_tables/blocked_holdout_summary.tsv", "x\n1\n")
        _write(root / "reports/core_tables/spatial_holdout_summary.tsv", "x\n1\n")
        _write(root / "data/analysis/calibration_metrics.tsv", "x\n1\n")
        _write(root / "reports/diagnostic_tables/negative_control_audit.tsv", "x\n1\n")
        _write(root / "reports/core_tables/literature_validation_matrix.tsv", "x\n1\n")

        result = validate_release_scientific_contract(root)
        assert result["status"] == "fail"
        errors = result.get("errors", [])
        assert isinstance(errors, list)
        assert any("missing claim levels" in str(err) for err in errors)
