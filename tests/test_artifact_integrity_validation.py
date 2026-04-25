from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plasmid_priority.validation.artifact_integrity import validate_release_artifact_integrity


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_release_artifact_integrity_passes_with_minimal_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(
            root / "docs" / "reproducibility_manifest.json",
            json.dumps(
                {
                    "protocol_hash": "p1",
                    "data_contract_sha": "d1",
                    "benchmarks_hash": "b1",
                    "candidate_evidence_dossiers": [
                        "reports/reviewer_pack/candidate_evidence_dossiers/index.md",
                    ],
                },
            ),
        )
        _write(root / "reports/reviewer_pack/run_reproducibility.sh", "#!/usr/bin/env bash\n")
        _write(root / "reports/reviewer_pack/candidate_evidence_dossiers/index.md", "# index\n")
        result = validate_release_artifact_integrity(root)
        assert result["status"] == "pass"
        assert result["errors"] == []


def test_validate_release_artifact_integrity_fails_for_missing_runner() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(
            root / "docs" / "reproducibility_manifest.json",
            json.dumps(
                {
                    "protocol_hash": "p1",
                    "data_contract_sha": "d1",
                    "benchmarks_hash": "b1",
                    "candidate_evidence_dossiers": [],
                },
            ),
        )
        result = validate_release_artifact_integrity(root)
        assert result["status"] == "fail"
        errors = result.get("errors", [])
        assert isinstance(errors, list)
        assert any("missing reproducibility runner" in str(err) for err in errors)
