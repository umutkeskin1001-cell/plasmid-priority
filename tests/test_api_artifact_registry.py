from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from plasmid_priority.api.artifact_registry import ArtifactRegistry, ArtifactUnavailableError


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_artifact_registry_scores_and_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        data_dir = root / "data"
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "priority_index": [0.9, 0.2],
                "operational_priority_index": [0.88, 0.22],
                "bio_priority_index": [0.85, 0.25],
                "evidence_support_index": [0.8, 0.4],
                "T_eff": [0.7, 0.3],
                "H_eff": [0.6, 0.2],
                "A_eff": [0.9, 0.1],
            },
        )
        (data_dir / "scores").mkdir(parents=True, exist_ok=True)
        scored.to_csv(data_dir / "scores" / "backbone_scored.tsv", sep="\t", index=False)
        _write(
            root / "docs" / "reproducibility_manifest.json",
            json.dumps(
                {
                    "protocol_hash": "p1",
                    "benchmarks_hash": "b1",
                    "data_contract_sha": "d1",
                    "claim_levels": ["proxy"],
                    "label_cards": [
                        {
                            "label_name": "spread_label",
                            "description": "desc",
                            "caveats": "caveat",
                        },
                    ],
                },
            ),
        )

        registry = ArtifactRegistry(project_root=root)
        models = registry.list_models()
        assert models and models[0]["protocol_hash"] == "p1"

        scores = registry.score_backbones(["bb1"])
        assert len(scores) == 1
        assert scores[0]["backbone_id"] == "bb1"
        assert scores[0]["priority_index"] == 0.9

        evidence = registry.get_evidence("bb1")
        assert evidence["claim_level"] == "literature_supported"
        assert evidence["scores"]["priority_index"] == 0.9

        explain = registry.explain_backbone("bb1")
        assert explain["components"]["T_eff"] == 0.7


def test_artifact_registry_requires_scored_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write(root / "docs" / "reproducibility_manifest.json", "{}")
        registry = ArtifactRegistry(project_root=root)
        try:
            registry.score_backbones(["bb1"])
        except ArtifactUnavailableError:
            pass
        else:  # pragma: no cover - safety
            raise AssertionError("expected ArtifactUnavailableError")


def test_score_endpoint_prefers_artifact_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from plasmid_priority.api import app as api_app

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        data_dir = root / "data"
        scored = pd.DataFrame(
            {
                "backbone_id": ["bb1", "bb2"],
                "priority_index": [0.9, 0.2],
                "operational_priority_index": [0.88, 0.22],
                "bio_priority_index": [0.85, 0.25],
                "evidence_support_index": [0.8, 0.4],
                "T_eff": [0.7, 0.3],
                "H_eff": [0.6, 0.2],
                "A_eff": [0.9, 0.1],
            },
        )
        (data_dir / "scores").mkdir(parents=True, exist_ok=True)
        scored.to_csv(data_dir / "scores" / "backbone_scored.tsv", sep="\t", index=False)
        _write(root / "docs" / "reproducibility_manifest.json", json.dumps({"protocol_hash": "p1"}))

        monkeypatch.setattr(
            api_app,
            "_build_artifact_registry",
            lambda: ArtifactRegistry(project_root=root),
        )
        monkeypatch.delenv("PLASMID_PRIORITY_API_KEY", raising=False)
        monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
        monkeypatch.delenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", raising=False)

        client = TestClient(api_app.app)
        response = client.post(
            "/score",
            json={"backbone_ids": ["bb1", "bb-missing"], "config_key": "geo_spread"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert [row["backbone_id"] for row in payload["scores"]] == ["bb1", "bb-missing"]
        assert payload["scores"][0]["priority_index"] == 0.9
        assert payload["scores"][1]["priority_index"] == 0.0
