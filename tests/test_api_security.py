from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from plasmid_priority.api import app as api_app


class _FakeArtifactRegistry:
    def score_backbones(self, backbone_ids: list[str]) -> list[dict[str, float | str]]:
        return [{"backbone_id": str(bid), "priority_index": 0.42} for bid in backbone_ids]

    def explain_backbone(self, backbone_id: str) -> dict[str, object]:
        return {"backbone_id": backbone_id, "components": {"T_eff": 0.2, "H_eff": 0.3, "A_eff": 0.4}}

    def get_evidence(self, backbone_id: str) -> dict[str, object]:
        return {"backbone_id": backbone_id, "claim_level": "proxy_only", "scores": {"priority_index": 0.42}}


def _reset_request_buckets() -> None:
    with api_app._REQUEST_BUCKETS_LOCK:
        api_app._REQUEST_BUCKETS.clear()


def test_scoring_surface_rejects_requests_when_api_key_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLASMID_PRIORITY_API_KEY", raising=False)
    monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", raising=False)
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: _FakeArtifactRegistry())
    _reset_request_buckets()

    client = TestClient(api_app.app)
    response = client.post("/score", json={"backbone_ids": ["bb1"], "config_key": "geo_spread"})
    assert response.status_code == 500
    assert response.json()["detail"] == "API key not configured on server"


def test_scoring_surface_auth_guard_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_API_KEY", "secret-token")
    monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", raising=False)
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: _FakeArtifactRegistry())
    _reset_request_buckets()
    client = TestClient(api_app.app)

    score = client.post("/score", json={"backbone_ids": ["bb1"], "config_key": "geo_spread"})
    explain = client.get("/explain/bb1")
    evidence = client.get("/evidence/bb1")

    assert score.status_code == 401
    assert explain.status_code == 401
    assert evidence.status_code == 401
    assert score.json()["detail"] == "Missing or invalid API key"

    authorized = client.post(
        "/score",
        json={"backbone_ids": ["bb1"], "config_key": "geo_spread"},
        headers={"x-api-key": "secret-token"},
    )
    assert authorized.status_code == 200


def test_scoring_surface_fails_closed_when_artifact_registry_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_API_KEY", "secret-token")
    monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", raising=False)
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: None)
    _reset_request_buckets()

    client = TestClient(api_app.app)
    response = client.post(
        "/score",
        json={"backbone_ids": ["bb1"], "config_key": "geo_spread"},
        headers={"x-api-key": "secret-token"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Artifact registry unavailable"


def test_score_request_size_guard_returns_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_API_KEY", "secret-token")
    monkeypatch.setenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", "128")
    monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: _FakeArtifactRegistry())
    _reset_request_buckets()

    client = TestClient(api_app.app)
    response = client.post(
        "/score",
        json={"backbone_ids": ["x" * 300], "config_key": "geo_spread"},
        headers={"x-api-key": "secret-token"},
    )
    assert response.status_code == 413
    assert "payload too large" in response.json()["detail"].lower()


def test_score_rate_limit_returns_429_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_API_KEY", "secret-token")
    monkeypatch.setenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", "4096")
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: _FakeArtifactRegistry())
    _reset_request_buckets()

    client = TestClient(api_app.app)
    headers = {"x-api-key": "secret-token"}
    first = client.post("/score", json={"backbone_ids": ["bb1"], "config_key": "geo_spread"}, headers=headers)
    second = client.post(
        "/score",
        json={"backbone_ids": ["bb1"], "config_key": "geo_spread"},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert "rate limit exceeded" in second.json()["detail"].lower()


def test_scoring_surface_accepts_bearer_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_API_KEY", "secret-token")
    monkeypatch.delenv("PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("PLASMID_PRIORITY_MAX_REQUEST_BYTES", raising=False)
    monkeypatch.setattr(api_app, "_build_artifact_registry", lambda: _FakeArtifactRegistry())
    _reset_request_buckets()

    client = TestClient(api_app.app)
    response = client.post(
        "/score",
        json={"backbone_ids": ["bb1"], "config_key": "geo_spread"},
        headers={"authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200
