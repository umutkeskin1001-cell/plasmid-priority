from __future__ import annotations

import importlib.util
import json
import tempfile
import time
from pathlib import Path
from unittest import mock

from plasmid_priority.api import app as api_app
from plasmid_priority.api.sdk import PlasmidPrioritySDK
from plasmid_priority.reporting.rag import load_rag_chunks, retrieve_rag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JURY_SPEC = importlib.util.spec_from_file_location(
    "jury_dashboard_script",
    PROJECT_ROOT / "scripts/44_build_jury_dashboard.py",
)
assert JURY_SPEC is not None and JURY_SPEC.loader is not None
jury_dashboard_script = importlib.util.module_from_spec(JURY_SPEC)
JURY_SPEC.loader.exec_module(jury_dashboard_script)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "independent_audit_script",
    PROJECT_ROOT / "scripts/45_generate_independent_audit_packet.py",
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
independent_audit_script = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(independent_audit_script)


def test_graphql_dispatch_supports_models_and_score_backbones() -> None:
    class _FakeRegistry:
        @staticmethod
        def list_models() -> list[dict[str, str]]:
            return [{"model_version": "v1"}]

        @staticmethod
        def score_backbones(backbone_ids: list[str]) -> list[dict[str, object]]:
            return [{"backbone_id": item, "priority_index": 0.5} for item in backbone_ids]

    with mock.patch.object(api_app, "REGISTRY", _FakeRegistry()):
        models_payload = api_app._execute_graphql_query("query { models }")
        assert models_payload["data"]["models"][0]["model_version"] == "v1"
        scores_payload = api_app._execute_graphql_query(
            "query($backboneIds: [String!]!) { scoreBackbones(backboneIds: $backboneIds) { backbone_id } }",
            {"backboneIds": ["bb1", "bb2"]},
        )
        assert len(scores_payload["data"]["scoreBackbones"]) == 2


def test_async_batch_job_reaches_completed_state() -> None:
    class _FakeRegistry:
        @staticmethod
        def score_backbones(backbone_ids: list[str]) -> list[dict[str, object]]:
            return [{"backbone_id": item, "priority_index": 0.7} for item in backbone_ids]

    with mock.patch.object(api_app, "REGISTRY", _FakeRegistry()):
        job_id = api_app._start_batch_score_job(["bb1"])
        deadline = time.time() + 2.0
        payload: dict[str, object] = {}
        while time.time() < deadline:
            payload = api_app._batch_job(job_id)
            if payload.get("status") == "completed":
                break
            time.sleep(0.02)
        assert payload.get("status") == "completed"
        result = payload.get("result", {})
        assert isinstance(result, dict)
        scores = result.get("scores", [])
        assert isinstance(scores, list)
        assert scores and scores[0]["backbone_id"] == "bb1"


def test_sdk_routes_requests_to_expected_paths() -> None:
    captured: list[tuple[str, str, dict[str, object] | None]] = []

    class _FakeSDK(PlasmidPrioritySDK):
        def _request(
            self, method: str, path: str, payload: dict[str, object] | None = None
        ) -> dict[str, object]:
            captured.append((method, path, payload))
            return {"status": "ok"}

    sdk = _FakeSDK(base_url="http://localhost:1234")
    sdk.health()
    sdk.score_backbones(["bb1"])
    sdk.start_score_batch(["bb2"])
    sdk.graphql("query { health }")
    assert ("GET", "/health", None) in captured
    assert ("POST", "/score/backbones", {"backbone_ids": ["bb1"]}) in captured
    assert ("POST", "/score/backbones/batch", {"backbone_ids": ["bb2"]}) in captured
    assert any(path == "/graphql" for _, path, _ in captured)


def test_local_rag_retrieval_returns_ranked_hits() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        doc_a = root / "a.md"
        doc_b = root / "b.md"
        doc_a.write_text("# Claim Levels\nExternal validation and evidence hierarchy.\n", encoding="utf-8")
        doc_b.write_text("# Runtime\nRuntime budget and profiling details.\n", encoding="utf-8")
        chunks = load_rag_chunks([doc_a, doc_b])
        hits = retrieve_rag(chunks, "external validation evidence", top_k=2)
        assert hits
        assert "Claim Levels" in str(hits[0]["title"])


def test_jury_dashboard_script_writes_json_and_html() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        output_dir = root / "reports" / "release"
        output_dir.mkdir(parents=True, exist_ok=True)
        (root / "config").mkdir(parents=True, exist_ok=True)
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "performance").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "scientific_protocol.md").write_text(
            "# Scientific Protocol\nValidation and reproducibility details.\n",
            encoding="utf-8",
        )
        (root / "config" / "rag_corpus.yaml").write_text(
            "paths:\n  - docs/scientific_protocol.md\n",
            encoding="utf-8",
        )
        (output_dir / "release_readiness_report.json").write_text(
            json.dumps({"status": "pass", "checks": {"x": True}}),
            encoding="utf-8",
        )
        (root / "reports" / "performance" / "workflow_performance_dashboard.json").write_text(
            json.dumps({"summaries": []}),
            encoding="utf-8",
        )
        result = jury_dashboard_script.main(
            ["--project-root", str(root), "--output-dir", str(output_dir)]
        )
        assert result == 0
        assert (output_dir / "jury_dashboard.json").exists()
        assert (output_dir / "jury_dashboard.html").exists()


def test_independent_audit_packet_script_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "pre_registration.md").write_text("# prereg\n", encoding="utf-8")
        output_dir = root / "reports" / "audits"
        with (
            mock.patch.object(
                independent_audit_script,
                "evaluate_release_readiness",
                return_value={"status": "pass"},
            ),
            mock.patch.object(
                independent_audit_script,
                "validate_release_scientific_contract",
                return_value={"status": "pass"},
            ),
            mock.patch.object(
                independent_audit_script,
                "validate_release_artifact_integrity",
                return_value={"status": "pass"},
            ),
        ):
            result = independent_audit_script.main(
                ["--project-root", str(root), "--output-dir", str(output_dir)]
            )
        assert result == 0
        assert (output_dir / "independent_audit_packet.json").exists()
        assert (output_dir / "independent_audit_packet.md").exists()
