from __future__ import annotations

import importlib.util
import json
import tempfile
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


def test_model_registry_scores_backbones_and_zero_fills_missing() -> None:
    class _FakeRegistry:
        @staticmethod
        def score_backbones(backbone_ids: list[str]) -> list[dict[str, object]]:
            assert backbone_ids == ["bb1", "bb2"]
            return [
                {
                    "backbone_id": "bb1",
                    "priority_index": "0.50",
                    "operational_priority_index": "0.40",
                    "bio_priority_index": "0.60",
                    "evidence_support_index": "0.70",
                },
            ]

    with mock.patch.object(api_app, "_build_artifact_registry", return_value=_FakeRegistry()):
        scores = api_app.ModelRegistry.score_backbones(["bb1", "bb2"])
    assert len(scores) == 2
    assert scores[0]["backbone_id"] == "bb1"
    assert scores[0]["priority_index"] == 0.5
    assert scores[1] == {"backbone_id": "bb2", "priority_index": 0.0}


def test_model_registry_fails_closed_when_artifact_registry_unavailable() -> None:
    if not api_app.FASTAPI_AVAILABLE:
        return
    with mock.patch.object(api_app, "_build_artifact_registry", return_value=None):
        try:
            api_app.ModelRegistry.score_backbones(["bb1"])
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 503
            assert "artifact registry unavailable" in str(getattr(exc, "detail", "")).lower()
        else:  # pragma: no cover - safety
            raise AssertionError("expected HTTPException when artifact registry is unavailable")


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
    sdk.config()
    sdk.models()
    sdk.score_backbones(["bb1"])
    sdk.explain_backbone("bb1")
    sdk.get_evidence("bb1")
    assert ("GET", "/health", None) in captured
    assert ("GET", "/config", None) in captured
    assert ("GET", "/models", None) in captured
    assert (
        "POST",
        "/score/backbones",
        {"backbone_ids": ["bb1"], "config_key": "geo_spread"},
    ) in captured
    assert ("GET", "/explain/bb1", None) in captured
    assert ("GET", "/evidence/bb1", None) in captured


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
