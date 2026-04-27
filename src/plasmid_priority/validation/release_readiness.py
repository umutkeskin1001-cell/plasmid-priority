"""Release go/no-go checklist synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasmid_priority.validation.artifact_integrity import validate_release_artifact_integrity
from plasmid_priority.validation.scientific_contract import validate_release_scientific_contract


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _has_all_claim_levels(manifest: dict[str, Any] | None) -> bool:
    if manifest is None:
        return False
    levels = manifest.get("claim_levels", [])
    if not isinstance(levels, list):
        return False
    required = {"observed", "proxy", "literature_supported", "externally_validated"}
    return required.issubset({str(item) for item in levels})


def _api_is_artifact_backed(app_source: str) -> bool:
    required_tokens = (
        "ArtifactRegistry",
        '"/score/backbones"',
        "ArtifactUnavailableError",
    )
    return all(token in app_source for token in required_tokens)


def _phase5_productization_checks(project_root: Path, app_source: str) -> dict[str, bool]:
    sdk_path = project_root / "src/plasmid_priority/api/sdk.py"
    sdk_source = sdk_path.read_text(encoding="utf-8") if sdk_path.exists() else ""
    return {
        "score_surface_present": '"/score"' in app_source,
        "score_backbones_surface_present": '"/score/backbones"' in app_source,
        "models_surface_present": '"/models"' in app_source,
        "explain_surface_present": '"/explain/{backbone_id}"' in app_source,
        "evidence_surface_present": '"/evidence/{backbone_id}"' in app_source,
        "sdk_present": sdk_path.exists(),
        "sdk_deprecated_surfaces_removed": (
            '"/graphql"' not in sdk_source and '"/score/backbones/batch"' not in sdk_source
        ),
        "jury_dashboard_html_present": (
            project_root / "reports/release/jury_dashboard.html"
        ).exists(),
        "pre_registration_present": (project_root / "docs/pre_registration.md").exists(),
        "independent_audit_script_present": (
            project_root / "scripts/45_generate_independent_audit_packet.py"
        ).exists(),
        "rag_corpus_config_present": (project_root / "config/rag_corpus.yaml").exists(),
    }


def _compute_target_proxy(project_root: Path) -> bool:
    """Proxy check: budget config exists and all required modes are present."""
    budget_path = project_root / "config" / "performance_budgets.yaml"
    if not budget_path.exists():
        return False
    content = budget_path.read_text(encoding="utf-8")
    required_modes = (
        "smoke-local",
        "dev-refresh",
        "model-refresh",
        "report-refresh",
        "release-full",
    )
    return all(mode in content for mode in required_modes)


def _sensitivity_reporting_incremental(project_root: Path) -> bool:
    required = [
        project_root / "src/plasmid_priority/sensitivity/variant_cache.py",
        project_root / "src/plasmid_priority/reporting/cache.py",
    ]
    return all(path.exists() for path in required)


def _canonical_verify_surface_present(project_root: Path) -> bool:
    required = [
        project_root / "Makefile",
        project_root / "Justfile",
        project_root / "scripts/46_verify_release.py",
    ]
    if not all(path.exists() for path in required):
        return False
    makefile = (project_root / "Makefile").read_text(encoding="utf-8")
    justfile = (project_root / "Justfile").read_text(encoding="utf-8")
    return "verify-release:" in makefile and "verify-release:" in justfile


def _runbook_surface_present(project_root: Path) -> bool:
    required = [
        project_root / "docs/runbooks/release_verification.md",
        project_root / "docs/runbooks/quality_scoreboard.md",
        project_root / "docs/runbooks/scientific_invariants.md",
        project_root / "docs/runbooks/security.md",
        project_root / "docs/runbooks/demo_path.md",
    ]
    return all(path.exists() for path in required)


def _stakeholder_overviews_present(project_root: Path) -> bool:
    required = [
        project_root / "docs/scientific_overview.md",
        project_root / "docs/cto_overview.md",
        project_root / "docs/investor_overview.md",
        project_root / "docs/product_strategy.md",
    ]
    return all(path.exists() for path in required)


def evaluate_release_readiness(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    docs_manifest = _read_json(root / "docs" / "reproducibility_manifest.json")
    scientific = validate_release_scientific_contract(root)
    artifact = validate_release_artifact_integrity(root)
    app_source = (root / "src/plasmid_priority/api/app.py").read_text(encoding="utf-8")
    runner_path = root / "reports/reviewer_pack/run_reproducibility.sh"

    checks = {
        "compute_targets_configured": _compute_target_proxy(root),
        "validation_gauntlet_present": scientific.get("status") == "pass",
        "claim_levels_complete": _has_all_claim_levels(docs_manifest),
        "api_artifact_backed_non_placeholder": _api_is_artifact_backed(app_source),
        "reviewer_package_one_command": runner_path.exists(),
        "sensitivity_reporting_incremental": _sensitivity_reporting_incremental(root),
        "canonical_verify_surface_present": _canonical_verify_surface_present(root),
        "runbook_surface_present": _runbook_surface_present(root),
        "stakeholder_overviews_present": _stakeholder_overviews_present(root),
        "artifact_integrity": artifact.get("status") == "pass",
    }
    checks.update(_phase5_productization_checks(root, app_source))
    failed = [name for name, ok in checks.items() if not bool(ok)]
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "scientific_contract_errors": scientific.get("errors", []),
        "artifact_integrity_errors": artifact.get("errors", []),
    }
