"""Release-time scientific contract checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasmid_priority.evidence import validate_claim_levels_present
from plasmid_priority.validation.label_cards import validate_label_cards

REQUIRED_VALIDATION_FILES: tuple[str, ...] = (
    "reports/diagnostic_tables/rolling_temporal_validation.tsv",
    "reports/core_tables/blocked_holdout_summary.tsv",
    "reports/core_tables/spatial_holdout_summary.tsv",
    "data/analysis/calibration_metrics.tsv",
    "reports/diagnostic_tables/negative_control_audit.tsv",
    "reports/core_tables/literature_validation_matrix.tsv",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_release_scientific_contract(project_root: Path) -> dict[str, object]:
    """Validate minimal scientific release contract before publishing artifacts."""
    docs_dir = project_root / "docs"
    reviewer_pack_dir = project_root / "reports" / "reviewer_pack"
    required_files = [
        docs_dir / "model_card.md",
        docs_dir / "data_card.md",
        docs_dir / "benchmark_contract.md",
        docs_dir / "scientific_protocol.md",
        docs_dir / "label_card_bundle.md",
        docs_dir / "reproducibility_manifest.json",
        reviewer_pack_dir / "canonical_metadata.json",
        reviewer_pack_dir / "run_reproducibility.sh",
    ]
    errors: list[str] = []
    for required in required_files:
        if not required.exists():
            errors.append(f"missing required scientific artifact: {required}")
            continue
        if required.suffix == ".md":
            text = required.read_text(encoding="utf-8").strip()
            if not text:
                errors.append(f"empty markdown artifact: {required}")
            if required.name == "label_card_bundle.md":
                if "## Label Cards" not in text:
                    errors.append("label_card_bundle.md missing '## Label Cards' section")
                if "## Claim Levels" not in text:
                    errors.append("label_card_bundle.md missing '## Claim Levels' section")

    reproducibility_manifest_path = docs_dir / "reproducibility_manifest.json"
    reproducibility_manifest = _read_json(reproducibility_manifest_path)
    if reproducibility_manifest is None:
        errors.append("reproducibility_manifest.json is missing or invalid JSON")
    else:
        claim_levels = reproducibility_manifest.get("claim_levels", [])
        if not isinstance(claim_levels, list):
            errors.append("reproducibility_manifest.json claim_levels must be a list")
        label_cards = reproducibility_manifest.get("label_cards", [])
        errors.extend(validate_label_cards(label_cards))

    for rel_path in REQUIRED_VALIDATION_FILES:
        path = project_root / rel_path
        if not path.exists():
            errors.append(f"missing required validation matrix artifact: {path}")
            continue
        if path.is_file() and path.stat().st_size <= 0:
            errors.append(f"empty validation matrix artifact: {path}")

    if reproducibility_manifest is not None:
        claim_levels = reproducibility_manifest.get("claim_levels", [])
        if isinstance(claim_levels, list):
            missing_levels = validate_claim_levels_present([str(item) for item in claim_levels])
            if missing_levels:
                errors.append(
                    "reproducibility_manifest.json missing claim levels: "
                    + ", ".join(missing_levels),
                )
        dossier_paths = reproducibility_manifest.get("candidate_evidence_dossiers", [])
        if not isinstance(dossier_paths, list) or not dossier_paths:
            errors.append("reproducibility_manifest.json missing candidate_evidence_dossiers")
        else:
            for rel in dossier_paths:
                dossier_path = project_root / str(rel)
                if not dossier_path.exists():
                    errors.append(f"missing candidate evidence dossier: {dossier_path}")
    dossier_dir = reviewer_pack_dir / "candidate_evidence_dossiers"
    if not dossier_dir.exists() or not any(dossier_dir.glob("*.md")):
        errors.append("missing candidate_evidence_dossiers markdown artifacts")
    return {"status": "pass" if not errors else "fail", "errors": errors}
