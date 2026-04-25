#!/usr/bin/env python3
"""Generate independent audit + pre-registration packet."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plasmid_priority.validation.artifact_integrity import validate_release_artifact_integrity
from plasmid_priority.validation.release_readiness import evaluate_release_readiness
from plasmid_priority.validation.scientific_contract import validate_release_scientific_contract


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Independent Audit Packet",
        "",
        f"Generated at: {payload.get('generated_at', '')}",
        f"Overall status: `{payload.get('status', 'fail')}`",
        "",
        "## Controls",
        "",
    ]
    checks = payload.get("checks", {})
    if isinstance(checks, dict):
        for key in sorted(checks):
            icon = "PASS" if checks.get(key) else "FAIL"
            lines.append(f"- {icon} `{key}`")
    prereg = payload.get("pre_registration_path", "")
    lines.extend(["", "## Pre-registration", "", f"- Path: `{prereg}`"])
    failures = payload.get("failed_checks", [])
    if isinstance(failures, list) and failures:
        lines.extend(["", "## Failed Checks", ""])
        for item in failures:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    output_dir = (args.output_dir or (root / "reports" / "audits")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prereg_path = root / "docs" / "pre_registration.md"
    release_readiness = evaluate_release_readiness(root)
    scientific = validate_release_scientific_contract(root)
    artifact = validate_release_artifact_integrity(root)

    checks = {
        "release_readiness_pass": release_readiness.get("status") == "pass",
        "scientific_contract_pass": scientific.get("status") == "pass",
        "artifact_integrity_pass": artifact.get("status") == "pass",
        "pre_registration_present": prereg_path.exists(),
    }
    failed_checks = [name for name, ok in checks.items() if not bool(ok)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not failed_checks else "fail",
        "checks": checks,
        "failed_checks": failed_checks,
        "pre_registration_path": str(prereg_path),
        "release_readiness": release_readiness,
        "scientific_contract": scientific,
        "artifact_integrity": artifact,
    }

    json_path = output_dir / "independent_audit_packet.json"
    md_path = output_dir / "independent_audit_packet.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
