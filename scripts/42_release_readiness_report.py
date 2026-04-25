#!/usr/bin/env python3
"""Build release go/no-go readiness report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plasmid_priority.utils.files import atomic_write_json, ensure_directory
from plasmid_priority.validation.release_readiness import evaluate_release_readiness


def _render_markdown(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines = [
        "# Release Readiness Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Overall status: `{payload.get('status', 'fail')}`",
        "",
        "## Checklist",
        "",
    ]
    for key in sorted(checks):
        icon = "PASS" if checks.get(key) else "FAIL"
        lines.append(f"- {icon} `{key}`")
    failed = payload.get("failed_checks", [])
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for item in failed:
            lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reports/release under project root.",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output_dir = ensure_directory(args.output_dir or (root / "reports" / "release"))

    payload = evaluate_release_readiness(root)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(output_dir / "release_readiness_report.json", payload)
    (output_dir / "release_readiness_report.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    print(f"Wrote release readiness report to: {output_dir}")
    if payload.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
