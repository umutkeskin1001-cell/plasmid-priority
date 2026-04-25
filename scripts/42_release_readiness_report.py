#!/usr/bin/env python3
"""Build release go/no-go readiness report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
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
    context = build_context(root)
    with ManagedScriptRun(context, "42_release_readiness_report") as run:
        json_path = output_dir / "release_readiness_report.json"
        md_path = output_dir / "release_readiness_report.md"
        run.record_output(json_path)
        run.record_output(md_path)
        payload = evaluate_release_readiness(root)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
        run.set_metric("failed_check_count", len(payload.get("failed_checks", [])))
        run.set_metric("status", payload.get("status", "fail"))
        print(f"Wrote release readiness report to: {output_dir}")
        if payload.get("status") != "pass":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
