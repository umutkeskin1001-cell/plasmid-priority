#!/usr/bin/env python3
"""Run the canonical release verification gauntlet."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from plasmid_priority.utils.files import atomic_write_json, ensure_directory


@dataclass(frozen=True)
class VerificationStep:
    name: str
    command: tuple[str, ...]


DEFAULT_STEPS: tuple[VerificationStep, ...] = (
    VerificationStep("protocol-freshness", ("make", "protocol-freshness")),
    VerificationStep("import-contract", ("make", "import-contract")),
    VerificationStep("runtime-budget-gate", ("make", "runtime-budget-gate")),
    VerificationStep("scientific-contract-gate", ("make", "scientific-contract-gate")),
    VerificationStep("artifact-integrity", ("make", "artifact-integrity")),
    VerificationStep("release-readiness", ("make", "release-readiness")),
    VerificationStep("lint", ("make", "lint")),
    VerificationStep("typecheck", ("make", "typecheck")),
    VerificationStep("test-cov", ("make", "test-cov")),
    VerificationStep("critical-coverage", ("make", "critical-coverage")),
    VerificationStep("security", ("make", "security")),
    VerificationStep("smoke", ("make", "smoke")),
    VerificationStep("docs-check", ("make", "docs-check")),
)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Release Verification Summary",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Overall status: `{payload['status']}`",
        "",
        "## Steps",
        "",
    ]
    for step in payload["steps"]:
        status = "PASS" if step["status"] == "pass" else "FAIL"
        lines.append(
            f"- {status} `{step['name']}`: `{step['command']}` "
            f"({step['duration_seconds']:.2f}s)"
        )
    failed = payload.get("failed_step")
    if failed:
        lines.extend(["", "## First Failure", "", f"- `{failed}`"])
    return "\n".join(lines) + "\n"


def _run_step(step: VerificationStep, *, project_root: Path, dry_run: bool) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    command_text = shlex.join(step.command)
    if dry_run:
        return {
            "name": step.name,
            "command": command_text,
            "status": "planned",
            "return_code": 0,
            "duration_seconds": 0.0,
            "started_at": started_at,
        }
    print(f"[verify-release] START {step.name}: {command_text}", flush=True)
    started_perf = perf_counter()
    completed = subprocess.run(step.command, cwd=project_root, check=False)
    duration_seconds = max(0.0, perf_counter() - started_perf)
    status = "pass" if completed.returncode == 0 else "fail"
    print(
        f"[verify-release] {status.upper()} {step.name} "
        f"(rc={completed.returncode}, {duration_seconds:.2f}s)",
        flush=True,
    )
    return {
        "name": step.name,
        "command": command_text,
        "status": status,
        "return_code": int(completed.returncode),
        "duration_seconds": duration_seconds,
        "started_at": started_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reports/release under project root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned steps only.")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    output_dir = ensure_directory(args.output_dir or (project_root / "reports" / "release"))

    step_results: list[dict[str, Any]] = []
    failed_step: str | None = None
    for step in DEFAULT_STEPS:
        result = _run_step(step, project_root=project_root, dry_run=args.dry_run)
        step_results.append(result)
        if result["status"] == "fail":
            failed_step = step.name
            break

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if failed_step is None else "fail",
        "failed_step": failed_step,
        "steps": step_results,
    }
    atomic_write_json(output_dir / "release_verification_summary.json", payload)
    (output_dir / "release_verification_summary.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    if failed_step is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
