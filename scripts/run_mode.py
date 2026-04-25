#!/usr/bin/env python3
"""Run a workflow with an explicit runtime mode and data root policy."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import DATA_ROOT_ENV_VAR
from plasmid_priority.runtime import (
    MODE_ALLOWED_WORKFLOWS,
    MODE_DEFAULT_WORKFLOW,
    prompt_for_data_root,
    resolve_mode_data_root,
    validate_mode_workflow,
)
from plasmid_priority.settings import get_settings
from plasmid_priority.snapshots import profile_has_content, sync_profile_outputs

WORKFLOW_DEFAULT_BUDGET_MODE: dict[str, str] = {
    "pipeline": "release-full",
    "pipeline-sequential": "release-full",
    "analysis-refresh": "dev-refresh",
    "analysis-refresh-sequential": "dev-refresh",
    "core-refresh": "model-refresh",
    "support-refresh": "dev-refresh",
    "reports-only": "report-refresh",
    "release": "release-full",
    "fetch-external": "smoke-local",
    "demo-pipeline": "smoke-local",
}


def _prepare_data_root(mode: str, data_root: Path) -> Path:
    resolved = data_root.expanduser().resolve()
    if mode == "full-local":
        if not resolved.exists():
            raise FileNotFoundError(
                f"Full-local data root does not exist: {resolved}. Mount the USB path first.",
            )
        if not resolved.is_dir():
            raise NotADirectoryError(f"Full-local data root is not a directory: {resolved}")
        return resolved
    if mode == "demo":
        # For demo mode, auto-generate sample data if missing
        manifest = resolved / "SAMPLE_DATA_MANIFEST.json"
        if not manifest.exists():
            print("\n[run_mode] Sample data not found. Generating demo dataset...", flush=True)
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_sample_data.py")],
                cwd=PROJECT_ROOT,
                check=True,
            )
            print("[run_mode] Sample data generated at " + str(resolved) + "\n", flush=True)
        return resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_workflow_command(
    workflow: str,
    *,
    data_root: Path,
    max_workers: int | None,
    dry_run: bool,
    budget_mode: str | None = None,
    strict_budget: bool = False,
    extra_env: dict[str, str] | None = None,
) -> int:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_workflow.py"), workflow]
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    if budget_mode:
        command.extend(["--budget-mode", str(budget_mode)])
    if strict_budget:
        command.append("--strict-budget")
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env[DATA_ROOT_ENV_VAR] = str(data_root)
    if extra_env:
        env.update(extra_env)
    timeout_seconds = max(
        1,
        int(os.environ.get("PLASMID_PRIORITY_WORKFLOW_TIMEOUT_SECONDS", "86400")),
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
        timeout=timeout_seconds,
    )
    return int(completed.returncode)


def _run_profile_and_dashboard(
    *,
    data_root: Path,
    budget_mode: str | None,
) -> None:
    env = os.environ.copy()
    env[DATA_ROOT_ENV_VAR] = str(data_root)
    budget_mode_value = budget_mode or "smoke-local"
    profile_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "profile_workflow.py"),
        "--budget-mode",
        budget_mode_value,
        "--top-n",
        "10",
    ]
    profile_result = subprocess.run(
        profile_command,
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
    )
    if int(profile_result.returncode) != 0:
        print(
            f"[run_mode] profile_workflow failed with code {int(profile_result.returncode)}",
            file=sys.stderr,
            flush=True,
        )
        return
    dashboard_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "41_build_performance_dashboard.py")],
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
    )
    if int(dashboard_result.returncode) != 0:
        print(
            f"[run_mode] performance dashboard build failed with code {int(dashboard_result.returncode)}",
            file=sys.stderr,
            flush=True,
        )
    readiness_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "42_release_readiness_report.py")],
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
    )
    if int(readiness_result.returncode) != 0:
        print(
            f"[run_mode] release readiness report failed with code {int(readiness_result.returncode)}",
            file=sys.stderr,
            flush=True,
        )
    jury_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "44_build_jury_dashboard.py")],
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
    )
    if int(jury_result.returncode) != 0:
        print(
            f"[run_mode] jury dashboard build failed with code {int(jury_result.returncode)}",
            file=sys.stderr,
            flush=True,
        )
    audit_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "45_generate_independent_audit_packet.py")],
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
    )
    if int(audit_result.returncode) != 0:
        print(
            f"[run_mode] independent audit packet generation failed with code {int(audit_result.returncode)}",
            file=sys.stderr,
            flush=True,
        )


def _prepare_source_data_root(source_data_root: str | Path) -> Path:
    resolved = Path(str(source_data_root)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Source data root does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Source data root is not a directory: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_DEFAULT_WORKFLOW) + ("demo",))
    parser.add_argument(
        "--workflow",
        default=None,
        help="Workflow to run. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="External or cache-backed data root. Required for full-local if stdin is not interactive.",
    )
    parser.add_argument(
        "--source-data-root",
        default=None,
        help="For fast-local, refresh the small local report cache from this full data root before running.",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--budget-mode",
        default=None,
        help="Override runtime budget mode passed to run_workflow.py",
    )
    parser.add_argument(
        "--strict-budget",
        action="store_true",
        help="Fail if runtime exceeds budget threshold.",
    )
    parser.add_argument(
        "--report-mode",
        choices=("report-fast", "report-full", "report-diff"),
        default=None,
        help="Exported as PLASMID_PRIORITY_REPORT_MODE for report build step.",
    )
    parser.add_argument(
        "--compute-tier",
        default=None,
        help="Exported as PLASMID_PRIORITY_COMPUTE_TIER for module A.",
    )
    parser.add_argument(
        "--no-profile-after-run",
        action="store_true",
        help="Skip profile_workflow + performance dashboard generation after successful run.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    default_workflow = MODE_DEFAULT_WORKFLOW[args.mode]
    if args.mode == "full-local" and settings.workflow_mode in MODE_ALLOWED_WORKFLOWS["full-local"]:
        default_workflow = settings.workflow_mode
    workflow = args.workflow or default_workflow
    validate_mode_workflow(args.mode, workflow)
    if args.max_workers is None:
        args.max_workers = max(1, int(settings.max_jobs))

    if args.mode == "full-local" and args.data_root in (None, ""):
        # Check env var first before prompting
        env_data_root = os.environ.get(DATA_ROOT_ENV_VAR)
        if env_data_root:
            args.data_root = env_data_root
        else:
            args.data_root = prompt_for_data_root()

    data_root = _prepare_data_root(args.mode, resolve_mode_data_root(args.mode, args.data_root))
    if args.mode == "fast-local":
        if args.source_data_root not in (None, ""):
            source_root = _prepare_source_data_root(str(args.source_data_root))
            sync_profile_outputs(source_root, data_root, "report-pack", clean_first=True)
        elif not profile_has_content(data_root, "report-pack"):
            raise FileNotFoundError(
                "Fast-local cache is empty. Provide --source-data-root once to seed the local report cache.",
            )
    resolved_budget_mode = args.budget_mode or WORKFLOW_DEFAULT_BUDGET_MODE.get(
        workflow, "smoke-local"
    )
    extra_env: dict[str, str] = {}
    if args.report_mode:
        extra_env["PLASMID_PRIORITY_REPORT_MODE"] = str(args.report_mode)
    if args.compute_tier:
        extra_env["PLASMID_PRIORITY_COMPUTE_TIER"] = str(args.compute_tier)

    result = _run_workflow_command(
        workflow,
        data_root=data_root,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        budget_mode=resolved_budget_mode,
        strict_budget=bool(args.strict_budget),
        extra_env=extra_env if extra_env else None,
    )
    if result == 0 and not args.dry_run and not args.no_profile_after_run:
        _run_profile_and_dashboard(data_root=data_root, budget_mode=resolved_budget_mode)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
