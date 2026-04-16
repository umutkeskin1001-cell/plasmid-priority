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
    MODE_DEFAULT_WORKFLOW,
    prompt_for_data_root,
    resolve_mode_data_root,
    validate_mode_workflow,
)
from plasmid_priority.snapshots import profile_has_content, sync_profile_outputs


def _prepare_data_root(mode: str, data_root: Path) -> Path:
    resolved = data_root.expanduser().resolve()
    if mode == "full-local":
        if not resolved.exists():
            raise FileNotFoundError(
                f"Full-local data root does not exist: {resolved}. Mount the USB path first."
            )
        if not resolved.is_dir():
            raise NotADirectoryError(f"Full-local data root is not a directory: {resolved}")
        return resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_workflow_command(
    workflow: str,
    *,
    data_root: Path,
    max_workers: int | None,
    dry_run: bool,
) -> int:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_workflow.py"), workflow]
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env[DATA_ROOT_ENV_VAR] = str(data_root)
    timeout_seconds = max(
        1, int(os.environ.get("PLASMID_PRIORITY_WORKFLOW_TIMEOUT_SECONDS", "86400"))
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        env=env,
        timeout=timeout_seconds,
    )
    return int(completed.returncode)


def _prepare_source_data_root(source_data_root: str | Path) -> Path:
    resolved = Path(str(source_data_root)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Source data root does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Source data root is not a directory: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_DEFAULT_WORKFLOW))
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
    args = parser.parse_args(argv)

    workflow = args.workflow or MODE_DEFAULT_WORKFLOW[args.mode]
    validate_mode_workflow(args.mode, workflow)

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
                "Fast-local cache is empty. Provide --source-data-root once to seed the local report cache."
            )
    return _run_workflow_command(
        workflow,
        data_root=data_root,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
