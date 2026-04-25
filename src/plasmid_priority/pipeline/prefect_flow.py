"""Prefect orchestration for the Phase 4 engineering workflow."""


import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasmid_priority.config import DATA_ROOT_ENV_VAR
from plasmid_priority.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_WORKFLOW_SCRIPT = PROJECT_ROOT / "scripts" / "run_workflow.py"
WORKFLOW_TIMEOUT_ENV_VAR = "PLASMID_PRIORITY_WORKFLOW_TIMEOUT_SECONDS"

try:
    from prefect import flow, task

    PREFECT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when Prefect is absent.
    PREFECT_AVAILABLE = False
    flow = None  # type: ignore[assignment]
    task = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PrefectStage:
    """Single DAG stage mapped to a run_workflow mode."""

    name: str
    mode: str
    deps: tuple[str, ...] = ()


def resolve_phase4_runtime_options(
    *,
    max_workers: int | None = None,
    data_root: str | None = None,
) -> tuple[int, str | None]:
    """Resolve runtime parameters from explicit args and AppSettings defaults."""
    settings = get_settings()
    resolved_workers = int(max_workers) if max_workers is not None else int(settings.max_jobs)
    if resolved_workers < 1:
        resolved_workers = 1

    resolved_data_root = data_root
    if resolved_data_root in (None, "") and settings.data_root is not None:
        resolved_data_root = str(settings.resolved_data_root(PROJECT_ROOT))
    return resolved_workers, resolved_data_root


def build_phase4_stage_plan(
    *,
    include_fetch: bool = False,
    run_release: bool = False,
) -> list[PrefectStage]:
    """Construct a deterministic DAG plan for Phase 4 orchestration."""
    stages: list[PrefectStage] = []
    if include_fetch:
        stages.append(PrefectStage(name="fetch_external", mode="fetch-external"))

    core_deps: tuple[str, ...] = ("fetch_external",) if include_fetch else ()
    stages.extend(
        [
            PrefectStage(name="core_refresh", mode="core-refresh", deps=core_deps),
            PrefectStage(name="support_refresh", mode="support-refresh", deps=("core_refresh",)),
            PrefectStage(
                name="reports_only",
                mode="reports-only",
                deps=("core_refresh", "support_refresh"),
            ),
        ],
    )
    if run_release:
        stages.append(PrefectStage(name="release_bundle", mode="release", deps=("reports_only",)))
    return stages


def render_phase4_stage_plan(stages: list[PrefectStage]) -> list[str]:
    """Render a human-readable DAG plan."""
    lines: list[str] = []
    for stage in stages:
        if stage.deps:
            dep_text = ",".join(stage.deps)
            lines.append(f"{stage.name}: mode={stage.mode} deps={dep_text}")
        else:
            lines.append(f"{stage.name}: mode={stage.mode}")
    return lines


def run_workflow_mode(
    mode: str,
    *,
    max_workers: int | None = None,
    data_root: str | None = None,
    dry_run: bool = False,
    python_bin: str = sys.executable,
) -> int:
    """Run a workflow mode through the existing CLI runner."""
    resolved_workers, resolved_data_root = resolve_phase4_runtime_options(
        max_workers=max_workers,
        data_root=data_root,
    )
    command = [
        python_bin,
        str(RUN_WORKFLOW_SCRIPT),
        mode,
        "--max-workers",
        str(resolved_workers),
    ]
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    if resolved_data_root not in (None, ""):
        env[DATA_ROOT_ENV_VAR] = str(resolved_data_root)
    timeout_seconds = max(1, int(os.environ.get(WORKFLOW_TIMEOUT_ENV_VAR, "86400")))
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return 124
    return int(completed.returncode)


def _run_stage_impl(
    stage: PrefectStage,
    *,
    max_workers: int,
    data_root: str | None,
    dry_run: bool,
) -> str:
    exit_code = run_workflow_mode(
        stage.mode,
        max_workers=max_workers,
        data_root=data_root,
        dry_run=dry_run,
    )
    if exit_code != 0:
        raise RuntimeError(f"Stage '{stage.name}' failed with exit code {exit_code}")
    return stage.name


if PREFECT_AVAILABLE:

    @task(name="run-workflow-stage")
    def run_stage_task(
        stage: PrefectStage,
        *,
        max_workers: int = 4,
        data_root: str | None = None,
        dry_run: bool = False,
    ) -> str:
        return _run_stage_impl(
            stage,
            max_workers=max_workers,
            data_root=data_root,
            dry_run=dry_run,
        )


    @flow(name="plasmid-priority-phase4")
    def run_phase4_prefect_flow(
        *,
        include_fetch: bool = False,
        run_release: bool = False,
        max_workers: int = 4,
        data_root: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Execute the Phase 4 DAG via Prefect."""
        resolved_workers, resolved_data_root = resolve_phase4_runtime_options(
            max_workers=max_workers,
            data_root=data_root,
        )
        stages = build_phase4_stage_plan(include_fetch=include_fetch, run_release=run_release)
        futures: dict[str, Any] = {}
        for stage in stages:
            wait_for = [futures[name] for name in stage.deps if name in futures]
            futures[stage.name] = run_stage_task.submit(  # type: ignore[call-overload]
                stage,
                max_workers=resolved_workers,
                data_root=resolved_data_root,
                dry_run=dry_run,
                wait_for=wait_for,
            )
        return [str(futures[stage.name].result()) for stage in stages]

else:

    def run_phase4_prefect_flow(  # type: ignore[misc]
        *,
        include_fetch: bool = False,
        run_release: bool = False,
        max_workers: int = 4,
        data_root: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        del include_fetch, run_release, max_workers, data_root, dry_run
        raise RuntimeError(
            "Prefect is not installed. Install with: "
            "`uv pip install 'plasmid-priority[engineering]'`.",
        )
