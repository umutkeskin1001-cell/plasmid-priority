"""Runtime mode helpers for choosing workflow and data roots."""

from __future__ import annotations

import sys
from pathlib import Path

from plasmid_priority.config import DATA_ROOT_ENV_VAR, find_project_root

MODE_DEFAULT_WORKFLOW: dict[str, str] = {
    "fast-local": "reports-only",
    "full-local": "pipeline",
}

MODE_ALLOWED_WORKFLOWS: dict[str, tuple[str, ...]] = {
    "fast-local": ("reports-only",),
    "full-local": (
        "pipeline",
        "pipeline-sequential",
        "analysis-refresh",
        "analysis-refresh-sequential",
        "core-refresh",
        "support-refresh",
        "reports-only",
        "release",
    ),
}


_RUNTIME_PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def default_mode_data_root(mode: str) -> Path:
    return Path.home() / ".cache" / "plasmid-priority" / mode / "data"


def resolve_mode_workflow(mode: str, workflow: str | None = None) -> str:
    """Resolve the workflow for a runtime mode and validate the choice."""
    chosen_workflow = workflow or MODE_DEFAULT_WORKFLOW[mode]
    validate_mode_workflow(mode, chosen_workflow)
    return chosen_workflow


def resolve_mode_data_root(mode: str, explicit_data_root: str | Path | None = None) -> Path:
    if explicit_data_root not in (None, ""):
        candidate = Path(str(explicit_data_root)).expanduser()
        if not candidate.is_absolute():
            candidate = _RUNTIME_PROJECT_ROOT / candidate
        return candidate.resolve()
    if mode == "full-local":
        raise ValueError(
            f"{DATA_ROOT_ENV_VAR} must be set or --data-root must be provided for full-local mode."
        )
    return default_mode_data_root(mode)


def prompt_for_data_root() -> str:
    if not sys.stdin.isatty():
        raise ValueError(
            "Missing --data-root for full-local mode and no interactive terminal is available."
        )
    value = input("Full-local data root path: ").strip()
    if not value:
        raise ValueError("Full-local mode requires a non-empty data root path.")
    return value


def validate_mode_workflow(mode: str, workflow: str) -> None:
    allowed = MODE_ALLOWED_WORKFLOWS[mode]
    if workflow not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"Workflow '{workflow}' is not supported for {mode}. Allowed: {joined}.")
