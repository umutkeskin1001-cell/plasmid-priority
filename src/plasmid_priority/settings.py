"""Runtime settings for workflow and API guard controls."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from plasmid_priority.config import DATA_ROOT_ENV_VAR, find_project_root


@dataclass(frozen=True)
class AppSettings:
    """Application runtime settings loaded from environment variables."""

    workflow_mode: str = field(default_factory=lambda: _workflow_mode_env())
    max_jobs: int = field(default_factory=lambda: _max_jobs_env())
    data_root: Path | None = field(default_factory=lambda: _data_root_env())

    def resolved_data_root(self, project_root: Path | None = None) -> Path:
        """Resolve configured data root against project root when relative."""
        if self.data_root is None:
            raise ValueError(f"{DATA_ROOT_ENV_VAR} is not configured.")
        candidate = self.data_root.expanduser()
        if not candidate.is_absolute():
            base_root = (
                project_root.resolve()
                if project_root is not None
                else find_project_root(Path(__file__).resolve())
            )
            candidate = base_root / candidate
        return candidate.resolve()


@dataclass(frozen=True)
class APISettings:
    """Environment-driven API guard settings."""

    api_key: str | None
    max_request_bytes: int
    rate_limit_per_minute: int


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, value)


def _workflow_mode_env() -> str:
    return os.getenv("PLASMID_PRIORITY_WORKFLOW_MODE", "pipeline").strip() or "pipeline"


def _max_jobs_env() -> int:
    return _int_env("PLASMID_PRIORITY_MAX_JOBS", default=1, minimum=1)


def _data_root_env() -> Path | None:
    raw_data_root = os.getenv(DATA_ROOT_ENV_VAR, "").strip()
    return Path(raw_data_root) if raw_data_root else None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load cached application settings from environment variables."""
    return AppSettings()


def load_api_settings() -> APISettings:
    """Load API guard settings from environment variables."""
    api_key = os.getenv("PLASMID_PRIORITY_API_KEY", "").strip() or None
    max_request_bytes = _int_env(
        "PLASMID_PRIORITY_MAX_REQUEST_BYTES",
        1_048_576,
        minimum=0,
    )
    rate_limit_per_minute = _int_env(
        "PLASMID_PRIORITY_RATE_LIMIT_PER_MINUTE",
        0,
        minimum=0,
    )
    return APISettings(
        api_key=api_key,
        max_request_bytes=max_request_bytes,
        rate_limit_per_minute=rate_limit_per_minute,
    )
