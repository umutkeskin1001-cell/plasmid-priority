"""Typed schema objects for workflow performance profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

CacheStatus = Literal["hit", "miss", "partial", "bypass"]


@dataclass(frozen=True)
class StepTelemetry:
    """Per-step telemetry snapshot captured by workflow orchestration."""

    step_name: str
    status: str
    cache_status: CacheStatus
    duration_seconds: float
    rows_in: int | None = None
    rows_out: int | None = None
    bytes_read: int | None = None
    bytes_written: int | None = None
    peak_rss_mb: float | None = None
    cpu_time_seconds: float | None = None
    io_wait_hint: float | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_seconds"] = round(float(self.duration_seconds), 6)
        if self.cpu_time_seconds is not None:
            payload["cpu_time_seconds"] = round(float(self.cpu_time_seconds), 6)
        if self.io_wait_hint is not None:
            payload["io_wait_hint"] = round(float(self.io_wait_hint), 6)
        if self.peak_rss_mb is not None:
            payload["peak_rss_mb"] = round(float(self.peak_rss_mb), 3)
        return payload


@dataclass(frozen=True)
class WorkflowProfile:
    """Workflow-level profile document."""

    mode: str
    status: str
    protocol_hash: str
    started_at: str
    finished_at: str
    duration_seconds: float
    steps: list[StepTelemetry]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "performance-profile-v1"

    @classmethod
    def now(
        cls,
        *,
        mode: str,
        status: str,
        protocol_hash: str,
        started_at: datetime,
        duration_seconds: float,
        steps: list[StepTelemetry],
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowProfile:
        finished_at = datetime.now(UTC)
        return cls(
            mode=mode,
            status=status,
            protocol_hash=protocol_hash,
            started_at=started_at.isoformat(timespec="seconds"),
            finished_at=finished_at.isoformat(timespec="seconds"),
            duration_seconds=duration_seconds,
            steps=steps,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_seconds"] = round(float(self.duration_seconds), 6)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload
