"""Workflow telemetry helpers."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from plasmid_priority.performance.profile_schema import CacheStatus, StepTelemetry

try:  # pragma: no cover - platform dependent
    import resource
except ImportError:  # pragma: no cover - windows fallback
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time runtime snapshot used for duration/CPU calculations."""

    perf_counter: float
    child_user: float
    child_system: float
    child_maxrss: int


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def capture_resource_snapshot() -> ResourceSnapshot:
    if resource is None:  # pragma: no cover - platform dependent
        return ResourceSnapshot(  # type: ignore[unreachable]
            perf_counter=time.perf_counter(),
            child_user=0.0,
            child_system=0.0,
            child_maxrss=0,
        )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ResourceSnapshot(
        perf_counter=time.perf_counter(),
        child_user=float(usage.ru_utime),
        child_system=float(usage.ru_stime),
        child_maxrss=int(usage.ru_maxrss),
    )


def _rss_mb(raw_value: int) -> float:
    if raw_value <= 0:
        return 0.0
    if sys.platform == "darwin":
        return raw_value / (1024 * 1024)
    return raw_value / 1024.0  # type: ignore[unreachable]


def _sum_rows(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        total = 0
        for item in value.values():
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                total += int(item)
            elif isinstance(item, str) and item.strip():
                try:
                    total += int(float(item))
                except ValueError:
                    continue
        return total
    return None


def _sum_manifest_bytes(manifest: object) -> int | None:
    if not isinstance(manifest, dict):
        return None
    total = 0
    seen = False
    for entry in manifest.values():
        if not isinstance(entry, dict):
            continue
        size = entry.get("size")
        if isinstance(size, bool):
            continue
        if isinstance(size, (int, float)):
            total += int(size)
            seen = True
        elif isinstance(size, str) and size.strip():
            try:
                total += int(float(size))
                seen = True
            except ValueError:
                continue
    return total if seen else None


def _manifest_hash(manifest: object) -> str | None:
    if not isinstance(manifest, dict):
        return None
    return _stable_hash(manifest)


def infer_cache_status_from_summary(summary: dict[str, Any]) -> CacheStatus:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics.get("cache_hit") is True:
        return "partial"
    return "miss"


def build_step_telemetry(
    *,
    step_name: str,
    status: str,
    cache_status: CacheStatus,
    summary: dict[str, Any],
    started: ResourceSnapshot,
    finished_perf_counter: float | None = None,
) -> StepTelemetry:
    finished = finished_perf_counter if finished_perf_counter is not None else time.perf_counter()
    duration_seconds = max(0.0, float(finished - started.perf_counter))

    cpu_time_seconds: float | None = None
    peak_rss_mb: float | None = None
    if resource is not None:  # pragma: no cover - platform dependent
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_delta = (float(usage.ru_utime) - started.child_user) + (
            float(usage.ru_stime) - started.child_system
        )
        cpu_time_seconds = max(0.0, cpu_delta)
        peak_rss_mb = _rss_mb(max(int(usage.ru_maxrss), int(started.child_maxrss)))

    io_wait_hint: float | None = None
    if cpu_time_seconds is not None:
        io_wait_hint = max(0.0, duration_seconds - cpu_time_seconds)

    input_manifest = summary.get("input_manifest", {})
    output_manifest = summary.get("output_manifest", {})
    return StepTelemetry(
        step_name=step_name,
        status=status,
        cache_status=cache_status,
        duration_seconds=duration_seconds,
        rows_in=_sum_rows(summary.get("n_rows_in")),
        rows_out=_sum_rows(summary.get("n_rows_out")),
        bytes_read=_sum_manifest_bytes(input_manifest),
        bytes_written=_sum_manifest_bytes(output_manifest),
        peak_rss_mb=peak_rss_mb,
        cpu_time_seconds=cpu_time_seconds,
        io_wait_hint=io_wait_hint,
        input_hash=_manifest_hash(input_manifest),
        output_hash=_manifest_hash(output_manifest),
    )


def upsert_summary_telemetry(summary: dict[str, Any], telemetry: StepTelemetry) -> dict[str, Any]:
    payload = dict(summary)
    telemetry_payload = telemetry.to_dict()
    payload["duration_seconds"] = telemetry_payload["duration_seconds"]
    payload["cache_status"] = telemetry_payload["cache_status"]
    payload["rows_in"] = telemetry_payload.get("rows_in")
    payload["rows_out"] = telemetry_payload.get("rows_out")
    payload["bytes_read"] = telemetry_payload.get("bytes_read")
    payload["bytes_written"] = telemetry_payload.get("bytes_written")
    payload["peak_rss_mb"] = telemetry_payload.get("peak_rss_mb")
    payload["cpu_time_seconds"] = telemetry_payload.get("cpu_time_seconds")
    payload["io_wait_hint"] = telemetry_payload.get("io_wait_hint")
    payload["input_hash"] = telemetry_payload.get("input_hash")
    payload["output_hash"] = telemetry_payload.get("output_hash")
    payload["telemetry"] = telemetry_payload
    return payload
