"""Trend analysis and regression detection for workflow telemetry history."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepTrend:
    """Trend summary for a single step across historical runs."""

    step_name: str
    n_observations: int
    mean_duration: float | None
    std_duration: float | None
    min_duration: float | None
    max_duration: float | None
    last_duration: float | None
    regression_detected: bool
    regression_zscore: float | None
    regression_threshold: float


@dataclass(frozen=True)
class CacheHitRate:
    """Aggregate cache performance summary."""

    total_steps: int
    cache_hits: int
    cache_misses: int
    cache_partials: int
    hit_rate: float
    by_step: dict[str, dict[str, int]]


@dataclass(frozen=True)
class TrendReport:
    """Full trend analysis report for a workflow profile."""

    mode: str | None
    n_runs: int
    steps: list[StepTrend]
    cache_summary: CacheHitRate
    overall_regression: bool
    metadata: dict[str, Any]


def _load_history_rows(history_path: Path) -> list[dict[str, Any]]:
    """Load all historical profile rows from a JSONL file."""
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        content = history_path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _extract_step_durations(rows: list[dict[str, Any]], step_name: str) -> list[float]:
    durations: list[float] = []
    for row in rows:
        steps = row.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("step_name", "")) != step_name:
                continue
            duration_value = step.get("duration_seconds")
            if duration_value is None:
                continue
            try:
                duration = float(duration_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration >= 0.0:
                durations.append(duration)
    return durations


def _summarize_cache_hits(rows: list[dict[str, Any]]) -> CacheHitRate:
    by_step: dict[str, dict[str, int]] = {}
    total_steps = 0
    cache_hits = 0
    cache_misses = 0
    cache_partials = 0
    for row in rows:
        steps = row.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_name = str(step.get("step_name", "")).strip() or "unknown"
            cache_status = str(step.get("cache_status", "")).strip().lower() or "unknown"
            bucket = by_step.setdefault(
                step_name,
                {"hit": 0, "miss": 0, "partial": 0, "bypass": 0, "unknown": 0},
            )
            if cache_status not in bucket:
                cache_status = "unknown"
            bucket[cache_status] += 1
            total_steps += 1
            if cache_status == "hit":
                cache_hits += 1
            elif cache_status == "miss":
                cache_misses += 1
            elif cache_status == "partial":
                cache_partials += 1

    hit_rate = (cache_hits / total_steps) if total_steps else 0.0
    return CacheHitRate(
        total_steps=total_steps,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_partials=cache_partials,
        hit_rate=hit_rate,
        by_step=by_step,
    )


def _step_trend(
    step_name: str,
    durations: list[float],
    *,
    regression_threshold: float,
) -> StepTrend:
    n_observations = len(durations)
    if n_observations == 0:
        return StepTrend(
            step_name=step_name,
            n_observations=0,
            mean_duration=None,
            std_duration=None,
            min_duration=None,
            max_duration=None,
            last_duration=None,
            regression_detected=False,
            regression_zscore=None,
            regression_threshold=regression_threshold,
        )

    mean_duration = float(statistics.fmean(durations))
    std_duration = float(statistics.pstdev(durations)) if n_observations > 1 else 0.0
    min_duration = float(min(durations))
    max_duration = float(max(durations))
    last_duration = float(durations[-1])

    regression_detected = False
    regression_zscore: float | None = None
    if n_observations >= 3:
        baseline = durations[:-1]
        baseline_mean = float(statistics.fmean(baseline))
        baseline_std = float(statistics.pstdev(baseline)) if len(baseline) > 1 else 0.0
        if baseline_std > 0.0:
            regression_zscore = (last_duration - baseline_mean) / baseline_std
            regression_detected = bool(regression_zscore >= regression_threshold)

    return StepTrend(
        step_name=step_name,
        n_observations=n_observations,
        mean_duration=mean_duration,
        std_duration=std_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        last_duration=last_duration,
        regression_detected=regression_detected,
        regression_zscore=regression_zscore,
        regression_threshold=regression_threshold,
    )


def analyze_workflow_trends(
    history_path: Path,
    *,
    mode: str | None = None,
    regression_threshold: float = 2.5,
) -> TrendReport:
    """Analyze workflow profile history and detect runtime regressions."""
    rows = _load_history_rows(history_path)
    if mode is not None:
        rows = [row for row in rows if str(row.get("mode", row.get("budget_mode", ""))) == mode]

    step_names: set[str] = set()
    for row in rows:
        steps = row.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_name = str(step.get("step_name", "")).strip()
            if step_name:
                step_names.add(step_name)

    step_summaries = [
        _step_trend(
            step_name,
            _extract_step_durations(rows, step_name),
            regression_threshold=regression_threshold,
        )
        for step_name in sorted(step_names)
    ]
    cache_summary = _summarize_cache_hits(rows)
    overall_regression = any(step.regression_detected for step in step_summaries)
    metadata: dict[str, Any] = {
        "history_path": str(history_path),
        "latest_generated_at": str(rows[-1].get("generated_at", "")) if rows else None,
    }
    return TrendReport(
        mode=mode,
        n_runs=len(rows),
        steps=step_summaries,
        cache_summary=cache_summary,
        overall_regression=overall_regression,
        metadata=metadata,
    )
