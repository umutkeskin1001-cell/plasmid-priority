#!/usr/bin/env python3
"""Summarize workflow telemetry and enforce runtime budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from plasmid_priority.config import DATA_ROOT_ENV_VAR, build_context
from plasmid_priority.performance.profile_schema import StepTelemetry
from plasmid_priority.utils.files import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_FILE_NAME = "workflow_profile_history.jsonl"


def _workflow_data_root() -> Path:
    context = build_context(PROJECT_ROOT)
    data_root = context.data_root if context.data_root is not None else context.data_dir
    return Path(data_root).resolve()


def _load_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _to_telemetry(summary: dict[str, Any]) -> StepTelemetry | None:
    telemetry = summary.get("telemetry")
    if isinstance(telemetry, dict):
        try:
            return StepTelemetry(
                step_name=str(telemetry.get("step_name", summary.get("script_name", ""))),
                status=str(telemetry.get("status", summary.get("status", "unknown"))),
                cache_status=str(telemetry.get("cache_status", "miss")),  # type: ignore
                duration_seconds=float(telemetry.get("duration_seconds", 0.0)),
                rows_in=int(telemetry["rows_in"]) if telemetry.get("rows_in") is not None else None,
                rows_out=int(telemetry["rows_out"])
                if telemetry.get("rows_out") is not None
                else None,
                bytes_read=(
                    int(telemetry["bytes_read"])
                    if telemetry.get("bytes_read") is not None
                    else None
                ),
                bytes_written=(
                    int(telemetry["bytes_written"])
                    if telemetry.get("bytes_written") is not None
                    else None
                ),
                peak_rss_mb=(
                    float(telemetry["peak_rss_mb"])
                    if telemetry.get("peak_rss_mb") is not None
                    else None
                ),
                cpu_time_seconds=(
                    float(telemetry["cpu_time_seconds"])
                    if telemetry.get("cpu_time_seconds") is not None
                    else None
                ),
                io_wait_hint=(
                    float(telemetry["io_wait_hint"])
                    if telemetry.get("io_wait_hint") is not None
                    else None
                ),
                input_hash=(
                    str(telemetry["input_hash"])
                    if telemetry.get("input_hash") is not None
                    else None
                ),
                output_hash=(
                    str(telemetry["output_hash"])
                    if telemetry.get("output_hash") is not None
                    else None
                ),
                metadata=dict(telemetry.get("metadata", {})),
            )
        except (TypeError, ValueError):
            return None
    if "duration_seconds" not in summary:
        return None
    step_name = str(summary.get("script_name", "unknown"))
    return StepTelemetry(
        step_name=step_name,
        status=str(summary.get("status", "unknown")),
        cache_status=str(summary.get("cache_status", "miss")),  # type: ignore
        duration_seconds=float(summary.get("duration_seconds", 0.0)),
        rows_in=summary.get("rows_in"),
        rows_out=summary.get("rows_out"),
        bytes_read=summary.get("bytes_read"),
        bytes_written=summary.get("bytes_written"),
        peak_rss_mb=summary.get("peak_rss_mb"),
        cpu_time_seconds=summary.get("cpu_time_seconds"),
        io_wait_hint=summary.get("io_wait_hint"),
        input_hash=summary.get("input_hash"),
        output_hash=summary.get("output_hash"),
    )


def _load_budgets(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        content = path.read_text(encoding="utf-8")
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


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _latest_previous_for_mode(
    history: list[dict[str, Any]],
    *,
    mode: str | None,
    current_timestamp: str,
) -> dict[str, Any] | None:
    filtered = []
    for row in history:
        if mode and str(row.get("budget_mode", "")) != str(mode):
            continue
        if str(row.get("generated_at", "")) >= current_timestamp:
            continue
        filtered.append(row)
    if not filtered:
        return None
    filtered.sort(key=lambda item: str(item.get("generated_at", "")))
    return filtered[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Override data root (default from ${DATA_ROOT_ENV_VAR} or project context).",
    )
    parser.add_argument(
        "--budgets",
        type=Path,
        default=PROJECT_ROOT / "config" / "performance_budgets.yaml",
        help="Path to budget configuration YAML.",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        default=None,
        help="Budget mode key (e.g., dev-refresh, release-full).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of heavy/slow steps to surface.",
    )
    parser.add_argument(
        "--strict-budget",
        action="store_true",
        help="Return non-zero when budget is exceeded.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve() if args.data_root else _workflow_data_root()
    logs_dir = data_root / "tmp" / "logs"
    summary_paths = sorted(logs_dir.glob("*_summary.json"))
    step_rows: list[StepTelemetry] = []
    for summary_path in summary_paths:
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        telemetry = _to_telemetry(summary)
        if telemetry is not None:
            step_rows.append(telemetry)

    if not step_rows:
        print("No telemetry found in workflow summaries.")
        return 1

    slowest = sorted(step_rows, key=lambda row: row.duration_seconds, reverse=True)[: args.top_n]
    heaviest = sorted(
        step_rows,
        key=lambda row: row.bytes_written or 0,
        reverse=True,
    )[: args.top_n]
    total_runtime = sum(row.duration_seconds for row in step_rows)

    print(f"Steps profiled: {len(step_rows)}")
    print(f"Total runtime (sum of step durations): {total_runtime:.2f}s")
    print("\nTop slow steps:")
    for row in slowest:
        print(f"  - {row.step_name}: {row.duration_seconds:.2f}s ({row.cache_status})")
    print("\nTop heavy steps:")
    for row in heaviest:
        bytes_written = row.bytes_written or 0
        print(f"  - {row.step_name}: {bytes_written / (1024 * 1024):.2f} MiB written")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps_profiled": len(step_rows),
        "total_duration_seconds": total_runtime,
        "top_slowest_steps": [row.to_dict() for row in slowest],
        "top_heaviest_steps": [row.to_dict() for row in heaviest],
        "budget_mode": args.budget_mode,
        "git_commit": _git_commit(PROJECT_ROOT),
    }
    report_path = logs_dir / "workflow_profile_report.json"
    atomic_write_json(report_path, report_payload)
    print(f"\nWrote profile report: {report_path}")

    history_path = logs_dir / _HISTORY_FILE_NAME
    history = _load_history(history_path)
    previous = _latest_previous_for_mode(
        history,
        mode=args.budget_mode,
        current_timestamp=str(report_payload.get("generated_at", "")),
    )
    _append_history(history_path, report_payload)
    if previous is not None:
        prev_runtime = float(previous.get("total_duration_seconds", 0.0))
        delta_seconds = total_runtime - prev_runtime
        delta_pct = (delta_seconds / prev_runtime * 100.0) if prev_runtime > 0 else 0.0
        prev_commit = str(previous.get("git_commit", "unknown"))
        print(
            "Runtime delta vs previous profile"
            f" (mode={args.budget_mode or 'unspecified'}, prev_commit={prev_commit[:12]}): "
            f"{delta_seconds:+.2f}s ({delta_pct:+.2f}%)",
        )
    print(f"Updated profile history: {history_path}")

    budgets = _load_budgets(args.budgets)
    budget_mode = args.budget_mode
    if budget_mode:
        mode_cfg = dict(budgets.get("modes", {})).get(budget_mode, {})
        budget_seconds = mode_cfg.get("budget_seconds")
        if isinstance(budget_seconds, (int, float)):
            tolerance = dict(budgets.get("enforcement", {})).get(
                "default_exceedance_tolerance", 0.1
            )
            threshold = float(budget_seconds) * (1.0 + float(tolerance))
            exceeds = total_runtime > threshold
            print(
                f"Budget check [{budget_mode}]: runtime={total_runtime:.2f}s, "
                f"budget={float(budget_seconds):.2f}s, threshold={threshold:.2f}s",
            )
            if exceeds and args.strict_budget:
                print("Budget exceeded.")
                return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
