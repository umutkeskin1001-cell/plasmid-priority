"""Freeze snapshot, invariant checking, and scientific equivalence helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from plasmid_priority.shared.provenance import dataframe_content_hash
from plasmid_priority.utils.files import ensure_directory, file_sha256


@dataclass(frozen=True)
class InvariantResult:
    name: str
    status: str
    value: float | None
    threshold: float | None
    message: str
    rollback: bool


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _top100(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "backbone_id" not in frame.columns:
        return pd.DataFrame(columns=["backbone_id", "freeze_rank"])
    working = frame.copy()
    if "freeze_rank" in working.columns:
        working["freeze_rank"] = pd.to_numeric(working["freeze_rank"], errors="coerce")
        working = working.sort_values("freeze_rank", kind="mergesort")
    columns = [column for column in ["backbone_id", "freeze_rank"] if column in working.columns]
    return working.loc[:, columns].head(100)


def _headline_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    metric_cols = [
        "summary_label",
        "model_name",
        "roc_auc",
        "average_precision",
        "brier_score",
        "ece",
        "scientific_acceptance_status",
        "selection_adjusted_empirical_p_roc_auc",
        "permutation_p_roc_auc",
    ]
    present = [c for c in metric_cols if c in frame.columns]
    rows = frame.loc[:, present].fillna("NA")
    output: dict[str, Any] = {}
    for _, row in rows.iterrows():
        label = str(row.get("summary_label", row.get("model_name", "unknown")))
        output[label] = {k: row[k] for k in present if k != "summary_label"}
    return output


def _artifact_rows_and_schema(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        frame = _read_table(path)
        result[str(path)] = {
            "exists": path.exists(),
            "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            "rows": int(len(frame)) if not frame.empty else 0,
            "columns": list(frame.columns.astype(str)) if not frame.empty else [],
            "content_hash": dataframe_content_hash(frame, sort_by=list(frame.columns[:1]))
            if not frame.empty
            else dataframe_content_hash(pd.DataFrame()),
        }
    return result


def _run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - started
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "duration_seconds": duration,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        "status": "pass" if completed.returncode == 0 else "fail",
    }


def build_freeze_snapshot(
    *,
    project_root: Path,
    run_quality_checks: bool,
) -> dict[str, Any]:
    headline_path = project_root / "reports" / "core_tables" / "headline_validation_summary.tsv"
    candidates_path = project_root / "reports" / "core_tables" / "candidate_universe.tsv"
    model_metrics_path = project_root / "reports" / "core_tables" / "model_metrics.tsv"
    scorecard_path = project_root / "reports" / "core_tables" / "model_selection_scorecard.tsv"
    protocol_doc_path = project_root / "docs" / "benchmark_contract.md"

    headline = _read_table(headline_path)
    candidates = _read_table(candidates_path)
    top_100 = _top100(candidates)

    artifacts = _artifact_rows_and_schema(
        [headline_path, candidates_path, model_metrics_path, scorecard_path, protocol_doc_path]
    )

    quality: dict[str, Any] = {}
    if run_quality_checks:
        python_executable = sys.executable or "python3"
        lint_command = [python_executable, "-m", "ruff", "check", "src/", "scripts/", "tests/"]
        typecheck_command = [python_executable, "-m", "mypy", "src/plasmid_priority/"]
        test_command = [python_executable, "-m", "pytest", "tests/", "-q", "--tb=short"]
        security_command = [python_executable, "-m", "pip_audit", "--desc"]
        quality = {
            "lint": _run_command(lint_command, project_root),
            "typecheck": _run_command(typecheck_command, project_root),
            "test": _run_command(test_command, project_root),
            "security": _run_command(security_command, project_root),
        }

    return {
        "headline_metrics": _headline_metrics(headline),
        "top_100_candidates": top_100.to_dict(orient="records"),
        "artifacts": artifacts,
        "runtime_and_ram": {
            "note": (
                "Runtime/RAM tracking is captured per gate command duration; "
                "RSS capture will be extended in workflow summaries."
            ),
            "quality_commands": {
                name: payload.get("duration_seconds") for name, payload in quality.items()
            },
        },
        "quality_outputs": quality,
    }


def compare_invariants(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
) -> list[InvariantResult]:
    invariants = contract.get("invariants", {}) if isinstance(contract, dict) else {}
    rollback = contract.get("rollback_conditions", {}) if isinstance(contract, dict) else {}

    baseline_top = {str(item.get("backbone_id")) for item in baseline.get("top_100_candidates", [])}
    candidate_top = {
        str(item.get("backbone_id")) for item in candidate.get("top_100_candidates", [])
    }
    union = baseline_top | candidate_top
    overlap_ratio = (len(baseline_top & candidate_top) / len(union)) if union else 1.0
    ranking_min_overlap = float(
        invariants.get("ranking_drift", {}).get("min_top100_overlap_ratio", 1.0)
    )
    ranking_rollback = bool(rollback.get("ranking_drift", {}).get("rollback_on_fail", True))

    results = [
        InvariantResult(
            name="ranking_drift",
            status="pass" if overlap_ratio >= ranking_min_overlap else "fail",
            value=overlap_ratio,
            threshold=ranking_min_overlap,
            message=f"Top-100 overlap ratio={overlap_ratio:.4f}",
            rollback=overlap_ratio < ranking_min_overlap and ranking_rollback,
        )
    ]

    metric_tol = float(invariants.get("metric_drift", {}).get("max_abs_delta", 0.0))
    metric_deltas: list[float] = []
    baseline_metrics = baseline.get("headline_metrics", {})
    candidate_metrics = candidate.get("headline_metrics", {})
    for key in sorted(set(baseline_metrics) & set(candidate_metrics)):
        b = baseline_metrics.get(key, {})
        c = candidate_metrics.get(key, {})
        for metric_name in ("roc_auc", "average_precision", "brier_score", "ece"):
            try:
                metric_deltas.append(abs(float(c.get(metric_name)) - float(b.get(metric_name))))
            except (TypeError, ValueError):
                continue
    max_metric_delta = max(metric_deltas) if metric_deltas else 0.0
    metric_rollback = bool(rollback.get("metric_drift", {}).get("rollback_on_fail", True))
    results.append(
        InvariantResult(
            name="metric_drift",
            status="pass" if max_metric_delta <= metric_tol else "fail",
            value=max_metric_delta,
            threshold=metric_tol,
            message=f"Max headline metric |delta|={max_metric_delta:.6f}",
            rollback=max_metric_delta > metric_tol and metric_rollback,
        )
    )

    max_row_delta = int(invariants.get("row_count_drift", {}).get("max_row_delta", 0))
    worst_row_delta = 0
    for path, baseline_artifact in baseline.get("artifacts", {}).items():
        candidate_artifact = candidate.get("artifacts", {}).get(path, {})
        try:
            candidate_rows = int(candidate_artifact.get("rows", 0))
            baseline_rows = int(baseline_artifact.get("rows", 0))
            delta = abs(candidate_rows - baseline_rows)
        except (TypeError, ValueError):
            delta = max_row_delta + 1
        worst_row_delta = max(worst_row_delta, delta)
    row_rollback = bool(rollback.get("row_count_drift", {}).get("rollback_on_fail", True))
    results.append(
        InvariantResult(
            name="row_count_drift",
            status="pass" if worst_row_delta <= max_row_delta else "fail",
            value=float(worst_row_delta),
            threshold=float(max_row_delta),
            message=f"Max row-count delta={worst_row_delta}",
            rollback=worst_row_delta > max_row_delta and row_rollback,
        )
    )

    schema_allow_additive = bool(
        invariants.get("schema_drift", {}).get("allow_additive_columns", False)
    )
    schema_failures = 0
    for path, baseline_artifact in baseline.get("artifacts", {}).items():
        base_cols = set(baseline_artifact.get("columns", []))
        cand_cols = set(candidate.get("artifacts", {}).get(path, {}).get("columns", []))
        if not schema_allow_additive and base_cols != cand_cols:
            schema_failures += 1
        elif schema_allow_additive and not base_cols.issubset(cand_cols):
            schema_failures += 1
    schema_rollback = bool(rollback.get("schema_drift", {}).get("rollback_on_fail", True))
    results.append(
        InvariantResult(
            name="schema_drift",
            status="pass" if schema_failures == 0 else "fail",
            value=float(schema_failures),
            threshold=0.0,
            message=f"Schema drifted in {schema_failures} artifact(s)",
            rollback=schema_failures > 0 and schema_rollback,
        )
    )

    return results


def scientific_equivalence(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    equivalence = contract.get("equivalence", {}) if isinstance(contract, dict) else {}
    metric_tol = float(equivalence.get("metric_abs_tolerance", 1e-12))
    ranking_min_overlap = float(equivalence.get("ranking_min_overlap_ratio", 1.0))

    invariant_results = compare_invariants(
        baseline=baseline,
        candidate=candidate,
        contract={
            "invariants": {
                "ranking_drift": {"min_top100_overlap_ratio": ranking_min_overlap},
                "metric_drift": {"max_abs_delta": metric_tol},
                "row_count_drift": {"max_row_delta": 0},
                "schema_drift": {"allow_additive_columns": False},
            },
            "rollback_conditions": {
                "ranking_drift": {"rollback_on_fail": True},
                "metric_drift": {"rollback_on_fail": True},
                "row_count_drift": {"rollback_on_fail": True},
                "schema_drift": {"rollback_on_fail": True},
            },
        },
    )

    provenance_equal = baseline.get("artifacts", {}) == candidate.get("artifacts", {})
    all_pass = all(result.status == "pass" for result in invariant_results)
    status = "pass" if all_pass and provenance_equal else "fail"
    numeric_diff = [
        result.__dict__
        for result in invariant_results
        if result.name in {"metric_drift", "row_count_drift", "schema_drift"}
    ]
    return {
        "status": status,
        "metric_equality": next(r.status for r in invariant_results if r.name == "metric_drift"),
        "report_numeric_diff": numeric_diff,
        "ranking_stability": next(r.status for r in invariant_results if r.name == "ranking_drift"),
        "provenance_equality": "pass" if provenance_equal else "fail",
        "rollback_required": any(r.rollback for r in invariant_results) or not provenance_equal,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_freeze_contract(path: Path) -> dict[str, Any]:
    return _load_contract(path)
