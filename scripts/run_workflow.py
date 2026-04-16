#!/usr/bin/env python3
"""Dependency-aware local workflow runner for sequential and light parallel refreshes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from plasmid_priority.config import DATA_ROOT_ENV_VAR
from plasmid_priority.utils.files import (
    atomic_write_json,
    path_signature_with_hash,
    project_python_source_paths,
)
from plasmid_priority.validation import validate_script_boundary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    script: str
    deps: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


STEP_LIBRARY: dict[str, WorkflowStep] = {
    "00_fetch_external_data": WorkflowStep("00_fetch_external_data", "00_fetch_external_data.py"),
    "01_check_inputs": WorkflowStep("01_check_inputs", "01_check_inputs.py"),
    "02_build_all_plasmids_fasta": WorkflowStep(
        "02_build_all_plasmids_fasta",
        "02_build_all_plasmids_fasta.py",
        deps=("01_check_inputs",),
        args=("--overwrite",),
    ),
    "03_build_bronze_table": WorkflowStep(
        "03_build_bronze_table",
        "03_build_bronze_table.py",
        deps=("02_build_all_plasmids_fasta",),
        args=("--overwrite",),
    ),
    "04_harmonize_metadata": WorkflowStep(
        "04_harmonize_metadata",
        "04_harmonize_metadata.py",
        deps=("03_build_bronze_table",),
    ),
    "05_deduplicate": WorkflowStep(
        "05_deduplicate", "05_deduplicate.py", deps=("04_harmonize_metadata",)
    ),
    "06_annotate_mobility": WorkflowStep(
        "06_annotate_mobility", "06_annotate_mobility.py", deps=("05_deduplicate",)
    ),
    "07_annotate_amr": WorkflowStep(
        "07_annotate_amr", "07_annotate_amr.py", deps=("06_annotate_mobility",)
    ),
    "08_build_amr_consensus": WorkflowStep(
        "08_build_amr_consensus", "08_build_amr_consensus.py", deps=("07_annotate_amr",)
    ),
    "09_assign_backbones": WorkflowStep(
        "09_assign_backbones", "09_assign_backbones.py", deps=("08_build_amr_consensus",)
    ),
    "10_compute_coherence": WorkflowStep(
        "10_compute_coherence", "10_compute_coherence.py", deps=("09_assign_backbones",)
    ),
    "11_compute_feature_T": WorkflowStep(
        "11_compute_feature_T", "11_compute_feature_T.py", deps=("10_compute_coherence",)
    ),
    "12_compute_feature_H": WorkflowStep(
        "12_compute_feature_H", "12_compute_feature_H.py", deps=("11_compute_feature_T",)
    ),
    "13_compute_feature_A": WorkflowStep(
        "13_compute_feature_A", "13_compute_feature_A.py", deps=("12_compute_feature_H",)
    ),
    "14_build_backbone_table": WorkflowStep(
        "14_build_backbone_table", "14_build_backbone_table.py", deps=("13_compute_feature_A",)
    ),
    "15_normalize_and_score": WorkflowStep(
        "15_normalize_and_score", "15_normalize_and_score.py", deps=("14_build_backbone_table",)
    ),
    "16_run_module_A": WorkflowStep(
        "16_run_module_A",
        "16_run_module_A.py",
        deps=("15_normalize_and_score",),
    ),
    "17_run_module_B": WorkflowStep(
        "17_run_module_B", "17_run_module_B.py", deps=("15_normalize_and_score",)
    ),
    "18_run_module_C_pathogen_detection": WorkflowStep(
        "18_run_module_C_pathogen_detection",
        "18_run_module_C_pathogen_detection.py",
        deps=("16_run_module_A",),
    ),
    "19_run_module_D_external_support": WorkflowStep(
        "19_run_module_D_external_support",
        "19_run_module_D_external_support.py",
        deps=("16_run_module_A",),
    ),
    "20_run_module_E_amrfinder_concordance": WorkflowStep(
        "20_run_module_E_amrfinder_concordance",
        "20_run_module_E_amrfinder_concordance.py",
        deps=("16_run_module_A",),
    ),
    "21_run_validation": WorkflowStep(
        "21_run_validation", "21_run_validation.py", deps=("16_run_module_A",)
    ),
    "22_run_sensitivity": WorkflowStep(
        "22_run_sensitivity",
        "22_run_sensitivity.py",
        deps=("15_normalize_and_score",),
    ),
    "23_run_module_f_enrichment": WorkflowStep(
        "23_run_module_f_enrichment",
        "23_run_module_f_enrichment.py",
        deps=("15_normalize_and_score",),
    ),
    "27_run_advanced_audits": WorkflowStep(
        "27_run_advanced_audits", "27_run_advanced_audits.py", deps=("21_run_validation",)
    ),
    "24_build_reports": WorkflowStep(
        "24_build_reports",
        "24_build_reports.py",
        deps=(
            "17_run_module_B",
            "18_run_module_C_pathogen_detection",
            "19_run_module_D_external_support",
            "20_run_module_E_amrfinder_concordance",
            "21_run_validation",
            "22_run_sensitivity",
            "23_run_module_f_enrichment",
            "27_run_advanced_audits",
        ),
    ),
    "25_export_tubitak_summary": WorkflowStep(
        "25_export_tubitak_summary", "25_export_tubitak_summary.py", deps=("24_build_reports",)
    ),
    "28_build_release_bundle": WorkflowStep(
        "28_build_release_bundle",
        "28_build_release_bundle.py",
        deps=("24_build_reports", "25_export_tubitak_summary"),
    ),
    "29_build_experiment_registry": WorkflowStep(
        "29_build_experiment_registry", "29_build_experiment_registry.py"
    ),
}

PIPELINE_STEP_NAMES: tuple[str, ...] = (
    "01_check_inputs",
    "02_build_all_plasmids_fasta",
    "03_build_bronze_table",
    "04_harmonize_metadata",
    "05_deduplicate",
    "06_annotate_mobility",
    "07_annotate_amr",
    "08_build_amr_consensus",
    "09_assign_backbones",
    "10_compute_coherence",
    "11_compute_feature_T",
    "12_compute_feature_H",
    "13_compute_feature_A",
    "14_build_backbone_table",
    "15_normalize_and_score",
    "16_run_module_A",
    "17_run_module_B",
    "18_run_module_C_pathogen_detection",
    "19_run_module_D_external_support",
    "20_run_module_E_amrfinder_concordance",
    "21_run_validation",
    "22_run_sensitivity",
    "23_run_module_f_enrichment",
    "27_run_advanced_audits",
    "24_build_reports",
    "25_export_tubitak_summary",
)

ANALYSIS_REFRESH_STEP_NAMES: tuple[str, ...] = (
    "15_normalize_and_score",
    "16_run_module_A",
    "17_run_module_B",
    "18_run_module_C_pathogen_detection",
    "19_run_module_D_external_support",
    "20_run_module_E_amrfinder_concordance",
    "21_run_validation",
    "22_run_sensitivity",
    "23_run_module_f_enrichment",
    "27_run_advanced_audits",
    "24_build_reports",
    "25_export_tubitak_summary",
)

CORE_REFRESH_STEP_NAMES: tuple[str, ...] = (
    "15_normalize_and_score",
    "16_run_module_A",
    "21_run_validation",
    "22_run_sensitivity",
    "27_run_advanced_audits",
    "24_build_reports",
    "25_export_tubitak_summary",
)

SUPPORT_REFRESH_STEP_NAMES: tuple[str, ...] = (
    "15_normalize_and_score",
    "16_run_module_A",
    "17_run_module_B",
    "18_run_module_C_pathogen_detection",
    "19_run_module_D_external_support",
    "20_run_module_E_amrfinder_concordance",
    "23_run_module_f_enrichment",
    "24_build_reports",
    "25_export_tubitak_summary",
)

RELEASE_STEP_NAMES: tuple[str, ...] = (
    "24_build_reports",
    "25_export_tubitak_summary",
    "28_build_release_bundle",
    "29_build_experiment_registry",
)

REPORTS_ONLY_STEP_NAMES: tuple[str, ...] = (
    "24_build_reports",
    "25_export_tubitak_summary",
)

FETCH_EXTERNAL_STEP_NAMES: tuple[str, ...] = ("00_fetch_external_data",)

SEQUENTIAL_WORKFLOW_MODES = {"pipeline-sequential", "analysis-refresh-sequential"}


MODE_STEP_NAMES: dict[str, tuple[str, ...]] = {
    "pipeline": PIPELINE_STEP_NAMES,
    "pipeline-sequential": PIPELINE_STEP_NAMES,
    "analysis-refresh": ANALYSIS_REFRESH_STEP_NAMES,
    "analysis-refresh-sequential": ANALYSIS_REFRESH_STEP_NAMES,
    "core-refresh": CORE_REFRESH_STEP_NAMES,
    "support-refresh": SUPPORT_REFRESH_STEP_NAMES,
    "reports-only": REPORTS_ONLY_STEP_NAMES,
    "release": RELEASE_STEP_NAMES,
    "fetch-external": FETCH_EXTERNAL_STEP_NAMES,
}


MODE_DEP_OVERRIDES: dict[str, dict[str, tuple[str, ...]]] = {
    "analysis-refresh": {
        "24_build_reports": (
            "17_run_module_B",
            "18_run_module_C_pathogen_detection",
            "19_run_module_D_external_support",
            "20_run_module_E_amrfinder_concordance",
            "21_run_validation",
            "22_run_sensitivity",
            "23_run_module_f_enrichment",
            "27_run_advanced_audits",
        ),
    },
    "core-refresh": {
        "24_build_reports": (
            "21_run_validation",
            "22_run_sensitivity",
            "27_run_advanced_audits",
        ),
    },
    "support-refresh": {
        "24_build_reports": (
            "17_run_module_B",
            "18_run_module_C_pathogen_detection",
            "19_run_module_D_external_support",
            "20_run_module_E_amrfinder_concordance",
            "23_run_module_f_enrichment",
        ),
    },
    "release": {
        "29_build_experiment_registry": ("28_build_release_bundle",),
    },
}


def _topologically_sorted(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    order = {step.name: index for index, step in enumerate(steps)}
    steps_by_name = {step.name: step for step in steps}
    indegree = {step.name: 0 for step in steps}
    dependents: dict[str, list[str]] = {step.name: [] for step in steps}

    for step in steps:
        for dep in step.deps:
            if dep not in steps_by_name:
                continue
            indegree[step.name] += 1
            dependents[dep].append(step.name)

    ready = sorted(
        [steps_by_name[name] for name, degree in indegree.items() if degree == 0],
        key=lambda step: order[step.name],
    )
    sorted_steps: list[WorkflowStep] = []

    while ready:
        step = ready.pop(0)
        sorted_steps.append(step)
        for dependent_name in dependents[step.name]:
            indegree[dependent_name] -= 1
            if indegree[dependent_name] == 0:
                ready.append(steps_by_name[dependent_name])
        ready.sort(key=lambda candidate: order[candidate.name])

    if len(sorted_steps) != len(steps):
        raise RuntimeError("Workflow graph contains a dependency cycle.")
    return sorted_steps


def _workflow_steps(mode: str) -> list[WorkflowStep]:
    try:
        selected_names = MODE_STEP_NAMES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow mode: {mode}") from exc

    overrides = MODE_DEP_OVERRIDES.get(mode, {})
    steps: list[WorkflowStep] = []
    for name in selected_names:
        base = STEP_LIBRARY[name]
        deps = tuple(dep for dep in overrides.get(name, base.deps) if dep in selected_names)
        steps.append(replace(base, deps=deps))
    return _topologically_sorted(steps)


def _workflow_data_root() -> Path:
    raw_value = os.environ.get(DATA_ROOT_ENV_VAR)
    if raw_value in (None, ""):
        return (PROJECT_ROOT / "data").resolve()
    candidate = Path(str(raw_value)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _workflow_checkpoint_path(mode: str) -> Path:
    return _workflow_data_root() / "tmp" / "workflow" / f"{mode}.json"


def _script_path(step: WorkflowStep) -> Path:
    return PROJECT_ROOT / "scripts" / step.script


def _source_signatures_for_step(step: WorkflowStep) -> list[dict[str, object]]:
    source_paths = project_python_source_paths(PROJECT_ROOT, script_path=_script_path(step))
    return [path_signature_with_hash(path, include_file_hash=True) for path in source_paths]


def _load_workflow_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_step_summary_path(step: WorkflowStep) -> Path:
    return _workflow_data_root() / "tmp" / "logs" / f"{step.name}_summary.json"


def _input_manifest_matches(summary: dict[str, object]) -> bool:
    manifest = summary.get("input_manifest")
    if not isinstance(manifest, dict) or not manifest:
        return True
    for entry in manifest.values():
        if not isinstance(entry, dict):
            return False
        path_value = entry.get("path")
        if not path_value:
            return False
        path = Path(str(path_value))
        if not path.exists():
            return False
        try:
            current = path_signature_with_hash(path, include_file_hash=True)
        except (OSError, ValueError):
            return False
        for key in ("path", "size", "mtime_ns", "kind", "sha256", "digest", "entry_count"):
            if entry.get(key) != current.get(key):
                return False
    return True


def _step_checkpoint_is_valid(step: WorkflowStep, checkpoint: dict[str, object]) -> bool:
    step_state = checkpoint.get("steps", {})
    if not isinstance(step_state, dict):
        return False
    entry = step_state.get(step.name)
    if not isinstance(entry, dict):
        return False
    if str(entry.get("status", "")).lower() != "ok":
        return False
    summary_path = Path(str(entry.get("summary_path", _current_step_summary_path(step))))
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict) or str(summary.get("status", "")).lower() != "ok":
        return False
    if not _input_manifest_matches(summary):
        return False
    output_files = summary.get("output_files_written", [])
    if not isinstance(output_files, list):
        return False
    for output_value in output_files:
        output_path = Path(str(output_value))
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        if not output_path.exists():
            return False
    source_signatures = entry.get("source_signatures", [])
    current_source_signatures = _source_signatures_for_step(step)
    return source_signatures == current_source_signatures


def _write_workflow_checkpoint(
    path: Path,
    *,
    mode: str,
    completed_steps: list[str],
    step: WorkflowStep | None = None,
    step_status: str | None = None,
    return_code: int | None = None,
) -> None:
    checkpoint = _load_workflow_checkpoint(path)
    steps = checkpoint.get("steps", {})
    if not isinstance(steps, dict):
        steps = {}
    if step is not None and step_status is not None:
        summary_path = _current_step_summary_path(step)
        step_payload: dict[str, object] = {
            "status": step_status,
            "summary_path": str(summary_path),
            "script_path": str(_script_path(step)),
            "source_signatures": _source_signatures_for_step(step),
        }
        if return_code is not None:
            step_payload["return_code"] = int(return_code)
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = {}
            if isinstance(summary, dict):
                step_payload["summary_status"] = summary.get("status")
                step_payload["output_files_written"] = summary.get("output_files_written", [])
                step_payload["input_manifest"] = summary.get("input_manifest", {})
        steps[step.name] = step_payload
    checkpoint.update(
        {
            "mode": mode,
            "completed_steps": completed_steps,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps": steps,
        }
    )
    atomic_write_json(path, checkpoint)


def _auto_job_cap(max_workers: int) -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(8, cpu_total // max(max_workers, 1)))


def _run_step(step: WorkflowStep, *, auto_job_cap: int | None = None) -> int:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / step.script), *step.args]
    print(f"[workflow] {step.name}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    if step.env:
        env.update(dict(step.env))
    if auto_job_cap is not None and "PLASMID_PRIORITY_MAX_JOBS" not in env:
        env["PLASMID_PRIORITY_MAX_JOBS"] = str(int(auto_job_cap))
    timeout_seconds = max(
        1,
        int(os.environ.get("PLASMID_PRIORITY_STEP_TIMEOUT_SECONDS", "86400")),
    )
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(f"[workflow] {step.name} timed out after {timeout_seconds}s", file=sys.stderr)
        return 124
    return int(completed.returncode)


def run_workflow(
    mode: str,
    *,
    max_workers: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
) -> int:
    steps = _workflow_steps(mode)
    if dry_run:
        for step in steps:
            dep_text = f" deps={','.join(step.deps)}" if step.deps else ""
            arg_text = " " + " ".join(step.args) if step.args else ""
            print(f"{step.name}{dep_text}: {step.script}{arg_text}")
        return 0

    checkpoint_path = _workflow_checkpoint_path(mode)
    checkpoint = _load_workflow_checkpoint(checkpoint_path) if resume else {}
    completed: set[str] = set()
    if resume and checkpoint:
        for step in steps:
            if _step_checkpoint_is_valid(step, checkpoint):
                completed.add(step.name)

    if mode in SEQUENTIAL_WORKFLOW_MODES:
        max_workers = 1
    elif max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    max_workers = max(1, min(int(max_workers), len(steps)))
    auto_job_cap = _auto_job_cap(max_workers)
    order = {step.name: index for index, step in enumerate(steps)}
    pending = {step.name: step for step in steps if step.name not in completed}
    running: dict[Future[int], WorkflowStep] = {}
    checkpoint_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            running_names = {running_step.name for running_step in running.values()}
            ready = [
                step
                for step in pending.values()
                if step.name not in running_names and all(dep in completed for dep in step.deps)
            ]
            ready.sort(key=lambda step: order[step.name])

            while ready and len(running) < max_workers:
                step = ready.pop(0)
                future = executor.submit(_run_step, step, auto_job_cap=auto_job_cap)
                running[future] = step

            if not running:
                blocked = {
                    name: [dep for dep in step.deps if dep not in completed]
                    for name, step in pending.items()
                }
                raise RuntimeError(f"Workflow deadlock detected: {blocked}")

            done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                return_code = int(future.result())
                if return_code != 0:
                    with checkpoint_lock:
                        _write_workflow_checkpoint(
                            checkpoint_path,
                            mode=mode,
                            completed_steps=sorted(completed, key=lambda name: order[name]),
                            step=step,
                            step_status="failed",
                            return_code=return_code,
                        )
                    print(
                        f"[workflow] {step.name} failed with exit code {return_code}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return return_code
                summary_path = _current_step_summary_path(step)
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    with checkpoint_lock:
                        _write_workflow_checkpoint(
                            checkpoint_path,
                            mode=mode,
                            completed_steps=sorted(completed, key=lambda name: order[name]),
                            step=step,
                            step_status="failed",
                            return_code=return_code,
                        )
                    print(
                        f"[workflow] {step.name} did not write a readable summary: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                boundary_result = validate_script_boundary(summary, project_root=PROJECT_ROOT)
                if boundary_result.get("status") != "pass":
                    with checkpoint_lock:
                        _write_workflow_checkpoint(
                            checkpoint_path,
                            mode=mode,
                            completed_steps=sorted(completed, key=lambda name: order[name]),
                            step=step,
                            step_status="failed",
                            return_code=return_code,
                        )
                    issues = [
                        entry.get("message", str(entry))
                        for entry in boundary_result.get("errors", [])
                    ]
                    print(
                        f"[workflow] {step.name} failed boundary validation: {'; '.join(issues)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                completed.add(step.name)
                pending.pop(step.name, None)
                with checkpoint_lock:
                    _write_workflow_checkpoint(
                        checkpoint_path,
                        mode=mode,
                        completed_steps=sorted(completed, key=lambda name: order[name]),
                        step=step,
                        step_status="ok",
                        return_code=return_code,
                    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_STEP_NAMES))
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Maximum number of concurrent steps."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved steps without executing scripts."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint resume and rerun every step from scratch.",
    )
    args = parser.parse_args(argv)
    return run_workflow(
        args.mode,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
