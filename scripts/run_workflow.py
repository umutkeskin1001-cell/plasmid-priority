#!/usr/bin/env python3
"""Dependency-aware local workflow runner for sequential and light parallel refreshes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import yaml

from plasmid_priority.cache import (
    ArtifactCache,
    build_step_cache_key,
    software_fingerprint,
    stable_hash,
)
from plasmid_priority.config import DATA_ROOT_ENV_VAR, build_context, context_config_paths
from plasmid_priority.modeling.plugin_system import registry
from plasmid_priority.performance import (
    ResourceSnapshot,
    StepTelemetry,
    WorkflowProfile,
    build_step_telemetry,
    capture_resource_snapshot,
    infer_cache_status_from_summary,
    upsert_summary_telemetry,
)
from plasmid_priority.pipeline.step_contract import StepResult, write_step_result
from plasmid_priority.protocol import ScientificProtocol, build_execution_hash, build_protocol_hash
from plasmid_priority.utils.files import (
    atomic_write_json,
    path_signature_with_hash,
    project_python_source_paths,
)
from plasmid_priority.utils.parallel import configure_blas_threads
from plasmid_priority.validation import validate_script_boundary
from plasmid_priority.validation.artifact_integrity import validate_release_artifact_integrity
from plasmid_priority.validation.release_readiness import evaluate_release_readiness
from plasmid_priority.validation.scientific_contract import validate_release_scientific_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CACHE_DISABLED_STEPS: frozenset[str] = frozenset({"00_fetch_external_data"})
PERFORMANCE_BUDGETS_PATH = PROJECT_ROOT / "config" / "performance_budgets.yaml"
DEFAULT_BUDGET_MODE_MAP: dict[str, str] = {
    "pipeline": "release-full",
    "pipeline-sequential": "release-full",
    "analysis-refresh": "dev-refresh",
    "analysis-refresh-sequential": "dev-refresh",
    "core-refresh": "model-refresh",
    "support-refresh": "dev-refresh",
    "reports-only": "report-refresh",
    "release": "release-full",
    "fetch-external": "smoke-local",
    "demo-pipeline": "smoke-local",
}
VALIDATION_SCREEN_SPLITS = str(os.environ.get("PLASMID_PRIORITY_VALIDATION_SCREEN_SPLITS", "1"))
VALIDATION_SCREEN_REPEATS = str(os.environ.get("PLASMID_PRIORITY_VALIDATION_SCREEN_REPEATS", "1"))
VALIDATION_FINALIST_SPLITS = str(os.environ.get("PLASMID_PRIORITY_VALIDATION_FINALIST_SPLITS", "1"))
VALIDATION_FINALIST_REPEATS = str(
    os.environ.get("PLASMID_PRIORITY_VALIDATION_FINALIST_REPEATS", "1")
)
VALIDATION_JOBS = str(os.environ.get("PLASMID_PRIORITY_VALIDATION_JOBS", "2"))
VALIDATION_SELECTION_ADJUSTED_PERMUTATIONS = str(
    os.environ.get("PLASMID_PRIORITY_VALIDATION_SELECTION_ADJUSTED_PERMUTATIONS", "10"),
)


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
        "05_deduplicate",
        "05_deduplicate.py",
        deps=("04_harmonize_metadata",),
    ),
    "06_annotate_mobility": WorkflowStep(
        "06_annotate_mobility",
        "06_annotate_mobility.py",
        deps=("05_deduplicate",),
    ),
    "07_annotate_amr": WorkflowStep(
        "07_annotate_amr",
        "07_annotate_amr.py",
        deps=("05_deduplicate",),
    ),
    "08_build_amr_consensus": WorkflowStep(
        "08_build_amr_consensus",
        "08_build_amr_consensus.py",
        deps=("07_annotate_amr",),
    ),
    "09_assign_backbones": WorkflowStep(
        "09_assign_backbones",
        "09_assign_backbones.py",
        deps=("08_build_amr_consensus",),
    ),
    "10_compute_coherence": WorkflowStep(
        "10_compute_coherence",
        "10_compute_coherence.py",
        deps=("09_assign_backbones",),
    ),
    "11_compute_feature_T": WorkflowStep(
        "11_compute_feature_T",
        "11_compute_feature_T.py",
        deps=("10_compute_coherence",),
    ),
    "12_compute_feature_H": WorkflowStep(
        "12_compute_feature_H",
        "12_compute_feature_H.py",
        deps=("10_compute_coherence",),
    ),
    "13_compute_feature_A": WorkflowStep(
        "13_compute_feature_A",
        "13_compute_feature_A.py",
        deps=("10_compute_coherence",),
    ),
    "14_build_backbone_table": WorkflowStep(
        "14_build_backbone_table",
        "14_build_backbone_table.py",
        deps=("11_compute_feature_T", "12_compute_feature_H", "13_compute_feature_A"),
    ),
    "15_normalize_and_score": WorkflowStep(
        "15_normalize_and_score",
        "15_normalize_and_score.py",
        deps=("14_build_backbone_table",),
    ),
    "16_run_module_A": WorkflowStep(
        "16_run_module_A",
        "16_run_module_A.py",
        deps=("15_normalize_and_score",),
    ),
    "17_run_module_B": WorkflowStep(
        "17_run_module_B",
        "17_run_module_B.py",
        deps=("15_normalize_and_score",),
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
        "21_run_validation",
        "21_run_validation.py",
        deps=("16_run_module_A",),
        args=(
            "--jobs",
            VALIDATION_JOBS,
            "--single-model-screen-splits",
            VALIDATION_SCREEN_SPLITS,
            "--single-model-screen-repeats",
            VALIDATION_SCREEN_REPEATS,
            "--single-model-finalist-splits",
            VALIDATION_FINALIST_SPLITS,
            "--single-model-finalist-repeats",
            VALIDATION_FINALIST_REPEATS,
            "--selection-adjusted-permutations",
            VALIDATION_SELECTION_ADJUSTED_PERMUTATIONS,
        ),
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
        "27_run_advanced_audits",
        "27b_run_advanced_audits.py",
        deps=("21_run_validation",),
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
        "25_export_tubitak_summary",
        "25_export_tubitak_summary.py",
        deps=("24_build_reports",),
    ),
    "31_generate_scientific_contracts": WorkflowStep(
        "31_generate_scientific_contracts",
        "31_generate_scientific_contracts.py",
        deps=("24_build_reports", "25_export_tubitak_summary"),
    ),
    "41_build_performance_dashboard": WorkflowStep(
        "41_build_performance_dashboard",
        "41_build_performance_dashboard.py",
        deps=("24_build_reports",),
    ),
    "52_build_official_release_artifacts": WorkflowStep(
        "52_build_official_release_artifacts",
        "build_official_release_artifacts.py",
        deps=("24_build_reports",),
    ),
    "42_release_readiness_report": WorkflowStep(
        "42_release_readiness_report",
        "42_release_readiness_report.py",
        deps=(
            "31_generate_scientific_contracts",
            "41_build_performance_dashboard",
            "52_build_official_release_artifacts",
        ),
    ),
    "28_build_release_bundle": WorkflowStep(
        "28_build_release_bundle",
        "28_build_release_bundle.py",
        deps=("42_release_readiness_report",),
    ),
    "29_build_experiment_registry": WorkflowStep(
        "29_build_experiment_registry",
        "29_build_experiment_registry.py",
    ),
}

# Phase 4.2: Resource-aware scheduling hints
STEP_RESOURCES: dict[str, dict[str, int]] = {
    "06_annotate_mobility": {"cpu": 4, "ram_gb": 4},
    "07_annotate_amr": {"cpu": 4, "ram_gb": 4},
    "16_run_module_A": {"cpu": -1, "ram_gb": 8},  # Use all available CPU
    "21_run_validation": {"cpu": -1, "ram_gb": 6},
    "22_run_sensitivity": {"cpu": -1, "ram_gb": 6},
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
    "31_generate_scientific_contracts",
    "41_build_performance_dashboard",
    "52_build_official_release_artifacts",
    "42_release_readiness_report",
    "28_build_release_bundle",
    "29_build_experiment_registry",
)

REPORTS_ONLY_STEP_NAMES: tuple[str, ...] = (
    "24_build_reports",
    "25_export_tubitak_summary",
)

FETCH_EXTERNAL_STEP_NAMES: tuple[str, ...] = ("00_fetch_external_data",)

# Demo pipeline - minimal end-to-end for demonstration (30-second target)
DEMO_PIPELINE_STEP_NAMES: tuple[str, ...] = (
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
    "24_build_reports",
)

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
    "demo-pipeline": DEMO_PIPELINE_STEP_NAMES,
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
        "42_release_readiness_report": (
            "31_generate_scientific_contracts",
            "41_build_performance_dashboard",
            "52_build_official_release_artifacts",
        ),
        "28_build_release_bundle": ("42_release_readiness_report",),
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


# Phase 2: Steps that can be skipped for fast dev iteration
VALIDATION_STEP_NAMES: frozenset[str] = frozenset({
    "21_run_validation",
    "22_run_sensitivity",
    "27_run_advanced_audits",
})
REPORT_STEP_NAMES: frozenset[str] = frozenset({
    "24_build_reports",
    "25_export_tubitak_summary",
    "31_generate_scientific_contracts",
    "41_build_performance_dashboard",
    "52_build_official_release_artifacts",
    "42_release_readiness_report",
    "28_build_release_bundle",
    "29_build_experiment_registry",
})


def _workflow_steps(
    mode: str,
    *,
    skip_validation: bool = False,
    skip_reports: bool = False,
) -> list[WorkflowStep]:
    try:
        selected_names = list(MODE_STEP_NAMES[mode])
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow mode: {mode}") from exc

    # Phase 2: Fast dev iteration — skip heavy steps
    if skip_validation:
        selected_names = [name for name in selected_names if name not in VALIDATION_STEP_NAMES]
    if skip_reports:
        selected_names = [name for name in selected_names if name not in REPORT_STEP_NAMES]

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
    return bool(source_signatures == current_source_signatures)


def _summary_mapping(summary: dict[str, object], key: str) -> dict[str, object]:
    value = summary.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _summary_list(summary: dict[str, object], key: str) -> list[object]:
    value = summary.get(key, [])
    return list(value) if isinstance(value, list) else []


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
        },
    )
    atomic_write_json(path, checkpoint)


def _auto_job_cap(max_workers: int) -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(8, cpu_total // max(max_workers, 1)))


def _load_performance_budgets(path: Path = PERFORMANCE_BUDGETS_PATH) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_budget_mode(workflow_mode: str, override: str | None = None) -> str:
    if override:
        return override
    return DEFAULT_BUDGET_MODE_MAP.get(workflow_mode, workflow_mode)


def _evaluate_runtime_budget(
    *,
    workflow_mode: str,
    total_runtime_seconds: float,
    budget_mode_override: str | None = None,
    strict_budget: bool = False,
    budgets: dict[str, object] | None = None,
) -> tuple[bool, str]:
    payload = budgets if budgets is not None else _load_performance_budgets()
    modes = payload.get("modes", {})
    enforcement = payload.get("enforcement", {})
    if not isinstance(modes, dict):
        modes = {}
    if not isinstance(enforcement, dict):
        enforcement = {}

    budget_mode = _resolve_budget_mode(workflow_mode, budget_mode_override)
    selected = modes.get(budget_mode, {})
    if not isinstance(selected, dict):
        selected = {}
    budget_seconds = selected.get("budget_seconds")
    if not isinstance(budget_seconds, (int, float)):
        return (
            True,
            f"[workflow] budget check skipped (no budget configured for mode '{budget_mode}')",
        )

    tolerance = enforcement.get("default_exceedance_tolerance", 0.1)
    tolerance_value = float(tolerance) if isinstance(tolerance, (int, float)) else 0.1
    threshold = float(budget_seconds) * (1.0 + tolerance_value)
    exceeded = float(total_runtime_seconds) > threshold
    budget_message = (
        f"[workflow] runtime-budget[{budget_mode}] "
        f"runtime={float(total_runtime_seconds):.2f}s "
        f"budget={float(budget_seconds):.2f}s "
        f"threshold={threshold:.2f}s "
        f"exceeded={str(exceeded).lower()}"
    )
    if not exceeded:
        return True, budget_message

    release_block = bool(enforcement.get("release_block_on_exceedance", True))
    should_fail = strict_budget or (workflow_mode == "release" and release_block)
    return (not should_fail), budget_message


def _step_cache_input_paths(step: WorkflowStep, context: object) -> list[Path]:
    data_dir = Path(getattr(context, "data_dir"))
    if step.name == "15_normalize_and_score":
        return [
            data_dir / "features/backbone_table.tsv",
            data_dir / "features/feature_T.tsv",
            data_dir / "features/feature_H.tsv",
            data_dir / "features/feature_A.tsv",
            *context_config_paths(context),
        ]
    if step.name == "21_run_validation":
        return [
            data_dir / "scores/backbone_scored.tsv",
            data_dir / "silver/plasmid_backbones.tsv",
            data_dir / "silver/plasmid_amr_consensus.tsv",
            data_dir / "analysis/module_a_metrics.json",
            data_dir / "analysis/module_a_predictions.tsv",
            *context_config_paths(context),
        ]
    if step.name == "22_run_sensitivity":
        return [
            data_dir / "scores/backbone_scored.tsv",
            data_dir / "silver/plasmid_backbones.tsv",
            data_dir / "silver/plasmid_amr_hits.tsv",
            *context_config_paths(context),
        ]
    return [*context_config_paths(context)]


def _infer_input_paths_from_summary(step: WorkflowStep) -> list[Path]:
    summary_path = _current_step_summary_path(step)
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    manifest = payload.get("input_manifest", {})
    if not isinstance(manifest, dict):
        return []
    inferred: list[Path] = []
    for entry in manifest.values():
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.exists():
            inferred.append(path)
    return inferred


def _infer_input_paths_from_deps(step: WorkflowStep) -> list[Path]:
    inferred: list[Path] = []
    for dep_name in step.deps:
        dep = STEP_LIBRARY.get(dep_name)
        if dep is None:
            continue
        dep_summary_path = _current_step_summary_path(dep)
        if not dep_summary_path.exists():
            continue
        try:
            payload = json.loads(dep_summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for output in payload.get("output_files_written", []):
            path = Path(str(output))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                inferred.append(path)
    return inferred


def _external_db_fingerprints(context: object) -> dict[str, object]:
    data_dir = Path(getattr(context, "data_dir"))
    manifest_path = data_dir / "raw" / "external" / "fetch_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    subset = {
        key: payload.get(key)
        for key in (
            "amrfinderplus_latest",
            "resfinder_latest",
            "vfdb_latest",
            "mobsuite_db",
            "card_data",
            "who_catalog",
        )
        if key in payload
    }
    return subset


def _build_step_cache_bundle(
    step: WorkflowStep,
    *,
    context: object,
    cache_protocol_hash: str,
) -> tuple[str, dict[str, object]] | None:
    if step.name in ARTIFACT_CACHE_DISABLED_STEPS:
        return None
    configured_paths = _step_cache_input_paths(step, context)
    inferred_paths = _infer_input_paths_from_summary(step)
    dep_paths = _infer_input_paths_from_deps(step)
    combined_paths = [*configured_paths, *inferred_paths, *dep_paths]
    deduped_paths: list[Path] = []
    seen: set[Path] = set()
    for path in combined_paths:
        resolved = Path(path).resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        deduped_paths.append(resolved)
    if not deduped_paths:
        return None
    input_signatures = [
        path_signature_with_hash(path, include_file_hash=True) for path in deduped_paths
    ]
    source_signatures = _source_signatures_for_step(step)
    config_paths = [path for path in context_config_paths(context) if path.exists()]
    config_signatures = [
        path_signature_with_hash(path, include_file_hash=True) for path in config_paths
    ]
    cache_key, cache_payload = build_step_cache_key(
        step_name=step.name,
        source_hash=stable_hash(source_signatures),
        input_manifest_hash=stable_hash(input_signatures),
        args=step.args,
        env=dict(step.env),
        config_hash=stable_hash(config_signatures),
        protocol_hash=cache_protocol_hash,
        software=software_fingerprint(),
        external_fingerprints=_external_db_fingerprints(context),
    )
    return cache_key, cache_payload


def _load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("step summary payload must be a JSON object")
    return payload


def _summary_output_paths(summary: dict[str, object]) -> list[Path]:
    output_values = summary.get("output_files_written", [])
    if not isinstance(output_values, list):
        return []
    paths: list[Path] = []
    for output_value in output_values:
        path = Path(str(output_value))
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    return paths


def _step_result_from_summary(
    *,
    step: WorkflowStep,
    protocol_hash: str,
    summary_path: Path,
    summary: dict[str, object],
    telemetry: StepTelemetry,
) -> StepResult:
    return StepResult(
        step_name=step.name,
        status="ok",
        inputs={
            key: str(value.get("sha256") or value.get("digest") or value.get("size"))
            for key, value in _summary_mapping(summary, "input_manifest").items()
            if isinstance(value, dict)
        },
        outputs={
            key: str(value.get("sha256") or value.get("digest") or value.get("size"))
            for key, value in _summary_mapping(summary, "output_manifest").items()
            if isinstance(value, dict)
        },
        protocol_hash=protocol_hash,
        rows_in=telemetry.rows_in,
        rows_out=telemetry.rows_out,
        duration_seconds=telemetry.duration_seconds,
        cache_status=telemetry.cache_status,
        bytes_read=telemetry.bytes_read,
        bytes_written=telemetry.bytes_written,
        peak_rss_mb=telemetry.peak_rss_mb,
        cpu_time_seconds=telemetry.cpu_time_seconds,
        io_wait_hint=telemetry.io_wait_hint,
        input_hash=telemetry.input_hash,
        output_hash=telemetry.output_hash,
        warnings=[str(item) for item in _summary_list(summary, "warnings")],
        scientific_notes=[str(item) for item in _summary_list(summary, "notes")],
        metadata={
            "run_id": summary.get("run_id"),
            "correlation_id": summary.get("correlation_id"),
            "summary_path": str(summary_path),
        },
    )


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


def _enforce_step_boundary(mode: str, step: WorkflowStep) -> bool:
    """Return whether release-grade script boundary validation should block the workflow."""
    return not (mode == "demo-pipeline" and step.name == "24_build_reports")


def run_workflow(
    mode: str,
    *,
    max_workers: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    budget_mode: str | None = None,
    strict_budget: bool = False,
    skip_validation: bool = False,
    skip_reports: bool = False,
) -> int:
    context = build_context(PROJECT_ROOT)
    protocol = ScientificProtocol.from_config(context.config)
    protocol_hash = build_protocol_hash(protocol)
    cache_protocol_hash = build_execution_hash(protocol.execution)
    registry.discover_entry_points()
    steps = _workflow_steps(mode, skip_validation=skip_validation, skip_reports=skip_reports)
    workflow_started_at = datetime.now(timezone.utc)
    workflow_started_perf = time.perf_counter()
    step_telemetry_records: list[StepTelemetry] = []
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

    artifact_cache = ArtifactCache(_workflow_data_root() / "tmp" / "artifact_cache")

    def _flush_workflow_profile(status: str) -> float:
        duration_seconds = max(0.0, time.perf_counter() - workflow_started_perf)
        profile = WorkflowProfile.now(
            mode=mode,
            status=status,
            protocol_hash=protocol_hash,
            started_at=workflow_started_at,
            duration_seconds=duration_seconds,
            steps=step_telemetry_records,
            metadata={"resume": resume, "max_workers": max_workers},
        )
        atomic_write_json(context.logs_dir / f"workflow_{mode}_profile.json", profile.to_dict())
        return duration_seconds

    if mode in SEQUENTIAL_WORKFLOW_MODES:
        max_workers = 1
    elif max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    max_workers = max(1, min(int(max_workers), len(steps)))
    auto_job_cap = _auto_job_cap(max_workers)
    order = {step.name: index for index, step in enumerate(steps)}
    pending = {step.name: step for step in steps if step.name not in completed}
    running: dict[Future[int], WorkflowStep] = {}
    running_started: dict[str, ResourceSnapshot] = {}
    step_cache_bundles: dict[str, tuple[str, dict[str, object]]] = {}
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
                cache_bundle = _build_step_cache_bundle(
                    step,
                    context=context,
                    cache_protocol_hash=cache_protocol_hash,
                )
                if cache_bundle is not None:
                    step_cache_bundles[step.name] = cache_bundle
                    cache_key, cache_key_payload = cache_bundle
                    cached_manifest = artifact_cache.load(cache_key)
                    if cached_manifest is not None:
                        restore_started = capture_resource_snapshot()
                        summary_path = _current_step_summary_path(step)
                        if artifact_cache.restore(cached_manifest, summary_path=summary_path):
                            try:
                                summary = _load_summary(summary_path)
                            except (OSError, ValueError) as exc:
                                print(
                                    f"[workflow] {step.name} cache restore summary unreadable: {exc}; rerunning",
                                    flush=True,
                                )
                            else:
                                telemetry = build_step_telemetry(
                                    step_name=step.name,
                                    status="ok",
                                    cache_status="hit",
                                    summary=summary,
                                    started=restore_started,
                                )
                                summary = upsert_summary_telemetry(summary, telemetry)
                                atomic_write_json(summary_path, summary)
                                boundary_result = validate_script_boundary(
                                    summary, project_root=PROJECT_ROOT
                                )
                                if boundary_result.get("status") == "pass":
                                    step_result_path = write_step_result(
                                        _step_result_from_summary(
                                            step=step,
                                            protocol_hash=protocol_hash,
                                            summary_path=summary_path,
                                            summary=summary,
                                            telemetry=telemetry,
                                        ),
                                        context.logs_dir,
                                    )
                                    completed.add(step.name)
                                    pending.pop(step.name, None)
                                    step_telemetry_records.append(telemetry)
                                    with checkpoint_lock:
                                        _write_workflow_checkpoint(
                                            checkpoint_path,
                                            mode=mode,
                                            completed_steps=sorted(
                                                completed,
                                                key=lambda name: order[name],
                                            ),
                                            step=step,
                                            step_status="ok",
                                            return_code=0,
                                        )
                                    print(
                                        f"[workflow] {step.name}: restored from artifact cache ({cache_key[:12]})",
                                        flush=True,
                                    )
                                    print(
                                        f"[workflow] wrote step contract: {step_result_path}",
                                        flush=True,
                                    )
                                    continue
                                print(
                                    f"[workflow] {step.name} cached outputs failed boundary validation; rerunning",
                                    flush=True,
                                )

                future = executor.submit(_run_step, step, auto_job_cap=auto_job_cap)
                running[future] = step
                running_started[step.name] = capture_resource_snapshot()

            if not running:
                blocked = {
                    name: [dep for dep in step.deps if dep not in completed]
                    for name, step in pending.items()
                }
                raise RuntimeError(f"Workflow deadlock detected: {blocked}")

            done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                started = running_started.pop(step.name, capture_resource_snapshot())
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
                    _flush_workflow_profile("failed")
                    return return_code
                summary_path = _current_step_summary_path(step)
                try:
                    summary = _load_summary(summary_path)
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
                    _flush_workflow_profile("failed")
                    return 1
                cache_status = infer_cache_status_from_summary(summary)
                telemetry = build_step_telemetry(
                    step_name=step.name,
                    status="ok",
                    cache_status=cache_status,
                    summary=summary,
                    started=started,
                )
                summary = upsert_summary_telemetry(summary, telemetry)
                atomic_write_json(summary_path, summary)
                step_result_path = write_step_result(
                    _step_result_from_summary(
                        step=step,
                        protocol_hash=protocol_hash,
                        summary_path=summary_path,
                        summary=summary,
                        telemetry=telemetry,
                    ),
                    context.logs_dir,
                )
                boundary_result = validate_script_boundary(summary, project_root=PROJECT_ROOT)
                if (
                    boundary_result.get("status") != "pass"
                    and _enforce_step_boundary(mode, step)
                ):
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
                    _flush_workflow_profile("failed")
                    return 1
                cache_bundle = step_cache_bundles.get(step.name)
                if cache_bundle is not None:
                    cache_key, cache_key_payload = cache_bundle
                    artifact_cache.publish(
                        step_name=step.name,
                        cache_key=cache_key,
                        cache_key_payload=cache_key_payload,
                        summary=summary,
                        output_paths=_summary_output_paths(summary),
                    )
                completed.add(step.name)
                pending.pop(step.name, None)
                step_telemetry_records.append(telemetry)
                with checkpoint_lock:
                    _write_workflow_checkpoint(
                        checkpoint_path,
                        mode=mode,
                        completed_steps=sorted(completed, key=lambda name: order[name]),
                        step=step,
                        step_status="ok",
                        return_code=return_code,
                    )
                print(f"[workflow] wrote step contract: {step_result_path}", flush=True)

    total_runtime_seconds = _flush_workflow_profile("ok")
    budget_ok, budget_message = _evaluate_runtime_budget(
        workflow_mode=mode,
        total_runtime_seconds=total_runtime_seconds,
        budget_mode_override=budget_mode,
        strict_budget=strict_budget,
    )
    print(budget_message, flush=True)
    if not budget_ok:
        print(
            "[workflow] runtime budget exceeded; blocking completion for this mode.",
            file=sys.stderr,
            flush=True,
        )
        return 3
    if mode == "release":
        scientific_contract = validate_release_scientific_contract(PROJECT_ROOT)
        if scientific_contract.get("status") != "pass":
            issues = [str(item) for item in _summary_list(scientific_contract, "errors")]
            print(
                "[workflow] scientific-contract gate failed: "
                + ("; ".join(issues) if issues else "unknown error"),
                file=sys.stderr,
                flush=True,
            )
            return 4
        artifact_integrity = validate_release_artifact_integrity(PROJECT_ROOT)
        if artifact_integrity.get("status") != "pass":
            issues = [str(item) for item in _summary_list(artifact_integrity, "errors")]
            print(
                "[workflow] artifact-integrity gate failed: "
                + ("; ".join(issues) if issues else "unknown error"),
                file=sys.stderr,
                flush=True,
            )
            return 5
        readiness = evaluate_release_readiness(PROJECT_ROOT)
        if readiness.get("status") != "pass":
            issues = [str(item) for item in _summary_list(readiness, "failed_checks")]
            print(
                "[workflow] release-readiness gate failed: "
                + ("; ".join(issues) if issues else "unknown error"),
                file=sys.stderr,
                flush=True,
            )
            return 6
    return 0


def main(argv: list[str] | None = None) -> int:
    # Phase 0 optimization: prevent nested BLAS parallelism chaos
    configure_blas_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_STEP_NAMES))
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved steps without executing scripts.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint resume and rerun every step from scratch.",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        default=None,
        help="Override performance budget mode key from config/performance_budgets.yaml.",
    )
    parser.add_argument(
        "--strict-budget",
        action="store_true",
        help="Fail workflow whenever runtime exceeds configured budget threshold.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Phase 2: Skip heavy validation steps (21, 22, 27) for fast dev iteration.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Phase 2: Skip report generation steps (24+) for fast dev iteration.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (also sets PLASMID_PRIORITY_MAX_JOBS).",
    )
    args = parser.parse_args(argv)
    # Phase 2: Forward --jobs to env var for child processes
    if args.jobs is not None:
        os.environ["PLASMID_PRIORITY_MAX_JOBS"] = str(args.jobs)
    return run_workflow(
        args.mode,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        budget_mode=args.budget_mode,
        strict_budget=args.strict_budget,
        skip_validation=args.skip_validation,
        skip_reports=args.skip_reports,
    )


if __name__ == "__main__":
    raise SystemExit(main())
