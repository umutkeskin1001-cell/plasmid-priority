#!/usr/bin/env python3
"""Dependency-aware local workflow runner for sequential and light parallel refreshes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    script: str
    deps: tuple[str, ...] = ()
    args: tuple[str, ...] = ()


STEP_LIBRARY: dict[str, WorkflowStep] = {
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
    "05_deduplicate": WorkflowStep("05_deduplicate", "05_deduplicate.py", deps=("04_harmonize_metadata",)),
    "06_annotate_mobility": WorkflowStep("06_annotate_mobility", "06_annotate_mobility.py", deps=("05_deduplicate",)),
    "07_annotate_amr": WorkflowStep("07_annotate_amr", "07_annotate_amr.py", deps=("06_annotate_mobility",)),
    "08_build_amr_consensus": WorkflowStep("08_build_amr_consensus", "08_build_amr_consensus.py", deps=("07_annotate_amr",)),
    "09_assign_backbones": WorkflowStep("09_assign_backbones", "09_assign_backbones.py", deps=("08_build_amr_consensus",)),
    "10_compute_coherence": WorkflowStep("10_compute_coherence", "10_compute_coherence.py", deps=("09_assign_backbones",)),
    "11_compute_feature_T": WorkflowStep("11_compute_feature_T", "11_compute_feature_T.py", deps=("10_compute_coherence",)),
    "12_compute_feature_H": WorkflowStep("12_compute_feature_H", "12_compute_feature_H.py", deps=("11_compute_feature_T",)),
    "13_compute_feature_A": WorkflowStep("13_compute_feature_A", "13_compute_feature_A.py", deps=("12_compute_feature_H",)),
    "14_build_backbone_table": WorkflowStep("14_build_backbone_table", "14_build_backbone_table.py", deps=("13_compute_feature_A",)),
    "15_normalize_and_score": WorkflowStep("15_normalize_and_score", "15_normalize_and_score.py", deps=("14_build_backbone_table",)),
    "16_run_module_A": WorkflowStep("16_run_module_A", "16_run_module_A.py", deps=("15_normalize_and_score",)),
    "17_run_module_B": WorkflowStep("17_run_module_B", "17_run_module_B.py", deps=("15_normalize_and_score",)),
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
    "21_run_validation": WorkflowStep("21_run_validation", "21_run_validation.py", deps=("16_run_module_A",)),
    "22_run_sensitivity": WorkflowStep("22_run_sensitivity", "22_run_sensitivity.py", deps=("15_normalize_and_score",)),
    "23_run_module_f_enrichment": WorkflowStep("23_run_module_f_enrichment", "23_run_module_f_enrichment.py", deps=("15_normalize_and_score",)),
    "27_run_advanced_audits": WorkflowStep("27_run_advanced_audits", "27_run_advanced_audits.py", deps=("21_run_validation",)),
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
    "25_export_tubitak_summary": WorkflowStep("25_export_tubitak_summary", "25_export_tubitak_summary.py", deps=("24_build_reports",)),
    "28_build_release_bundle": WorkflowStep(
        "28_build_release_bundle",
        "28_build_release_bundle.py",
        deps=("24_build_reports", "25_export_tubitak_summary"),
    ),
    "29_build_experiment_registry": WorkflowStep("29_build_experiment_registry", "29_build_experiment_registry.py"),
}


MODE_STEP_NAMES: dict[str, tuple[str, ...]] = {
    "pipeline": (
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
    ),
    "analysis-refresh": (
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
    ),
    "core-refresh": (
        "15_normalize_and_score",
        "16_run_module_A",
        "21_run_validation",
        "22_run_sensitivity",
        "27_run_advanced_audits",
        "24_build_reports",
        "25_export_tubitak_summary",
    ),
    "support-refresh": (
        "15_normalize_and_score",
        "16_run_module_A",
        "17_run_module_B",
        "18_run_module_C_pathogen_detection",
        "19_run_module_D_external_support",
        "20_run_module_E_amrfinder_concordance",
        "23_run_module_f_enrichment",
        "24_build_reports",
        "25_export_tubitak_summary",
    ),
    "release": (
        "24_build_reports",
        "25_export_tubitak_summary",
        "28_build_release_bundle",
        "29_build_experiment_registry",
    ),
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


def _run_step(step: WorkflowStep) -> int:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / step.script), *step.args]
    print(f"[workflow] {step.name}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def run_workflow(mode: str, *, max_workers: int = 1, dry_run: bool = False) -> int:
    steps = _workflow_steps(mode)
    if dry_run:
        for step in steps:
            dep_text = f" deps={','.join(step.deps)}" if step.deps else ""
            arg_text = " " + " ".join(step.args) if step.args else ""
            print(f"{step.name}{dep_text}: {step.script}{arg_text}")
        return 0

    max_workers = max(1, min(int(max_workers), len(steps)))
    order = {step.name: index for index, step in enumerate(steps)}
    pending = {step.name: step for step in steps}
    completed: set[str] = set()
    running: dict[object, WorkflowStep] = {}

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
                future = executor.submit(_run_step, step)
                running[future] = step

            if not running:
                blocked = {
                    name: [dep for dep in step.deps if dep not in completed]
                    for name, step in pending.items()
                }
                raise RuntimeError(f"Workflow deadlock detected: {blocked}")

            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                return_code = int(future.result())
                if return_code != 0:
                    print(f"[workflow] {step.name} failed with exit code {return_code}", file=sys.stderr, flush=True)
                    return return_code
                completed.add(step.name)
                pending.pop(step.name, None)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_STEP_NAMES))
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of concurrent steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved steps without executing scripts.")
    args = parser.parse_args(argv)
    return run_workflow(args.mode, max_workers=args.max_workers, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
