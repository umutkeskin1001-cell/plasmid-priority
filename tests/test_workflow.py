from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_workflow_script",
    PROJECT_ROOT / "scripts/run_workflow.py",
)
assert SPEC is not None and SPEC.loader is not None
run_workflow_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_workflow_script
SPEC.loader.exec_module(run_workflow_script)


class WorkflowTests(unittest.TestCase):
    def test_workflow_banner_summarizes_mode_and_capacity(self) -> None:
        self.assertEqual(
            run_workflow_script._workflow_banner(
                "pipeline", step_count=25, max_workers=4, auto_job_cap=2
            ),
            "[workflow] mode=pipeline kind=parallel steps=25 workers=4 auto_job_cap=2",
        )

    def test_pipeline_sequential_workflow_keeps_full_dependency_chain(self) -> None:
        steps = run_workflow_script._workflow_steps("pipeline-sequential")
        names = [step.name for step in steps]
        self.assertEqual(
            names,
            [
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
            ],
        )

    def test_core_refresh_workflow_keeps_only_headline_analysis_tail(self) -> None:
        steps = run_workflow_script._workflow_steps("core-refresh")
        names = [step.name for step in steps]
        self.assertEqual(
            names,
            [
                "15_normalize_and_score",
                "16_run_module_A",
                "21_run_validation",
                "22_run_sensitivity",
                "27_run_advanced_audits",
                "24_build_reports",
                "25_export_tubitak_summary",
            ],
        )

    def test_analysis_refresh_sequential_workflow_keeps_analysis_tail(self) -> None:
        steps = run_workflow_script._workflow_steps("analysis-refresh-sequential")
        names = [step.name for step in steps]
        self.assertEqual(
            names,
            [
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
            ],
        )

    def test_support_refresh_workflow_excludes_core_validation_steps(self) -> None:
        steps = run_workflow_script._workflow_steps("support-refresh")
        names = {step.name for step in steps}
        self.assertIn("17_run_module_B", names)
        self.assertIn("20_run_module_E_amrfinder_concordance", names)
        self.assertNotIn("21_run_validation", names)
        self.assertNotIn("22_run_sensitivity", names)
        self.assertNotIn("27_run_advanced_audits", names)

    def test_release_workflow_contains_bundle_and_registry_steps(self) -> None:
        steps = run_workflow_script._workflow_steps("release")
        names = [step.name for step in steps]
        self.assertIn("28_build_release_bundle", names)
        self.assertIn("29_build_experiment_registry", names)
        release_step = next(step for step in steps if step.name == "28_build_release_bundle")
        self.assertEqual(release_step.deps, ("24_build_reports", "25_export_tubitak_summary"))

    def test_reports_only_workflow_keeps_report_tail(self) -> None:
        steps = run_workflow_script._workflow_steps("reports-only")
        names = [step.name for step in steps]
        self.assertEqual(names, ["24_build_reports", "25_export_tubitak_summary"])

    def test_auto_job_cap_splits_cpu_budget_across_workers(self) -> None:
        with mock.patch.object(run_workflow_script.os, "cpu_count", return_value=12):
            self.assertEqual(run_workflow_script._auto_job_cap(4), 3)
            self.assertEqual(run_workflow_script._auto_job_cap(1), 8)


if __name__ == "__main__":
    unittest.main()
