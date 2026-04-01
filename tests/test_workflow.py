from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
