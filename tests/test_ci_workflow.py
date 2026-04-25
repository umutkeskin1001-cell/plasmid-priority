from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


class CiWorkflowTests(unittest.TestCase):
    def _get_install_step(self) -> dict[str, Any]:
        workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["quality"]["steps"]
        return next(step for step in steps if step["name"] == "Install package and dev tooling")

    def test_ci_installs_dev_dependencies(self) -> None:
        install_step = self._get_install_step()
        self.assertIn('-e ".[analysis,dev,tree-models]"', install_step["run"])

    def test_ci_installs_tree_models(self) -> None:
        """Primary model (discovery_boosted/LightGBM) must be installed in CI."""
        install_step = self._get_install_step()
        self.assertIn(
            "tree-models",
            install_step["run"],
            (
                "CI must install tree-models extra so LightGBM (primary model) is tested. "
                "Without this, CI silently tests a different model than production."
            ),
        )

    def test_ci_verifies_lightgbm_import(self) -> None:
        """CI must verify LightGBM can be imported after installation."""
        install_step = self._get_install_step()
        self.assertIn(
            "import lightgbm",
            install_step["run"],
            (
                "CI install step must verify LightGBM importability with: "
                "python -c 'import lightgbm'"
            ),
        )

    def test_ci_runs_make_quality(self) -> None:
        workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["quality"]["steps"]
        run_commands = {str(step.get("run", "")).strip() for step in steps}
        self.assertIn("make quality", run_commands)

    def test_ci_uses_multiple_python_versions(self) -> None:
        """CI must test on at least Python 3.12."""
        workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        matrix = workflow["jobs"]["quality"]["strategy"]["matrix"]
        python_versions = matrix.get("python-version", [])
        self.assertIn("3.12", python_versions, "CI must test on Python 3.12")

    def test_ci_runs_critical_path_coverage_gate(self) -> None:
        workflow_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("critical-path coverage", workflow_text.lower())
        self.assertIn("tests/test_probabilistic_labels.py", workflow_text)
        self.assertIn("tests/test_modeling_temporal_cv.py", workflow_text)
        self.assertIn("python -m coverage run", workflow_text)
        self.assertIn("src/plasmid_priority/labels/probabilistic.py", workflow_text)


if __name__ == "__main__":
    unittest.main()
