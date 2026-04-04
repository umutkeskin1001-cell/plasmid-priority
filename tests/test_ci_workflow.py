from __future__ import annotations

import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


class CiWorkflowTests(unittest.TestCase):
    def test_ci_installs_dev_dependencies(self) -> None:
        workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["quality"]["steps"]
        install_step = next(
            step for step in steps if step["name"] == "Install package and dev tooling"
        )
        self.assertIn('-e ".[analysis,dev]"', install_step["run"])

    def test_ci_runs_make_quality(self) -> None:
        workflow = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["quality"]["steps"]
        quality_step = next(step for step in steps if step["name"] == "Run quality gate")
        self.assertEqual(quality_step["run"], "make quality")


if __name__ == "__main__":
    unittest.main()
