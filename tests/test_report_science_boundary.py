from __future__ import annotations

import ast
import importlib.util
import inspect
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


_BUILD_REPORTS_SPEC = importlib.util.spec_from_file_location(
    "build_reports_script",
    PROJECT_ROOT / "scripts/24_build_reports.py",
)
assert _BUILD_REPORTS_SPEC is not None and _BUILD_REPORTS_SPEC.loader is not None
build_reports_script = importlib.util.module_from_spec(_BUILD_REPORTS_SPEC)
_BUILD_REPORTS_SPEC.loader.exec_module(build_reports_script)


class ReportScienceBoundaryTests(unittest.TestCase):
    def test_report_builder_contains_no_model_fitting_calls(self) -> None:
        source = (PROJECT_ROOT / "scripts/24_build_reports.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned_calls: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"fit", "predict_proba"}:
                banned_calls.append(func.attr)
            elif isinstance(func, ast.Name) and func.id in {
                "GridSearchCV",
                "cross_val_score",
                "cross_validate",
                "train_test_split",
            }:
                banned_calls.append(func.id)

        self.assertEqual(
            banned_calls,
            [],
            f"Report builder should only assemble artifacts, but found fitting calls: {banned_calls}",
        )

    def test_report_builder_does_not_call_fit_functions(self) -> None:
        """Verify the report builder does not call model fitting functions."""
        source = inspect.getsource(build_reports_script)

        # These are the main model fitting functions that should not be called
        forbidden_patterns = [
            "fit_full_model_predictions(",
            "fit_feature_columns_predictions(",
            "fit_predict_model_holdout(",
            "run_module_a(",
        ]

        for pattern in forbidden_patterns:
            self.assertNotIn(
                pattern,
                source,
                f"Report builder should not call {pattern} as it should only assemble persisted artifacts",
            )


if __name__ == "__main__":
    unittest.main()
