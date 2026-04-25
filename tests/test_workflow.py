from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import TracebackType
from typing import Literal
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

VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "run_validation_script",
    PROJECT_ROOT / "scripts/21_run_validation.py",
)
assert VALIDATION_SPEC is not None and VALIDATION_SPEC.loader is not None
run_validation_script = importlib.util.module_from_spec(VALIDATION_SPEC)
sys.modules[VALIDATION_SPEC.name] = run_validation_script
VALIDATION_SPEC.loader.exec_module(run_validation_script)

RELEASE_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "build_release_bundle_script",
    PROJECT_ROOT / "scripts/28_build_release_bundle.py",
)
assert RELEASE_BUNDLE_SPEC is not None and RELEASE_BUNDLE_SPEC.loader is not None
build_release_bundle_script = importlib.util.module_from_spec(RELEASE_BUNDLE_SPEC)
sys.modules[RELEASE_BUNDLE_SPEC.name] = build_release_bundle_script
RELEASE_BUNDLE_SPEC.loader.exec_module(build_release_bundle_script)


class _FakeValidationRun:
    def __init__(self) -> None:
        self.inputs: list[Path] = []
        self.outputs: list[Path] = []
        self.metrics: dict[str, object] = {}
        self.notes: list[str] = []

    def __enter__(self) -> _FakeValidationRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False

    def record_input(self, path: Path) -> None:
        self.inputs.append(Path(path))

    def record_output(self, path: Path) -> None:
        self.outputs.append(Path(path))

    def note(self, message: str) -> None:
        self.notes.append(message)

    def set_metric(self, key: str, value: object) -> None:
        self.metrics[key] = value


class _FakePipelineSettings:
    split_year = 2015
    min_new_countries_for_spread = 3


class _FakeValidationContext:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.pipeline_settings = _FakePipelineSettings()


class WorkflowTests(unittest.TestCase):
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
        self.assertIn("52_build_official_release_artifacts", names)
        self.assertIn("28_build_release_bundle", names)
        self.assertIn("29_build_experiment_registry", names)
        self.assertIn("41_build_performance_dashboard", names)
        self.assertIn("42_release_readiness_report", names)
        official_step = next(
            step for step in steps if step.name == "52_build_official_release_artifacts"
        )
        readiness_step = next(step for step in steps if step.name == "42_release_readiness_report")
        release_step = next(step for step in steps if step.name == "28_build_release_bundle")
        self.assertEqual(official_step.deps, ("24_build_reports",))
        self.assertIn("52_build_official_release_artifacts", readiness_step.deps)
        self.assertEqual(release_step.deps, ("42_release_readiness_report",))

    def test_reports_only_workflow_keeps_report_tail(self) -> None:
        steps = run_workflow_script._workflow_steps("reports-only")
        names = [step.name for step in steps]
        self.assertEqual(names, ["24_build_reports", "25_export_tubitak_summary"])

    def test_all_workflow_step_scripts_exist(self) -> None:
        missing = [
            step.name
            for step in run_workflow_script.STEP_LIBRARY.values()
            if not (PROJECT_ROOT / "scripts" / step.script).exists()
        ]
        self.assertEqual(missing, [])

    def test_demo_pipeline_does_not_enforce_full_report_boundary(self) -> None:
        report_step = run_workflow_script.STEP_LIBRARY["24_build_reports"]
        check_step = run_workflow_script.STEP_LIBRARY["01_check_inputs"]

        self.assertFalse(run_workflow_script._enforce_step_boundary("demo-pipeline", report_step))
        self.assertTrue(run_workflow_script._enforce_step_boundary("demo-pipeline", check_step))
        self.assertTrue(run_workflow_script._enforce_step_boundary("pipeline", report_step))

    def test_auto_job_cap_splits_cpu_budget_across_workers(self) -> None:
        with mock.patch.object(run_workflow_script.os, "cpu_count", return_value=12):
            self.assertEqual(run_workflow_script._auto_job_cap(4), 3)
            self.assertEqual(run_workflow_script._auto_job_cap(1), 8)

    def test_workflow_checkpoint_skips_completed_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            logs_dir = data_root / "tmp" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            output_path = data_root / "tmp" / "outputs" / "step-one.tsv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("ok\n", encoding="utf-8")
            step = run_workflow_script.WorkflowStep("01_check_inputs", "01_check_inputs.py")
            checkpoint_path = data_root / "tmp" / "workflow" / "pipeline.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            source_signatures = run_workflow_script._source_signatures_for_step(step)
            summary_path = logs_dir / "01_check_inputs_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "script_name": "01_check_inputs",
                        "status": "ok",
                        "output_files_written": [str(output_path)],
                        "input_manifest": {},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "mode": "pipeline",
                        "completed_steps": ["01_check_inputs"],
                        "steps": {
                            "01_check_inputs": {
                                "status": "ok",
                                "summary_path": str(summary_path),
                                "script_path": str(PROJECT_ROOT / "scripts" / step.script),
                                "source_signatures": source_signatures,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    run_workflow_script.os.environ,
                    {"PLASMID_PRIORITY_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                mock.patch.object(run_workflow_script, "_workflow_steps", return_value=[step]),
                mock.patch.object(run_workflow_script, "_run_step", return_value=0) as run_step,
            ):
                result = run_workflow_script.run_workflow("pipeline", max_workers=1, resume=True)

            self.assertEqual(result, 0)
            run_step.assert_not_called()

    def test_workflow_resume_reruns_invalid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            logs_dir = data_root / "tmp" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            step = run_workflow_script.WorkflowStep("01_check_inputs", "01_check_inputs.py")
            checkpoint_path = data_root / "tmp" / "workflow" / "pipeline.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path = logs_dir / "01_check_inputs_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "script_name": "01_check_inputs",
                        "status": "failed",
                        "output_files_written": [],
                        "input_manifest": {},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "mode": "pipeline",
                        "completed_steps": [],
                        "steps": {
                            "01_check_inputs": {
                                "status": "failed",
                                "summary_path": str(summary_path),
                                "script_path": str(PROJECT_ROOT / "scripts" / step.script),
                                "source_signatures": [],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    run_workflow_script.os.environ,
                    {"PLASMID_PRIORITY_DATA_ROOT": str(data_root)},
                    clear=False,
                ),
                mock.patch.object(run_workflow_script, "_workflow_steps", return_value=[step]),
                mock.patch.object(run_workflow_script, "_run_step", return_value=0) as run_step,
            ):
                result = run_workflow_script.run_workflow("pipeline", max_workers=1, resume=True)

            self.assertEqual(result, 0)
            run_step.assert_called_once()

    def test_validation_script_records_single_model_pareto_screen_output(self) -> None:
        fake_run = _FakeValidationRun()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            context = _FakeValidationContext(root)
            with (
                mock.patch.object(run_validation_script, "build_context", return_value=context),
                mock.patch.object(run_validation_script, "ManagedScriptRun", return_value=fake_run),
                mock.patch.object(
                    run_validation_script, "project_python_source_paths", return_value=[]
                ),
                mock.patch.object(
                    run_validation_script, "load_signature_manifest", return_value=True
                ),
            ):
                result = run_validation_script.main()

        self.assertEqual(result, 0)
        self.assertIn(
            Path(tmp_dir) / "data/analysis/single_model_pareto_screen.tsv",
            fake_run.outputs,
        )
        self.assertIn(
            Path(tmp_dir) / "data/analysis/single_model_pareto_finalists.tsv",
            fake_run.outputs,
        )
        self.assertIn(
            Path(tmp_dir) / "data/analysis/single_model_official_decision.tsv",
            fake_run.outputs,
        )
        self.assertEqual(fake_run.metrics.get("cache_hit"), True)

    def test_release_bundle_includes_single_model_report_surface(self) -> None:
        release_files = set(build_release_bundle_script.RELEASE_FILES)
        self.assertIn("reports/headline_validation_summary.md", release_files)
        self.assertIn("reports/core_tables/headline_validation_summary.tsv", release_files)
        self.assertIn("reports/core_tables/single_model_official_decision.tsv", release_files)
        self.assertIn("reports/core_tables/official_model_scores.tsv", release_files)
        self.assertIn("reports/core_tables/official_model_scorecard.tsv", release_files)
        self.assertIn("reports/core_tables/official_candidate_decisions.tsv", release_files)
        self.assertIn("reports/core_tables/official_model_summary.json", release_files)
        self.assertIn("docs/reproducibility_manifest.json", release_files)
        self.assertIn("reports/reviewer_pack/run_reproducibility.sh", release_files)
        self.assertIn("reports/performance/workflow_performance_dashboard.md", release_files)
        self.assertIn("reports/release/release_readiness_report.md", release_files)

    def test_release_info_includes_official_model_family_summary_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "reports/core_tables").mkdir(parents=True)
            (root / "reports/core_tables/model_metrics.tsv").write_text(
                "\n".join(
                    [
                        "model_name\troc_auc\troc_auc_ci_lower\troc_auc_ci_upper\taverage_precision\taverage_precision_ci_lower\taverage_precision_ci_upper",
                        "pareto_model\t0.83\t0.80\t0.86\t0.75\t0.71\t0.79",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "reports/core_tables/official_model_summary.json").write_text(
                '{"official_model_family_status": "pass", "candidate_count": 12, "review_not_rank_count": 3, "decision_surface": "conservative_evidence_consensus"}\n',
                encoding="utf-8",
            )

            release_info = build_release_bundle_script._build_release_info(root)

        self.assertIn("Official model family status: pass", release_info)
        self.assertIn("Official decision surface: conservative_evidence_consensus", release_info)
        self.assertIn("Official candidate decisions: 12 total; 3 review_not_rank", release_info)

    def test_release_bundle_includes_candidate_dossier_glob_surface(self) -> None:
        glob_patterns = set(build_release_bundle_script.RELEASE_GLOB_FILES)
        self.assertIn("reports/reviewer_pack/candidate_evidence_dossiers/*.md", glob_patterns)

    def test_release_workflow_fails_when_artifact_integrity_gate_fails(self) -> None:
        with (
            mock.patch.object(run_workflow_script.registry, "discover_entry_points"),
            mock.patch.object(run_workflow_script, "_workflow_steps", return_value=[]),
            mock.patch.object(
                run_workflow_script,
                "validate_release_scientific_contract",
                return_value={"status": "pass", "errors": []},
            ),
            mock.patch.object(
                run_workflow_script,
                "validate_release_artifact_integrity",
                return_value={"status": "fail", "errors": ["broken"]},
            ),
            mock.patch.object(
                run_workflow_script,
                "evaluate_release_readiness",
                return_value={"status": "pass", "failed_checks": []},
            ),
        ):
            result = run_workflow_script.run_workflow("release", max_workers=1, resume=False)
        self.assertEqual(result, 5)

    def test_release_workflow_fails_when_readiness_gate_fails(self) -> None:
        with (
            mock.patch.object(run_workflow_script.registry, "discover_entry_points"),
            mock.patch.object(run_workflow_script, "_workflow_steps", return_value=[]),
            mock.patch.object(
                run_workflow_script,
                "validate_release_scientific_contract",
                return_value={"status": "pass", "errors": []},
            ),
            mock.patch.object(
                run_workflow_script,
                "validate_release_artifact_integrity",
                return_value={"status": "pass", "errors": []},
            ),
            mock.patch.object(
                run_workflow_script,
                "evaluate_release_readiness",
                return_value={"status": "fail", "failed_checks": ["x"]},
            ),
        ):
            result = run_workflow_script.run_workflow("release", max_workers=1, resume=False)
        self.assertEqual(result, 6)

    def test_workflow_discovers_entry_point_plugins_before_execution(self) -> None:
        with mock.patch.object(run_workflow_script.registry, "discover_entry_points") as discover:
            with mock.patch.object(run_workflow_script, "_workflow_steps", return_value=[]):
                result = run_workflow_script.run_workflow(
                    "reports-only", max_workers=1, resume=False
                )

        self.assertEqual(result, 0)
        discover.assert_called_once()

    def test_release_info_uses_single_model_official_decision_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "reports/core_tables").mkdir(parents=True)
            (root / "reports/core_tables/model_metrics.tsv").write_text(
                "\n".join(
                    [
                        "model_name\troc_auc\troc_auc_ci_lower\troc_auc_ci_upper\taverage_precision\taverage_precision_ci_lower\taverage_precision_ci_upper\tselection_adjusted_empirical_p_roc_auc\tpermutation_p_roc_auc\tn_permutations\tn_permutations_selection_adjusted",
                        "legacy_model\t0.71\t0.68\t0.74\t0.61\t0.57\t0.65\t0.040\t0.050\t200\t400",
                        "pareto_model\t0.83\t0.80\t0.86\t0.75\t0.71\t0.79\t0.006\t0.012\t200\t500",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "reports/core_tables/single_model_official_decision.tsv").write_text(
                "\n".join(
                    [
                        "official_model_name\tdecision_reason\tscientific_acceptance_status",
                        "pareto_model\taccepted_with_best_reliability_power_tradeoff\tpass",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            release_info = build_release_bundle_script._build_release_info(root)

        self.assertIn("Primary model: pareto_model", release_info)
        self.assertIn("Single-model decision status: pass", release_info)
        self.assertIn(
            "Single-model decision reason: accepted_with_best_reliability_power_tradeoff",
            release_info,
        )
        self.assertIn("Selection-adjusted permutation p = 0.006 (n=500)", release_info)
        self.assertIn("Fixed-score permutation p = 0.012 (n=200)", release_info)

    def test_release_info_marks_missing_fixed_score_permutations_as_na(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "reports/core_tables").mkdir(parents=True)
            (root / "reports/core_tables/model_metrics.tsv").write_text(
                "\n".join(
                    [
                        "model_name\troc_auc\troc_auc_ci_lower\troc_auc_ci_upper\taverage_precision\taverage_precision_ci_lower\taverage_precision_ci_upper\tselection_adjusted_empirical_p_roc_auc\tn_permutations_selection_adjusted",
                        "phylo_support_fusion_priority\t0.83\t0.80\t0.86\t0.75\t0.71\t0.79\t0.005\t200",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "reports/core_tables/single_model_official_decision.tsv").write_text(
                "\n".join(
                    [
                        "official_model_name\tdecision_reason\tscientific_acceptance_status",
                        "discovery_12f_source__pruned\tlowest_failure_severity_with_competitive_auc\tfail",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            release_info = build_release_bundle_script._build_release_info(root)

        self.assertIn("Primary model: phylo_support_fusion_priority", release_info)
        self.assertIn("Selection-adjusted permutation p = 0.005 (n=200)", release_info)
        self.assertIn("Fixed-score permutation p NA (n=NA)", release_info)

    def test_runtime_budget_blocks_release_mode_when_exceeded(self) -> None:
        budgets = {
            "modes": {"release-full": {"budget_seconds": 10}},
            "enforcement": {
                "default_exceedance_tolerance": 0.1,
                "release_block_on_exceedance": True,
            },
        }
        ok, message = run_workflow_script._evaluate_runtime_budget(
            workflow_mode="release",
            total_runtime_seconds=20.0,
            budgets=budgets,
        )
        self.assertFalse(ok)
        self.assertIn("exceeded=true", message)
        self.assertIn("release-full", message)

    def test_runtime_budget_does_not_block_non_release_without_strict_flag(self) -> None:
        budgets = {
            "modes": {"dev-refresh": {"budget_seconds": 10}},
            "enforcement": {
                "default_exceedance_tolerance": 0.0,
                "release_block_on_exceedance": True,
            },
        }
        ok, message = run_workflow_script._evaluate_runtime_budget(
            workflow_mode="analysis-refresh",
            total_runtime_seconds=15.0,
            budgets=budgets,
        )
        self.assertTrue(ok)
        self.assertIn("exceeded=true", message)


if __name__ == "__main__":
    unittest.main()
