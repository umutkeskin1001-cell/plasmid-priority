from __future__ import annotations

import importlib.util
import json
import tempfile
import time
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

from plasmid_priority.config import build_context
from plasmid_priority.reporting import (
    ManagedScriptRun,
    build_report_overview_table,
    validate_report_artifact,
)
from plasmid_priority.reporting.figures import (
    _candidate_tick_label,
    plot_calibration_threshold_summary,
)
from plasmid_priority.reporting.model_audit import (
    build_candidate_risk_table,
    build_threshold_flip_table,
)
from plasmid_priority.utils.files import (
    atomic_write_json,
    load_signature_manifest,
    write_signature_manifest,
)


class ReportingTests(unittest.TestCase):
    def test_report_model_metrics_keeps_audit_models_and_marks_official_surface(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "status": "ok",
                    "roc_auc": 0.83,
                    "average_precision": 0.77,
                    "brier_score": 0.16,
                    "positive_prevalence": 0.36,
                    "n_backbones": 100,
                    "n_positive": 36,
                },
                {
                    "model_name": "governance_model",
                    "status": "ok",
                    "roc_auc": 0.79,
                    "average_precision": 0.71,
                    "brier_score": 0.17,
                    "positive_prevalence": 0.36,
                    "n_backbones": 100,
                    "n_positive": 36,
                },
                {
                    "model_name": "baseline_both",
                    "status": "ok",
                    "roc_auc": 0.72,
                    "average_precision": 0.65,
                    "brier_score": 0.19,
                    "positive_prevalence": 0.36,
                    "n_backbones": 100,
                    "n_positive": 36,
                },
                {
                    "model_name": "audit_only_model",
                    "status": "ok",
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "brier_score": 0.18,
                    "positive_prevalence": 0.36,
                    "n_backbones": 100,
                    "n_positive": 36,
                },
            ]
        )
        confirmatory = pd.DataFrame(
            [
                {
                    "cohort_name": "confirmatory_internal",
                    "model_name": "primary_model",
                    "status": "ok",
                    "roc_auc": 0.78,
                    "average_precision": 0.66,
                    "brier_score": 0.15,
                    "positive_prevalence": 0.30,
                    "n_backbones": 70,
                    "n_positive": 21,
                    "share_of_primary_eligible": 0.70,
                }
            ]
        )

        result = build_reports_script._build_report_model_metrics(
            model_metrics,
            confirmatory_cohort_summary=confirmatory,
            primary_model_name="primary_model",
            governance_model_name="governance_model",
        )

        self.assertEqual(
            result["model_name"].head(4).tolist(),
            [
                "primary_model",
                "governance_model",
                "baseline_both",
                "internal_high_integrity_subset_primary_model",
            ],
        )
        self.assertIn("audit_only_model", set(result["model_name"].astype(str)))
        self.assertIn("brier_skill_score", result.columns)
        self.assertIn("report_visibility", result.columns)
        primary_row = result.loc[result["model_name"] == "primary_model"].iloc[0]
        audit_row = result.loc[result["model_name"] == "audit_only_model"].iloc[0]
        self.assertAlmostEqual(float(primary_row["brier_skill_score"]), 1.0 - (0.16 / (0.36 * 0.64)))
        self.assertEqual(str(primary_row["report_visibility"]), "official")
        self.assertEqual(str(audit_row["report_visibility"]), "audit_only")

    def test_headline_validation_summary_uses_internal_subset_label(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "roc_auc": 0.83,
                    "average_precision": 0.77,
                    "scientific_acceptance_status": "fail",
                },
                {"model_name": "governance_model", "roc_auc": 0.79, "average_precision": 0.71},
                {"model_name": "baseline_both", "roc_auc": 0.72, "average_precision": 0.65},
                {
                    "model_name": "internal_high_integrity_subset_primary_model",
                    "roc_auc": 0.78,
                    "average_precision": 0.66,
                },
            ]
        )
        blocked_holdout = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "blocked_holdout_group_columns": "dominant_source,dominant_region_train",
                    "blocked_holdout_roc_auc": 0.74,
                    "blocked_holdout_group_count": 5,
                    "worst_blocked_holdout_group": "dominant_region_train:Asia",
                    "worst_blocked_holdout_group_roc_auc": 0.70,
                }
            ]
        )
        country_missingness_bounds = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "eligible_for_country_bounds": True,
                    "label_observed": 1,
                    "label_midpoint": 1,
                    "label_optimistic": 1,
                    "label_weighted": 1,
                },
                {
                    "backbone_id": "bb2",
                    "eligible_for_country_bounds": True,
                    "label_observed": 0,
                    "label_midpoint": 1,
                    "label_optimistic": 1,
                    "label_weighted": 0,
                },
            ]
        )
        country_missingness_sensitivity = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "outcome_name": "label_observed",
                    "roc_auc": 0.74,
                    "average_precision": 0.63,
                },
                {
                    "model_name": "primary_model",
                    "outcome_name": "label_midpoint",
                    "roc_auc": 0.77,
                    "average_precision": 0.66,
                },
                {
                    "model_name": "primary_model",
                    "outcome_name": "label_optimistic",
                    "roc_auc": 0.78,
                    "average_precision": 0.68,
                },
                {
                    "model_name": "primary_model",
                    "outcome_name": "label_weighted",
                    "roc_auc": 0.75,
                    "average_precision": 0.64,
                },
            ]
        )

        result = build_reports_script._build_headline_validation_summary(
            model_metrics,
            primary_model_name="primary_model",
            governance_model_name="governance_model",
        )

        self.assertIn("internal_high_integrity_subset", result["summary_label"].tolist())
        self.assertIn("scientific_acceptance_status", result.columns)
        self.assertIn("scientific_acceptance_failed_criteria", result.columns)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "headline_validation_summary.md"
            build_reports_script._write_headline_validation_summary(
                output_path,
                result,
                primary_model_name="primary_model",
                governance_model_name="governance_model",
                blocked_holdout_summary=blocked_holdout,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
            )
            content = output_path.read_text(encoding="utf-8")
        self.assertIn("Blocked Holdout Audit", content)
        self.assertIn("Benchmark scope note", content)
        self.assertIn("benchmark-limited", content)
        self.assertIn("dominant_source", content)
        self.assertIn("dominant_region_train", content)
        self.assertIn("dominant_region_train:Asia", content)
        self.assertIn("Country Missingness", content)
        self.assertIn("country_missingness_bounds.tsv", content)
        self.assertIn("country_missingness_sensitivity.tsv", content)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", content)
        self.assertIn("nonlinear_deconfounding_audit.tsv", content)
        self.assertIn("ordinal_outcome_audit.tsv", content)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", content)
        self.assertIn("macro_region_jump_outcome.tsv", content)
        self.assertIn("prospective_candidate_freeze.tsv", content)
        self.assertIn("annual_candidate_freeze_summary.tsv", content)
        self.assertIn("mash_similarity_graph.tsv", content)
        self.assertIn("counterfactual_shortlist_comparison.tsv", content)
        self.assertIn("geographic_jump_distance_outcome.tsv", content)
        self.assertIn("amr_uncertainty_summary.tsv", content)

    def test_headline_validation_summary_includes_single_model_pareto_decision(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "primary_model",
                    "roc_auc": 0.83,
                    "average_precision": 0.77,
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:source_holdout",
                },
                {
                    "model_name": "governance_model",
                    "roc_auc": 0.79,
                    "average_precision": 0.71,
                },
                {"model_name": "baseline_both", "roc_auc": 0.72, "average_precision": 0.65},
                {
                    "model_name": "pareto_model",
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "scientific_acceptance_status": "pass",
                    "scientific_acceptance_failed_criteria": "",
                },
            ]
        )
        single_model_official_decision = pd.DataFrame(
            [
                {
                    "official_model_name": "pareto_model",
                    "decision_reason": "accepted_with_best_reliability_power_tradeoff",
                    "scientific_acceptance_status": "pass",
                    "scientific_acceptance_failed_criteria": "",
                    "failure_severity": 0.0,
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "weighted_objective_score": 0.84,
                    "screen_fit_seconds": 9.25,
                    "compute_efficiency_score": 0.68,
                    "selected_from_n_finalists": 3,
                }
            ]
        )
        single_model_pareto_finalists = pd.DataFrame(
            [
                {
                    "model_name": "pareto_model",
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "ece": 0.031,
                    "spatial_holdout_roc_auc": 0.76,
                    "selection_adjusted_empirical_p_roc_auc": 0.006,
                    "scientific_acceptance_status": "pass",
                    "scientific_acceptance_failed_criteria": "",
                }
            ]
        )

        result = build_reports_script._build_headline_validation_summary(
            model_metrics,
            primary_model_name="primary_model",
            governance_model_name="governance_model",
            single_model_official_decision=single_model_official_decision,
            single_model_pareto_finalists=single_model_pareto_finalists,
        )

        pareto_row = result.loc[
            result["summary_label"].astype(str).eq("single_model_pareto_official")
        ].iloc[0]
        self.assertEqual(str(pareto_row["model_name"]), "pareto_model")
        self.assertEqual(
            str(pareto_row["decision_reason"]),
            "accepted_with_best_reliability_power_tradeoff",
        )
        self.assertEqual(int(pareto_row["selected_from_n_finalists"]), 3)
        self.assertAlmostEqual(float(pareto_row["ece"]), 0.031)
        self.assertAlmostEqual(float(pareto_row["selection_adjusted_empirical_p_roc_auc"]), 0.006)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "headline_validation_summary.md"
            build_reports_script._write_headline_validation_summary(
                output_path,
                result,
                primary_model_name="primary_model",
                governance_model_name="governance_model",
            )
            content = output_path.read_text(encoding="utf-8")
        self.assertIn("Single-Model Pareto Decision", content)
        self.assertIn("pareto_model", content)
        self.assertIn("accepted_with_best_reliability_power_tradeoff", content)

    def test_attach_single_model_decision_summary_adds_prefixed_fields(self) -> None:
        import pandas as pd

        model_selection_summary = pd.DataFrame(
            [
                {
                    "published_primary_model": "primary_model",
                    "published_primary_roc_auc": 0.83,
                }
            ]
        )
        single_model_official_decision = pd.DataFrame(
            [
                {
                    "official_model_name": "pareto_model",
                    "decision_reason": "lowest_failure_severity_with_competitive_auc",
                    "scientific_acceptance_status": "fail",
                    "scientific_acceptance_failed_criteria": "fail:source_holdout",
                    "failure_severity": 0.04,
                    "roc_auc": 0.81,
                    "average_precision": 0.73,
                    "weighted_objective_score": 0.78,
                    "screen_fit_seconds": 12.5,
                    "compute_efficiency_score": 0.61,
                    "selected_from_n_finalists": 2,
                }
            ]
        )

        result = build_reports_script._attach_single_model_decision_summary(
            model_selection_summary,
            single_model_official_decision,
            published_primary_model="primary_model",
        )

        row = result.iloc[0]
        self.assertEqual(str(row["single_model_official_model"]), "pareto_model")
        self.assertEqual(
            str(row["single_model_official_decision_reason"]),
            "lowest_failure_severity_with_competitive_auc",
        )
        self.assertEqual(str(row["single_model_official_scientific_acceptance_status"]), "fail")
        self.assertEqual(str(row["single_model_official_failed_criteria"]), "fail:source_holdout")
        self.assertEqual(int(row["single_model_selected_from_n_finalists"]), 2)
        self.assertEqual(bool(row["single_model_official_matches_published_primary"]), False)

    def test_prune_shadowed_report_tables_preserves_single_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            core_dir = root / "core"
            diag_dir = root / "diag"
            analysis_dir = root / "analysis"
            core_dir.mkdir()
            diag_dir.mkdir()
            analysis_dir.mkdir()
            preserved = diag_dir / "single_model_pareto_finalists.tsv"
            preserved.write_text("x\n1\n", encoding="utf-8")
            shadowed = diag_dir / "ordinary_shadowed.tsv"
            shadowed.write_text("x\n1\n", encoding="utf-8")
            (analysis_dir / "single_model_pareto_finalists.tsv").write_text("x\n2\n", encoding="utf-8")
            (analysis_dir / "ordinary_shadowed.tsv").write_text("x\n2\n", encoding="utf-8")

            build_reports_script._prune_shadowed_report_tables(
                core_dir,
                diag_dir,
                analysis_dir,
                preserve_file_names={"single_model_pareto_finalists.tsv"},
            )

            self.assertTrue(preserved.exists())
            self.assertFalse(shadowed.exists())

    def test_weighting_sensitivity_table_uses_explicit_sample_weight_modes(self) -> None:
        sensitivity = {
            "default": {
                "sample_weight_mode": "class_balanced",
                "roc_auc": 0.81,
                "average_precision": 0.74,
                "average_precision_lift": 0.37,
                "brier_score": 0.17,
                "precision_at_top_25": 1.0,
                "recall_at_top_25": 0.07,
                "positive_prevalence": 0.36,
            },
            "class_plus_knownness_balanced_primary": {
                "sample_weight_mode": "class_balanced+knownness_balanced",
                "roc_auc": 0.80,
                "average_precision": 0.73,
                "average_precision_lift": 0.36,
                "brier_score": 0.18,
                "precision_at_top_25": 1.0,
                "recall_at_top_25": 0.07,
                "positive_prevalence": 0.36,
            },
            "knownness_balanced_primary": {
                "sample_weight_mode": "knownness_balanced",
                "roc_auc": 0.79,
                "average_precision": 0.72,
                "average_precision_lift": 0.35,
                "brier_score": 0.16,
                "precision_at_top_25": 0.96,
                "recall_at_top_25": 0.06,
                "positive_prevalence": 0.36,
            },
        }

        result = build_reports_script._build_weighting_sensitivity_table(sensitivity)

        self.assertEqual(
            result["variant"].tolist(),
            ["default", "class_plus_knownness_balanced_primary", "knownness_balanced_primary"],
        )
        self.assertEqual(
            result["sample_weight_mode"].tolist(),
            [
                "class_balanced",
                "class_balanced+knownness_balanced",
                "knownness_balanced",
            ],
        )
        self.assertEqual(result["brier_score"].tolist(), [0.17, 0.18, 0.16])

    def test_l2_sensitivity_table_keeps_resolved_sample_weight_mode(self) -> None:
        sensitivity = {
            "primary_l2_0p5": {
                "l2": 0.5,
                "sample_weight_mode": "class_balanced+knownness_balanced",
                "roc_auc": 0.81,
                "average_precision": 0.74,
                "average_precision_lift": 0.37,
                "brier_score": 0.17,
                "precision_at_top_25": 1.0,
                "recall_at_top_25": 0.07,
            }
        }

        result = build_reports_script._build_l2_sensitivity_table(sensitivity)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "sample_weight_mode"], "class_balanced+knownness_balanced")
        self.assertEqual(float(result.loc[0, "brier_score"]), 0.17)

    def test_atomic_write_json_sanitizes_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "payload.json"
            atomic_write_json(output_path, {"value": float("nan"), "nested": [1.0, float("inf")]})
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIsNone(payload["value"])
        self.assertEqual(payload["nested"], [1.0, None])

    def test_summary_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "data/manifests").mkdir(parents=True)
            (root / "reports").mkdir()
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [],
                    }
                ),
                encoding="utf-8",
            )
            context = build_context(root)
            with ManagedScriptRun(context, "unit_test_script") as run:
                run.note("hello")
            summary_path = root / "data/tmp/logs/unit_test_script_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["notes"], ["hello"])

    def test_signature_manifest_round_trip_and_invalidates_on_input_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.tsv"
            output_path = root / "output.tsv"
            source_path = root / "script.py"
            manifest_path = root / "cache.manifest.json"
            input_path.write_text("a\n", encoding="utf-8")
            output_path.write_text("result\n", encoding="utf-8")
            source_path.write_text("print('x')\n", encoding="utf-8")
            metadata = {"pipeline_settings": {"split_year": 2015}}

            write_signature_manifest(
                manifest_path,
                input_paths=[input_path],
                output_paths=[output_path],
                source_paths=[source_path],
                metadata=metadata,
            )
            self.assertTrue(
                load_signature_manifest(
                    manifest_path,
                    input_paths=[input_path],
                    source_paths=[source_path],
                    metadata=metadata,
                )
            )

            time.sleep(0.01)
            input_path.write_text("b\n", encoding="utf-8")
            self.assertFalse(
                load_signature_manifest(
                    manifest_path,
                    input_paths=[input_path],
                    source_paths=[source_path],
                    metadata=metadata,
                )
            )

    def test_signature_manifest_invalidates_on_nested_directory_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "db"
            nested_dir = input_dir / "release"
            output_path = root / "output.tsv"
            source_path = root / "script.py"
            manifest_path = root / "cache.manifest.json"
            nested_dir.mkdir(parents=True)
            (nested_dir / "index.tsv").write_text("a\n", encoding="utf-8")
            output_path.write_text("result\n", encoding="utf-8")
            source_path.write_text("print('x')\n", encoding="utf-8")

            write_signature_manifest(
                manifest_path,
                input_paths=[input_dir],
                output_paths=[output_path],
                source_paths=[source_path],
                metadata={},
            )
            self.assertTrue(
                load_signature_manifest(
                    manifest_path,
                    input_paths=[input_dir],
                    source_paths=[source_path],
                    metadata={},
                )
            )

            time.sleep(0.01)
            (nested_dir / "index.tsv").write_text("b\n", encoding="utf-8")
            self.assertFalse(
                load_signature_manifest(
                    manifest_path,
                    input_paths=[input_dir],
                    source_paths=[source_path],
                    metadata={},
                )
            )

    def test_threshold_flip_table_handles_nan_inputs_and_recomputes_default_status(self) -> None:
        import pandas as pd

        scored = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "member_count_train": float("nan"),
                    "n_countries_train": 1.0,
                    "n_new_countries": 2.0,
                    "priority_index": 0.7,
                    "spread_label": 999.0,
                }
            ]
        )

        result = build_threshold_flip_table(
            scored, candidate_ids=["AA001"], thresholds=(1, 2, 3, 4), default_threshold=3
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.loc[0, "member_count_train"]), 0)
        self.assertEqual(int(result.loc[0, "label_ge_2"]), 1)
        self.assertEqual(int(result.loc[0, "label_ge_3"]), 0)
        self.assertEqual(int(result.loc[0, "spread_label_default"]), 0)
        self.assertEqual(int(result.loc[0, "threshold_flip_count"]), 2)

    def test_candidate_risk_table_tolerates_missing_source_columns_and_freeze_rank(self) -> None:
        import pandas as pd

        dossier = pd.DataFrame(
            [
                {
                    "backbone_id": "AA002",
                    "candidate_confidence_tier": "watchlist",
                    "coherence_score": 0.4,
                    "member_count_train": 1,
                    "n_countries_train": 1,
                    "support_profile_available": True,
                    "external_support_modalities_count": 0,
                    "primary_minus_conservative_prediction": 0.2,
                }
            ]
        )

        result = build_candidate_risk_table(dossier)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "backbone_id"], "AA002")
        self.assertEqual(result.loc[0, "false_positive_risk_tier"], "high")
        self.assertIn("weak_external_support_risk", result.loc[0, "risk_flags"])
        self.assertIn("proxy_gap_risk", result.loc[0, "risk_flags"])
        self.assertIn("candidate_confidence_score", result.columns)
        self.assertIn("low_candidate_confidence_risk", result.columns)
        self.assertTrue(bool(result.loc[0, "low_candidate_confidence_risk"]))

    def test_candidate_tick_label_accepts_namedtuple_rows(self) -> None:
        from collections import namedtuple

        Row = namedtuple("Row", ["backbone_id"])

        self.assertEqual(_candidate_tick_label(Row(backbone_id="AA003")), "AA003")

    def test_candidate_brief_table_uses_explicit_novelty_watchlist_language(self) -> None:
        import pandas as pd

        candidate_portfolio = pd.DataFrame(
            [
                {
                    "portfolio_track": "novel_signal",
                    "track_rank": 1,
                    "backbone_id": "AA276",
                    "uncertainty_review_tier": "review",
                    "member_count_train": 1,
                    "n_countries_train": 1,
                    "n_new_countries": 4,
                    "candidate_confidence_score": 0.63,
                    "candidate_explanation_summary": "Confidence 0.63; primary mobility; review review.",
                    "bootstrap_top_10_frequency": 0.68,
                    "variant_top_10_frequency": 0.63,
                    "multiverse_stability_score": 0.66,
                    "multiverse_stability_tier": "moderately_stable",
                    "source_support_tier": "refseq_dominant",
                    "evidence_tier": "novelty_watchlist",
                    "action_tier": "low_confidence_backlog",
                    "in_consensus_top50": True,
                    "consensus_rank": 41,
                }
            ]
        )
        backbones = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq1",
                    "backbone_id": "AA276",
                    "resolved_year": 2014,
                    "country": "TR",
                    "species": "Klebsiella pneumoniae",
                    "genus": "Klebsiella",
                    "primary_replicon": "IncFIB",
                    "record_origin": "RefSeq",
                },
                {
                    "sequence_accession": "seq2",
                    "backbone_id": "AA276",
                    "resolved_year": 2018,
                    "country": "DE",
                    "species": "Klebsiella pneumoniae",
                    "genus": "Klebsiella",
                    "primary_replicon": "IncFIB",
                    "record_origin": "RefSeq",
                },
            ]
        )
        amr_consensus = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq1",
                    "amr_gene_symbols": "blaOXA-1",
                    "amr_drug_classes": "BETA-LACTAM",
                }
            ]
        )
        model_selection_summary = pd.DataFrame(
            [
                {
                    "published_primary_model": "primary_priority",
                    "conservative_model_name": "conservative_priority",
                    "governance_primary_model": "governance_priority",
                    "published_primary_decision_utility_score": 0.31,
                    "published_primary_optimal_decision_threshold": 0.42,
                    "conservative_decision_utility_score": 0.18,
                    "conservative_optimal_decision_threshold": 0.50,
                    "governance_primary_decision_utility_score": 0.29,
                    "governance_primary_optimal_decision_threshold": 0.38,
                }
            ]
        )
        decision_yield = pd.DataFrame(
            [
                {
                    "model_name": "primary_priority",
                    "top_k": 10,
                    "precision_at_k": 0.50,
                    "recall_at_k": 0.60,
                },
                {
                    "model_name": "primary_priority",
                    "top_k": 25,
                    "precision_at_k": 0.44,
                    "recall_at_k": 0.78,
                },
                {
                    "model_name": "conservative_priority",
                    "top_k": 10,
                    "precision_at_k": 0.42,
                    "recall_at_k": 0.52,
                },
                {
                    "model_name": "governance_priority",
                    "top_k": 10,
                    "precision_at_k": 0.57,
                    "recall_at_k": 0.69,
                },
            ]
        )
        benchmark_protocol = pd.DataFrame(
            [
                {"benchmark_role": "primary_benchmark"},
                {"benchmark_role": "governance_benchmark"},
                {"benchmark_role": "conservative_benchmark"},
                {"benchmark_role": "counts_baseline"},
                {"benchmark_role": "source_control"},
            ]
        )
        official_context = build_reports_script._build_official_benchmark_context(
            model_selection_summary,
            decision_yield,
            benchmark_protocol=benchmark_protocol,
        )
        candidate_universe = build_reports_script._attach_official_benchmark_context(
            pd.DataFrame({"backbone_id": ["AA276"]}), official_context
        )
        candidate_portfolio = build_reports_script._attach_official_benchmark_context(
            candidate_portfolio, official_context
        )

        result = build_reports_script._build_candidate_brief_table(
            candidate_portfolio,
            backbones,
            amr_consensus,
            model_selection_summary=model_selection_summary,
            decision_yield=decision_yield,
        )
        case_studies = build_reports_script._build_candidate_case_studies(result, per_track=1)
        summary_en = str(result.loc[0, "candidate_summary_en"])
        summary_tr = str(result.loc[0, "candidate_summary_tr"])

        self.assertIn("Confidence score", summary_en)
        self.assertIn("Rank stability", summary_en)
        self.assertIn("Official benchmark yield", summary_en)
        self.assertIn("Decision utility", summary_en)
        self.assertIn("Case summary", summary_en)
        self.assertIn("coklu model uzlasi top-50", summary_tr)
        self.assertIn("ayri erken sinyal izleme hatti", summary_tr)
        self.assertIn("Resmi benchmark verimi", summary_tr)
        self.assertIn("Karar faydasi", summary_tr)
        self.assertNotIn("Consensus kisa listesinde", summary_tr)
        self.assertIn("uncertainty_review_tier", result.columns)
        self.assertEqual(str(result.loc[0, "uncertainty_review_tier"]), "review")
        self.assertIn("candidate_confidence_score", result.columns)
        self.assertIn("official_primary_top_10_precision", result.columns)
        self.assertIn("official_governance_top_10_precision", result.columns)
        self.assertIn("official_benchmark_panel_size", result.columns)
        self.assertEqual(int(result.loc[0, "official_benchmark_panel_size"]), 5)
        self.assertIn("official_primary_model", candidate_universe.columns)
        self.assertIn("official_primary_decision_utility_score", candidate_universe.columns)
        self.assertIn(
            "official_primary_optimal_decision_threshold", candidate_universe.columns
        )
        self.assertEqual(
            str(candidate_universe.loc[0, "official_governance_model"]),
            "governance_priority",
        )
        self.assertIn("official_primary_model", case_studies.columns)
        self.assertIn("official_benchmark_panel_size", case_studies.columns)
        self.assertIn("official_primary_decision_utility_score", case_studies.columns)
        self.assertIn(
            "official_primary_optimal_decision_threshold", case_studies.columns
        )
        self.assertIn("candidate_explanation_summary", result.columns)
        self.assertIn("low_candidate_confidence_risk", result.columns)
        self.assertIn("bootstrap_top_10_frequency", case_studies.columns)
        self.assertIn("variant_top_10_frequency", case_studies.columns)
        self.assertIn("multiverse_stability_score", case_studies.columns)
        self.assertIn("multiverse_stability_tier", case_studies.columns)
        self.assertIn("uncertainty_review_tier", case_studies.columns)
        self.assertEqual(str(case_studies.loc[0, "uncertainty_review_tier"]), "review")
        self.assertIn("candidate_confidence_score", case_studies.columns)
        self.assertIn("candidate_explanation_summary", case_studies.columns)
        self.assertIn("low_candidate_confidence_risk", case_studies.columns)
        self.assertIn("Belirsizlik inceleme seviyesi", str(result.loc[0, "candidate_summary_tr"]))
        self.assertIn("Guven skoru", str(result.loc[0, "candidate_summary_tr"]))
        self.assertIn("Vaka ozeti", str(result.loc[0, "candidate_summary_tr"]))

    def test_report_overview_table_summarizes_core_outputs(self) -> None:
        import pandas as pd

        model_selection_summary = pd.DataFrame(
            [
                {
                    "published_primary_model": "primary_priority",
                    "governance_primary_model": "governance_priority",
                    "conservative_model_name": "conservative_priority",
                    "published_primary_roc_auc": 0.75,
                    "published_primary_average_precision": 0.68,
                    "governance_primary_roc_auc": 0.72,
                    "published_primary_decision_utility_score": 0.31,
                    "published_primary_optimal_decision_threshold": 0.42,
                    "governance_primary_decision_utility_score": 0.29,
                    "governance_primary_optimal_decision_threshold": 0.38,
                }
            ]
        )
        decision_yield = pd.DataFrame(
            [
                {
                    "model_name": "primary_priority",
                    "top_k": 10,
                    "precision_at_k": 0.50,
                    "recall_at_k": 0.60,
                },
                {
                    "model_name": "governance_priority",
                    "top_k": 10,
                    "precision_at_k": 0.57,
                    "recall_at_k": 0.69,
                },
            ]
        )
        threshold_utility = pd.DataFrame(
            [
                {
                    "model_name": "primary_priority",
                    "optimal_threshold": 0.42,
                    "optimal_threshold_utility_per_sample": 0.31,
                },
                {
                    "model_name": "governance_priority",
                    "optimal_threshold": 0.38,
                    "optimal_threshold_utility_per_sample": 0.29,
                },
            ]
        )
        overview = build_report_overview_table(
            model_selection_summary=model_selection_summary,
            decision_yield=decision_yield,
            threshold_utility_summary=threshold_utility,
            candidate_portfolio=pd.DataFrame({"backbone_id": ["aa1", "aa2"]}),
            candidate_case_studies=pd.DataFrame({"backbone_id": ["aa1"]}),
            false_negative_audit=pd.DataFrame({"backbone_id": []}),
        )
        self.assertIn("panel_item", overview.columns)
        self.assertIn("metric_value", overview.columns)
        self.assertIn("primary_priority_utility", set(overview["panel_item"].astype(str)))
        self.assertIn("candidate_portfolio_size", set(overview["panel_item"].astype(str)))

    def test_report_artifact_validator_rejects_duplicates(self) -> None:
        import pandas as pd

        with self.assertRaises(ValueError):
            validate_report_artifact(
                pd.DataFrame({"backbone_id": ["a", "a"], "candidate_confidence_score": [0.3, 0.4]}),
                artifact_name="candidate_portfolio",
                required_columns=("backbone_id", "candidate_confidence_score"),
                unique_key="backbone_id",
                probability_columns=("candidate_confidence_score",),
            )

    def test_plot_calibration_threshold_summary_writes_compact_figure(self) -> None:
        import pandas as pd

        calibration = pd.DataFrame(
            [
                {"mean_prediction": 0.12, "observed_rate": 0.10, "n_backbones": 12},
                {"mean_prediction": 0.34, "observed_rate": 0.30, "n_backbones": 18},
                {"mean_prediction": 0.66, "observed_rate": 0.70, "n_backbones": 22},
            ]
        )
        threshold_sensitivity = pd.DataFrame(
            [
                {
                    "new_country_threshold": 2,
                    "roc_auc": 0.74,
                    "roc_auc_ci_lower": 0.70,
                    "roc_auc_ci_upper": 0.78,
                    "average_precision": 0.63,
                    "n_eligible_backbones": 120,
                },
                {
                    "new_country_threshold": 3,
                    "roc_auc": 0.77,
                    "roc_auc_ci_lower": 0.73,
                    "roc_auc_ci_upper": 0.81,
                    "average_precision": 0.66,
                    "n_eligible_backbones": 115,
                },
                {
                    "new_country_threshold": 4,
                    "roc_auc": 0.73,
                    "roc_auc_ci_lower": 0.69,
                    "roc_auc_ci_upper": 0.77,
                    "average_precision": 0.61,
                    "n_eligible_backbones": 108,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "calibration_threshold_summary.png"
            plot_calibration_threshold_summary(
                calibration,
                threshold_sensitivity,
                output_path,
                "primary_model",
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_candidate_evidence_matrix_combines_portfolio_and_threshold_context(self) -> None:
        import pandas as pd

        candidate_portfolio = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "portfolio_track": "established_high_risk",
                    "track_rank": 1,
                    "candidate_confidence_score": 0.86,
                    "candidate_explanation_summary": "Confidence 0.86; primary mobility; review clear.",
                    "candidate_confidence_tier": "tier_a",
                    "evidence_tier": "tier_a",
                    "action_tier": "core_surveillance",
                    "false_positive_risk_tier": "low",
                    "risk_flag_count": 1,
                    "consensus_rank": 3,
                    "consensus_support_count": 2,
                    "primary_model_candidate_score": 0.91,
                    "baseline_both_candidate_score": 0.72,
                    "novelty_margin_vs_baseline": 0.19,
                    "operational_risk_score": 0.84,
                    "risk_spread_probability": 0.93,
                    "risk_uncertainty": 0.12,
                    "risk_decision_tier": "action",
                    "risk_abstain_flag": False,
                    "bootstrap_top_10_frequency": 0.9,
                    "variant_top_10_frequency": 0.8,
                    "external_support_modalities_count": 2,
                    "source_support_tier": "cross_source_supported",
                    "module_f_enriched_signature_count": 1,
                    "n_new_countries": 4,
                    "spread_label": 1,
                }
            ]
        )
        candidate_briefs = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "dominant_genus": "Escherichia",
                    "dominant_species": "Escherichia coli",
                    "top_amr_classes": "BETA-LACTAM",
                    "top_amr_genes": "blaCTX-M",
                }
            ]
        )
        candidate_threshold_flip = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "threshold_flip_count": 2,
                    "eligible_for_threshold_audit": True,
                    "default_threshold": 3,
                    "spread_label_default": 1,
                    "label_ge_2": 1,
                    "label_ge_3": 1,
                    "label_ge_4": 0,
                }
            ]
        )

        matrix = build_reports_script._build_candidate_evidence_matrix(
            candidate_portfolio,
            candidate_briefs,
            candidate_threshold_flip,
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix.loc[0, "portfolio_track"], "established_high_risk")
        self.assertEqual(int(matrix.loc[0, "threshold_flip_count"]), 2)
        self.assertEqual(matrix.loc[0, "dominant_species"], "Escherichia coli")
        self.assertEqual(matrix.loc[0, "top_amr_classes"], "BETA-LACTAM")
        self.assertIn("candidate_confidence_score", matrix.columns)
        self.assertIn("candidate_explanation_summary", matrix.columns)

    def test_jury_brief_uses_guardrail_language_for_knownness_and_model_choice(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "roc_auc": 0.765,
                    "average_precision": 0.675,
                },
                {
                    "model_name": "evidence_aware_priority",
                    "roc_auc": 0.803,
                    "average_precision": 0.731,
                },
                {"model_name": "bio_clean_priority", "roc_auc": 0.768, "average_precision": 0.680},
                {"model_name": "baseline_both", "roc_auc": 0.729, "average_precision": 0.651},
                {"model_name": "source_only", "roc_auc": 0.448, "average_precision": 0.330},
            ]
        )
        country_missingness_bounds = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "eligible_for_country_bounds": True,
                    "label_observed": 1,
                    "label_midpoint": 1,
                    "label_optimistic": 1,
                    "label_weighted": 1,
                }
            ]
        )
        country_missingness_sensitivity = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "outcome_name": "label_observed",
                    "roc_auc": 0.74,
                    "average_precision": 0.63,
                },
                {
                    "model_name": "parsimonious_priority",
                    "outcome_name": "label_midpoint",
                    "roc_auc": 0.77,
                    "average_precision": 0.66,
                },
            ]
        )
        family_summary = pd.DataFrame()
        dropout_table = pd.DataFrame(
            [
                {"feature_name": "__full_model__", "roc_auc_drop_vs_full": 0.0},
                {"feature_name": "T_eff_norm", "roc_auc_drop_vs_full": 0.035},
            ]
        )
        scored = pd.DataFrame({"backbone_id": ["bb1", "bb2"]})
        candidate_portfolio = pd.DataFrame(
            {"portfolio_track": ["established_high_risk", "novel_signal"]}
        )
        decision_yield = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "top_k": 10,
                    "precision_at_k": 0.8,
                    "recall_at_k": 0.022,
                },
                {
                    "model_name": "parsimonious_priority",
                    "top_k": 25,
                    "precision_at_k": 0.92,
                    "recall_at_k": 0.064,
                },
                {
                    "model_name": "evidence_aware_priority",
                    "top_k": 10,
                    "precision_at_k": 1.0,
                    "recall_at_k": 0.028,
                },
                {
                    "model_name": "bio_clean_priority",
                    "top_k": 10,
                    "precision_at_k": 0.9,
                    "recall_at_k": 0.025,
                },
                {
                    "model_name": "baseline_both",
                    "top_k": 10,
                    "precision_at_k": 0.9,
                    "recall_at_k": 0.025,
                },
            ]
        )
        model_selection_summary = pd.DataFrame(
            [
                {
                    "selection_rationale": "published primary chosen for simpler, more interpretable, lower-proxy headline reporting; the strongest support-heavy alternative overlaps on only 0/10 top candidates, recovering to 9/25 and 26/50, so the audit keeps both views explicit",
                    "primary_vs_strongest_top_10_overlap_count": 0,
                    "primary_vs_strongest_top_25_overlap_count": 9,
                    "primary_vs_strongest_top_50_overlap_count": 26,
                    "governance_primary_model": "evidence_aware_priority",
                    "governance_selection_rationale": "governance track is kept separate from discovery so that guardrail loss, not headline AUC, governs the policy readout",
                }
            ]
        )
        model_selection_scorecard = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "selection_rank": 2,
                    "strict_knownness_acceptance_flag": True,
                    "knownness_matched_gap": -0.002,
                    "source_holdout_gap": -0.004,
                    "guardrail_loss": 0.006,
                    "governance_priority_score": 0.754,
                    "leakage_review_required": False,
                },
                {
                    "model_name": "evidence_aware_priority",
                    "selection_rank": 1,
                    "strict_knownness_acceptance_flag": False,
                    "knownness_matched_gap": -0.018,
                    "source_holdout_gap": -0.007,
                    "guardrail_loss": 0.025,
                    "governance_priority_score": 0.778,
                    "leakage_review_required": False,
                },
            ]
        )
        knownness_summary = pd.DataFrame(
            [
                {
                    "lowest_knownness_quartile_primary_roc_auc": 0.593,
                    "top_k_lower_half_knownness_count": 0,
                }
            ]
        )
        source_balance_resampling = pd.DataFrame({"roc_auc": [0.70, 0.71]})
        novelty_specialist_metrics = pd.DataFrame(
            [
                {
                    "cohort_name": "lowest_knownness_quartile",
                    "model_name": "novelty_specialist_priority",
                    "status": "ok",
                    "roc_auc": 0.679,
                }
            ]
        )
        adaptive_gated_metrics = pd.DataFrame(
            [
                {
                    "model_name": "adaptive_natural_priority",
                    "status": "ok",
                    "roc_auc": 0.811,
                    "average_precision": 0.716,
                }
            ]
        )
        operational_risk_watchlist = pd.DataFrame(
            [
                {"backbone_id": "bb1", "risk_decision_tier": "action"},
                {"backbone_id": "bb2", "risk_decision_tier": "review"},
                {"backbone_id": "bb3", "risk_decision_tier": "abstain"},
            ]
        )
        blocked_holdout = pd.DataFrame(
            [
                {
                    "model_name": "parsimonious_priority",
                    "blocked_holdout_group_columns": "dominant_source,dominant_region_train",
                    "blocked_holdout_roc_auc": 0.76,
                    "blocked_holdout_group_count": 4,
                    "worst_blocked_holdout_group": "dominant_source:insd_leaning",
                    "worst_blocked_holdout_group_roc_auc": 0.71,
                }
            ]
        )
        rank_stability = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "bootstrap_top_k_frequency": 0.92,
                    "bootstrap_top_10_frequency": 0.94,
                }
            ]
        )
        variant_consistency = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "variant_top_k_frequency": 0.89,
                    "variant_top_10_frequency": 0.91,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "jury_brief.md"
            build_reports_script._write_jury_brief(
                output_path,
                primary_model_name="parsimonious_priority",
                conservative_model_name="bio_clean_priority",
                model_metrics=model_metrics,
                family_summary=family_summary,
                dropout_table=dropout_table,
                scored=scored,
                candidate_portfolio=candidate_portfolio,
                decision_yield=decision_yield,
                model_selection_summary=model_selection_summary,
                model_selection_scorecard=model_selection_scorecard,
                knownness_summary=knownness_summary,
                source_balance_resampling=source_balance_resampling,
                novelty_specialist_metrics=novelty_specialist_metrics,
                adaptive_gated_metrics=adaptive_gated_metrics,
                operational_risk_watchlist=operational_risk_watchlist,
                outcome_threshold=3,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                blocked_holdout_summary=blocked_holdout,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
            )
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("## Interpretation Guardrails", content)
        self.assertIn("## Formal Hypotheses", content)
        self.assertNotIn("## Why This Is Defensible", content)
        self.assertIn("Discovery benchmark", content)
        self.assertIn("Governance watch-only", content)
        self.assertIn("Governance track", content)
        self.assertIn("top-25 overlap: `9/25`", content)
        self.assertIn("top-50 overlap: `26/50`", content)
        self.assertIn("shortlist-prioritization benchmark", content)
        self.assertIn("sampling saturation / knownness signals", content)
        self.assertIn("AMRFinder is optional", content)
        self.assertIn("adaptive_natural_priority", content)
        self.assertIn("operational_risk_watchlist.tsv", content)
        self.assertIn("Current operational watchlist mix", content)
        self.assertIn("## Zero-Floor Component Behavior", content)
        self.assertIn("## OLS Residual Approach", content)
        self.assertIn("Only three models are official", content)
        self.assertIn("No external validation claim is made", content)
        self.assertIn("Blocked Holdout Audit", content)
        self.assertIn("dominant_source + dominant_region_train", content)
        self.assertIn("internal source/region stress test", content)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", content)
        self.assertIn("nonlinear_deconfounding_audit.tsv", content)
        self.assertIn("ordinal_outcome_audit.tsv", content)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", content)
        self.assertIn("macro_region_jump_outcome.tsv", content)
        self.assertIn("prospective_candidate_freeze.tsv", content)
        self.assertIn("annual_candidate_freeze_summary.tsv", content)
        self.assertIn("future_sentinel_audit.tsv", content)
        self.assertIn("mash_similarity_graph.tsv", content)
        self.assertIn("counterfactual_shortlist_comparison.tsv", content)
        self.assertIn("geographic_jump_distance_outcome.tsv", content)
        self.assertIn("amr_uncertainty_summary.tsv", content)
        self.assertIn("## Ranking Stability", content)
        self.assertIn("candidate_rank_stability.tsv", content)
        self.assertIn("candidate_variant_consistency.tsv", content)
        self.assertIn("## Release Surface", content)
        self.assertIn("blocked_holdout_summary.tsv", content)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", content)
        self.assertIn("nonlinear_deconfounding_audit.tsv", content)
        self.assertIn("ordinal_outcome_audit.tsv", content)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", content)
        self.assertIn("macro_region_jump_outcome.tsv", content)
        self.assertIn("candidate_rank_stability.tsv", content)
        self.assertIn("candidate_variant_consistency.tsv", content)
        self.assertIn("calibration_threshold_summary.png", content)
        self.assertIn("Country Missingness", content)
        self.assertIn("country_missingness_bounds.tsv", content)
        self.assertIn("country_missingness_sensitivity.tsv", content)

        bundle_jury_brief = (
            PROJECT_ROOT / "reports/release/bundle/reports/jury_brief.md"
        ).read_text(encoding="utf-8")
        self.assertIn("## Release Surface", bundle_jury_brief)
        self.assertIn("blocked_holdout_summary.tsv", bundle_jury_brief)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", bundle_jury_brief)
        self.assertIn("nonlinear_deconfounding_audit.tsv", bundle_jury_brief)
        self.assertIn("ordinal_outcome_audit.tsv", bundle_jury_brief)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", bundle_jury_brief)
        self.assertIn("macro_region_jump_outcome.tsv", bundle_jury_brief)
        self.assertIn("prospective_candidate_freeze.tsv", bundle_jury_brief)
        self.assertIn("annual_candidate_freeze_summary.tsv", bundle_jury_brief)
        self.assertIn("future_sentinel_audit.tsv", bundle_jury_brief)
        self.assertIn("mash_similarity_graph.tsv", bundle_jury_brief)
        self.assertIn("counterfactual_shortlist_comparison.tsv", bundle_jury_brief)
        self.assertIn("geographic_jump_distance_outcome.tsv", bundle_jury_brief)
        self.assertIn("amr_uncertainty_summary.tsv", bundle_jury_brief)
        self.assertIn("candidate_rank_stability.tsv", bundle_jury_brief)
        self.assertIn("candidate_variant_consistency.tsv", bundle_jury_brief)
        self.assertIn("calibration_threshold_summary.png", bundle_jury_brief)
        self.assertIn("Country Missingness", bundle_jury_brief)
        self.assertIn("country_missingness_bounds.tsv", bundle_jury_brief)
        self.assertIn("country_missingness_sensitivity.tsv", bundle_jury_brief)

    def test_executive_summary_reports_confirmatory_cohort_and_case_studies(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "seer_model",
                    "roc_auc": 0.82,
                    "average_precision": 0.74,
                    "scientific_acceptance_status": "fail",
                },
                {"model_name": "guard_model", "roc_auc": 0.79, "average_precision": 0.70},
                {"model_name": "baseline_both", "roc_auc": 0.72, "average_precision": 0.65},
            ]
        )
        country_missingness_bounds = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "eligible_for_country_bounds": True,
                    "label_observed": 1,
                    "label_midpoint": 1,
                    "label_optimistic": 1,
                    "label_weighted": 1,
                }
            ]
        )
        country_missingness_sensitivity = pd.DataFrame(
            [
                {
                    "model_name": "seer_model",
                    "outcome_name": "label_observed",
                    "roc_auc": 0.74,
                    "average_precision": 0.63,
                },
                {
                    "model_name": "seer_model",
                    "outcome_name": "label_midpoint",
                    "roc_auc": 0.77,
                    "average_precision": 0.66,
                },
            ]
        )
        confirmatory = pd.DataFrame(
            [
                {
                    "cohort_name": "confirmatory_internal",
                    "model_name": "seer_model",
                    "status": "ok",
                    "n_backbones": 42,
                    "roc_auc": 0.81,
                    "average_precision": 0.72,
                    "share_of_primary_eligible": 0.35,
                }
            ]
        )
        false_negative_audit = pd.DataFrame(
            [{"backbone_id": "bb1", "miss_driver_flags": "low_knownness,threshold_fragile"}]
        )
        case_studies = pd.DataFrame([{"backbone_id": "bb1"}])
        rank_stability = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "bootstrap_top_k_frequency": 0.91,
                    "bootstrap_top_10_frequency": 0.93,
                }
            ]
        )
        variant_consistency = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "variant_top_k_frequency": 0.88,
                    "variant_top_10_frequency": 0.90,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "executive_summary.md"
            build_reports_script._write_executive_summary(
                output_path,
                primary_model_name="seer_model",
                governance_model_name="guard_model",
                baseline_model_name="baseline_both",
                model_metrics=model_metrics,
                confirmatory_cohort_summary=confirmatory,
                false_negative_audit=false_negative_audit,
                candidate_case_studies=case_studies,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                blocked_holdout_summary=pd.DataFrame(
                    [
                        {
                            "model_name": "seer_model",
                            "blocked_holdout_group_columns": "dominant_source,dominant_region_train",
                            "blocked_holdout_roc_auc": 0.77,
                            "blocked_holdout_group_count": 3,
                            "worst_blocked_holdout_group": "dominant_region_train:Europe",
                            "worst_blocked_holdout_group_roc_auc": 0.71,
                        }
                    ]
                ),
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
            )
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("The Seer", content)
        self.assertIn("The Guard", content)
        self.assertIn("No external validation claim is made", content)
        self.assertIn("Benchmark scope:", content)
        self.assertIn("conditional benchmark candidate", content)
        self.assertIn("fixed-bin ECE", content)
        self.assertIn("Internal high-integrity subset audit", content)
        self.assertIn("## Ranking Stability", content)
        self.assertIn("candidate_rank_stability.tsv", content)
        self.assertIn("candidate_variant_consistency.tsv", content)
        self.assertIn("candidate_case_studies.tsv", content)
        self.assertIn("candidate_rank_stability.tsv", content)
        self.assertIn("candidate_variant_consistency.tsv", content)
        self.assertIn("blocked_holdout_summary.tsv", content)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", content)
        self.assertIn("nonlinear_deconfounding_audit.tsv", content)
        self.assertIn("ordinal_outcome_audit.tsv", content)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", content)
        self.assertIn("macro_region_jump_outcome.tsv", content)
        self.assertIn("prospective_candidate_freeze.tsv", content)
        self.assertIn("annual_candidate_freeze_summary.tsv", content)
        self.assertIn("future_sentinel_audit.tsv", content)
        self.assertIn("mash_similarity_graph.tsv", content)
        self.assertIn("counterfactual_shortlist_comparison.tsv", content)
        self.assertIn("geographic_jump_distance_outcome.tsv", content)
        self.assertIn("amr_uncertainty_summary.tsv", content)
        self.assertIn("calibration_threshold_summary.png", content)
        self.assertIn("Country Missingness", content)
        self.assertIn("country_missingness_bounds.tsv", content)
        self.assertIn("country_missingness_sensitivity.tsv", content)

    def test_turkish_summary_reports_release_surface_assets(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "seer_model",
                    "roc_auc": 0.82,
                    "average_precision": 0.74,
                    "brier_skill_score": 0.19,
                },
                {
                    "model_name": "guard_model",
                    "roc_auc": 0.79,
                    "average_precision": 0.70,
                    "brier_skill_score": 0.17,
                },
                {
                    "model_name": "baseline_both",
                    "roc_auc": 0.72,
                    "average_precision": 0.65,
                    "brier_skill_score": 0.0,
                },
            ]
        )
        country_missingness_bounds = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "eligible_for_country_bounds": True,
                    "label_observed": 1,
                    "label_midpoint": 1,
                    "label_optimistic": 1,
                    "label_weighted": 1,
                }
            ]
        )
        country_missingness_sensitivity = pd.DataFrame(
            [
                {
                    "model_name": "seer_model",
                    "outcome_name": "label_observed",
                    "roc_auc": 0.74,
                    "average_precision": 0.63,
                },
                {
                    "model_name": "seer_model",
                    "outcome_name": "label_midpoint",
                    "roc_auc": 0.77,
                    "average_precision": 0.66,
                },
            ]
        )
        blocked_holdout = pd.DataFrame(
            [
                {
                    "model_name": "seer_model",
                    "blocked_holdout_group_columns": "dominant_source,dominant_region_train",
                    "blocked_holdout_roc_auc": 0.77,
                    "blocked_holdout_group_count": 3,
                    "worst_blocked_holdout_group": "dominant_region_train:Europe",
                    "worst_blocked_holdout_group_roc_auc": 0.71,
                }
            ]
        )
        rank_stability = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "bootstrap_top_k_frequency": 0.91,
                    "bootstrap_top_10_frequency": 0.93,
                }
            ]
        )
        variant_consistency = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "top_k": 10,
                    "variant_top_k_frequency": 0.88,
                    "variant_top_10_frequency": 0.90,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ozet_tr.md"
            build_reports_script._write_turkish_summary(
                output_path,
                primary_model_name="seer_model",
                conservative_model_name="guard_model",
                model_metrics=model_metrics,
                candidate_briefs=pd.DataFrame(),
                candidate_portfolio=pd.DataFrame(),
                decision_yield=pd.DataFrame(),
                model_selection_summary=pd.DataFrame(),
                knownness_summary=pd.DataFrame(),
                source_balance_resampling=pd.DataFrame(),
                novelty_specialist_metrics=pd.DataFrame(),
                adaptive_gated_metrics=pd.DataFrame(),
                outcome_threshold=3,
                blocked_holdout_summary=blocked_holdout,
                country_missingness_bounds=country_missingness_bounds,
                country_missingness_sensitivity=country_missingness_sensitivity,
                rank_stability=rank_stability,
                variant_consistency=variant_consistency,
            )
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("## Sürüm Yüzeyi", content)
        self.assertIn("## Sıralama Kararlılığı", content)
        self.assertIn("blocked_holdout_summary.tsv", content)
        self.assertIn("frozen_scientific_acceptance_audit.tsv", content)
        self.assertIn("nonlinear_deconfounding_audit.tsv", content)
        self.assertIn("ordinal_outcome_audit.tsv", content)
        self.assertIn("exposure_adjusted_event_outcomes.tsv", content)
        self.assertIn("macro_region_jump_outcome.tsv", content)
        self.assertIn("prospective_candidate_freeze.tsv", content)
        self.assertIn("annual_candidate_freeze_summary.tsv", content)
        self.assertIn("future_sentinel_audit.tsv", content)
        self.assertIn("mash_similarity_graph.tsv", content)
        self.assertIn("counterfactual_shortlist_comparison.tsv", content)
        self.assertIn("geographic_jump_distance_outcome.tsv", content)
        self.assertIn("amr_uncertainty_summary.tsv", content)
        self.assertIn("candidate_rank_stability.tsv", content)
        self.assertIn("candidate_variant_consistency.tsv", content)
        self.assertIn("calibration_threshold_summary.png", content)
        self.assertIn("bloke edilmiş holdout denetimi", content)
        self.assertIn("dominant_source + dominant_region_train", content)
        self.assertIn("country_missingness_bounds.tsv", content)
        self.assertIn("country_missingness_sensitivity.tsv", content)
        self.assertIn("ülke eksikliği varsayımlarına göre", content)

    def test_operational_risk_watchlist_prefers_action_rows_and_keeps_primary_model(self) -> None:
        import pandas as pd

        risk_dictionary = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "model_name": "primary_model",
                    "operational_risk_score": 0.91,
                    "risk_spread_probability": 0.88,
                    "risk_event_within_3y": 0.82,
                    "risk_macro_region_jump_3y": 0.79,
                    "risk_three_countries_within_5y": 0.76,
                    "risk_uncertainty": 0.20,
                    "risk_decision_tier": "action",
                    "knownness_score": 0.4,
                    "source_band": "source_mixed",
                },
                {
                    "backbone_id": "bb2",
                    "model_name": "primary_model",
                    "operational_risk_score": 0.85,
                    "risk_spread_probability": 0.80,
                    "risk_event_within_3y": 0.75,
                    "risk_macro_region_jump_3y": 0.60,
                    "risk_three_countries_within_5y": 0.55,
                    "risk_uncertainty": 0.15,
                    "risk_decision_tier": "review",
                    "knownness_score": 0.7,
                    "source_band": "cross_source_supported",
                },
                {
                    "backbone_id": "bb3",
                    "model_name": "other_model",
                    "operational_risk_score": 0.99,
                    "risk_spread_probability": 0.99,
                    "risk_event_within_3y": 0.99,
                    "risk_macro_region_jump_3y": 0.99,
                    "risk_three_countries_within_5y": 0.99,
                    "risk_uncertainty": 0.05,
                    "risk_decision_tier": "action",
                },
            ]
        )
        candidate_portfolio = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "portfolio_track": "established_high_risk",
                    "track_rank": 1,
                    "candidate_confidence_tier": "tier_a",
                },
                {
                    "backbone_id": "bb2",
                    "portfolio_track": "novel_signal",
                    "track_rank": 1,
                    "candidate_confidence_tier": "watchlist",
                },
            ]
        )

        watchlist = build_reports_script._build_operational_risk_watchlist(
            risk_dictionary,
            primary_model_name="primary_model",
            candidate_portfolio=candidate_portfolio,
            top_k=10,
        )

        self.assertEqual(watchlist["backbone_id"].tolist(), ["bb1", "bb2"])
        self.assertEqual(str(watchlist.loc[0, "risk_decision_tier"]), "action")
        self.assertIn("portfolio_track", watchlist.columns)

    def test_operational_risk_watchlist_is_tiered_across_action_review_abstain(self) -> None:
        import pandas as pd

        risk_dictionary = pd.DataFrame(
            [
                {
                    "backbone_id": f"bb{i}",
                    "model_name": "primary_model",
                    "operational_risk_score": 0.95 - i * 0.01,
                    "risk_spread_probability": 0.90 - i * 0.01,
                    "risk_event_within_3y": 0.80 - i * 0.01,
                    "risk_macro_region_jump_3y": 0.75 - i * 0.01,
                    "risk_three_countries_within_5y": 0.70 - i * 0.01,
                    "risk_uncertainty": 0.10 + i * 0.02,
                    "risk_uncertainty_quantile": 0.10 + i * 0.08,
                    "risk_decision_tier": "action" if i < 6 else "review" if i < 10 else "abstain",
                    "knownness_score": 0.5,
                    "source_band": "source_mixed",
                }
                for i in range(12)
            ]
        )

        watchlist = build_reports_script._build_operational_risk_watchlist(
            risk_dictionary,
            primary_model_name="primary_model",
            top_k=6,
        )

        self.assertEqual(len(watchlist), 6)
        self.assertIn("risk_uncertainty_quantile", watchlist.columns)
        self.assertTrue(
            {"action", "review", "abstain"}.issubset(
                set(watchlist["risk_decision_tier"].astype(str))
            )
        )
        self.assertEqual(watchlist["operational_risk_rank"].tolist(), list(range(1, 7)))

    def test_candidate_multiverse_stability_requires_more_than_threshold_only_signal(self) -> None:
        import pandas as pd

        candidate_stability = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "bootstrap_top_10_frequency": float("nan"),
                    "bootstrap_top_25_frequency": float("nan"),
                    "variant_top_10_frequency": float("nan"),
                    "variant_top_25_frequency": float("nan"),
                    "primary_model_candidate_score": 0.8,
                },
                {
                    "backbone_id": "AA002",
                    "bootstrap_top_10_frequency": 0.9,
                    "bootstrap_top_25_frequency": 0.9,
                    "variant_top_10_frequency": 0.8,
                    "variant_top_25_frequency": 0.8,
                    "primary_model_candidate_score": 0.7,
                },
            ]
        )
        candidate_threshold_flip = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "threshold_flip_count": 0,
                    "eligible_for_threshold_audit": True,
                },
                {
                    "backbone_id": "AA002",
                    "threshold_flip_count": 0,
                    "eligible_for_threshold_audit": True,
                },
            ]
        )

        result = build_reports_script._build_candidate_multiverse_stability(
            candidate_stability,
            candidate_threshold_flip,
        )

        threshold_only = result.loc[result["backbone_id"] == "AA001"].iloc[0]
        fully_supported = result.loc[result["backbone_id"] == "AA002"].iloc[0]

        self.assertAlmostEqual(float(threshold_only["bootstrap_top_25_frequency"]), 0.0)
        self.assertAlmostEqual(float(threshold_only["variant_top_25_frequency"]), 0.0)
        self.assertAlmostEqual(
            float(threshold_only["multiverse_stability_score"]), 1.0 / 3.0, places=6
        )
        self.assertEqual(str(threshold_only["multiverse_stability_tier"]), "fragile")
        self.assertEqual(int(threshold_only["multiverse_component_count"]), 3)

        self.assertGreater(float(fully_supported["multiverse_stability_score"]), 0.8)
        self.assertEqual(str(fully_supported["multiverse_stability_tier"]), "stable")

    def test_claim_discipline_failed_model_not_presented_as_accepted(self) -> None:
        """Test that a failed strict acceptance model cannot be presented as accepted in narrative."""
        from plasmid_priority.reporting.narrative_utils import benchmark_scope_note, strict_acceptance_status
        import pandas as pd

        # Test that failed status produces conditional language
        failed_row = pd.Series({"scientific_acceptance_status": "fail"})
        failed_status = strict_acceptance_status(failed_row)
        failed_note = benchmark_scope_note(failed_status)
        
        # Failed models should produce conditional language
        self.assertIn("conditional", failed_note.lower())
        self.assertIn("benchmark-limited", failed_note.lower())
        self.assertNotIn("accepted", failed_note.lower())

        # Test that passed status allows accepted language
        passed_row = pd.Series({"scientific_acceptance_status": "pass"})
        passed_status = strict_acceptance_status(passed_row)
        passed_note = benchmark_scope_note(passed_status)
        
        # Passed models should reference the benchmark contract
        self.assertIn("accepted language is allowed", passed_note.lower())
        self.assertIn("benchmark contract", passed_note.lower())

        # Test that not_scored status produces conditional language
        not_scored_row = pd.Series({"scientific_acceptance_status": "not_scored"})
        not_scored_status = strict_acceptance_status(not_scored_row)
        not_scored_note = benchmark_scope_note(not_scored_status)
        
        # Not-scored models should produce conditional language
        self.assertIn("conditional", not_scored_note.lower())
        self.assertIn("benchmark-limited", not_scored_note.lower())


if __name__ == "__main__":
    unittest.main()
