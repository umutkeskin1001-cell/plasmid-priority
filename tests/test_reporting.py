from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_BUILD_REPORTS_SPEC = importlib.util.spec_from_file_location(
    "build_reports_script",
    PROJECT_ROOT / "scripts/24_build_reports.py",
)
assert _BUILD_REPORTS_SPEC is not None and _BUILD_REPORTS_SPEC.loader is not None
build_reports_script = importlib.util.module_from_spec(_BUILD_REPORTS_SPEC)
_BUILD_REPORTS_SPEC.loader.exec_module(build_reports_script)

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.reporting.model_audit import build_candidate_risk_table, build_threshold_flip_table
from plasmid_priority.reporting.figures import _candidate_tick_label
from plasmid_priority.utils.files import atomic_write_json


class ReportingTests(unittest.TestCase):
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
                json.dumps({"version": 1, "created_on": "2026-03-22", "download_date": "2026-03-22", "assets": []}),
                encoding="utf-8",
            )
            context = build_context(root)
            with ManagedScriptRun(context, "unit_test_script") as run:
                run.note("hello")
            summary_path = root / "data/tmp/logs/unit_test_script_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["notes"], ["hello"])

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

        result = build_threshold_flip_table(scored, candidate_ids=["AA001"], thresholds=(1, 2, 3, 4), default_threshold=3)

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
                    "member_count_train": 1,
                    "n_countries_train": 1,
                    "n_new_countries": 4,
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
            [{"sequence_accession": "seq1", "amr_gene_symbols": "blaOXA-1", "amr_drug_classes": "BETA-LACTAM"}]
        )

        result = build_reports_script._build_candidate_brief_table(candidate_portfolio, backbones, amr_consensus)
        summary_tr = str(result.loc[0, "candidate_summary_tr"])

        self.assertIn("coklu model uzlasi top-50", summary_tr)
        self.assertIn("ayri erken sinyal izleme hatti", summary_tr)
        self.assertNotIn("Consensus kisa listesinde", summary_tr)

    def test_jury_brief_uses_guardrail_language_for_knownness_and_model_choice(self) -> None:
        import pandas as pd

        model_metrics = pd.DataFrame(
            [
                {"model_name": "parsimonious_priority", "roc_auc": 0.765, "average_precision": 0.675},
                {"model_name": "evidence_aware_priority", "roc_auc": 0.803, "average_precision": 0.731},
                {"model_name": "bio_clean_priority", "roc_auc": 0.768, "average_precision": 0.680},
                {"model_name": "baseline_both", "roc_auc": 0.729, "average_precision": 0.651},
                {"model_name": "source_only", "roc_auc": 0.448, "average_precision": 0.330},
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
        candidate_portfolio = pd.DataFrame({"portfolio_track": ["established_high_risk", "novel_signal"]})
        decision_yield = pd.DataFrame(
            [
                {"model_name": "parsimonious_priority", "top_k": 10, "precision_at_k": 0.8, "recall_at_k": 0.022},
                {"model_name": "parsimonious_priority", "top_k": 25, "precision_at_k": 0.92, "recall_at_k": 0.064},
                {"model_name": "evidence_aware_priority", "top_k": 10, "precision_at_k": 1.0, "recall_at_k": 0.028},
                {"model_name": "bio_clean_priority", "top_k": 10, "precision_at_k": 0.9, "recall_at_k": 0.025},
                {"model_name": "baseline_both", "top_k": 10, "precision_at_k": 0.9, "recall_at_k": 0.025},
            ]
        )
        model_selection_summary = pd.DataFrame(
            [
                {
                    "selection_rationale": "published primary chosen for simpler, more interpretable, lower-proxy headline reporting; the strongest support-heavy alternative overlaps on only 0/10 top candidates, recovering to 9/25 and 26/50, so the audit keeps both views explicit",
                    "primary_vs_strongest_top_10_overlap_count": 0,
                    "primary_vs_strongest_top_25_overlap_count": 9,
                    "primary_vs_strongest_top_50_overlap_count": 26,
                }
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
                knownness_summary=knownness_summary,
                source_balance_resampling=source_balance_resampling,
                novelty_specialist_metrics=novelty_specialist_metrics,
                adaptive_gated_metrics=adaptive_gated_metrics,
                outcome_threshold=3,
            )
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("## Interpretation Guardrails", content)
        self.assertIn("## Formal Hypotheses", content)
        self.assertNotIn("## Why This Is Defensible", content)
        self.assertIn("top-25 overlap: `9/25`", content)
        self.assertIn("top-50 overlap: `26/50`", content)
        self.assertIn("shortlist-prioritization benchmark", content)
        self.assertIn("sampling saturation / knownness signals", content)
        self.assertIn("AMRFinder is optional", content)
        self.assertIn("adaptive_natural_priority", content)
        self.assertIn("## Zero-Floor Component Behavior", content)
        self.assertIn("## OLS Residual Approach", content)

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
                {"backbone_id": "AA001", "threshold_flip_count": 0, "eligible_for_threshold_audit": True},
                {"backbone_id": "AA002", "threshold_flip_count": 0, "eligible_for_threshold_audit": True},
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
        self.assertAlmostEqual(float(threshold_only["multiverse_stability_score"]), 1.0 / 3.0, places=6)
        self.assertEqual(str(threshold_only["multiverse_stability_tier"]), "fragile")
        self.assertEqual(int(threshold_only["multiverse_component_count"]), 3)

        self.assertGreater(float(fully_supported["multiverse_stability_score"]), 0.8)
        self.assertEqual(str(fully_supported["multiverse_stability_tier"]), "stable")


if __name__ == "__main__":
    unittest.main()
