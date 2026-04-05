from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReportSchemaTests(unittest.TestCase):
    def _read_table(self, relative_path: str) -> pd.DataFrame:
        path = PROJECT_ROOT / relative_path
        self.assertTrue(path.exists(), f"missing generated artifact: {relative_path}")
        return pd.read_csv(path, sep="\t")

    def test_model_metrics_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/model_metrics.tsv")
        required = {"model_name", "roc_auc", "average_precision", "precision_at_top_25"}
        self.assertTrue(required.issubset(frame.columns))
        self.assertFalse(frame.empty)

    def test_model_selection_scorecard_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/model_selection_scorecard.tsv")
        required = {
            "model_name",
            "selection_rank",
            "selection_composite_score",
            "prediction_vs_knownness_spearman",
            "guardrail_loss",
            "governance_priority_score",
            "governance_rank",
        }
        self.assertTrue(required.issubset(frame.columns))
        self.assertGreaterEqual(int(frame["selection_rank"].min()), 1)

    def test_model_selection_summary_has_governance_columns(self) -> None:
        frame = self._read_table("reports/core_tables/model_selection_summary.tsv")
        required = {
            "published_primary_model",
            "published_primary_track",
            "governance_primary_model",
            "governance_primary_track",
            "governance_primary_guardrail_loss",
            "governance_primary_governance_priority_score",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_candidate_portfolio_has_decision_columns(self) -> None:
        frame = self._read_table("reports/core_tables/candidate_portfolio.tsv")
        required = {
            "backbone_id",
            "portfolio_track",
            "candidate_confidence_tier",
            "candidate_confidence_score",
            "candidate_explanation_summary",
            "multiverse_stability_score",
            "multiverse_stability_tier",
            "recommended_monitoring_tier",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_benchmark_protocol_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/benchmark_protocol.tsv")
        required = {
            "model_name",
            "benchmark_role",
            "benchmark_status",
            "benchmark_track",
            "model_family",
            "roc_auc",
            "average_precision",
            "selection_rationale",
            "benchmark_guardrail_status",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_operational_risk_watchlist_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/operational_risk_watchlist.tsv")
        required = {
            "backbone_id",
            "operational_risk_rank",
            "operational_risk_score",
            "risk_event_within_3y",
            "risk_macro_region_jump_3y",
            "risk_three_countries_within_5y",
            "risk_uncertainty_quantile",
            "risk_uncertainty",
            "risk_decision_tier",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_confirmatory_cohort_summary_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/confirmatory_cohort_summary.tsv")
        required = {
            "cohort_name",
            "model_name",
            "status",
            "n_backbones",
            "positive_prevalence",
            "share_of_primary_eligible",
        }
        self.assertTrue(required.issubset(frame.columns))

    def test_candidate_case_studies_has_required_columns(self) -> None:
        frame = self._read_table("reports/core_tables/candidate_case_studies.tsv")
        required = {
            "portfolio_track",
            "track_rank",
            "backbone_id",
            "candidate_summary_en",
            "candidate_summary_tr",
            "candidate_confidence_score",
            "candidate_explanation_summary",
            "bootstrap_top_10_frequency",
            "variant_top_10_frequency",
            "multiverse_stability_score",
            "multiverse_stability_tier",
        }
        self.assertTrue(required.issubset(frame.columns))


if __name__ == "__main__":
    unittest.main()
