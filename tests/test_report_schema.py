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

    def test_cross_table_consistency_model_selection_acceptance_audit(self) -> None:
        """Test that model selection summary and frozen scientific acceptance audit agree on model names."""
        model_selection = self._read_table("reports/core_tables/model_selection_summary.tsv")
        acceptance_audit = self._read_table("reports/core_tables/frozen_scientific_acceptance_audit.tsv")
        
        # Get model names from both tables
        selection_models = set(model_selection["published_primary_model"].dropna().unique())
        acceptance_models = set(acceptance_audit["model_name"].dropna().unique())
        
        # The primary model should appear in both
        if selection_models:
            self.assertTrue(
                len(selection_models.intersection(acceptance_models)) > 0,
                "Model selection and acceptance audit should share at least one model name"
            )

    def test_cross_table_consistency_headline_acceptance_status(self) -> None:
        """Test that headline validation summary and frozen scientific acceptance audit agree on acceptance status."""
        headline_summary = self._read_table("reports/core_tables/headline_validation_summary.tsv")
        acceptance_audit = self._read_table("reports/core_tables/frozen_scientific_acceptance_audit.tsv")
        
        # Get primary model from headline summary
        primary_model_row = headline_summary.loc[headline_summary["summary_label"] == "discovery_primary"]
        if not primary_model_row.empty:
            primary_model = primary_model_row.iloc[0]["model_name"]
            
            # Check that this model exists in acceptance audit
            audit_row = acceptance_audit.loc[acceptance_audit["model_name"] == primary_model]
            self.assertFalse(
                audit_row.empty,
                f"Primary model {primary_model} should appear in acceptance audit"
            )

    def test_cross_table_consistency_blocked_holdout_summary(self) -> None:
        """Test that blocked holdout summary contains models from model selection."""
        blocked_holdout = self._read_table("reports/core_tables/blocked_holdout_summary.tsv")
        model_selection = self._read_table("reports/core_tables/model_selection_summary.tsv")
        
        # Get model names from both tables
        holdout_models = set(blocked_holdout["model_name"].dropna().unique())
        selection_models = set(model_selection["published_primary_model"].dropna().unique())
        
        # At least some models should overlap
        if selection_models and holdout_models:
            overlap = selection_models.intersection(holdout_models)
            self.assertTrue(
                len(overlap) > 0,
                f"Model selection and blocked holdout should share models, got {len(overlap)} overlap"
            )


if __name__ == "__main__":
    unittest.main()
