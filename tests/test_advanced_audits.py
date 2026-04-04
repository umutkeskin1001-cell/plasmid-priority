from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from plasmid_priority.reporting.advanced_audits import (
    build_amr_uncertainty_table,
    build_confirmatory_cohort_summary,
    build_consensus_shortlist,
    build_count_outcome_alignment,
    build_counterfactual_shortlist_comparison,
    build_country_missingness_bounds,
    build_duplicate_completeness_change_audit,
    build_event_timing_outcomes,
    build_exposure_adjusted_event_table,
    build_exposure_adjusted_outcome_audit,
    build_false_negative_audit,
    build_geographic_jump_distance_table,
    build_knownness_matched_validation,
    build_macro_region_jump_table,
    build_mash_similarity_graph_table,
    build_matched_stratum_propensity_audit,
    build_metadata_quality_table,
    build_missingness_sensitivity_performance,
    build_nonlinear_deconfounding_audit,
    build_operational_risk_dictionary,
    build_ordinal_outcome_alignment,
    build_secondary_outcome_performance,
)


class AdvancedAuditTests(unittest.TestCase):
    def test_build_macro_region_jump_table_counts_new_regions(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "resolved_year": 2014,
                    "country": "USA",
                    "TAXONOMY_family": "Enterobacteriaceae",
                    "TAXONOMY_order": "Enterobacterales",
                },
                {
                    "backbone_id": "AA001",
                    "resolved_year": 2017,
                    "country": "Germany",
                    "TAXONOMY_family": "Pseudomonadaceae",
                    "TAXONOMY_order": "Pseudomonadales",
                },
                {
                    "backbone_id": "AA001",
                    "resolved_year": 2018,
                    "country": "Canada",
                    "TAXONOMY_family": "Pseudomonadaceae",
                    "TAXONOMY_order": "Pseudomonadales",
                },
                {
                    "backbone_id": "AA002",
                    "resolved_year": 2014,
                    "country": "USA",
                    "TAXONOMY_family": "Enterobacteriaceae",
                    "TAXONOMY_order": "Enterobacterales",
                },
                {
                    "backbone_id": "AA002",
                    "resolved_year": 2017,
                    "country": "USA",
                    "TAXONOMY_family": "Enterobacteriaceae",
                    "TAXONOMY_order": "Enterobacterales",
                },
            ]
        )
        propensity = pd.DataFrame(
            [
                {
                    "resolved_year": 2017,
                    "country": "Germany",
                    "inverse_upload_weight": 0.8,
                    "rarity_weight": 0.7,
                },
                {
                    "resolved_year": 2018,
                    "country": "Canada",
                    "inverse_upload_weight": 0.5,
                    "rarity_weight": 0.3,
                },
            ]
        )

        result = build_macro_region_jump_table(
            records, propensity, split_year=2015, test_year_end=2023
        )
        row = result.loc[result["backbone_id"] == "AA001"].iloc[0]

        self.assertEqual(int(row["n_new_macro_regions"]), 1)
        self.assertEqual(int(row["macro_region_jump_label"]), 1)
        self.assertEqual(int(row["n_new_host_families"]), 1)
        self.assertEqual(int(row["host_family_jump_label"]), 1)
        self.assertGreater(float(row["weighted_new_country_burden"]), 0.0)

    def test_build_secondary_outcome_performance_returns_model_rows(self) -> None:
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.9},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.2},
                {"backbone_id": "AA003", "model_name": "primary", "oof_prediction": 0.8},
                {"backbone_id": "AA004", "model_name": "primary", "oof_prediction": 0.1},
            ]
        )
        outcomes = pd.DataFrame(
            [
                {"backbone_id": "AA001", "macro_region_jump_label": 1},
                {"backbone_id": "AA002", "macro_region_jump_label": 0},
                {"backbone_id": "AA003", "macro_region_jump_label": 1},
                {"backbone_id": "AA004", "macro_region_jump_label": 0},
            ]
        )

        result = build_secondary_outcome_performance(
            predictions,
            outcomes,
            outcome_columns=["macro_region_jump_label"],
            model_names=["primary"],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "status"], "ok")
        self.assertGreater(float(result.loc[0, "roc_auc"]), 0.9)

    def test_build_count_outcome_alignment_returns_correlations(self) -> None:
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.9},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.7},
                {"backbone_id": "AA003", "model_name": "primary", "oof_prediction": 0.2},
                {"backbone_id": "AA004", "model_name": "primary", "oof_prediction": 0.1},
                {"backbone_id": "AA005", "model_name": "primary", "oof_prediction": 0.8},
            ]
        )
        outcomes = pd.DataFrame(
            [
                {"backbone_id": "AA001", "n_new_countries_recomputed": 10},
                {"backbone_id": "AA002", "n_new_countries_recomputed": 7},
                {"backbone_id": "AA003", "n_new_countries_recomputed": 1},
                {"backbone_id": "AA004", "n_new_countries_recomputed": 0},
                {"backbone_id": "AA005", "n_new_countries_recomputed": 8},
            ]
        )
        result = build_count_outcome_alignment(
            predictions,
            outcomes,
            count_column="n_new_countries_recomputed",
            model_names=["primary"],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "status"], "ok")
        self.assertGreater(float(result.loc[0, "spearman_corr"]), 0.8)
        self.assertIn("spearman_ci_lower", result.columns)
        self.assertIn("spearman_ci_upper", result.columns)

    def test_build_exposure_adjusted_audits_return_rate_rows(self) -> None:
        records = pd.DataFrame(
            [
                {"backbone_id": "AA001", "resolved_year": 2014, "country": "USA"},
                {"backbone_id": "AA001", "resolved_year": 2016, "country": "Germany"},
                {"backbone_id": "AA001", "resolved_year": 2018, "country": "Japan"},
                {"backbone_id": "AA002", "resolved_year": 2014, "country": "USA"},
                {"backbone_id": "AA002", "resolved_year": 2017, "country": "USA"},
                {"backbone_id": "AA003", "resolved_year": 2014, "country": "USA"},
                {"backbone_id": "AA003", "resolved_year": 2016, "country": "Canada"},
                {"backbone_id": "AA004", "resolved_year": 2014, "country": "USA"},
                {"backbone_id": "AA004", "resolved_year": 2019, "country": "France"},
                {"backbone_id": "AA005", "resolved_year": 2014, "country": "USA"},
                {"backbone_id": "AA005", "resolved_year": 2020, "country": "Brazil"},
            ]
        )
        propensity = pd.DataFrame(
            [
                {
                    "resolved_year": 2016,
                    "country": "Germany",
                    "inverse_upload_weight": 0.8,
                    "rarity_weight": 0.7,
                },
                {
                    "resolved_year": 2018,
                    "country": "Japan",
                    "inverse_upload_weight": 0.9,
                    "rarity_weight": 0.8,
                },
                {
                    "resolved_year": 2016,
                    "country": "Canada",
                    "inverse_upload_weight": 0.6,
                    "rarity_weight": 0.5,
                },
                {
                    "resolved_year": 2019,
                    "country": "France",
                    "inverse_upload_weight": 0.4,
                    "rarity_weight": 0.3,
                },
                {
                    "resolved_year": 2020,
                    "country": "Brazil",
                    "inverse_upload_weight": 0.2,
                    "rarity_weight": 0.2,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.95},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.10},
                {"backbone_id": "AA003", "model_name": "primary", "oof_prediction": 0.70},
                {"backbone_id": "AA004", "model_name": "primary", "oof_prediction": 0.40},
                {"backbone_id": "AA005", "model_name": "primary", "oof_prediction": 0.20},
            ]
        )

        exposure = build_exposure_adjusted_event_table(
            records, propensity, split_year=2015, test_year_end=2023
        )
        audit = build_exposure_adjusted_outcome_audit(
            predictions, exposure, model_names=["primary"]
        )

        self.assertIn("new_country_rate_per_year", set(audit["outcome_name"]))
        self.assertTrue((audit["status"] == "ok").all())

    def test_build_knownness_matched_validation_returns_weighted_row(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "backbone_id": f"AA{i:03d}",
                    "spread_label": int(i % 2 == 0),
                    "member_count_train": 1 + (i % 4),
                    "n_countries_train": 1 + (i % 3),
                    "refseq_share_train": 0.8 if i % 4 in (0, 1) else 0.2,
                }
                for i in range(24)
            ]
        )
        pred_rows = []
        for i in range(24):
            pred_rows.append(
                {
                    "backbone_id": f"AA{i:03d}",
                    "model_name": "primary",
                    "oof_prediction": 0.8 if i % 2 == 0 else 0.2,
                }
            )
            pred_rows.append(
                {
                    "backbone_id": f"AA{i:03d}",
                    "model_name": "baseline_both",
                    "oof_prediction": 0.6 if i % 2 == 0 else 0.4,
                }
            )
        predictions = pd.DataFrame(pred_rows)

        result = build_knownness_matched_validation(
            scored, predictions, model_names=["primary", "baseline_both"]
        )

        self.assertIn("__weighted_overall__", set(result["matched_stratum"].astype(str)))
        weighted = result.loc[
            (result["matched_stratum"] == "__weighted_overall__")
            & (result["model_name"] == "primary")
        ].iloc[0]
        self.assertGreater(float(weighted["weighted_mean_roc_auc"]), 0.7)

    def test_build_matched_stratum_propensity_audit_returns_weighted_row(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "backbone_id": f"AA{i:03d}",
                    "spread_label": int(i % 3 != 0),
                    "member_count_train": 1 + (i % 5),
                    "n_countries_train": 1 + (i % 4),
                    "refseq_share_train": 0.8 if i % 4 in (0, 1) else 0.2,
                }
                for i in range(40)
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "backbone_id": f"AA{i:03d}",
                    "model_name": "primary",
                    "oof_prediction": 0.85 if i % 3 != 0 else 0.25,
                }
                for i in range(40)
            ]
        )

        result = build_matched_stratum_propensity_audit(
            scored,
            predictions,
            model_names=["primary"],
        )

        self.assertIn("__weighted_overall__", set(result["matched_stratum"].astype(str)))
        weighted = result.loc[
            (result["matched_stratum"] == "__weighted_overall__")
            & (result["model_name"] == "primary")
        ].iloc[0]
        self.assertEqual(str(weighted["status"]), "ok")
        self.assertIn("ipw_risk_difference", result.columns)
        self.assertGreaterEqual(float(weighted["treated_outcome_ipw"]), 0.0)

    def test_build_confirmatory_cohort_summary_returns_internal_cohort_metrics(self) -> None:
        scored = pd.DataFrame(
            [
                {"backbone_id": "AA001", "spread_label": 1},
                {"backbone_id": "AA002", "spread_label": 0},
                {"backbone_id": "AA003", "spread_label": 1},
                {"backbone_id": "AA004", "spread_label": 0},
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.95},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.10},
                {"backbone_id": "AA003", "model_name": "primary", "oof_prediction": 0.90},
                {"backbone_id": "AA004", "model_name": "primary", "oof_prediction": 0.05},
            ]
        )
        metadata_quality = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "metadata_quality_score": 0.90,
                    "country_coverage_fraction": 0.80,
                    "duplicate_fraction": 0.00,
                },
                {
                    "backbone_id": "AA002",
                    "metadata_quality_score": 0.88,
                    "country_coverage_fraction": 0.70,
                    "duplicate_fraction": 0.05,
                },
                {
                    "backbone_id": "AA003",
                    "metadata_quality_score": 0.76,
                    "country_coverage_fraction": 0.55,
                    "duplicate_fraction": 0.10,
                },
                {
                    "backbone_id": "AA004",
                    "metadata_quality_score": 0.40,
                    "country_coverage_fraction": 0.20,
                    "duplicate_fraction": 0.40,
                },
            ]
        )

        result = build_confirmatory_cohort_summary(
            scored,
            predictions,
            model_names=["primary"],
            metadata_quality=metadata_quality,
        )

        self.assertIn("confirmatory_internal", set(result["cohort_name"].astype(str)))
        row = result.loc[
            (result["cohort_name"] == "confirmatory_internal") & (result["model_name"] == "primary")
        ].iloc[0]
        self.assertEqual(str(row["status"]), "ok")
        self.assertGreaterEqual(int(row["n_backbones"]), 3)
        self.assertGreater(float(row["roc_auc"]), 0.9)
        self.assertIn("max_calibration_error", result.columns)

    def test_build_nonlinear_deconfounding_audit_returns_rows(self) -> None:
        rows = []
        for i in range(30):
            label = int(i % 2 == 0)
            rows.append(
                {
                    "backbone_id": f"AA{i:03d}",
                    "spread_label": label,
                    "T_raw_norm": 0.7 if label else 0.3,
                    "H_specialization_norm": 0.6 if label else 0.4,
                    "A_raw_norm": 0.65 if label else 0.35,
                    "orit_support": 1.0 if label else 0.0,
                    "H_support_norm": 0.2 + 0.02 * i,
                    "H_support_norm_residual": np.sin(i / 5.0),
                    "log1p_member_count_train": float(i % 5),
                    "log1p_n_countries_train": float(i % 3),
                    "refseq_share_train": 0.8 if i % 2 == 0 else 0.2,
                }
            )
        scored = pd.DataFrame(rows)

        result = build_nonlinear_deconfounding_audit(scored)

        self.assertEqual(
            set(result["deconfounding_method"]), {"linear_existing", "quadratic", "stratified"}
        )
        self.assertTrue((result["status"] == "ok").all())

    def test_build_metadata_quality_table_scores_backbones(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq1",
                    "backbone_id": "AA001",
                    "nuccore_uid": "1",
                    "resolved_year": 2014,
                    "country": "USA",
                    "species": "Escherichia coli",
                    "TAXONOMY_family": "Enterobacteriaceae",
                    "typing_num_contigs": 1,
                    "typing_gc": 50.0,
                    "typing_size": 100000.0,
                },
                {
                    "sequence_accession": "seq2",
                    "backbone_id": "AA001",
                    "nuccore_uid": "2",
                    "resolved_year": 2014,
                    "country": "",
                    "species": "",
                    "TAXONOMY_family": "",
                    "typing_num_contigs": 12,
                    "typing_gc": 0.0,
                    "typing_size": 0.0,
                },
            ]
        )
        scored = pd.DataFrame([{"backbone_id": "AA001", "assignment_confidence_score": 0.8}])
        assembly = pd.DataFrame(
            [
                {
                    "NUCCORE_UID": "1",
                    "ASSEMBLY_Status": "Complete Genome",
                    "ASSEMBLY_coverage": "40x",
                    "ASSEMBLY_SeqReleaseDate": "2020-01-01",
                },
                {
                    "NUCCORE_UID": "2",
                    "ASSEMBLY_Status": "",
                    "ASSEMBLY_coverage": "",
                    "ASSEMBLY_SeqReleaseDate": "",
                },
            ]
        )
        biosample = pd.DataFrame(
            [
                {
                    "NUCCORE_UID": "1",
                    "BIOSAMPLE_pathogenicity": "pathogenic",
                    "DISEASE_tags": "sepsis",
                    "ECOSYSTEM_tags": "clinical",
                },
                {
                    "NUCCORE_UID": "2",
                    "BIOSAMPLE_pathogenicity": "",
                    "DISEASE_tags": "",
                    "ECOSYSTEM_tags": "",
                },
            ]
        )
        nucc = pd.DataFrame(
            [
                {
                    "NUCCORE_ACC": "seq1",
                    "NUCCORE_Completeness": "complete",
                    "NUCCORE_DuplicatedEntry": "false",
                },
                {
                    "NUCCORE_ACC": "seq2",
                    "NUCCORE_Completeness": "",
                    "NUCCORE_DuplicatedEntry": "true",
                },
            ]
        )

        result = build_metadata_quality_table(records, scored, assembly, biosample, nucc)

        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(float(result.loc[0, "metadata_quality_score"]), 0.0)

    def test_build_operational_risk_dictionary_returns_calibrated_risk_columns(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "backbone_id": f"AA{i:03d}",
                    "member_count_train": 1 + i,
                    "n_countries_train": 1 + (i // 2),
                    "refseq_share_train": 0.2 * i,
                }
                for i in range(6)
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": f"AA{i:03d}", "model_name": "primary", "oof_prediction": score}
                for i, score in enumerate([0.10, 0.20, 0.35, 0.55, 0.75, 0.90])
            ]
        )
        outcomes = pd.DataFrame(
            [
                {
                    "backbone_id": f"AA{i:03d}",
                    "spread_severity_bin": float(min(i, 3)),
                    "macro_region_jump_label": float(1 if i >= 3 else 0),
                    "event_within_3y_label": float(1 if i >= 2 else 0),
                    "three_countries_within_5y_label": float(1 if i >= 4 else 0),
                }
                for i in range(6)
            ]
        )

        result = build_operational_risk_dictionary(
            predictions, outcomes, scored=scored, model_names=["primary"]
        )

        self.assertEqual(len(result), 6)
        for column in (
            "risk_spread_probability",
            "risk_spread_severity",
            "risk_macro_region_jump_3y",
            "risk_event_within_3y",
            "risk_three_countries_within_5y",
            "operational_risk_score",
            "risk_uncertainty",
        ):
            self.assertIn(column, result.columns)
            self.assertTrue(result[column].between(0.0, 1.0).all())
        self.assertIn("risk_abstain_flag", result.columns)
        self.assertIn("source_band", result.columns)
        self.assertIn("knownness_score", result.columns)
        ordered = result.sort_values("risk_spread_probability").reset_index(drop=True)
        self.assertLessEqual(
            float(ordered.loc[1, "risk_macro_region_jump_3y"]),
            float(ordered.loc[4, "risk_macro_region_jump_3y"]) + 1e-9,
        )
        self.assertLessEqual(
            float(ordered.loc[1, "risk_event_within_3y"]),
            float(ordered.loc[4, "risk_event_within_3y"]) + 1e-9,
        )

    def test_build_consensus_shortlist_prefers_portfolio_and_fills_from_consensus(self) -> None:
        consensus = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "consensus_rank": 1,
                    "consensus_candidate_score": 0.95,
                    "consensus_support_count": 3,
                    "primary_candidate_score": 0.91,
                },
                {
                    "backbone_id": "AA002",
                    "consensus_rank": 2,
                    "consensus_candidate_score": 0.90,
                    "consensus_support_count": 3,
                    "primary_candidate_score": 0.88,
                },
                {
                    "backbone_id": "AA003",
                    "consensus_rank": 3,
                    "consensus_candidate_score": 0.85,
                    "consensus_support_count": 2,
                    "primary_candidate_score": 0.80,
                },
            ]
        )
        portfolio = pd.DataFrame(
            [
                {
                    "backbone_id": "AA002",
                    "portfolio_track": "established_high_risk",
                    "track_rank": 1,
                    "candidate_confidence_tier": "tier_a",
                },
            ]
        )
        stability = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "multiverse_stability_tier": "stable",
                    "multiverse_stability_score": 0.8,
                },
                {
                    "backbone_id": "AA002",
                    "multiverse_stability_tier": "stable",
                    "multiverse_stability_score": 0.9,
                },
            ]
        )

        result = build_consensus_shortlist(consensus, portfolio, stability, top_k=2)

        self.assertEqual(result.loc[0, "backbone_id"], "AA002")
        self.assertEqual(result.loc[0, "selection_origin"], "portfolio")
        self.assertEqual(result.loc[1, "backbone_id"], "AA001")
        self.assertEqual(result.loc[1, "selection_origin"], "consensus_fill")

    def test_build_false_negative_audit_flags_missed_positive_rows(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "spread_label": 1,
                    "n_new_countries": 6,
                    "member_count_train": 1,
                    "n_countries_train": 1,
                    "log1p_member_count_train": 0.1,
                    "log1p_n_countries_train": 0.1,
                    "refseq_share_train": 0.1,
                    "backbone_purity_score": 0.40,
                    "assignment_confidence_score": 0.35,
                    "H_external_host_range_support": 0.10,
                },
                {
                    "backbone_id": "AA002",
                    "spread_label": 1,
                    "n_new_countries": 3,
                    "member_count_train": 4,
                    "n_countries_train": 2,
                    "log1p_member_count_train": 1.0,
                    "log1p_n_countries_train": 0.7,
                    "refseq_share_train": 0.8,
                    "backbone_purity_score": 0.80,
                    "assignment_confidence_score": 0.90,
                    "H_external_host_range_support": 0.80,
                },
                {
                    "backbone_id": "AA003",
                    "spread_label": 0,
                    "n_new_countries": 0,
                    "member_count_train": 5,
                    "n_countries_train": 2,
                    "log1p_member_count_train": 1.2,
                    "log1p_n_countries_train": 0.7,
                    "refseq_share_train": 0.8,
                    "backbone_purity_score": 0.90,
                    "assignment_confidence_score": 0.95,
                    "H_external_host_range_support": 0.90,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.9},
                {"backbone_id": "AA003", "model_name": "primary", "oof_prediction": 0.7},
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.2},
            ]
        )
        metadata = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "metadata_quality_score": 0.30,
                    "metadata_quality_tier": "low",
                },
                {
                    "backbone_id": "AA002",
                    "metadata_quality_score": 0.90,
                    "metadata_quality_tier": "high",
                },
            ]
        )
        threshold_flip = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "threshold_flip_count": 3,
                    "eligible_for_threshold_audit": True,
                },
                {
                    "backbone_id": "AA002",
                    "threshold_flip_count": 0,
                    "eligible_for_threshold_audit": True,
                },
            ]
        )

        result = build_false_negative_audit(
            scored,
            predictions,
            primary_model_name="primary",
            metadata_quality=metadata,
            candidate_threshold_flip=threshold_flip,
            shortlist_cutoffs=(1, 2),
            top_n=5,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "backbone_id"], "AA001")
        self.assertTrue(bool(result.loc[0, "missed_by_top_1"]))
        self.assertIn("low_metadata_quality", str(result.loc[0, "miss_driver_flags"]))

    def test_event_timing_missingness_and_ordinal_outputs(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq1",
                    "resolved_year": 2014,
                    "country": "USA",
                },
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq2",
                    "resolved_year": 2016,
                    "country": "Germany",
                },
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq3",
                    "resolved_year": 2017,
                    "country": "France",
                },
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq4",
                    "resolved_year": 2018,
                    "country": "Japan",
                },
                {
                    "backbone_id": "AA002",
                    "sequence_accession": "seq5",
                    "resolved_year": 2014,
                    "country": "USA",
                },
                {
                    "backbone_id": "AA002",
                    "sequence_accession": "seq6",
                    "resolved_year": 2017,
                    "country": "",
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.9},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.2},
            ]
        )

        timing = build_event_timing_outcomes(records, split_year=2015, test_year_end=2023)
        bounds = build_country_missingness_bounds(
            records, split_year=2015, test_year_end=2023, threshold=3
        )
        ordinal = build_ordinal_outcome_alignment(
            predictions, timing, ordinal_column="spread_severity_bin", model_names=["primary"]
        )
        missingness = build_missingness_sensitivity_performance(
            predictions, bounds, model_names=["primary"]
        )

        row = timing.loc[timing["backbone_id"] == "AA001"].iloc[0]
        self.assertEqual(int(row["event_within_1y_label"]), 1)
        self.assertEqual(int(row["three_countries_within_5y_label"]), 1)
        bound_row = bounds.loc[bounds["backbone_id"] == "AA002"].iloc[0]
        self.assertGreaterEqual(
            float(bound_row["optimistic_new_countries"]), float(bound_row["observed_new_countries"])
        )
        self.assertEqual(len(ordinal), 1)
        self.assertTrue((missingness["status"] == "ok").all())

    def test_geographic_duplicate_amr_graph_and_counterfactual_audits(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq1",
                    "resolved_year": 2014,
                    "country": "USA",
                    "LOCATION_lat": 40.0,
                    "LOCATION_lng": -74.0,
                },
                {
                    "backbone_id": "AA001",
                    "sequence_accession": "seq2",
                    "resolved_year": 2017,
                    "country": "Japan",
                    "LOCATION_lat": 35.0,
                    "LOCATION_lng": 139.0,
                },
                {
                    "backbone_id": "AA002",
                    "sequence_accession": "seq3",
                    "resolved_year": 2014,
                    "country": "USA",
                    "LOCATION_lat": 37.0,
                    "LOCATION_lng": -122.0,
                },
                {
                    "backbone_id": "AA002",
                    "sequence_accession": "seq4",
                    "resolved_year": 2017,
                    "country": "Canada",
                    "LOCATION_lat": 43.0,
                    "LOCATION_lng": -79.0,
                },
            ]
        )
        nucc = pd.DataFrame(
            [
                {
                    "NUCCORE_ACC": "seq1",
                    "NUCCORE_Completeness": "complete",
                    "NUCCORE_DuplicatedEntry": "false",
                },
                {
                    "NUCCORE_ACC": "seq2",
                    "NUCCORE_Completeness": "",
                    "NUCCORE_DuplicatedEntry": "true",
                },
                {
                    "NUCCORE_ACC": "seq3",
                    "NUCCORE_Completeness": "complete",
                    "NUCCORE_DuplicatedEntry": "false",
                },
            ]
        )
        changes = pd.DataFrame(
            [
                {"NUCCORE_ACC": "seq2", "Flag": "updated", "Comment": ""},
            ]
        )
        amr_hits = pd.DataFrame(
            [
                {
                    "NUCCORE_ACC": "seq1",
                    "analysis_software_name": "amrfinderplus",
                    "gene_symbol": "blaA",
                    "drug_class": "BETA-LACTAM",
                },
                {
                    "NUCCORE_ACC": "seq1",
                    "analysis_software_name": "rgi",
                    "gene_symbol": "blaA",
                    "drug_class": "BETA-LACTAM",
                },
                {
                    "NUCCORE_ACC": "seq3",
                    "analysis_software_name": "amrfinderplus",
                    "gene_symbol": "tetA",
                    "drug_class": "TETRACYCLINE",
                },
            ]
        )
        mash_pairs = pd.DataFrame(
            [
                ("seq1", "seq2"),
                ("seq1", "seq3"),
                ("seq3", "seq4"),
            ],
            columns=["source_accession", "target_accession"],
        )
        scored = pd.DataFrame(
            [
                {
                    "backbone_id": "AA001",
                    "spread_label": 1,
                    "n_new_countries": 5,
                    "knownness_score": 0.8,
                    "log1p_member_count_train": 1.0,
                    "log1p_n_countries_train": 0.7,
                    "refseq_share_train": 0.8,
                },
                {
                    "backbone_id": "AA002",
                    "spread_label": 0,
                    "n_new_countries": 1,
                    "knownness_score": 0.2,
                    "log1p_member_count_train": 0.2,
                    "log1p_n_countries_train": 0.1,
                    "refseq_share_train": 0.2,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {"backbone_id": "AA001", "model_name": "primary", "oof_prediction": 0.9},
                {"backbone_id": "AA002", "model_name": "primary", "oof_prediction": 0.2},
                {"backbone_id": "AA001", "model_name": "baseline_both", "oof_prediction": 0.8},
                {"backbone_id": "AA002", "model_name": "baseline_both", "oof_prediction": 0.3},
            ]
        )

        geo = build_geographic_jump_distance_table(records, split_year=2015, test_year_end=2023)
        duplicate = build_duplicate_completeness_change_audit(
            records, nucc, changes, split_year=2015
        )
        amr_uncertainty = build_amr_uncertainty_table(records, amr_hits, split_year=2015)
        mash_graph = build_mash_similarity_graph_table(records, mash_pairs, split_year=2015)
        counterfactual = build_counterfactual_shortlist_comparison(
            scored,
            predictions,
            primary_model_name="primary",
            baseline_model_name="baseline_both",
            top_ks=(1,),
        )

        self.assertEqual(len(geo), 2)
        self.assertGreaterEqual(
            float(geo.loc[geo["backbone_id"] == "AA001", "max_jump_distance_km"].iloc[0]), 1000.0
        )
        self.assertEqual(len(duplicate), 2)
        self.assertEqual(len(amr_uncertainty), 2)
        self.assertEqual(len(mash_graph), 2)
        self.assertEqual(
            set(counterfactual["selection_mode"]),
            {
                "primary_natural",
                "baseline_natural",
                "primary_matched_to_baseline",
                "baseline_matched_to_primary",
            },
        )


if __name__ == "__main__":
    unittest.main()
