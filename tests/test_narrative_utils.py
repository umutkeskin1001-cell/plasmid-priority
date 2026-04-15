from __future__ import annotations

import pandas as pd

from plasmid_priority.reporting.narrative_utils import (
    benchmark_scope_note,
    blocked_holdout_summary_text,
    blocked_holdout_summary_text_tr,
    candidate_stability_summary_text,
    country_missingness_summary_text,
    rolling_temporal_summary,
    select_confirmatory_row,
    strict_acceptance_status,
    summarize_false_negative_audit,
    top_sign_stable_features,
)


def test_strict_acceptance_status_normalizes_text() -> None:
    assert strict_acceptance_status(pd.Series({"scientific_acceptance_status": " PASS "})) == "pass"
    assert strict_acceptance_status(pd.Series({})) == "not_scored"


def test_benchmark_scope_note_depends_on_acceptance_status() -> None:
    assert "clears the frozen scientific acceptance gate" in benchmark_scope_note("pass")
    assert "conditional and benchmark-limited" in benchmark_scope_note("fail")


def test_blocked_holdout_summary_text_literally_separates_group_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_name": "primary_model",
                "blocked_holdout_group_columns": "dominant_source,dominant_region_train",
                "blocked_holdout_roc_auc": 0.71,
                "blocked_holdout_group_count": 3,
                "worst_blocked_holdout_group": "source_a",
                "worst_blocked_holdout_group_roc_auc": 0.62,
            }
        ]
    )
    text = blocked_holdout_summary_text(frame, model_name="primary_model")
    text_tr = blocked_holdout_summary_text_tr(frame, model_name="primary_model")
    assert "dominant_source + dominant_region_train" in text
    assert "dominant_source + dominant_region_train" in text_tr
    assert "across `3` blocked groups" in text


def test_country_missingness_summary_text_describes_label_variants() -> None:
    bounds = pd.DataFrame(
        [
            {
                "backbone_id": "bb1",
                "eligible_for_country_bounds": True,
                "label_observed": 1,
                "label_midpoint": 0,
                "label_optimistic": 1,
                "label_weighted": 1,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "model_name": "primary_model",
                "outcome_name": "label_observed",
                "roc_auc": 0.6,
                "average_precision": 0.5,
            },
            {
                "model_name": "primary_model",
                "outcome_name": "label_weighted",
                "roc_auc": 0.7,
                "average_precision": 0.55,
            },
        ]
    )
    text = country_missingness_summary_text(bounds, sensitivity, model_name="primary_model")
    assert "country-missingness audit" in text
    assert "Sensitivity across those label variants spans ROC AUC" in text


def test_candidate_stability_summary_text_prefers_the_top_stable_backbone() -> None:
    frame = pd.DataFrame(
        [
            {
                "backbone_id": "bb1",
                "top_k": 10,
                "bootstrap_top_10_frequency": 0.8,
                "variant_top_10_frequency": 0.6,
            },
            {
                "backbone_id": "bb2",
                "top_k": 10,
                "bootstrap_top_10_frequency": 0.9,
                "variant_top_10_frequency": 0.7,
            },
        ]
    )
    text = candidate_stability_summary_text(
        frame,
        file_name="candidate_rank_stability.tsv",
        frequency_column="bootstrap_top_10_frequency",
    )
    assert "candidate rank stability" in text
    assert "bb2" in text


def test_summarize_false_negative_audit_returns_top_drivers() -> None:
    frame = pd.DataFrame(
        [
            {"miss_driver_flags": "low_knownness,source_shift"},
            {"miss_driver_flags": "low_knownness"},
            {"miss_driver_flags": "none"},
        ]
    )
    count, drivers = summarize_false_negative_audit(frame)
    assert count == 3
    assert drivers.startswith("low_knownness")


def test_select_confirmatory_row_returns_ok_row_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "cohort_name": "confirmatory_internal",
                "model_name": "primary",
                "status": "ok",
                "roc_auc": 0.7,
            },
            {
                "cohort_name": "confirmatory_internal",
                "model_name": "primary",
                "status": "fail",
                "roc_auc": 0.2,
            },
        ]
    )
    row = select_confirmatory_row(frame, cohort_name="confirmatory_internal", model_name="primary")
    assert float(row["roc_auc"]) == 0.7


def test_rolling_temporal_summary_aggregates_ok_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "status": "ok",
                "split_year": 2015,
                "horizon_years": 3,
                "backbone_assignment_mode": "training_only",
                "roc_auc": 0.6,
                "average_precision": 0.5,
                "brier_score": 0.2,
            },
            {
                "status": "ok",
                "split_year": 2016,
                "horizon_years": 4,
                "backbone_assignment_mode": "all_records",
                "roc_auc": 0.8,
                "average_precision": 0.7,
                "brier_score": 0.3,
            },
        ]
    )
    summary = rolling_temporal_summary(frame)
    assert summary["split_year_min"] == 2015
    assert summary["split_year_max"] == 2016
    assert summary["assignment_modes"] == "all_records,training_only"


def test_top_sign_stable_features_prefers_low_cv_features() -> None:
    frame = pd.DataFrame(
        [
            {"feature": "a", "sign_stable": True, "cv_of_coef": 0.2, "abs_mean_coefficient": 0.3},
            {"feature": "b", "sign_stable": True, "cv_of_coef": 0.1, "abs_mean_coefficient": 0.4},
            {"feature": "c", "sign_stable": False, "cv_of_coef": 0.05, "abs_mean_coefficient": 0.9},
        ]
    )
    assert top_sign_stable_features(frame, top_n=2) == "b, a"
