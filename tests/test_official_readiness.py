from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.modeling.official.artifacts import (
    build_official_model_scorecard,
    build_official_release_artifacts,
)
from plasmid_priority.modeling.official.runner import score_official_model_family


def _fit_ready_frame() -> pd.DataFrame:
    n = 12
    return pd.DataFrame(
        {
            "backbone_id": [f"bb_{idx}" for idx in range(n)],
            "log1p_member_count_train": np.linspace(0.0, 1.8, n),
            "log1p_n_countries_train": np.linspace(0.0, 1.4, n),
            "T_eff_norm": np.linspace(0.05, 0.95, n),
            "H_obs_specialization_norm": np.linspace(0.95, 0.05, n),
            "A_eff_norm": np.linspace(0.1, 0.9, n),
            "evidence_support_index": np.linspace(0.2, 1.0, n),
            "evidence_tier": ["moderate"] * 6 + ["high"] * 6,
            "uncertainty_tier": ["moderate"] * 6 + ["low"] * 6,
            "label": ([0, 1] * 6),
        },
    )


def test_score_official_model_family_emits_supervised_readiness_metadata() -> None:
    scores = score_official_model_family(_fit_ready_frame(), label_column="label")

    assert scores["official_supervised_model_status"].eq("fit").all()
    assert scores["official_supervised_feature_count"].eq(6).all()
    assert scores["official_supervised_requested_feature_count"].eq(6).all()
    assert scores["official_supervised_missing_feature_count"].eq(0).all()
    assert scores["official_supervised_labeled_count"].eq(12).all()
    assert scores["official_supervised_positive_count"].eq(6).all()
    assert scores["official_supervised_negative_count"].eq(6).all()
    assert scores["official_supervised_models_used"].eq(
        "sparse_calibrated_logistic,bounded_monotonic_tree"
    ).all()


def test_score_official_model_family_fails_closed_when_default_features_are_missing() -> None:
    frame = _fit_ready_frame().drop(columns=["evidence_support_index"])

    scores = score_official_model_family(frame, label_column="label")

    assert scores["official_supervised_model_status"].eq(
        "not_fit_missing_supervised_features",
    ).all()
    assert scores["official_supervised_missing_feature_count"].eq(1).all()
    assert scores["official_supervised_missing_features"].eq("evidence_support_index").all()
    assert scores["sparse_calibrated_logistic"].isna().all()
    assert scores["bounded_monotonic_tree"].isna().all()
    assert scores["conservative_consensus_score"].between(0.0, 1.0).all()


def test_score_official_model_family_rejects_tiny_labeled_surface() -> None:
    frame = _fit_ready_frame().iloc[:8].copy()

    scores = score_official_model_family(frame, label_column="label")

    assert scores["official_supervised_model_status"].eq(
        "not_fit_too_few_labeled_rows",
    ).all()
    assert scores["official_supervised_labeled_count"].eq(8).all()
    assert scores["official_supervised_min_labeled_rows"].eq(12).all()
    assert scores["sparse_calibrated_logistic"].isna().all()


def test_official_model_scorecard_carries_supervised_gate_metadata() -> None:
    frame = _fit_ready_frame().drop(columns=["evidence_support_index"])
    scores = score_official_model_family(frame, label_column="label")

    scorecard = build_official_model_scorecard(scores)
    primary_row = scorecard.loc[scorecard["model_name"] == "sparse_calibrated_logistic"].iloc[0]

    assert primary_row["official_family_status"] == "not_fit_missing_supervised_features"
    assert int(primary_row["supervised_missing_feature_count"]) == 1
    assert primary_row["supervised_missing_features"] == "evidence_support_index"
    assert int(primary_row["supervised_requested_feature_count"]) == 6


def test_official_release_artifacts_summary_carries_supervised_readiness() -> None:
    frame = _fit_ready_frame().drop(columns=["evidence_support_index"])

    artifacts = build_official_release_artifacts(
        frame,
        id_column="backbone_id",
        label_column="label",
    )

    summary = artifacts["summary"]
    assert summary["official_supervised_model_status"] == "not_fit_missing_supervised_features"
    assert summary["official_supervised_missing_feature_count"] == 1
    assert summary["official_supervised_missing_features"] == "evidence_support_index"
    assert summary["official_supervised_models_used"] == ""
