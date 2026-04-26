from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from plasmid_priority.decisions.candidate_policy import build_candidate_decision
from plasmid_priority.features.registry import FeatureRegistry, FeatureSpec
from plasmid_priority.features.safety_gate import FeatureSafetyGate
from plasmid_priority.modeling.official import (
    build_official_candidate_decisions,
    build_official_model_scorecard,
    build_official_release_artifacts,
    conservative_evidence_consensus,
    fit_bounded_tree_challenger,
    fit_sparse_calibrated_logistic,
    frozen_biological_prior_score,
    score_official_model_family,
    visibility_baseline_score,
    write_official_release_artifacts,
)
from plasmid_priority.modeling.official_registry import (
    OFFICIAL_MODEL_FAMILY,
    OfficialModelRole,
    validate_official_model_family,
)
from plasmid_priority.validation.label_quality import assess_label_quality
from plasmid_priority.validation.model_selection import select_gate_first_model


def _float_cell(frame: pd.DataFrame, row: int, column: str) -> float:
    return float(cast(float, frame.loc[row, column]))


def _row_by_string_value(frame: pd.DataFrame, column: str, value: str) -> dict[str, object]:
    records = frame.loc[frame[column].astype(str).eq(value)].to_dict("records")
    return cast(dict[str, object], records[0])


def _governance_classifier() -> Callable[[float, float, float, float, bool, float | None], str]:
    module = importlib.import_module("scripts.run_phase_61_governance_pruning")
    return cast(
        Callable[[float, float, float, float, bool, float | None], str],
        getattr(module, "classify_governance_candidate"),
    )


def _official_artifact_script_runner() -> Callable[[Path, Path, str, str | None], dict[str, Path]]:
    module = importlib.import_module("scripts.build_official_release_artifacts")
    return cast(
        Callable[[Path, Path, str, str | None], dict[str, Path]],
        getattr(module, "build_official_artifacts_from_tsv"),
    )


def _official_artifact_script_main() -> Callable[[list[str] | None], int]:
    module = importlib.import_module("scripts.build_official_release_artifacts")
    return cast(Callable[[list[str] | None], int], getattr(module, "main"))


def _official_artifact_frame_preparer() -> Callable[[pd.DataFrame], pd.DataFrame]:
    module = importlib.import_module("scripts.build_official_release_artifacts")
    return cast(
        Callable[[pd.DataFrame], pd.DataFrame],
        getattr(module, "prepare_official_candidate_frame"),
    )


def test_official_model_family_is_small_role_complete_and_non_experimental() -> None:
    summary = validate_official_model_family(OFFICIAL_MODEL_FAMILY)

    assert summary["status"] == "pass"
    assert cast(int, summary["model_count"]) <= 5
    assert summary["primary_model_name"] == "sparse_calibrated_logistic"
    assert summary["decision_model_name"] == "conservative_evidence_consensus"
    assert summary["experimental_model_count"] == 0
    assert {model.role for model in OFFICIAL_MODEL_FAMILY.models} == {
        OfficialModelRole.VISIBILITY_BASELINE,
        OfficialModelRole.BIOLOGICAL_PRIOR,
        OfficialModelRole.PRIMARY,
        OfficialModelRole.CHALLENGER,
        OfficialModelRole.DECISION_SURFACE,
    }


def test_feature_safety_gate_rejects_unknown_and_post_split_features() -> None:
    registry = FeatureRegistry(
        [
            FeatureSpec(
                name="T_eff_norm",
                group="mobility",
                temporal_scope="pre_split_only",
                prediction_time_available=True,
                source_scope="training_only",
                leakage_risk="low",
                allowed_in_official_models=True,
            ),
            FeatureSpec(
                name="future_country_count",
                group="geography",
                temporal_scope="post_split",
                prediction_time_available=False,
                source_scope="future",
                leakage_risk="critical",
                allowed_in_official_models=False,
            ),
        ],
    )

    gate = FeatureSafetyGate(registry)

    accepted = gate.evaluate(["T_eff_norm"])
    rejected = gate.evaluate(["T_eff_norm", "future_country_count", "unknown_feature"])

    assert accepted.status == "pass"
    assert rejected.status == "fail"
    assert "future_country_count" in rejected.blocked_features
    assert "unknown_feature" in rejected.unknown_features


def test_label_quality_gate_blocks_low_prevalence_and_high_missingness() -> None:
    frame = pd.DataFrame(
        {
            "good_label": [0, 1, 0, 1, 0, 1, 0, 1],
            "rare_label": [0, 0, 0, 0, 0, 0, 0, 1],
            "missing_label": [0, None, None, None, 1, None, None, None],
        },
    )

    good = assess_label_quality(frame, "good_label", min_positive=2, max_missing_fraction=0.25)
    rare = assess_label_quality(frame, "rare_label", min_positive=2, max_missing_fraction=0.25)
    missing = assess_label_quality(
        frame,
        "missing_label",
        min_positive=1,
        max_missing_fraction=0.25,
    )

    assert good.status == "pass"
    assert rare.status == "fail"
    assert "insufficient_positive_cases" in rare.reasons
    assert missing.status == "fail"
    assert "missing_fraction_too_high" in missing.reasons


def test_gate_first_model_selection_prefers_calibrated_robust_model_over_auc_leader() -> None:
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "high_auc_bad_gate",
                "roc_auc": 0.91,
                "average_precision": 0.80,
                "brier_score": 0.30,
                "expected_calibration_error": 0.12,
                "decision_utility": 0.70,
                "feature_count": 8,
                "feature_safety_pass": True,
                "label_quality_pass": True,
                "temporal_gate_pass": True,
                "source_holdout_gate_pass": True,
                "knownness_gate_pass": True,
                "calibration_gate_pass": False,
                "abstention_gate_pass": True,
            },
            {
                "model_name": "sparse_calibrated_logistic",
                "roc_auc": 0.84,
                "average_precision": 0.75,
                "brier_score": 0.16,
                "expected_calibration_error": 0.025,
                "decision_utility": 0.68,
                "feature_count": 5,
                "feature_safety_pass": True,
                "label_quality_pass": True,
                "temporal_gate_pass": True,
                "source_holdout_gate_pass": True,
                "knownness_gate_pass": True,
                "calibration_gate_pass": True,
                "abstention_gate_pass": True,
            },
        ],
    )

    winner = select_gate_first_model(scorecard)

    assert winner["model_name"] == "sparse_calibrated_logistic"
    assert winner["selection_status"] == "selected"
    assert winner["failed_gate_count"] == 0


def test_gate_first_model_selection_fails_when_no_model_passes_all_hard_gates() -> None:
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "unsafe_model",
                "roc_auc": 0.99,
                "feature_safety_pass": False,
                "label_quality_pass": True,
                "temporal_gate_pass": True,
                "source_holdout_gate_pass": True,
                "knownness_gate_pass": True,
                "calibration_gate_pass": True,
                "abstention_gate_pass": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="No candidate model passed all hard gates"):
        select_gate_first_model(scorecard)


def test_governance_pruning_classification_rejects_failed_gates_even_with_auc_gain() -> None:
    classification = _governance_classifier()(
        0.92,
        0.86,
        0.03,
        0.04,
        False,
        0.0,
    )

    assert classification == "REJECTED"


def test_candidate_decision_abstains_for_low_knownness_and_model_disagreement() -> None:
    decision = build_candidate_decision(
        backbone_id="bb_001",
        calibrated_priority_score=0.86,
        evidence_tier="moderate",
        uncertainty_tier="high",
        model_agreement=0.42,
        knownness_score=0.18,
        source_dominance_risk=True,
        annotation_conflict_risk=False,
    )

    assert decision["recommended_monitoring_tier"] == "review_not_rank"
    assert decision["allowed_claim_language"] == "insufficient_evidence"
    assert decision["abstention_reason"] == "low_knownness_and_model_disagreement"


def test_candidate_decision_allows_strong_claim_only_when_evidence_is_strong() -> None:
    decision = build_candidate_decision(
        backbone_id="bb_002",
        calibrated_priority_score=0.91,
        evidence_tier="high",
        uncertainty_tier="low",
        model_agreement=0.91,
        knownness_score=0.72,
        source_dominance_risk=False,
        annotation_conflict_risk=False,
    )

    assert decision["recommended_monitoring_tier"] == "core_surveillance"
    assert decision["allowed_claim_language"] == "strong_surveillance_candidate"
    assert decision["abstention_reason"] == ""


def test_visibility_baseline_is_bounded_and_monotone_with_training_visibility() -> None:
    frame = pd.DataFrame(
        {
            "log1p_member_count_train": [0.0, 1.0, 3.0],
            "log1p_n_countries_train": [0.0, 0.5, 2.0],
        },
    )

    scores = visibility_baseline_score(frame)

    assert scores.between(0.0, 1.0).all()
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]


def test_frozen_biological_prior_is_bounded_and_uses_biological_evidence_axes() -> None:
    frame = pd.DataFrame(
        {
            "T_eff_norm": [0.05, 0.90],
            "H_obs_specialization_norm": [0.10, 0.80],
            "A_eff_norm": [0.05, 0.85],
            "evidence_support_index": [0.20, 0.95],
        },
    )

    scores = frozen_biological_prior_score(frame)

    assert scores.between(0.0, 1.0).all()
    assert scores.iloc[1] > scores.iloc[0] + 0.50


def test_sparse_calibrated_logistic_fits_small_official_feature_set() -> None:
    frame = pd.DataFrame(
        {
            "x_signal": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "x_support": [0.0, 0.2, 0.1, 0.7, 0.8, 1.0],
            "label": [0, 0, 0, 1, 1, 1],
        },
    )

    model = fit_sparse_calibrated_logistic(frame, label_column="label", feature_columns=["x_signal", "x_support"])
    probabilities = model.predict_proba(frame)

    assert model.model_name == "sparse_calibrated_logistic"
    assert model.feature_columns == ("x_signal", "x_support")
    assert probabilities.between(0.0, 1.0).all()
    assert probabilities.iloc[-1] > probabilities.iloc[0]


def test_bounded_tree_challenger_returns_bounded_probabilities_without_large_compute() -> None:
    frame = pd.DataFrame(
        {
            "x_signal": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "x_support": [0.0, 0.2, 0.1, 0.7, 0.8, 1.0],
            "label": [0, 0, 0, 1, 1, 1],
        },
    )

    model = fit_bounded_tree_challenger(frame, label_column="label", feature_columns=["x_signal", "x_support"])
    probabilities = model.predict_proba(frame)

    assert model.model_name == "bounded_monotonic_tree"
    assert model.max_depth <= 3
    assert probabilities.between(0.0, 1.0).all()
    assert probabilities.iloc[-1] >= probabilities.iloc[0]


def test_conservative_evidence_consensus_penalizes_disagreement_and_weak_evidence() -> None:
    frame = pd.DataFrame(
        {
            "visibility_baseline": [0.60, 0.95],
            "frozen_biological_prior": [0.62, 0.20],
            "sparse_calibrated_logistic": [0.64, 0.98],
            "bounded_monotonic_tree": [0.61, 0.15],
            "evidence_tier": ["high", "low"],
            "uncertainty_tier": ["low", "high"],
        },
    )

    consensus = conservative_evidence_consensus(
        frame,
        score_columns=[
            "visibility_baseline",
            "frozen_biological_prior",
            "sparse_calibrated_logistic",
            "bounded_monotonic_tree",
        ],
    )

    assert consensus["conservative_consensus_score"].between(0.0, 1.0).all()
    assert _float_cell(consensus, 0, "conservative_consensus_score") > _float_cell(
        consensus,
        1,
        "conservative_consensus_score",
    )
    assert bool(consensus.loc[1, "consensus_review_flag"])


def test_score_official_model_family_runs_all_models_when_labels_are_available() -> None:
    frame = pd.DataFrame(
        {
            "log1p_member_count_train": np.linspace(0.0, 1.7, 12),
            "log1p_n_countries_train": np.linspace(0.0, 1.4, 12),
            "T_eff_norm": np.linspace(0.0, 1.0, 12),
            "H_obs_specialization_norm": np.linspace(1.0, 0.0, 12),
            "A_eff_norm": np.linspace(0.0, 1.0, 12),
            "evidence_support_index": np.linspace(0.2, 1.0, 12),
            "evidence_tier": ["moderate"] * 6 + ["high"] * 6,
            "uncertainty_tier": ["moderate"] * 6 + ["low"] * 6,
            "label": [0, 1] * 6,
        },
    )

    scores = score_official_model_family(frame, label_column="label")

    assert scores["official_supervised_model_status"].eq("fit").all()
    assert scores["conservative_consensus_score"].between(0.0, 1.0).all()
    assert _float_cell(scores, 11, "conservative_consensus_score") > _float_cell(
        scores,
        0,
        "conservative_consensus_score",
    )


def test_score_official_model_family_degrades_to_auditable_unsupervised_scores_without_labels() -> None:
    frame = pd.DataFrame(
        {
            "log1p_member_count_train": [0.0, 1.0],
            "log1p_n_countries_train": [0.0, 1.0],
            "T_eff_norm": [0.1, 0.9],
            "H_obs_specialization_norm": [0.2, 0.8],
            "A_eff_norm": [0.1, 0.9],
            "evidence_support_index": [0.2, 0.9],
        },
    )

    scores = score_official_model_family(frame)

    assert scores["official_supervised_model_status"].eq("not_fit_label_unavailable").all()
    assert scores["sparse_calibrated_logistic"].isna().all()
    assert scores["bounded_monotonic_tree"].isna().all()
    assert scores["conservative_consensus_score"].between(0.0, 1.0).all()
    assert _float_cell(scores, 1, "conservative_consensus_score") > _float_cell(
        scores,
        0,
        "conservative_consensus_score",
    )


def test_official_model_scorecard_exposes_roles_status_and_decision_surface() -> None:
    frame = pd.DataFrame(
        {
            "log1p_member_count_train": [0.0, 1.0],
            "log1p_n_countries_train": [0.0, 1.0],
            "T_eff_norm": [0.1, 0.9],
            "H_obs_specialization_norm": [0.2, 0.8],
            "A_eff_norm": [0.1, 0.9],
            "evidence_support_index": [0.2, 0.9],
        },
    )
    scores = score_official_model_family(frame)

    scorecard = build_official_model_scorecard(scores)

    assert scorecard["model_name"].tolist() == [
        model.name for model in OFFICIAL_MODEL_FAMILY.models
    ]
    assert set(scorecard["official_family_status"]) == {"available", "not_fit_label_unavailable"}
    decision_surface = _row_by_string_value(
        scorecard,
        "model_name",
        "conservative_evidence_consensus",
    )
    assert decision_surface["score_column"] == "conservative_consensus_score"
    assert bool(decision_surface["has_bounded_scores"])


def test_official_candidate_decisions_convert_scores_into_claim_limited_release_rows() -> None:
    frame = pd.DataFrame(
        {
            "backbone_id": ["bb_strong", "bb_review"],
            "log1p_member_count_train": [1.2, 0.1],
            "log1p_n_countries_train": [1.1, 0.1],
            "T_eff_norm": [0.95, 0.25],
            "H_obs_specialization_norm": [0.92, 0.20],
            "A_eff_norm": [0.97, 0.20],
            "evidence_support_index": [0.95, 0.20],
            "evidence_tier": ["high", "low"],
            "uncertainty_tier": ["low", "high"],
            "knownness_score": [0.90, 0.10],
            "source_dominance_risk": [False, False],
            "annotation_conflict_risk": [False, False],
        },
    )
    scores = score_official_model_family(frame)

    decisions = build_official_candidate_decisions(frame, scores, id_column="backbone_id")

    assert decisions["backbone_id"].tolist() == ["bb_strong", "bb_review"]
    assert decisions.loc[0, "allowed_claim_language"] == "strong_surveillance_candidate"
    assert decisions.loc[1, "recommended_monitoring_tier"] == "review_not_rank"
    assert decisions.loc[1, "allowed_claim_language"] == "insufficient_evidence"


def test_official_release_artifacts_bundle_scores_scorecard_decisions_and_summary() -> None:
    frame = pd.DataFrame(
        {
            "backbone_id": ["bb_low", "bb_high"],
            "log1p_member_count_train": [0.0, 1.0],
            "log1p_n_countries_train": [0.0, 1.0],
            "T_eff_norm": [0.1, 0.95],
            "H_obs_specialization_norm": [0.2, 0.95],
            "A_eff_norm": [0.1, 0.95],
            "evidence_support_index": [0.2, 0.95],
            "evidence_tier": ["low", "high"],
            "uncertainty_tier": ["high", "low"],
            "knownness_score": [0.10, 0.90],
        },
    )

    artifacts = build_official_release_artifacts(frame, id_column="backbone_id")

    assert set(artifacts) == {"scores", "scorecard", "candidate_decisions", "summary"}
    assert artifacts["scores"]["backbone_id"].tolist() == ["bb_low", "bb_high"]
    assert artifacts["summary"]["official_model_family_status"] == "pass"
    assert artifacts["summary"]["candidate_count"] == 2
    assert artifacts["summary"]["review_not_rank_count"] >= 1


def test_write_official_release_artifacts_exports_reusable_release_tables(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "backbone_id": ["bb_low", "bb_high"],
            "log1p_member_count_train": [0.0, 1.0],
            "log1p_n_countries_train": [0.0, 1.0],
            "T_eff_norm": [0.1, 0.95],
            "H_obs_specialization_norm": [0.2, 0.95],
            "A_eff_norm": [0.1, 0.95],
            "evidence_support_index": [0.2, 0.95],
            "evidence_tier": ["low", "high"],
            "uncertainty_tier": ["high", "low"],
            "knownness_score": [0.10, 0.90],
        },
    )
    artifacts = build_official_release_artifacts(frame, id_column="backbone_id")

    written = write_official_release_artifacts(artifacts, tmp_path)

    assert set(written) == {"scores", "scorecard", "candidate_decisions", "summary"}
    assert written["scores"].name == "official_model_scores.tsv"
    assert written["scorecard"].read_text(encoding="utf-8").startswith("official_model_order\t")
    assert '"official_model_family_status": "pass"' in written["summary"].read_text(
        encoding="utf-8",
    )


def test_official_artifact_script_builds_release_tables_from_tsv(tmp_path: Path) -> None:
    input_path = tmp_path / "official_input.tsv"
    output_dir = tmp_path / "core_tables"
    pd.DataFrame(
        {
            "backbone_id": ["bb_low", "bb_high"],
            "log1p_member_count_train": [0.0, 1.0],
            "log1p_n_countries_train": [0.0, 1.0],
            "T_eff_norm": [0.1, 0.95],
            "H_obs_specialization_norm": [0.2, 0.95],
            "A_eff_norm": [0.1, 0.95],
            "evidence_support_index": [0.2, 0.95],
            "evidence_tier": ["low", "high"],
            "uncertainty_tier": ["high", "low"],
            "knownness_score": [0.10, 0.90],
        },
    ).to_csv(input_path, sep="\t", index=False)

    written = _official_artifact_script_runner()(input_path, output_dir, "backbone_id", None)

    assert (output_dir / "official_model_scores.tsv").exists()
    assert (output_dir / "official_model_scorecard.tsv").exists()
    assert (output_dir / "official_candidate_decisions.tsv").exists()
    assert written["summary"] == output_dir / "official_model_summary.json"


def test_official_artifact_script_fails_closed_when_input_is_missing(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing_candidate_portfolio.tsv"
    output_dir = tmp_path / "core_tables"

    exit_code = _official_artifact_script_main()(
        ["--input", str(missing_input), "--output-dir", str(output_dir)],
    )

    assert exit_code == 2
    assert not (output_dir / "official_model_summary.json").exists()
    assert not (output_dir / "official_model_scorecard.tsv").exists()


def test_official_artifact_script_prepares_existing_candidate_portfolio_columns() -> None:
    raw = pd.DataFrame(
        {
            "backbone_id": ["bb_legacy"],
            "member_count_train": [9],
            "n_countries_train": [4],
            "bio_priority_index": [0.82],
            "priority_index": [0.88],
            "evidence_support_index": [0.91],
            "evidence_tier": ["watchlist"],
            "model_prediction_uncertainty": [0.12],
        },
    )

    prepared = _official_artifact_frame_preparer()(raw)

    assert "log1p_member_count_train" in prepared.columns
    assert "log1p_n_countries_train" in prepared.columns
    assert _float_cell(prepared, 0, "T_eff_norm") == 0.82
    assert _float_cell(prepared, 0, "H_obs_specialization_norm") == 0.82
    assert _float_cell(prepared, 0, "A_eff_norm") == 0.88
    assert prepared.loc[0, "evidence_tier"] == "high"
    assert prepared.loc[0, "uncertainty_tier"] == "low"


def test_official_artifact_script_writes_workflow_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "candidate_portfolio.tsv"
    output_dir = tmp_path / "core_tables"
    pd.DataFrame(
        {
            "backbone_id": ["bb_high"],
            "log1p_member_count_train": [1.0],
            "log1p_n_countries_train": [1.0],
            "T_eff_norm": [0.95],
            "H_obs_specialization_norm": [0.95],
            "A_eff_norm": [0.95],
            "evidence_support_index": [0.95],
            "evidence_tier": ["high"],
            "uncertainty_tier": ["low"],
            "knownness_score": [0.90],
        },
    ).to_csv(input_path, sep="\t", index=False)
    data_root = tmp_path / "runtime-data"

    with mock.patch.dict(
        os.environ,
        {"PLASMID_PRIORITY_DATA_ROOT": str(data_root)},
        clear=False,
    ):
        exit_code = _official_artifact_script_main()(
            ["--input", str(input_path), "--output-dir", str(output_dir)],
        )

    assert exit_code == 0
    summary_path = data_root / "tmp" / "logs" / "52_build_official_release_artifacts_summary.json"
    assert summary_path.exists()
