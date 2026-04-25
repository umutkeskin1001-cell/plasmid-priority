"""Tests for probabilistic labels, Dawid-Skene, Co-teaching, and causal estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plasmid_priority.labels.coteaching import CoTeachingTrainer
from plasmid_priority.labels.counterfactual import CausalLabelEstimator
from plasmid_priority.labels.probabilistic import (
    DawidSkeneLabelFuser,
    build_probabilistic_labels,
)


class TestDawidSkeneLabelFuser:
    def test_perfect_agreement(self) -> None:
        votes = pd.DataFrame(
            {
                "rater_a": [0, 0, 1, 1],
                "rater_b": [0, 0, 1, 1],
                "rater_c": [0, 0, 1, 1],
            }
        )
        fuser = DawidSkeneLabelFuser(n_classes=2, max_iter=10)
        fuser.fit(votes)
        proba = fuser.predict_proba(votes)
        # Perfect agreement => high confidence (Laplace smoothing prevents 1.0)
        assert proba.max(axis=1).min() > 0.90
        pred = fuser.predict(votes)
        assert pred.tolist() == [0, 0, 1, 1]

    def test_noisy_rater_detected(self) -> None:
        # Rater C is a coin flip
        votes = pd.DataFrame(
            {
                "rater_a": [0, 0, 1, 1],
                "rater_b": [0, 0, 1, 1],
                "rater_c": [1, 0, 0, 1],  # 50% accurate
            }
        )
        fuser = DawidSkeneLabelFuser(n_classes=2, max_iter=20)
        fuser.fit(votes)
        reliability = fuser.rater_reliability
        rater_c = reliability.loc[reliability["rater"] == "rater_c"]
        assert rater_c["mean_accuracy"].iloc[0] < 0.8  # Noisy rater detected

    def test_missing_votes(self) -> None:
        votes = pd.DataFrame(
            {
                "rater_a": [0, 0, 1, np.nan],
                "rater_b": [0, np.nan, 1, 1],
            }
        )
        fuser = DawidSkeneLabelFuser(n_classes=2, max_iter=10)
        fuser.fit(votes)
        proba = fuser.predict_proba(votes)
        assert len(proba) == 4
        assert proba.min().min() >= 0.0
        assert proba.max().max() <= 1.0


class TestBuildProbabilisticLabels:
    def test_basic(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 8,
                "resolved_year": [2010, 2011, 2012, 2013, 2016, 2017, 2018, 2019],
                "country": ["USA", "CAN", "MEX", "BRA", "GBR", "FRA", "DEU", "ITA"],
                "host_genus": ["Ecoli"] * 4 + ["Klebsiella"] * 4,
                "clinical_context": ["environmental"] * 8,
            }
        )
        result = build_probabilistic_labels(records, split_year=2015)
        assert len(result) == 1
        assert result["backbone_id"].iloc[0] == "BB1"
        # 4 new countries after 2015 => hard label = 1
        assert result["hard_spread_label"].iloc[0] == 1
        # Probabilistic labels should be between 0 and 1
        assert 0 <= result["prob_spread_0"].iloc[0] <= 1
        assert 0 <= result["prob_spread_1"].iloc[0] <= 1
        assert result["label_confidence"].iloc[0] > 0.5

    def test_empty(self) -> None:
        result = build_probabilistic_labels(pd.DataFrame())
        assert result.empty

    def test_no_spread(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 3,
                "resolved_year": [2010, 2011, 2012],
                "country": ["USA"] * 3,
            }
        )
        result = build_probabilistic_labels(records, split_year=2015)
        assert len(result) == 1
        assert result["hard_spread_label"].iloc[0] == 0

    def test_rater_reliability_logged(self) -> None:
        records = pd.DataFrame(
            {
                "backbone_id": ["BB1"] * 6 + ["BB2"] * 6,
                "resolved_year": [2010, 2011, 2012, 2016, 2017, 2018] * 2,
                "country": ["USA", "CAN", "MEX", "GBR", "FRA", "DEU"] * 2,
            }
        )
        result = build_probabilistic_labels(records, split_year=2015)
        assert "label_confidence" in result.columns
        assert "label_noise_estimate" in result.columns


class TestCoTeachingTrainer:
    def test_fit_predict(self) -> None:
        pytest.importorskip("torch")
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10).astype(np.float32)
        true_y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Add 20% label noise
        noise_mask = np.random.rand(n) < 0.2
        noisy_y = true_y.copy()
        noisy_y[noise_mask] = 1 - noisy_y[noise_mask]

        trainer = CoTeachingTrainer(
            input_dim=10,
            hidden_dim=32,
            forget_rate=0.2,
            num_epochs=20,
            batch_size=32,
            device="cpu",
        )
        trainer.fit(X, noisy_y)

        proba = trainer.predict_proba(X)
        assert len(proba) == n
        assert np.isfinite(proba).all()
        assert (proba >= 0).all() and (proba <= 1).all()

        pred = trainer.predict(X)
        assert len(pred) == n
        assert set(np.unique(pred)).issubset({0, 1})

    def test_ensemble_averaging(self) -> None:
        pytest.importorskip("torch")
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        trainer = CoTeachingTrainer(
            input_dim=5,
            hidden_dim=16,
            forget_rate=0.1,
            num_epochs=10,
            device="cpu",
        )
        trainer.fit(X, y)
        proba = trainer.predict_proba(X)
        # Ensemble should give moderate probabilities
        assert proba.mean() > 0.2 and proba.mean() < 0.8


class TestCausalLabelEstimator:
    def test_ate_estimation(self) -> None:
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "backbone_id": [f"BB{i}" for i in range(n)],
                "treatment": np.random.binomial(1, 0.5, n),
                "outcome": np.random.binomial(1, 0.3, n),
                "covariate_1": np.random.randn(n),
                "covariate_2": np.random.randn(n),
            }
        )
        estimator = CausalLabelEstimator(
            treatment_col="treatment",
            outcome_col="outcome",
            covariate_cols=["covariate_1", "covariate_2"],
        )
        estimator.fit(df)
        ate_df = estimator.estimate_ate(df)
        assert "ate" in ate_df.columns
        assert np.isfinite(ate_df["ate"].iloc[0])
        assert "propensity_score" in ate_df.columns
        assert ate_df["propensity_score"].between(0.01, 0.99).all()

    def test_counterfactual_labels(self) -> None:
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "backbone_id": [f"BB{i}" for i in range(n)],
                "treatment": np.random.binomial(1, 0.5, n),
                "outcome": np.random.binomial(1, 0.3, n),
                "covariate_1": np.random.randn(n),
            }
        )
        estimator = CausalLabelEstimator(
            treatment_col="treatment",
            outcome_col="outcome",
        )
        estimator.fit(df)
        cf = estimator.estimate_counterfactual_labels(df)
        assert "counterfactual_spread_prob_treated" in cf.columns
        assert "counterfactual_spread_prob_control" in cf.columns
        assert "causal_effect" in cf.columns
        assert cf["counterfactual_spread_prob_treated"].between(0, 1).all()
        assert cf["counterfactual_spread_prob_control"].between(0, 1).all()

    def test_missing_treatment_raises(self) -> None:
        df = pd.DataFrame({"wrong": [1, 2, 3]})
        estimator = CausalLabelEstimator(treatment_col="treatment")
        with pytest.raises(KeyError):
            estimator.fit(df)


def test_probabilistic_labels_abstain_for_empty_clinical_windows() -> None:
    records = pd.DataFrame(
        {
            "backbone_id": ["B1", "B1"],
            "resolved_year": [2010, 2011],
            "country": ["TR", "TR"],
            "host_genus": ["Escherichia", "Escherichia"],
            "clinical_context": ["clinical", "clinical"],
        }
    )

    labels = build_probabilistic_labels(records, split_year=2015, horizon_years=5)

    assert "rater_clinical_proxy_observed" in labels.columns
    assert labels["rater_clinical_proxy_observed"].tolist() == [False]


def test_dawid_skene_rejects_out_of_range_rater_values() -> None:
    votes = pd.DataFrame({"rater_a": [0, 1, 2]})
    fuser = DawidSkeneLabelFuser(n_classes=2)

    with pytest.raises(ValueError, match="outside"):
        fuser.fit(votes)
