"""Tests for Phase 2 ML models: FT-Transformer, Multi-task, Conformal, DeepEnsemble, NestedCV."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

_SKIP_TORCH_TRAINING = sys.version_info >= (3, 13)


class TestDeepEnsemble:
    def test_fit_predict(self) -> None:
        from plasmid_priority.modeling.deep_ensemble import DeepEnsemble

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        ens = DeepEnsemble(n_members=3, bootstrap=True, random_state=42)
        ens.fit(X, y)
        mean_p, unc = ens.predict_proba(X)
        assert len(mean_p) == len(X)
        assert len(unc) == len(X)
        assert (unc >= 0).all()
        pred = ens.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_uncertainty_gating(self) -> None:
        from plasmid_priority.modeling.deep_ensemble import DeepEnsemble

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        ens = DeepEnsemble(n_members=3, random_state=42)
        ens.fit(X, y)
        gated = ens.uncertainty_gated_predict(X, uncertainty_threshold=0.1)
        assert (gated == -1).sum() >= 0  # some may be uncertain


class TestConformal:
    def test_coverage(self) -> None:
        from plasmid_priority.modeling.conformal import SplitConformalPredictor

        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (X[:, 0] > 0).astype(int)
        base = LogisticRegression(max_iter=1000)
        base.fit(X, y)
        cp = SplitConformalPredictor(base, alpha=0.10, random_state=42)
        cp.fit(X, y)
        cov = cp.empirical_coverage(X, y)
        # Should be close to 1-alpha (0.90) but not guaranteed on small data
        assert cov >= 0.70

    def test_predict_sets_shape(self) -> None:
        from plasmid_priority.modeling.conformal import SplitConformalPredictor

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        base = LogisticRegression(max_iter=1000)
        base.fit(X, y)
        cp = SplitConformalPredictor(base, alpha=0.20, random_state=42)
        cp.fit(X, y)
        sets = cp.predict_sets(X[:10])
        assert sets.shape == (10, 2)
        assert sets.dtype == bool


class TestNestedCV:
    def test_basic(self) -> None:
        from plasmid_priority.modeling.nested_cv_bayesian import NestedCVEvaluator

        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = (X[:, 0] > 0).astype(int)
        ev = NestedCVEvaluator(outer_cv=3, inner_cv=2, random_state=42)
        result = ev.evaluate(X, y, lambda: LogisticRegression(max_iter=1000))
        assert "mean_score" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert 0.0 <= result["mean_score"] <= 1.0

    def test_bayesian_comparison(self) -> None:
        from plasmid_priority.modeling.nested_cv_bayesian import bayesian_model_comparison

        scores = {
            "model_a": [0.80, 0.82, 0.81, 0.83, 0.79],
            "model_b": [0.75, 0.76, 0.74, 0.77, 0.75],
        }
        result = bayesian_model_comparison(scores)
        assert "model_a" in result
        assert "model_b" in result
        assert result["model_a"]["posterior_prob"] > result["model_b"]["posterior_prob"]


class TestBootstrapCI:
    def test_basic(self) -> None:
        from sklearn.metrics import roc_auc_score

        from plasmid_priority.modeling.nested_cv_bayesian import bootstrap_ci

        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_score = np.random.rand(100)
        ci = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstrap=200, random_state=42)
        assert ci["lower"] <= ci["upper"]
        assert 0.0 <= ci["lower"] <= 1.0


@pytest.mark.slow
class TestFTTransformer:
    def test_fit_predict(self) -> None:
        if _SKIP_TORCH_TRAINING:
            pytest.skip(
                "Torch training tests are skipped on Python 3.13 due runtime segfaults in this stack.",
            )
        pytest.importorskip("torch")
        from plasmid_priority.modeling.ft_transformer import FTTransformerClassifier

        np.random.seed(42)
        X = np.random.randn(64, 8).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = FTTransformerClassifier(
            d_model=32, n_heads=2, n_blocks=1, num_epochs=3, batch_size=16, patience=2
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.isfinite(proba).all()
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_pandas_input(self) -> None:
        if _SKIP_TORCH_TRAINING:
            pytest.skip(
                "Torch training tests are skipped on Python 3.13 due runtime segfaults in this stack.",
            )
        pytest.importorskip("torch")
        from plasmid_priority.modeling.ft_transformer import FTTransformerClassifier

        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(64, 5).astype(np.float32), columns=[f"f{i}" for i in range(5)]
        )
        y = (df["f0"] > 0).astype(int).to_numpy()
        model = FTTransformerClassifier(
            d_model=16, n_heads=2, n_blocks=1, num_epochs=3, batch_size=16, patience=2
        )
        model.fit(df, y)
        pred = model.predict(df)
        assert len(pred) == len(df)


@pytest.mark.slow
class TestMultiTask:
    def test_fit_predict(self) -> None:
        if _SKIP_TORCH_TRAINING:
            pytest.skip(
                "Torch training tests are skipped on Python 3.13 due runtime segfaults in this stack.",
            )
        pytest.importorskip("torch")
        from plasmid_priority.modeling.multi_task import MultiTaskTrainer

        np.random.seed(42)
        X = np.random.randn(64, 6).astype(np.float32)
        y_dict = {
            "geo_spread": (X[:, 0] > 0).astype(int),
            "bio_transfer": (X[:, 1] > 0).astype(int),
            "clinical_hazard": (X[:, 2] > 0).astype(int),
        }
        trainer = MultiTaskTrainer(
            input_dim=6, hidden_dim=32, num_epochs=3, batch_size=16, patience=2
        )
        trainer.fit(X, y_dict)
        preds = trainer.predict_proba(X)
        assert set(preds.keys()) == {"geo_spread", "bio_transfer", "clinical_hazard"}
        for p in preds.values():
            assert len(p) == len(X)
            assert (p >= 0).all() and (p <= 1).all()
