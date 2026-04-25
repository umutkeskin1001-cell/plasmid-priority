"""Tests for Phase 3: Attention Consensus, Calibration, OOD, F2 Threshold."""

from __future__ import annotations

import numpy as np


class TestAttentionConsensus:
    def test_fit_predict(self) -> None:
        from plasmid_priority.modeling.attention_consensus import AttentionConsensus

        np.random.seed(42)
        branch_probs = np.random.rand(100, 3)
        y = (branch_probs.mean(axis=1) > 0.5).astype(int)
        consensus = AttentionConsensus(use_meta=False)
        consensus.fit(branch_probs, y)
        pred = consensus.predict(branch_probs)
        assert len(pred) == len(y)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_with_meta(self) -> None:
        from plasmid_priority.modeling.attention_consensus import AttentionConsensus

        np.random.seed(42)
        branch_probs = np.random.rand(50, 3)
        meta = np.random.rand(50, 2)
        y = (branch_probs.mean(axis=1) > 0.5).astype(int)
        consensus = AttentionConsensus(meta_dim=2, use_meta=True)
        consensus.fit(branch_probs, y, meta_features=meta)
        pred = consensus.predict(branch_probs, meta_features=meta)
        assert len(pred) == len(y)


class TestBetaCalibration:
    def test_beta_transform(self) -> None:
        from plasmid_priority.modeling.calibration import BetaCalibration

        np.random.seed(42)
        probs = np.clip(np.random.rand(200), 0.01, 0.99)
        y = (probs > 0.5).astype(int)
        cal = BetaCalibration(method="beta")
        cal.fit(probs, y)
        calibrated = cal.predict(probs)
        assert len(calibrated) == len(probs)
        assert (calibrated >= 0).all() and (calibrated <= 1).all()

    def test_isotonic(self) -> None:
        from plasmid_priority.modeling.calibration import BetaCalibration

        np.random.seed(42)
        probs = np.clip(np.random.rand(100), 0.01, 0.99)
        y = (probs > 0.5).astype(int)
        cal = BetaCalibration(method="isotonic")
        cal.fit(probs, y)
        calibrated = cal.predict(probs)
        assert len(calibrated) == len(probs)


class TestTemperatureScaling:
    def test_basic(self) -> None:
        from plasmid_priority.modeling.calibration import TemperatureScaling

        np.random.seed(42)
        logits = np.random.randn(100)
        y = (logits > 0).astype(int)
        ts = TemperatureScaling()
        ts.fit(logits, y)
        assert ts.temperature > 0
        calibrated = ts.predict(logits)
        assert len(calibrated) == len(logits)
        assert (calibrated >= 0).all() and (calibrated <= 1).all()


class TestMahalanobisOOD:
    def test_fit_predict(self) -> None:
        from plasmid_priority.modeling.ood_detection import MahalanobisOODDetector

        np.random.seed(42)
        X_in = np.random.randn(100, 4)
        y_in = np.random.binomial(1, 0.5, 100)
        det = MahalanobisOODDetector()
        det.fit(X_in, y_in)
        scores = det.score_samples(X_in)
        assert len(scores) == len(X_in)
        assert (scores >= 0).all()
        pred = det.predict(X_in)
        assert set(np.unique(pred)).issubset({-1, 1})

    def test_auroc(self) -> None:
        from plasmid_priority.modeling.ood_detection import MahalanobisOODDetector

        np.random.seed(42)
        X_in = np.random.randn(100, 4)
        y_in = np.random.binomial(1, 0.5, 100)
        X_out = np.random.randn(50, 4) + 5.0  # Far from training
        det = MahalanobisOODDetector()
        det.fit(X_in, y_in)
        auroc = det.compute_auroc(X_in, X_out)
        assert 0.0 <= auroc <= 1.0
        assert auroc > 0.5  # Should distinguish


class TestF2Threshold:
    def test_fit_predict(self) -> None:
        from plasmid_priority.modeling.threshold import F2ThresholdOptimizer

        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 100)
        y_score = np.random.rand(100)
        opt = F2ThresholdOptimizer(beta=2.0, cost_fn=5.0)
        opt.fit(y_true, y_score)
        assert 0.0 <= opt.best_threshold <= 1.0
        pred = opt.predict(y_score)
        assert len(pred) == len(y_true)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_tiered_risk(self) -> None:
        from plasmid_priority.modeling.threshold import F2ThresholdOptimizer

        opt = F2ThresholdOptimizer()
        scores = np.array([0.1, 0.4, 0.6, 0.8, 0.95])
        tiers = opt.tiered_risk(scores)
        assert tiers.tolist() == [0, 0, 1, 2, 3]
