"""Tests for experiment-level acceptance gates and honest reporting."""

from __future__ import annotations

import unittest

from plasmid_priority.modeling.experiment_gates import (
    TIE_NOISE_THRESHOLD,
    MEANINGFUL_GAIN_THRESHOLD,
    ConfigCandidate,
    ExperimentAcceptanceGates,
    HonestModelResult,
    compute_honest_result,
    evaluate_experiment_gates,
    interpret_gain,
)
from plasmid_priority.modeling import (
    assert_all_discovery_safe,
    get_research_models_by_track,
)
from plasmid_priority.reporting.model_audit import FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS


class TestInterpretGain(unittest.TestCase):
    """Tests for gain interpretation function."""

    def test_interpret_gain_noise(self) -> None:
        """delta < 0.015 should return TIE/NOISE."""
        self.assertEqual(interpret_gain(0.012), "TIE/NOISE")
        self.assertEqual(interpret_gain(0.014), "TIE/NOISE")
        self.assertEqual(interpret_gain(0.0), "TIE/NOISE")
        self.assertEqual(interpret_gain(-0.01), "TIE/NOISE")

    def test_interpret_gain_marginal(self) -> None:
        """0.015 <= delta < 0.025 should return MARGINAL."""
        self.assertEqual(interpret_gain(0.015), "MARGINAL")
        self.assertEqual(interpret_gain(0.020), "MARGINAL")
        self.assertEqual(interpret_gain(0.024), "MARGINAL")

    def test_interpret_gain_meaningful(self) -> None:
        """delta >= 0.025 should return MEANINGFUL."""
        self.assertEqual(interpret_gain(0.025), "MEANINGFUL")
        self.assertEqual(interpret_gain(0.030), "MEANINGFUL")
        self.assertEqual(interpret_gain(0.050), "MEANINGFUL")

    def test_threshold_constants(self) -> None:
        """Verify threshold constants match plan specification."""
        self.assertEqual(TIE_NOISE_THRESHOLD, 0.015)
        self.assertEqual(MEANINGFUL_GAIN_THRESHOLD, 0.025)


class TestComputeHonestResult(unittest.TestCase):
    """Tests for honest result computation with top-3 selection."""

    def _make_candidate(
        self,
        name: str,
        auc: float,
        ci: tuple[float, float] = (0.70, 0.80),
        ece: float = 0.05,
        p: float = 0.01,
        leakage: bool = True,
    ) -> ConfigCandidate:
        return ConfigCandidate(
            config_name=name,
            raw_auc=auc,
            raw_ci=ci,
            ece=ece,
            selection_adjusted_p=p,
            leakage_review_pass=leakage,
        )

    def test_honest_result_top3_mean(self) -> None:
        """Top-3 selection reduces to mean of top 3 AUCs."""
        candidates = [
            self._make_candidate("config_a", 0.800, (0.78, 0.82)),
            self._make_candidate("config_b", 0.790, (0.77, 0.81)),
            self._make_candidate("config_c", 0.780, (0.76, 0.80)),
            self._make_candidate("config_d", 0.750, (0.73, 0.77)),
        ]
        result = compute_honest_result(candidates)

        # Should select top 3
        self.assertEqual(len(result.top_k_configs), 3)
        # Reported AUC should be mean of top 3
        expected_mean = (0.800 + 0.790 + 0.780) / 3
        self.assertAlmostEqual(result.reported_selection_adjusted_auc, expected_mean, places=6)

    def test_honest_result_uses_raw_for_selection_but_reports_top3_mean(self) -> None:
        """selected_config uses max raw_auc; reported_selection_adjusted_auc is top-3 mean (not max)."""
        candidates = [
            self._make_candidate("winner", 0.850, (0.84, 0.86)),
            self._make_candidate("second", 0.840, (0.83, 0.85)),
            self._make_candidate("third", 0.830, (0.82, 0.84)),
        ]
        result = compute_honest_result(candidates)

        # selected_config should be the winner (highest raw AUC)
        self.assertEqual(result.selected_config.config_name, "winner")
        self.assertEqual(result.selected_config.raw_auc, 0.850)

        # But reported AUC should be mean, not max
        expected_mean = (0.850 + 0.840 + 0.830) / 3
        self.assertAlmostEqual(result.reported_selection_adjusted_auc, expected_mean, places=6)
        self.assertNotEqual(result.reported_selection_adjusted_auc, 0.850)

    def test_honest_result_ci_envelope(self) -> None:
        """CI should be min/max envelope of top-3 CIs."""
        candidates = [
            self._make_candidate("a", 0.800, (0.75, 0.85)),
            self._make_candidate("b", 0.790, (0.78, 0.80)),
            self._make_candidate("c", 0.780, (0.76, 0.82)),
        ]
        result = compute_honest_result(candidates)

        # CI lower should be min of all lowers
        self.assertEqual(result.reported_ci[0], 0.75)
        # CI upper should be max of all uppers
        self.assertEqual(result.reported_ci[1], 0.85)

    def test_honest_result_with_fewer_than_3(self) -> None:
        """Should work with fewer than 3 candidates."""
        candidates = [
            self._make_candidate("a", 0.800, (0.78, 0.82)),
            self._make_candidate("b", 0.790, (0.77, 0.81)),
        ]
        result = compute_honest_result(candidates)

        self.assertEqual(len(result.top_k_configs), 2)
        expected_mean = (0.800 + 0.790) / 2
        self.assertAlmostEqual(result.reported_selection_adjusted_auc, expected_mean, places=6)

    def test_honest_result_empty_list_raises(self) -> None:
        """Empty candidates list should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_honest_result([])


class TestExperimentGates(unittest.TestCase):
    """Tests for experiment-level acceptance gate evaluation."""

    def _make_result(
        self,
        ece: float = 0.05,
        p: float = 0.01,
        leakage: bool = True,
    ) -> HonestModelResult:
        candidate = ConfigCandidate(
            config_name="test",
            raw_auc=0.800,
            raw_ci=(0.78, 0.82),
            ece=ece,
            selection_adjusted_p=p,
            leakage_review_pass=leakage,
        )
        return HonestModelResult(
            selected_config=candidate,
            top_k_configs=(candidate,),
            reported_selection_adjusted_auc=0.800,
            reported_ci=(0.78, 0.82),
        )

    def test_experiment_gate_ece_reject(self) -> None:
        """ECE > 0.10 should cause rejection."""
        result = self._make_result(ece=0.15)
        gates = ExperimentAcceptanceGates(ece_max=0.10)
        evaluation = evaluate_experiment_gates(result, gates)

        self.assertFalse(evaluation["ece"])
        self.assertFalse(evaluation["overall"])

    def test_experiment_gate_ece_pass(self) -> None:
        """ECE < 0.10 should pass."""
        result = self._make_result(ece=0.05)
        gates = ExperimentAcceptanceGates(ece_max=0.10)
        evaluation = evaluate_experiment_gates(result, gates)

        self.assertTrue(evaluation["ece"])

    def test_experiment_gate_p_reject(self) -> None:
        """p > 0.05 should cause rejection."""
        result = self._make_result(p=0.10)
        gates = ExperimentAcceptanceGates(selection_adjusted_p_max=0.05)
        evaluation = evaluate_experiment_gates(result, gates)

        self.assertFalse(evaluation["selection_adjusted_p"])
        self.assertFalse(evaluation["overall"])

    def test_experiment_gate_p_pass(self) -> None:
        """p < 0.05 should pass."""
        result = self._make_result(p=0.01)
        gates = ExperimentAcceptanceGates(selection_adjusted_p_max=0.05)
        evaluation = evaluate_experiment_gates(result, gates)

        self.assertTrue(evaluation["selection_adjusted_p"])

    def test_governance_rolling_origin_gap_reject(self) -> None:
        """gap > 0.040 should cause rejection when provided."""
        result = self._make_result()
        gates = ExperimentAcceptanceGates(rolling_origin_gap_max=0.040)
        evaluation = evaluate_experiment_gates(result, gates, rolling_origin_gap=0.050)

        self.assertFalse(evaluation["rolling_origin_gap"])
        self.assertFalse(evaluation["overall"])

    def test_rolling_origin_gap_skipped_when_none(self) -> None:
        """gap=None should return None (not evaluated)."""
        result = self._make_result()
        gates = ExperimentAcceptanceGates(rolling_origin_gap_max=0.040)
        evaluation = evaluate_experiment_gates(result, gates, rolling_origin_gap=None)

        self.assertIsNone(evaluation["rolling_origin_gap"])
        # overall should still pass since other gates pass
        self.assertTrue(evaluation["overall"])

    def test_all_gates_pass(self) -> None:
        """All clear should result in overall=True."""
        result = self._make_result(ece=0.05, p=0.01, leakage=True)
        gates = ExperimentAcceptanceGates()
        evaluation = evaluate_experiment_gates(result, gates)

        self.assertTrue(evaluation["ece"])
        self.assertTrue(evaluation["selection_adjusted_p"])
        self.assertTrue(evaluation["leakage_review"])
        self.assertTrue(evaluation["overall"])

    def test_experiment_gates_do_not_mutate_frozen_thresholds(self) -> None:
        """FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS should be unchanged after gate evaluation."""
        original_ece = FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["ece_max"]
        original_p = FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["selection_adjusted_p_max"]

        result = self._make_result()
        gates = ExperimentAcceptanceGates()
        evaluate_experiment_gates(result, gates)

        # Verify frozen thresholds unchanged
        self.assertEqual(
            FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["ece_max"], original_ece
        )
        self.assertEqual(
            FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS["selection_adjusted_p_max"], original_p
        )


class TestKnownnessGapOptional(unittest.TestCase):
    """Tests for knownness_gap being optional."""

    def test_knownness_gap_optional_does_not_break_gate_eval(self) -> None:
        """missing knownness_gap still evaluates gates; gap reported but not gated."""
        candidate = ConfigCandidate(
            config_name="test",
            raw_auc=0.800,
            raw_ci=(0.78, 0.82),
            ece=0.05,
            selection_adjusted_p=0.01,
            leakage_review_pass=True,
            knownness_gap=None,  # Not provided
        )
        result = HonestModelResult(
            selected_config=candidate,
            top_k_configs=(candidate,),
            reported_selection_adjusted_auc=0.800,
            reported_ci=(0.78, 0.82),
        )
        gates = ExperimentAcceptanceGates()
        evaluation = evaluate_experiment_gates(result, gates)

        # Gates should still evaluate
        self.assertTrue(evaluation["overall"])
        # knownness_gap is in candidate but not part of gate evaluation
        self.assertIsNone(candidate.knownness_gap)


class TestAssertAllDiscoverySafe(unittest.TestCase):
    """Tests for discovery-safe feature assertion."""

    def test_assert_all_discovery_safe_accepts_clean_list(self) -> None:
        """All-discovery list should not raise error."""
        # These are all in _DISCOVERY_FEATURES
        features = ["T_eff_norm", "H_obs_specialization_norm", "A_eff_norm", "orit_support"]
        assert_all_discovery_safe(features)  # Should not raise

    def test_assert_all_discovery_safe_rejects_governance_feature(self) -> None:
        """Governance feature in list should raise ValueError."""
        # H_external_host_range_support is in _GOVERNANCE_FEATURES
        features = ["T_eff_norm", "H_external_host_range_support"]
        with self.assertRaises(ValueError) as context:
            assert_all_discovery_safe(features)
        self.assertIn("H_external_host_range_support", str(context.exception))


class TestGetResearchModelsByTrack(unittest.TestCase):
    """Tests for track-based filtering of research models."""

    def test_get_research_models_by_track_filters_correctly(self) -> None:
        """Should return only models matching requested track."""
        # The new discovery models should be in discovery track
        discovery_models = get_research_models_by_track("discovery")
        self.assertIn("discovery_4f_hybrid", discovery_models)
        self.assertIn("discovery_9f_source", discovery_models)

        # Governance model should not be in discovery
        self.assertNotIn("governance_15f_pruned", discovery_models)

        # Governance model should be in governance track
        governance_models = get_research_models_by_track("governance")
        self.assertIn("governance_15f_pruned", governance_models)

        # Discovery models should not be in governance
        self.assertNotIn("discovery_4f_hybrid", governance_models)


if __name__ == "__main__":
    unittest.main()
