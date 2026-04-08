"""Experiment-level acceptance gates for discrimination and acceptance optimization.

This module provides honest reporting infrastructure for model selection and
gain interpretation, separate from the frozen scientific acceptance thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Thresholds for gain interpretation (from v7 optimization plan)
TIE_NOISE_THRESHOLD = 0.015  # AUC differences < 0.015 are indistinguishable from noise
MEANINGFUL_GAIN_THRESHOLD = 0.025  # AUC gains >= 0.025 are considered meaningful


@dataclass(frozen=True)
class ConfigCandidate:
    """Single candidate config from a grid search.

    This represents one model configuration with all required fields for
    honest reporting and gate evaluation.

    Attributes:
        config_name: Identifier for the configuration (e.g., "discovery_9f_source")
        raw_auc: Out-of-fold AUC of this configuration
        raw_ci: 95% bootstrap confidence interval (lower, upper)
        ece: Expected Calibration Error
        selection_adjusted_p: p-value vs baseline from permutation null
        leakage_review_pass: Whether leakage review passed
        knownness_gap: Knownness gap (monitored, not gated); None if not computed
    """

    config_name: str
    raw_auc: float
    raw_ci: tuple[float, float]
    ece: float
    selection_adjusted_p: float
    leakage_review_pass: bool
    knownness_gap: float | None = None


@dataclass(frozen=True)
class HonestModelResult:
    """Clear separation of selected config vs. reported score.

    This dataclass implements the "selection bias mitigation" strategy from
    the v7 optimization plan: the config with the highest raw AUC is recorded
    as selected_config, but the reported AUC is the mean of the top-3 configs
    to reduce optimistic selection bias.

    Attributes:
        selected_config: The winning config (highest raw_auc)
        top_k_configs: Top-3 configs by raw AUC
        reported_selection_adjusted_auc: Mean AUC of top-3 (reduces selection bias)
        reported_ci: Pooled CI of top-3 (min/max envelope)
    """

    selected_config: ConfigCandidate
    top_k_configs: tuple[ConfigCandidate, ...]
    reported_selection_adjusted_auc: float
    reported_ci: tuple[float, float]


@dataclass(frozen=True)
class ExperimentAcceptanceGates:
    """Experiment-level acceptance gates (relaxed thresholds for optimization).

    These gates use relaxed thresholds (ECE < 0.10, p < 0.05) compared to the
    frozen scientific acceptance thresholds (ECE <= 0.05, p <= 0.01).

    Attributes:
        ece_max: Maximum acceptable ECE (default 0.10)
        selection_adjusted_p_max: Maximum acceptable p-value (default 0.05)
        require_leakage_pass: Whether leakage review is required (default True)
        rolling_origin_gap_max: Max rolling-origin gap for governance (default 0.040)
            Note: This is a placeholder contract; enforced only when rolling_origin
            infrastructure exists (Phase 0).
        hybrid_review_fraction_max: Max review fraction for hybrid models (default 0.25)
            Note: This is a placeholder contract; enforced only when hybrid
            calibration infrastructure exists.
    """

    ece_max: float = 0.10
    selection_adjusted_p_max: float = 0.05
    require_leakage_pass: bool = True
    rolling_origin_gap_max: float | None = 0.040
    hybrid_review_fraction_max: float | None = 0.25


def interpret_gain(delta_auc: float, delta_ci: float = 0.030) -> str:
    """Interpret an AUC gain according to the v7 optimization plan criteria.

    Args:
        delta_auc: The observed AUC gain
        delta_ci: The uncertainty (half-width of 95% CI), default 0.030

    Returns:
        "TIE/NOISE" if delta_auc < 0.015 (indistinguishable from baseline)
        "MARGINAL" if 0.015 <= delta_auc < 0.025 (technically positive but small)
        "MEANINGFUL" if delta_auc >= 0.025 (exceeds uncertainty floor)

    Examples:
        >>> interpret_gain(0.012)
        'TIE/NOISE'
        >>> interpret_gain(0.018)
        'MARGINAL'
        >>> interpret_gain(0.028)
        'MEANINGFUL'
    """
    if delta_auc < TIE_NOISE_THRESHOLD:
        return "TIE/NOISE"
    elif delta_auc < MEANINGFUL_GAIN_THRESHOLD:
        return "MARGINAL"
    else:
        return "MEANINGFUL"


def compute_honest_result(candidates: list[ConfigCandidate]) -> HonestModelResult:
    """Compute honest model result from a list of config candidates.

    This implements the selection bias mitigation strategy from the v7 plan:
    - Rank configs by raw AUC
    - Select top-3
    - Report mean AUC of top-3 (not the maximum)
    - CI is pooled as min/max envelope (conservative)

    Args:
        candidates: List of ConfigCandidate objects. Must have at least 1 entry.

    Returns:
        HonestModelResult with selected config and bias-mitigated report.

    Raises:
        ValueError: If candidates list is empty.
    """
    if not candidates:
        raise ValueError("candidates list must have at least 1 entry")

    # Sort by raw_auc descending
    sorted_candidates = sorted(
        candidates, key=lambda c: c.raw_auc, reverse=True
    )

    # Take top-3 (or fewer if less available)
    top_k = min(3, len(sorted_candidates))
    top_3 = tuple(sorted_candidates[:top_k])

    # Selected config is the winner (max raw AUC)
    selected = top_3[0]

    # Reported AUC is mean of top-3 (reduces selection bias)
    reported_auc = float(np.mean([c.raw_auc for c in top_3]))

    # CI is pooled as min/max envelope (conservative)
    lower_bounds = [c.raw_ci[0] for c in top_3]
    upper_bounds = [c.raw_ci[1] for c in top_3]
    reported_ci = (float(min(lower_bounds)), float(max(upper_bounds)))

    return HonestModelResult(
        selected_config=selected,
        top_k_configs=top_3,
        reported_selection_adjusted_auc=reported_auc,
        reported_ci=reported_ci,
    )


def evaluate_experiment_gates(
    result: HonestModelResult,
    gates: ExperimentAcceptanceGates,
    *,
    rolling_origin_gap: float | None = None,
    hybrid_review_fraction: float | None = None,
) -> dict[str, Any]:
    """Evaluate experiment acceptance gates for a model result.

    Args:
        result: The HonestModelResult to evaluate
        gates: ExperimentAcceptanceGates with thresholds
        rolling_origin_gap: Optional rolling-origin gap for governance models.
            If None, the rolling_origin_gap gate returns None (not evaluated).
        hybrid_review_fraction: Optional review fraction for hybrid models.
            If None, the hybrid_review_fraction gate returns None (not evaluated).

    Returns:
        Dictionary with per-gate results:
        - ece: bool | None - ECE gate pass/fail
        - selection_adjusted_p: bool | None - p-value gate pass/fail
        - leakage_review: bool | None - Leakage review gate pass/fail
        - rolling_origin_gap: bool | None - Rolling-origin gap gate (None if not provided)
        - hybrid_review_fraction: bool | None - Hybrid review fraction gate (None if not provided)
        - overall: bool | None - Overall acceptance (all gates must pass)
        - gate_details: dict - Detailed values for each gate

    Notes:
        - Placeholder contract gates (rolling_origin_gap, hybrid_review_fraction)
          return None when their input value is None.
        - This function does NOT modify FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS.
    """
    selected = result.selected_config

    # Evaluate each gate
    ece_pass = (
        selected.ece <= gates.ece_max
        if not np.isnan(selected.ece)
        else None
    )

    p_pass = (
        selected.selection_adjusted_p <= gates.selection_adjusted_p_max
        if not np.isnan(selected.selection_adjusted_p)
        else None
    )

    leakage_pass = (
        selected.leakage_review_pass
        if gates.require_leakage_pass
        else None
    )

    # Placeholder contract: rolling_origin_gap
    if rolling_origin_gap is not None and gates.rolling_origin_gap_max is not None:
        rolling_pass = rolling_origin_gap <= gates.rolling_origin_gap_max
    else:
        rolling_pass = None

    # Placeholder contract: hybrid_review_fraction
    if hybrid_review_fraction is not None and gates.hybrid_review_fraction_max is not None:
        hybrid_pass = hybrid_review_fraction <= gates.hybrid_review_fraction_max
    else:
        hybrid_pass = None

    # Overall: all evaluated gates must pass
    evaluated_gates = [
        g for g in [ece_pass, p_pass, leakage_pass, rolling_pass, hybrid_pass]
        if g is not None
    ]
    overall = all(evaluated_gates) if evaluated_gates else None

    return {
        "ece": ece_pass,
        "selection_adjusted_p": p_pass,
        "leakage_review": leakage_pass,
        "rolling_origin_gap": rolling_pass,
        "hybrid_review_fraction": hybrid_pass,
        "overall": overall,
        "gate_details": {
            "ece_value": selected.ece,
            "ece_threshold": gates.ece_max,
            "p_value": selected.selection_adjusted_p,
            "p_threshold": gates.selection_adjusted_p_max,
            "rolling_origin_gap_value": rolling_origin_gap,
            "rolling_origin_gap_threshold": gates.rolling_origin_gap_max,
            "hybrid_review_fraction_value": hybrid_review_fraction,
            "hybrid_review_fraction_threshold": gates.hybrid_review_fraction_max,
        },
    }
