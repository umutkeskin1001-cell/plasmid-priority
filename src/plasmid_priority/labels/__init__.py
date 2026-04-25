"""Probabilistic label engineering and noise-robust training.

Provides Dawid-Skene multi-rater label fusion, Co-teaching for noisy labels,
and causal counterfactual label estimation.
"""

from __future__ import annotations

from plasmid_priority.labels.counterfactual import CausalLabelEstimator
from plasmid_priority.labels.probabilistic import (
    DawidSkeneLabelFuser,
    build_probabilistic_labels,
)

try:
    from plasmid_priority.labels.coteaching import CoTeachingTrainer
except ImportError:
    CoTeachingTrainer = None  # type: ignore[assignment,misc]

__all__ = [
    "DawidSkeneLabelFuser",
    "build_probabilistic_labels",
    "CoTeachingTrainer",
    "CausalLabelEstimator",
]
