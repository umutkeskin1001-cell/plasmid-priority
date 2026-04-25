"""Survival analysis module for Plasmid Priority.

Provides Cox Proportional Hazards, competing risks (Fine-Gray),
time-dependent ROC, and Brier score evaluation for continuous-time
plasmid spread prediction.
"""

from __future__ import annotations

from plasmid_priority.survival.adaptive_split import AdaptiveTemporalSplitter
from plasmid_priority.survival.competing_risks import FineGrayCompetingRisks
from plasmid_priority.survival.cox_ph import (
    CoxPHSurvivalModel,
    compute_td_roc,
    time_dependent_brier_score,
)
from plasmid_priority.survival.lead_time import LeadTimeBiasCorrector

__all__ = [
    "CoxPHSurvivalModel",
    "compute_td_roc",
    "time_dependent_brier_score",
    "FineGrayCompetingRisks",
    "AdaptiveTemporalSplitter",
    "LeadTimeBiasCorrector",
]
