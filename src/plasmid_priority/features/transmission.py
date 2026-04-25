"""Transmission (T) feature computation.

This module provides functions for computing transmission-related features
for plasmid backbone analysis.
"""

from __future__ import annotations

import pandas as pd


def _support_factor(n: int, pseudocount: float = 3.0) -> float:
    """Compute support factor based on sample size with pseudocount smoothing."""
    return n / (n + pseudocount)


def compute_feature_t(training_canonical: pd.DataFrame) -> pd.DataFrame:
    """Compute the mobility component at backbone level.

    Args:
        training_canonical: Training canonical dataframe

    Returns:
        DataFrame with transmission features T_raw and T_eff
    """
    summary = (
        training_canonical.groupby("backbone_id", sort=False)
        .agg(
            member_count_train=("canonical_id", "size"),
            relaxase_support=("has_relaxase", "mean"),
            mpf_support=("has_mpf", "mean"),
            orit_support=("has_orit", "mean"),
            mobilizable_support=("is_mobilizable", "mean"),
        )
        .reset_index()
    )
    summary["support_shrinkage"] = summary["member_count_train"].map(
        lambda value: _support_factor(int(value)),
    )
    summary["T_raw"] = (
        (1.0 * summary["relaxase_support"])
        + (1.0 * summary["mpf_support"])
        + (0.75 * summary["orit_support"])
        + (0.50 * summary["mobilizable_support"])
    ) / 3.25
    summary["T_eff"] = summary["T_raw"] * summary["support_shrinkage"]
    return summary
