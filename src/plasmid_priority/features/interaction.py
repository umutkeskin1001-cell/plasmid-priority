"""Interaction feature utilities for modeling.

This module provides functions for computing interaction features between
different feature sets, particularly T×A (Transmission × AMR) and H×A (Host × AMR).
"""

from __future__ import annotations

import pandas as pd


def build_interaction_features(
    scored: pd.DataFrame,
    *,
    include_TxA: bool = True,
    include_HxA: bool = True,
) -> pd.DataFrame:
    """Build interaction features between different feature sets.

    Args:
        scored: Input dataframe with base features
        include_TxA: Whether to include Transmission × AMR interaction features
        include_HxA: Whether to include Host × AMR interaction features

    Returns:
        Dataframe with additional interaction feature columns
    """
    working = scored.copy()

    if include_TxA:
        working = _build_TxA_features(working)

    if include_HxA:
        working = _build_HxA_features(working)

    return working


def _build_TxA_features(scored: pd.DataFrame) -> pd.DataFrame:
    """Build Transmission × AMR interaction features.

    Interaction between transmission-related features (T_eff, T_phylogenetic, etc.)
    and AMR-related features.
    """
    working = scored.copy()

    # Transmission features
    T_cols = [col for col in working.columns if col.startswith("T_")]
    # AMR features
    A_cols = [col for col in working.columns if col.startswith("A_")]

    for t_col in T_cols:
        for a_col in A_cols:
            if t_col in working.columns and a_col in working.columns:
                interaction_name = f"{t_col}_x_{a_col}"
                working[interaction_name] = pd.to_numeric(working[t_col], errors="coerce").fillna(
                    0.0
                ) * pd.to_numeric(working[a_col], errors="coerce").fillna(0.0)

    return working


def _build_HxA_features(scored: pd.DataFrame) -> pd.DataFrame:
    """Build Host × AMR interaction features.

    Interaction between host-related features (H_obs, H_phylogenetic, etc.)
    and AMR-related features.
    """
    working = scored.copy()

    # Host features
    H_cols = [col for col in working.columns if col.startswith("H_")]
    # AMR features
    A_cols = [col for col in working.columns if col.startswith("A_")]

    for h_col in H_cols:
        for a_col in A_cols:
            if h_col in working.columns and a_col in working.columns:
                interaction_name = f"{h_col}_x_{a_col}"
                working[interaction_name] = pd.to_numeric(working[h_col], errors="coerce").fillna(
                    0.0
                ) * pd.to_numeric(working[a_col], errors="coerce").fillna(0.0)

    return working
