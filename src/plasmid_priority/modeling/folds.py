"""Stratified fold generation utilities.

This module provides functions for pre-computing stratified folds for
cross-validation, ensuring consistent splits across different model runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def precompute_stratified_folds(
    df: pd.DataFrame,
    *,
    label_column: str = "spread_label",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pre-compute stratified fold assignments for a dataset.

    Args:
        df: Input dataframe with samples
        label_column: Column containing the target label
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with original data plus 'fold' column indicating fold assignment
    """
    working = df.copy()

    if label_column not in working.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe")

    # Filter to labeled samples only
    labeled = working.loc[working[label_column].notna()].copy()
    labeled[label_column] = labeled[label_column].astype(int)

    if len(labeled) < n_splits:
        raise ValueError(
            f"Not enough samples ({len(labeled)}) for {n_splits} folds",
        )

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Assign fold numbers
    fold_assignments = np.full(len(working), -1, dtype=int)

    for fold_idx, (_, test_idx) in enumerate(skf.split(labeled, labeled[label_column])):
        # Map back to original indices
        original_indices = labeled.iloc[test_idx].index
        fold_assignments[working.index.get_indexer(original_indices)] = fold_idx

    working["fold"] = fold_assignments

    return working


def get_fold_data(
    df: pd.DataFrame,
    fold: int,
    *,
    mode: str = "train",
) -> pd.DataFrame:
    """Get train or validation data for a specific fold.

    Args:
        df: DataFrame with pre-computed fold assignments (must have 'fold' column)
        fold: Fold number to retrieve
        mode: 'train' or 'val'

    Returns:
        Subset of dataframe for the requested fold and mode
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame must have 'fold' column (use precompute_stratified_folds)")

    if mode == "train":
        return df.loc[df["fold"] != fold]
    elif mode == "val":
        return df.loc[df["fold"] == fold]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")


def get_fold_splits(
    df: pd.DataFrame,
    n_splits: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Get all train/validation splits for stratified folds.

    Args:
        df: DataFrame with pre-computed fold assignments
        n_splits: Number of splits (auto-detected if None)

    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame must have 'fold' column")

    if n_splits is None:
        n_splits = int(df["fold"].max()) + 1

    splits = []
    for fold in range(n_splits):
        train_df = get_fold_data(df, fold, mode="train")
        val_df = get_fold_data(df, fold, mode="val")
        splits.append((train_df, val_df))

    return splits
