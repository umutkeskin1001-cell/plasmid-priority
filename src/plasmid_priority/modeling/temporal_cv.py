from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.utils.temporal import coerce_required_years


def _require_binary_labels(labels: pd.Series) -> None:
    observed = sorted(pd.to_numeric(labels, errors="coerce").dropna().astype(int).unique().tolist())
    if observed != [0, 1]:
        raise ValueError("Temporal group folds require both classes in the eligible frame")


def temporal_group_folds(
    frame: pd.DataFrame,
    *,
    label_column: str = "spread_label",
    year_column: str = "resolved_year",
    group_column: str = "backbone_id",
    n_splits: int = 5,
    min_train_years: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")

    years = coerce_required_years(frame, year_column, context="temporal_group_folds")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    eligible = frame.loc[labels.notna()].copy()
    eligible["_year"] = years.loc[eligible.index].to_numpy(dtype=int)
    eligible["_label"] = labels.loc[eligible.index].astype(int)
    _require_binary_labels(eligible["_label"])

    unique_years = sorted(eligible["_year"].unique().tolist())
    if len(unique_years) < max(int(n_splits), 2):
        raise ValueError("Not enough distinct years for temporal group folds")

    candidate_test_years = unique_years[int(min_train_years) :]
    if not candidate_test_years:
        raise ValueError("Temporal group folds require at least one test year")

    selected_test_years = candidate_test_years[-int(n_splits) :]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_year in selected_test_years:
        train_mask = eligible["_year"] < int(test_year)
        test_mask = eligible["_year"] == int(test_year)
        train_groups = set(eligible.loc[train_mask, group_column].astype(str))
        test_groups = set(eligible.loc[test_mask, group_column].astype(str))
        overlap = train_groups & test_groups
        if overlap:
            train_mask &= ~eligible[group_column].astype(str).isin(overlap)
        train_labels = eligible.loc[train_mask, "_label"]
        test_labels = eligible.loc[test_mask, "_label"]
        if sorted(train_labels.unique().tolist()) != [0, 1]:
            continue
        if sorted(test_labels.unique().tolist()) != [0, 1]:
            continue
        train_idx = frame.index.get_indexer(eligible.loc[train_mask].index)
        test_idx = frame.index.get_indexer(eligible.loc[test_mask].index)
        folds.append((train_idx.astype(int), test_idx.astype(int)))

    if not folds:
        raise ValueError("No temporal group folds with both classes in train and test windows")
    return folds
