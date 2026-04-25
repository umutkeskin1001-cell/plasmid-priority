from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.modeling.temporal_cv import temporal_group_folds


def test_temporal_group_folds_have_no_group_overlap() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2010, 2011, 2011, 2012, 2012, 2013, 2013],
            "backbone_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "spread_label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    folds = temporal_group_folds(
        frame,
        label_column="spread_label",
        year_column="resolved_year",
        group_column="backbone_id",
        n_splits=3,
    )

    assert folds
    for train_idx, test_idx in folds:
        train_groups = set(frame.iloc[train_idx]["backbone_id"])
        test_groups = set(frame.iloc[test_idx]["backbone_id"])
        assert train_groups.isdisjoint(test_groups)
        assert frame.iloc[train_idx]["resolved_year"].max() < frame.iloc[test_idx]["resolved_year"].min()


def test_temporal_group_folds_require_both_classes_in_test_window() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2011, 2012, 2013],
            "backbone_id": ["A", "B", "C", "D"],
            "spread_label": [0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="both classes"):
        temporal_group_folds(frame, n_splits=2)


def test_temporal_group_folds_require_label_and_group_columns() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2011],
            "backbone_id": ["A", "B"],
            "spread_label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Missing label column"):
        temporal_group_folds(frame.drop(columns=["spread_label"]))
    with pytest.raises(ValueError, match="Missing group column"):
        temporal_group_folds(frame.drop(columns=["backbone_id"]))


def test_temporal_group_folds_require_enough_years() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2010, 2011, 2011],
            "backbone_id": ["A", "B", "C", "D"],
            "spread_label": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="Not enough distinct years"):
        temporal_group_folds(frame, n_splits=3)


def test_temporal_group_folds_require_available_test_year_after_training_window() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2010, 2011, 2011],
            "backbone_id": ["A", "B", "C", "D"],
            "spread_label": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="at least one test year"):
        temporal_group_folds(frame, n_splits=2, min_train_years=2)


def test_temporal_group_folds_fail_when_group_overlap_removes_training_classes() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2010, 2011, 2011],
            "backbone_id": ["A", "B", "A", "C"],
            "spread_label": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="No temporal group folds"):
        temporal_group_folds(frame, n_splits=2)
