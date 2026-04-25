from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.shared.temporal import (
    TemporalMetadataError,
    coerce_required_years,
    future_window_mask,
    pre_split_mask,
    resolve_window_bounds,
    split_year_window_mask,
)


def test_coerce_required_years_rejects_missing_years() -> None:
    frame = pd.DataFrame({"resolved_year": [2010, None, "bad"]})

    with pytest.raises(TemporalMetadataError, match="resolved_year"):
        coerce_required_years(frame, "resolved_year", context="unit-test")


def test_pre_split_mask_does_not_treat_missing_as_training() -> None:
    years = pd.Series([2014, None, "bad", 2016])

    mask = pre_split_mask(years, split_year=2015)

    assert mask.tolist() == [True, False, False, False]


def test_coerce_required_years_rejects_missing_column() -> None:
    frame = pd.DataFrame({"year": [2010]})

    with pytest.raises(TemporalMetadataError, match="missing required year column"):
        coerce_required_years(frame, "resolved_year", context="unit-test")


def test_resolve_window_bounds_rejects_negative_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_years"):
        resolve_window_bounds(split_year=2015, horizon_years=-1)


def test_future_window_masks_handle_sequences_and_open_windows() -> None:
    years = [2014, 2015, 2016, 2018, None, "bad"]

    open_mask = split_year_window_mask(years, split_year=2015)
    bounded_mask = future_window_mask(years, split_year=2015, horizon_years=2)

    assert open_mask.tolist() == [False, False, True, True, False, False]
    assert bounded_mask.tolist() == [False, False, True, False, False, False]
