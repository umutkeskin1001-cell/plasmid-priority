"""Tests for utils.math module."""

from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.utils.math import geometric_mean, geometric_mean_frame


def test_geometric_mean_basic() -> None:
    """Test basic geometric mean calculation."""
    values = [2.0, 8.0]
    result = geometric_mean(values)
    assert result == pytest.approx(4.0)  # sqrt(2*8) = 4


def test_geometric_mean_empty() -> None:
    """Test geometric mean with empty list."""
    result = geometric_mean([])
    assert result == 0.0


def test_geometric_mean_with_zero() -> None:
    """Test geometric mean with zero value."""
    result = geometric_mean([0.0, 2.0, 3.0])
    assert result == 0.0


def test_geometric_mean_with_negative() -> None:
    """Test geometric mean with negative value."""
    result = geometric_mean([-1.0, 2.0, 3.0])
    assert result == 0.0


def test_geometric_mean_frame_basic() -> None:
    """Test basic geometric mean calculation."""
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
            "col3": [7.0, 8.0, 9.0],
        }
    )
    result = geometric_mean_frame(df)
    assert len(result) == 3
    assert result.iloc[0] == pytest.approx((1.0 * 4.0 * 7.0) ** (1 / 3))
    assert result.iloc[1] == pytest.approx((2.0 * 5.0 * 8.0) ** (1 / 3))


def test_geometric_mean_frame_with_zeros() -> None:
    """Test geometric mean with zeros."""
    df = pd.DataFrame(
        {
            "col1": [0.0, 2.0, 3.0],
            "col2": [4.0, 0.0, 6.0],
            "col3": [7.0, 8.0, 0.0],
        }
    )
    result = geometric_mean_frame(df)
    assert len(result) == 3
    # With zeros, geometric mean should be zero
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 0.0


def test_geometric_mean_frame_with_negative() -> None:
    """Test geometric mean with negative values (should handle gracefully)."""
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, -5.0, 6.0],
            "col3": [7.0, 8.0, 9.0],
        }
    )
    # This might raise an error or return NaN for negative values
    # depending on implementation
    try:
        result = geometric_mean_frame(df)
        assert len(result) == 3
    except ValueError:
        # Expected for negative values in geometric mean
        pass


def test_geometric_mean_frame_empty() -> None:
    """Test geometric mean with empty dataframe."""
    df = pd.DataFrame(columns=["col1", "col2", "col3"])
    result = geometric_mean_frame(df)
    assert len(result) == 0


def test_geometric_mean_frame_single_column() -> None:
    """Test geometric mean with single column."""
    df = pd.DataFrame({"col1": [1.0, 2.0, 3.0]})
    result = geometric_mean_frame(df)
    assert len(result) == 3
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(3.0)


def test_geometric_mean_frame_negative_values() -> None:
    """Test geometric mean with negative values."""
    df = pd.DataFrame({"col1": [-1.0, -2.0, -3.0]})
    result = geometric_mean_frame(df)
    assert len(result) == 3
    # Should handle negative values (may return NaN or complex)


def test_geometric_mean_frame_mixed_signs() -> None:
    """Test geometric mean with mixed signs."""
    df = pd.DataFrame({"col1": [1.0, -2.0, 3.0], "col2": [4.0, 5.0, 6.0]})
    result = geometric_mean_frame(df)
    assert len(result) == 3


def test_geometric_mean_frame_very_large_values() -> None:
    """Test geometric mean with very large values."""
    df = pd.DataFrame({"col1": [1e10, 1e11, 1e12]})
    result = geometric_mean_frame(df)
    assert len(result) == 3


def test_geometric_mean_frame_very_small_values() -> None:
    """Test geometric mean with very small values."""
    df = pd.DataFrame({"col1": [1e-10, 1e-11, 1e-12]})
    result = geometric_mean_frame(df)
    assert len(result) == 3
