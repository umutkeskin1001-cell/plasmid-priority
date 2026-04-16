"""Property-based tests for core mathematical/normalization behavior using Hypothesis.

These tests verify key invariants of core utility functions without requiring
hand-crafted examples. Run with: pytest tests/test_properties.py -v
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from plasmid_priority.features.core import _normalized_shannon_evenness, _support_factor
from plasmid_priority.scoring.core import _empirical_percentile, _robust_sigmoid
from plasmid_priority.utils.dataframe import coalescing_left_merge


@st.composite
def _merge_case(draw) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = list(range(6))
    left_keys = draw(
        st.lists(st.sampled_from(keys), min_size=1, max_size=12)
    )
    right_keys = draw(
        st.lists(st.sampled_from(keys), unique=True, min_size=1, max_size=6)
    )
    optional_text = st.one_of(st.none(), st.text(min_size=1, max_size=3))
    left_shared = draw(st.lists(optional_text, min_size=len(left_keys), max_size=len(left_keys)))
    left_extra = draw(st.lists(optional_text, min_size=len(left_keys), max_size=len(left_keys)))
    right_shared = draw(st.lists(optional_text, min_size=len(right_keys), max_size=len(right_keys)))
    right_extra = draw(st.lists(optional_text, min_size=len(right_keys), max_size=len(right_keys)))
    left = pd.DataFrame(
        {
            "backbone_id": left_keys,
            "shared": left_shared,
            "left_only": left_extra,
        }
    )
    right = pd.DataFrame(
        {
            "backbone_id": right_keys,
            "shared": right_shared,
            "right_only": right_extra,
        }
    )
    return left, right


class TestSupportFactorProperties:
    """Property-based tests for the support factor (shrinkage) function."""

    @given(n=st.integers(min_value=0, max_value=1000000))
    @settings(max_examples=200)
    def test_support_factor_range(self, n: int) -> None:
        """Support factor must always be in [0, 1]."""
        result = _support_factor(n)
        assert 0.0 <= result <= 1.0

    @given(n=st.integers(min_value=1, max_value=1000000))
    @settings(max_examples=200)
    def test_support_factor_monotonicity(self, n: int) -> None:
        """Support factor increases with n (monotonicity)."""
        result_n = _support_factor(n)
        result_n_plus_1 = _support_factor(n + 1)
        assert result_n_plus_1 > result_n

    @given(n=st.integers(min_value=0, max_value=1000000))
    @settings(max_examples=200)
    def test_support_factor_zero_is_zero(self, n: int) -> None:
        """Support factor at n=0 is 0, otherwise positive."""
        if n == 0:
            assert _support_factor(n) == 0.0
        else:
            assert _support_factor(n) > 0.0


class TestShannonEvennessProperties:
    """Property-based tests for normalized Shannon evenness."""

    @given(
        values=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()), min_size=1, max_size=100
        )
    )
    @settings(max_examples=200)
    def test_evenness_range(self, values: list[str]) -> None:
        """Normalized Shannon evenness must be in [0, 1]."""
        result = _normalized_shannon_evenness(values)
        assert 0.0 <= result <= 1.0

    @given(values=st.lists(st.sampled_from(["A", "B"]), min_size=2, max_size=100))
    @settings(max_examples=100)
    def test_evenness_maximum_for_uniform(self, values: list[str]) -> None:
        """Evenness is 1.0 when all categories have equal count (and > 1 unique)."""
        # Need at least 2 unique values for evenness to be meaningful
        unique_counts = {v: values.count(v) for v in set(values)}
        if (
            len(unique_counts) >= 2 and len(set(unique_counts.values())) == 1
        ):  # All equal, multiple uniques
            result = _normalized_shannon_evenness(values)
            assert result == 1.0

    @given(
        base=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip()), min_size=2, max_size=20
        ),
        dominant=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_evenness_low_for_dominant(self, base: list[str], dominant: str) -> None:
        """Evenness is low when one category dominates."""
        # Create a list where one value dominates (90% of entries)
        n = len(base) * 9  # 90% will be dominant
        dominated_list = [dominant] * n + base
        result = _normalized_shannon_evenness(dominated_list)
        assert result < 0.5  # Should be low for dominated distribution

    @given(values=st.lists(st.text(), max_size=1))
    @settings(max_examples=50)
    def test_evenness_insufficient_data(self, values: list[str]) -> None:
        """Evenness is 0 for insufficient data (< 2 unique values)."""
        result = _normalized_shannon_evenness(values)
        assert result == 0.0


class TestEmpiricalPercentileProperties:
    """Property-based tests for empirical percentile normalization."""

    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
        ref_values=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
    )
    @settings(max_examples=200)
    def test_percentile_range(self, values: list[float], ref_values: list[float]) -> None:
        """Empirical percentile must be in [0, 1]."""
        s_values = pd.Series(values)
        s_ref = pd.Series(ref_values)
        result = _empirical_percentile(s_values, s_ref)
        assert (result >= 0.0).all() and (result <= 1.0).all()

    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=50,
        ).filter(lambda x: max(x) > 0)  # Need at least one positive value
    )
    @settings(max_examples=100)
    def test_percentile_self_reference_max(self, values: list[float]) -> None:
        """Maximum value against self-reference should have percentile near 1."""
        s = pd.Series(values)
        result = _empirical_percentile(s, s)
        max_result = result.max()
        assert max_result >= 0.99  # Close to 1.0


class TestRobustSigmoidProperties:
    """Property-based tests for robust sigmoid normalization."""

    @given(
        values=st.lists(
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
        ref_values=st.lists(
            st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=100,
        ),
    )
    @settings(max_examples=200)
    def test_sigmoid_range(self, values: list[float], ref_values: list[float]) -> None:
        """Robust sigmoid output must be in [0, 1]."""
        s_values = pd.Series(values)
        s_ref = pd.Series(ref_values)
        result = _robust_sigmoid(s_values, s_ref)
        assert (result >= 0.0).all() and (result <= 1.0).all()

    @given(
        values=st.lists(
            st.floats(min_value=-1000.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
        ref_values=st.lists(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_sigmoid_negative_input_zero(
        self, values: list[float], ref_values: list[float]
    ) -> None:
        """Negative inputs to sigmoid should yield zero output."""
        s_values = pd.Series(values)
        s_ref = pd.Series(ref_values)
        result = _robust_sigmoid(s_values, s_ref)
        assert (result == 0.0).all()


class TestNaNHandlingProperties:
    """Property-based tests for NaN handling invariants."""

    @given(
        values=st.lists(
            st.one_of(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False), st.just(float("nan"))
            ),
            min_size=5,
            max_size=50,
        ),
        ref_values=st.lists(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False), min_size=5, max_size=50
        ),
    )
    @settings(max_examples=100)
    def test_empirical_percentile_nan_handling(
        self, values: list[float], ref_values: list[float]
    ) -> None:
        """NaN inputs should result in 0.0 output (filled)."""
        s_values = pd.Series(values)
        s_ref = pd.Series(ref_values)
        result = _empirical_percentile(s_values, s_ref)
        # NaN positions should have 0.0
        nan_positions = s_values.isna()
        if nan_positions.any():
            assert (result[nan_positions] == 0.0).all()


class TestDataframeMergeProperties:
    """Property-based tests for merge coalescing behavior."""

    @given(case=_merge_case())
    @settings(max_examples=100)
    def test_coalescing_left_merge_matches_left_join_and_coalesce(
        self, case: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        left, right = case
        expected = left.merge(right, on="backbone_id", how="left", suffixes=("", "__incoming"))
        for column in ("shared", "right_only"):
            if column in left.columns and f"{column}__incoming" in expected.columns:
                incoming = f"{column}__incoming"
                expected[column] = expected[column].where(
                    expected[column].notna(), expected[incoming]
                )
                expected = expected.drop(columns=incoming)
        actual = coalescing_left_merge(left, right, on="backbone_id")
        pd.testing.assert_frame_equal(actual, expected)


# Smoke tests for quick validation
class TestSmokeProperties:
    """Quick smoke tests that always run."""

    def test_support_factor_basic(self) -> None:
        assert _support_factor(0) == 0.0
        assert _support_factor(10) > 0.0
        assert _support_factor(100) > _support_factor(10)

    def test_evenness_basic(self) -> None:
        assert _normalized_shannon_evenness([]) == 0.0
        assert _normalized_shannon_evenness(["A"]) == 0.0
        assert _normalized_shannon_evenness(["A", "B"]) == 1.0

    def test_percentile_basic(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _empirical_percentile(s, s)
        assert (result >= 0.0).all() and (result <= 1.0).all()
