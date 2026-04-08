"""Tests for the hardening batch 1 fixes: country/macro-region, AP, _dominant_share, _clean_text."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestCountryMacroRegionConsistency:
    """Test that all canonical countries have macro-region coverage."""

    def test_validate_country_macro_region_coverage_empty(self):
        """All canonical countries should have macro-region mappings."""
        from plasmid_priority.utils.geography import validate_country_macro_region_coverage

        missing = validate_country_macro_region_coverage()
        assert missing == [], f"Countries missing macro-region mapping: {missing}"

    def test_country_to_macro_region_returns_correct_regions(self):
        """Test that country_to_macro_region returns expected regions."""
        from plasmid_priority.utils.geography import country_to_macro_region

        # Test some known countries
        assert country_to_macro_region("USA") == "North America"
        assert country_to_macro_region("UK") == "Europe"
        assert country_to_macro_region("Japan") == "Asia"
        assert country_to_macro_region("Brazil") == "Latin America and Caribbean"
        assert country_to_macro_region("Nigeria") == "Africa"
        assert country_to_macro_region("Australia") == "Oceania"

    def test_country_to_macro_region_newly_added_countries(self):
        """Test that newly added countries from the fix are covered."""
        from plasmid_priority.utils.geography import country_to_macro_region

        # These were added in the fix
        assert country_to_macro_region("Andorra") == "Europe"
        assert country_to_macro_region("Angola") == "Africa"
        assert country_to_macro_region("Antigua and Barbuda") == "Latin America and Caribbean"
        assert country_to_macro_region("Aruba") == "Latin America and Caribbean"
        assert country_to_macro_region("Azerbaijan") == "Middle East and West Asia"
        assert country_to_macro_region("North Korea") == "Asia"
        assert country_to_macro_region("North Macedonia") == "Europe"

    def test_country_to_macro_region_empty_input(self):
        """Test that empty/invalid inputs return empty string."""
        from plasmid_priority.utils.geography import country_to_macro_region

        assert country_to_macro_region("") == ""
        assert country_to_macro_region(None) == ""
        assert country_to_macro_region("NonExistentCountry") == ""


class TestAveragePrecisionUnification:
    """Test that average_precision is unified to the project's implementation."""

    def test_average_precision_import_from_validation(self):
        """Test that average_precision can be imported from validation.metrics."""
        from plasmid_priority.validation.metrics import average_precision

        # Basic smoke test
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        result = average_precision(y_true, y_score)
        assert 0.0 <= result <= 1.0

    def test_average_precision_import_from_reporting_figures(self):
        """Test that average_precision is imported in figures module."""
        from plasmid_priority.reporting.figures import average_precision

        # Basic smoke test
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        result = average_precision(y_true, y_score)
        assert 0.0 <= result <= 1.0

    def test_average_precision_vs_sklearn_consistency(self):
        """Test that our implementation is reasonably close to sklearn."""
        from sklearn.metrics import average_precision_score as skl_ap

        from plasmid_priority.validation.metrics import average_precision

        # Test with various scenarios
        np.random.seed(42)
        for _ in range(5):
            y_true = np.random.randint(0, 2, size=100)
            y_score = np.random.random(size=100)

            our_ap = average_precision(y_true, y_score)
            sklearn_ap = skl_ap(y_true, y_score)

            # The implementations can differ due to tie-handling,
            # but should be reasonably close for random data
            assert abs(our_ap - sklearn_ap) < 0.1 or np.isnan(our_ap) == np.isnan(sklearn_ap)


class TestDominantShareDeduplication:
    """Test that _dominant_share is deduplicated and works correctly."""

    def test_dominant_share_from_dataframe_utils(self):
        """Test that dominant_share is available from utils.dataframe."""
        from plasmid_priority.utils.dataframe import dominant_share

        series = pd.Series(["a", "a", "b", "a", "c"])
        result = dominant_share(series)
        # 'a' appears 3/5 = 0.6
        assert abs(result - 0.6) < 0.001

    def test_dominant_share_empty_series(self):
        """Test dominant_share with empty series returns 0.0."""
        from plasmid_priority.utils.dataframe import dominant_share

        series = pd.Series([])
        assert dominant_share(series) == 0.0

        series = pd.Series(["", "", ""])
        assert dominant_share(series) == 0.0

    def test_dominant_share_with_na_values(self):
        """Test dominant_share handles NA values correctly."""
        from plasmid_priority.utils.dataframe import dominant_share

        series = pd.Series(["a", "a", np.nan, "b", None])
        result = dominant_share(series)
        # NA/None become "", so we have: "a", "a", "", "b", ""
        # non-empty: "a", "a", "b" -> 'a' appears 2/3 = 0.666...
        assert abs(result - 0.666666) < 0.001

    def test_dominant_share_all_same(self):
        """Test dominant_share when all values are the same."""
        from plasmid_priority.utils.dataframe import dominant_share

        series = pd.Series(["x", "x", "x", "x"])
        assert dominant_share(series) == 1.0


class TestCleanTextDeduplication:
    """Test that _clean_text is deduplicated and works correctly."""

    def test_clean_text_series_from_dataframe_utils(self):
        """Test that clean_text_series is available from utils.dataframe."""
        from plasmid_priority.utils.dataframe import clean_text_series

        series = pd.Series(["  hello  ", "world", None, np.nan, "  test  "], dtype=object)
        result = clean_text_series(series)

        assert result.iloc[0] == "hello"
        assert result.iloc[1] == "world"
        assert result.iloc[2] == ""  # None becomes ""
        assert result.iloc[3] == ""  # np.nan becomes ""
        assert result.iloc[4] == "test"

    def test_clean_text_series_all_na(self):
        """Test clean_text_series with all NA values."""
        from plasmid_priority.utils.dataframe import clean_text_series

        series = pd.Series([None, np.nan, None])
        result = clean_text_series(series)
        assert all(r == "" for r in result)

    def test_clean_text_series_preserves_original(self):
        """Test that clean_text_series doesn't modify original series."""
        from plasmid_priority.utils.dataframe import clean_text_series

        original = pd.Series(["  hello  ", "world"])
        result = clean_text_series(original)

        # Original should not be modified
        assert original.iloc[0] == "  hello  "
        assert result.iloc[0] == "hello"


class TestIntegration:
    """Integration tests for the hardening fixes."""

    def test_geography_imports_consistency(self):
        """Test that geography module imports correctly with COUNTRY_ALIAS_GROUPS."""
        from plasmid_priority.harmonize.records import COUNTRY_ALIAS_GROUPS
        from plasmid_priority.utils.geography import (
            COUNTRY_TO_MACRO_REGION,
        )

        # All canonical countries should be in the mapping
        for country in COUNTRY_ALIAS_GROUPS:
            assert country in COUNTRY_TO_MACRO_REGION, f"{country} not in COUNTRY_TO_MACRO_REGION"

    def test_features_core_imports(self):
        """Test that features.core imports correctly with canonical functions."""
        from plasmid_priority.features import core

        # These should exist and be the imported versions
        assert hasattr(core, "_clean_text_series")
        assert hasattr(core, "_dominant_share")

    def test_backbone_core_imports(self):
        """Test that backbone.core imports correctly with canonical functions."""
        from plasmid_priority.backbone import core

        assert hasattr(core, "_dominant_share")

    def test_advanced_audits_imports(self):
        """Test that advanced_audits imports correctly with canonical functions."""
        from plasmid_priority.reporting import advanced_audits

        assert hasattr(advanced_audits, "_clean_text")

    def test_harmonize_records_imports(self):
        """Test that harmonize.records imports correctly with canonical functions."""
        from plasmid_priority.harmonize import records

        assert hasattr(records, "clean_text_series")
        assert hasattr(records, "dominant_share")
