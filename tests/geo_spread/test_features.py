from __future__ import annotations

import unittest

from plasmid_priority.geo_spread import (
    GEO_SPREAD_ALLOWED_FEATURES,
    classify_geo_spread_feature,
    validate_geo_spread_feature_set,
)


class GeoSpreadFeatureTests(unittest.TestCase):
    def test_classify_geo_spread_feature_returns_expected_category(self) -> None:
        self.assertEqual(classify_geo_spread_feature("T_eff_norm"), "intrinsic")
        self.assertEqual(classify_geo_spread_feature("log1p_member_count_train"), "sampling_proxy")
        self.assertIn("coherence_score", GEO_SPREAD_ALLOWED_FEATURES)

    def test_validate_geo_spread_feature_set_rejects_unknown_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported features"):
            validate_geo_spread_feature_set(
                ["T_eff_norm", "imaginary_feature"],
                label="unit-test geo feature set",
            )


if __name__ == "__main__":
    unittest.main()
