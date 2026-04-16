from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from plasmid_priority.modeling.plugin_system import FeaturePlugin, ModelPlugin, PluginRegistry


class _FakeFeaturePlugin(FeaturePlugin):
    def build_features(self, training_canonical: pd.DataFrame) -> pd.DataFrame:
        return training_canonical.assign(fake_feature=1.0)


class _FakeModelPlugin(ModelPlugin):
    def fit_predict(self, scored: pd.DataFrame, **kwargs):  # type: ignore[override]
        return scored.assign(prediction=0.5)


class PluginSystemTests(unittest.TestCase):
    def test_registry_registers_and_discovers_entry_points(self) -> None:
        registry = PluginRegistry()

        registry.register_feature("fake_feature")(_FakeFeaturePlugin)
        registry.register_model("fake_model")(_FakeModelPlugin)
        self.assertIs(registry.get_feature("fake_feature"), _FakeFeaturePlugin)
        self.assertIs(registry.get_model("fake_model"), _FakeModelPlugin)

        fake_feature_ep = SimpleNamespace(name="entry_feature", load=lambda: _FakeFeaturePlugin)
        fake_model_ep = SimpleNamespace(name="entry_model", load=lambda: _FakeModelPlugin)

        with mock.patch(
            "plasmid_priority.modeling.plugin_system.entry_points",
            side_effect=lambda group=None: [fake_feature_ep] if group == "plasmid_priority.feature" else [fake_model_ep],
        ):
            registry.discover_entry_points()

        self.assertIs(registry.get_feature("entry_feature"), _FakeFeaturePlugin)
        self.assertIs(registry.get_model("entry_model"), _FakeModelPlugin)


if __name__ == "__main__":
    unittest.main()
