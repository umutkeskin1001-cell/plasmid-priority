"""Plugin registry for feature engineering and model backends.

Provides a lightweight, entry-point-based plugin system so that external
packages can register custom feature builders and model backends without
modifying the core codebase.

Usage::

    from plasmid_priority.modeling.plugin_system import FeaturePlugin, ModelPlugin, registry

    @registry.register_feature("custom_transfer")
    class CustomTransferPlugin(FeaturePlugin):
        def build_features(self, training_canonical: pd.DataFrame) -> pd.DataFrame:
            ...

    @registry.register_model("custom_model")
    class CustomModelPlugin(ModelPlugin):
        def fit_predict(self, scored: pd.DataFrame, **kwargs: Any) -> Any:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Any, Callable

import pandas as pd

_log = logging.getLogger(__name__)


class FeaturePlugin(ABC):
    """Base class for feature engineering plugins."""

    @abstractmethod
    def build_features(self, training_canonical: pd.DataFrame) -> pd.DataFrame:
        """Build feature columns from the training canonical table."""
        ...


class ModelPlugin(ABC):
    """Base class for model backend plugins."""

    @abstractmethod
    def fit_predict(self, scored: pd.DataFrame, **kwargs: Any) -> Any:
        """Fit the model and return predictions."""
        ...


class PluginRegistry:
    """Central registry for feature and model plugins."""

    def __init__(self) -> None:
        self._features: dict[str, type[FeaturePlugin]] = {}
        self._models: dict[str, type[ModelPlugin]] = {}

    # -- Feature plugins -------------------------------------------------------

    def register_feature(self, name: str) -> Callable[[type[FeaturePlugin]], type[FeaturePlugin]]:
        """Decorator to register a feature plugin."""

        def decorator(cls: type[FeaturePlugin]) -> type[FeaturePlugin]:
            self._features[name] = cls
            _log.info("Registered feature plugin: %s", name)
            return cls

        return decorator

    def get_feature(self, name: str) -> type[FeaturePlugin] | None:
        """Look up a feature plugin by name."""
        return self._features.get(name)

    @property
    def feature_names(self) -> list[str]:
        return list(self._features)

    # -- Model plugins ---------------------------------------------------------

    def register_model(self, name: str) -> Callable[[type[ModelPlugin]], type[ModelPlugin]]:
        """Decorator to register a model plugin."""

        def decorator(cls: type[ModelPlugin]) -> type[ModelPlugin]:
            self._models[name] = cls
            _log.info("Registered model plugin: %s", name)
            return cls

        return decorator

    def get_model(self, name: str) -> type[ModelPlugin] | None:
        """Look up a model plugin by name."""
        return self._models.get(name)

    @property
    def model_names(self) -> list[str]:
        return list(self._models)

    # -- Entry-point discovery -------------------------------------------------

    def discover_entry_points(self) -> None:
        """Load plugins registered via ``plasmid_priority.feature`` and
        ``plasmid_priority.model`` entry points.
        """
        for group, store, base_cls in [
            ("plasmid_priority.feature", self._features, FeaturePlugin),
            ("plasmid_priority.model", self._models, ModelPlugin),
        ]:
            for ep in entry_points(group=group):
                try:
                    cls = ep.load()
                    if isinstance(cls, type) and issubclass(cls, base_cls):
                        if base_cls is FeaturePlugin:
                            self._features[ep.name] = cls  # type: ignore[assignment]
                        else:
                            self._models[ep.name] = cls  # type: ignore[assignment]
                        _log.info("Discovered plugin %s via entry point %s", ep.name, group)
                except Exception as exc:
                    _log.warning("Failed to load entry point %s: %s", ep.name, exc)


# Module-level singleton
registry = PluginRegistry()
