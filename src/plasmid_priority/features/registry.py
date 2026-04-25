from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    temporal_scope: str
    prediction_time_available: bool
    source_scope: str
    leakage_risk: str
    allowed_in_official_models: bool
    allowed_in_experimental_models: bool = True
    biological_meaning: str = ""
    known_limitations: tuple[str, ...] = ()

    def is_safe_for_official_model(self) -> bool:
        if not self.allowed_in_official_models:
            return False
        if self.temporal_scope != "pre_split_only":
            return False
        if not self.prediction_time_available:
            return False
        return self.leakage_risk not in {"high", "critical"}


class FeatureRegistry:
    def __init__(self, specs: Iterable[FeatureSpec] = ()) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> FeatureSpec | None:
        return self._specs.get(str(name))

    def require(self, name: str) -> FeatureSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown feature: {name}")
        return spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def add(self, spec: FeatureSpec) -> FeatureRegistry:
        return FeatureRegistry((*self._specs.values(), spec))
