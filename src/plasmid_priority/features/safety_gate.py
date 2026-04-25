from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from plasmid_priority.features.registry import FeatureRegistry


@dataclass(frozen=True)
class FeatureSafetyResult:
    status: str
    checked_features: tuple[str, ...]
    blocked_features: tuple[str, ...]
    unknown_features: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


class FeatureSafetyGate:
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    def evaluate(self, feature_names: Iterable[str]) -> FeatureSafetyResult:
        checked: list[str] = []
        blocked: list[str] = []
        unknown: list[str] = []
        reasons: list[str] = []

        for raw_name in feature_names:
            name = str(raw_name)
            checked.append(name)
            spec = self.registry.get(name)
            if spec is None:
                unknown.append(name)
                reasons.append(f"{name}:unknown_feature")
                continue
            if not spec.is_safe_for_official_model():
                blocked.append(name)
                reasons.append(
                    f"{name}:temporal_scope={spec.temporal_scope},"
                    f"prediction_time_available={spec.prediction_time_available},"
                    f"leakage_risk={spec.leakage_risk},"
                    f"allowed={spec.allowed_in_official_models}",
                )

        status = "pass" if not blocked and not unknown else "fail"
        return FeatureSafetyResult(
            status=status,
            checked_features=tuple(checked),
            blocked_features=tuple(blocked),
            unknown_features=tuple(unknown),
            reasons=tuple(reasons),
        )
