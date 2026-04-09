"""Canonical scientific protocol metadata for Plasmid Priority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_SELECTION_ADJUSTED_P_MAX = 0.01


def _coerce_int(value: object, *, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _coerce_float(value: object, *, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _coerce_name_tuple(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    try:
        return tuple(str(item) for item in value if str(item).strip())
    except TypeError:
        return (str(value),)


def _deduplicate(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(name) for name in names if str(name)))


def build_protocol_snapshot(protocol: "ScientificProtocol") -> dict[str, Any]:
    return {
        "acceptance_thresholds": protocol.acceptance_thresholds,
        "ablation_model_names": list(protocol.ablation_model_names),
        "conservative_model_name": protocol.conservative_model_name,
        "core_model_names": list(protocol.core_model_names),
        "governance_model_fallback": protocol.governance_model_fallback,
        "governance_model_name": protocol.governance_model_name,
        "governance_model_policy": protocol.governance_model_policy,
        "min_new_countries_for_spread": int(protocol.min_new_countries_for_spread),
        "official_model_names": list(protocol.official_model_names),
        "outcome_definition": protocol.outcome_definition,
        "primary_model_fallback": protocol.primary_model_fallback,
        "primary_model_name": protocol.primary_model_name,
        "research_model_names": list(protocol.research_model_names),
        "selection_adjusted_p_max": float(protocol.selection_adjusted_p_max),
        "split_year": int(protocol.split_year),
    }


def build_protocol_hash(protocol: "ScientificProtocol") -> str:
    payload = json.dumps(
        build_protocol_snapshot(protocol),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScientificProtocol:
    """Immutable protocol describing the canonical scientific surface."""

    split_year: int
    min_new_countries_for_spread: int
    primary_model_name: str
    primary_model_fallback: str
    conservative_model_name: str
    governance_model_name: str
    governance_model_fallback: str
    core_model_names: tuple[str, ...]
    research_model_names: tuple[str, ...]
    ablation_model_names: tuple[str, ...]
    selection_adjusted_p_max: float = DEFAULT_SELECTION_ADJUSTED_P_MAX

    @property
    def outcome_definition(self) -> dict[str, object]:
        return {
            "outcome_label": "spread_label",
            "split_year": int(self.split_year),
            "min_new_countries_for_spread": int(self.min_new_countries_for_spread),
        }

    @property
    def governance_model_policy(self) -> dict[str, str]:
        return {
            "primary_model_name": self.primary_model_name,
            "primary_model_fallback": self.primary_model_fallback,
            "governance_model_name": self.governance_model_name,
            "governance_model_fallback": self.governance_model_fallback,
            "conservative_model_name": self.conservative_model_name,
        }

    @property
    def acceptance_thresholds(self) -> dict[str, float]:
        return {
            "selection_adjusted_p_max": float(self.selection_adjusted_p_max),
        }

    @property
    def official_model_names(self) -> tuple[str, ...]:
        return _deduplicate(
            [
                self.primary_model_name,
                self.governance_model_name,
                "baseline_both",
            ]
        )

    def validate(self) -> None:
        core_names = set(self.core_model_names)
        research_names = set(self.research_model_names)
        if self.primary_model_name not in core_names:
            raise ValueError(
                "Configured primary model must be part of the official core surface; "
                f"`{self.primary_model_name}` is not listed in core_model_names."
            )
        if self.governance_model_name not in core_names:
            raise ValueError(
                "Configured governance model must be part of the official core surface; "
                f"`{self.governance_model_name}` is not listed in core_model_names."
            )
        if "baseline_both" not in core_names:
            raise ValueError(
                "Configured protocol must include `baseline_both` in the official core surface."
            )
        if self.primary_model_name in research_names and self.primary_model_name not in core_names:
            raise ValueError(
                "Configured primary model cannot live only in research_model_names."
            )
        if (
            self.governance_model_name in research_names
            and self.governance_model_name not in core_names
        ):
            raise ValueError(
                "Configured governance model cannot live only in research_model_names."
            )
        if not self.official_model_names:
            raise ValueError("Scientific protocol must define at least one official model name.")
        if not 0.0 < float(self.selection_adjusted_p_max) <= 1.0:
            raise ValueError("selection_adjusted_p_max must be in the interval (0, 1].")

    def resolve_primary_model_name(
        self,
        model_names: Iterable[str],
        *,
        allow_fallbacks: bool = False,
    ) -> str:
        names = {str(name) for name in model_names}
        if self.primary_model_name in names:
            return self.primary_model_name
        if allow_fallbacks and self.primary_model_fallback in names:
            return self.primary_model_fallback
        raise ValueError(
            "Configured primary model is missing from the provided model surface: "
            f"{sorted(names)}"
        )

    def resolve_governance_model_name(
        self,
        model_names: Iterable[str],
        *,
        allow_fallbacks: bool = False,
    ) -> str:
        names = {str(name) for name in model_names}
        if self.governance_model_name in names:
            return self.governance_model_name
        if allow_fallbacks and self.governance_model_fallback in names:
            return self.governance_model_fallback
        if allow_fallbacks and self.primary_model_fallback in names:
            return self.primary_model_fallback
        raise ValueError(
            "Configured governance model is missing from the provided model surface: "
            f"{sorted(names)}"
        )

    def resolve_official_model_names(
        self,
        model_names: Iterable[str],
        *,
        require_complete: bool = True,
    ) -> tuple[str, ...]:
        names = {str(name) for name in model_names}
        official = tuple(name for name in self.official_model_names if name in names)
        if require_complete and official != self.official_model_names:
            missing = [name for name in self.official_model_names if name not in names]
            raise ValueError(
                "Provided model surface is missing official model names: "
                f"{', '.join(sorted(missing))}"
            )
        return official

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ScientificProtocol":
        models = config.get("models", {}) if isinstance(config, dict) else {}
        pipeline = config.get("pipeline", {}) if isinstance(config, dict) else {}
        if not isinstance(models, dict):
            models = {}
        if not isinstance(pipeline, dict):
            pipeline = {}
        protocol = cls(
            split_year=_coerce_int(pipeline.get("split_year"), default=2015),
            min_new_countries_for_spread=_coerce_int(
                pipeline.get("min_new_countries_for_spread"), default=3
            ),
            primary_model_name=str(models.get("primary_model_name", "bio_clean_priority")),
            primary_model_fallback=str(
                models.get("primary_model_fallback", "parsimonious_priority")
            ),
            conservative_model_name=str(
                models.get("conservative_model_name", "parsimonious_priority")
            ),
            governance_model_name=str(
                models.get("governance_model_name", "phylo_support_fusion_priority")
            ),
            governance_model_fallback=str(
                models.get("governance_model_fallback", "support_synergy_priority")
            ),
            core_model_names=_coerce_name_tuple(models.get("core_model_names", ())),
            research_model_names=_coerce_name_tuple(models.get("research_model_names", ())),
            ablation_model_names=_coerce_name_tuple(models.get("ablation_model_names", ())),
            selection_adjusted_p_max=_coerce_float(
                models.get("selection_adjusted_p_max"), default=DEFAULT_SELECTION_ADJUSTED_P_MAX
            ),
        )
        protocol.validate()
        return protocol
