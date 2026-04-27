"""Canonical scientific protocol metadata for Plasmid Priority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SELECTION_ADJUSTED_P_MAX = 0.01
DEFAULT_MATCHED_KNOWNNESS_GAP_MIN = -0.005
DEFAULT_SOURCE_HOLDOUT_GAP_MIN = -0.005
DEFAULT_SPATIAL_HOLDOUT_GAP_MIN = -0.03
DEFAULT_ECE_MAX = 0.05
DEFAULT_OFFICIAL_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "matched_knownness_gap_min": DEFAULT_MATCHED_KNOWNNESS_GAP_MIN,
    "source_holdout_gap_min": DEFAULT_SOURCE_HOLDOUT_GAP_MIN,
    "spatial_holdout_gap_min": DEFAULT_SPATIAL_HOLDOUT_GAP_MIN,
    "ece_max": DEFAULT_ECE_MAX,
    "selection_adjusted_p_max": DEFAULT_SELECTION_ADJUSTED_P_MAX,
}
DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "reliability": 0.4,
    "predictive_power": 0.4,
    "compute_efficiency": 0.2,
}
BENCHMARK_CONTRACT_VERSION = "2025-07-15"


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
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _deduplicate(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(name) for name in names if str(name)))


def _coerce_float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _coerce_string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        result[key_text] = str(raw)
    return result


def _coerce_rules_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in value.items()}


def _default_eligibility_rules() -> dict[str, Any]:
    return {
        "required_assignment_mode": "training_only",
        "require_training_only_assignment": True,
        "require_temporal_metadata": True,
    }


@dataclass(frozen=True)
class ExecutionProtocol:
    """Parameters affecting data processing and model training."""

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
    min_new_host_genera_for_transfer: int = 2
    min_new_host_families_for_transfer: int = 1
    horizon_years: int = 5
    forbidden_features: tuple[str, ...] = field(default_factory=tuple)
    eligibility_rules: dict[str, Any] = field(default_factory=dict)

    def build_snapshot(self) -> dict[str, Any]:
        return {
            "split_year": int(self.split_year),
            "min_new_countries_for_spread": int(self.min_new_countries_for_spread),
            "min_new_host_genera_for_transfer": int(self.min_new_host_genera_for_transfer),
            "min_new_host_families_for_transfer": int(self.min_new_host_families_for_transfer),
            "horizon_years": int(self.horizon_years),
            "primary_model_name": self.primary_model_name,
            "primary_model_fallback": self.primary_model_fallback,
            "conservative_model_name": self.conservative_model_name,
            "governance_model_name": self.governance_model_name,
            "governance_model_fallback": self.governance_model_fallback,
            "core_model_names": list(self.core_model_names),
            "research_model_names": list(self.research_model_names),
            "ablation_model_names": list(self.ablation_model_names),
            "forbidden_features": list(self.forbidden_features),
            "eligibility_rules": self.eligibility_rules,
        }


@dataclass(frozen=True)
class EvaluationProtocol:
    """Parameters affecting thresholds, metadata, and reporting."""

    clinical_escalation_thresholds: dict[str, float] = field(default_factory=dict)
    label_proxy_caveats: dict[str, str] = field(default_factory=dict)
    matched_knownness_gap_min: float = DEFAULT_MATCHED_KNOWNNESS_GAP_MIN
    source_holdout_gap_min: float = DEFAULT_SOURCE_HOLDOUT_GAP_MIN
    spatial_holdout_gap_min: float = DEFAULT_SPATIAL_HOLDOUT_GAP_MIN
    ece_max: float = DEFAULT_ECE_MAX
    selection_adjusted_p_max: float = DEFAULT_SELECTION_ADJUSTED_P_MAX

    def build_snapshot(self) -> dict[str, Any]:
        return {
            "clinical_escalation_thresholds": self.clinical_escalation_thresholds,
            "label_proxy_caveats": self.label_proxy_caveats,
            "matched_knownness_gap_min": float(self.matched_knownness_gap_min),
            "source_holdout_gap_min": float(self.source_holdout_gap_min),
            "spatial_holdout_gap_min": float(self.spatial_holdout_gap_min),
            "ece_max": float(self.ece_max),
            "selection_adjusted_p_max": float(self.selection_adjusted_p_max),
        }


def build_execution_hash(protocol: ExecutionProtocol) -> str:
    payload = json.dumps(
        protocol.build_snapshot(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_evaluation_hash(protocol: EvaluationProtocol) -> str:
    payload = json.dumps(
        protocol.build_snapshot(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_protocol_snapshot(protocol: ScientificProtocol) -> dict[str, Any]:
    snapshot = protocol.execution.build_snapshot()
    snapshot.update(protocol.evaluation.build_snapshot())
    # Keep backward-compatible reporting/provenance surface while preserving
    # execution-vs-evaluation split for hash composition internals.
    snapshot["benchmark_contract_version"] = protocol.benchmark_contract_version
    snapshot["outcome_definition"] = protocol.outcome_definition
    snapshot["official_model_names"] = list(protocol.official_model_names)
    snapshot["benchmark_scope"] = protocol.benchmark_scope
    snapshot["governance_model_policy"] = protocol.governance_model_policy
    snapshot["acceptance_thresholds"] = protocol.acceptance_thresholds
    snapshot["single_model_objective_weights"] = protocol.single_model_objective_weights
    return snapshot


def build_protocol_hash(protocol: ScientificProtocol) -> str:
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

    execution: ExecutionProtocol
    evaluation: EvaluationProtocol

    # Delegated properties for backward compatibility
    @property
    def split_year(self) -> int:
        return self.execution.split_year

    @property
    def min_new_countries_for_spread(self) -> int:
        return self.execution.min_new_countries_for_spread

    @property
    def primary_model_name(self) -> str:
        return self.execution.primary_model_name

    @property
    def primary_model_fallback(self) -> str:
        return self.execution.primary_model_fallback

    @property
    def conservative_model_name(self) -> str:
        return self.execution.conservative_model_name

    @property
    def governance_model_name(self) -> str:
        return self.execution.governance_model_name

    @property
    def governance_model_fallback(self) -> str:
        return self.execution.governance_model_fallback

    @property
    def core_model_names(self) -> tuple[str, ...]:
        return self.execution.core_model_names

    @property
    def research_model_names(self) -> tuple[str, ...]:
        return self.execution.research_model_names

    @property
    def ablation_model_names(self) -> tuple[str, ...]:
        return self.execution.ablation_model_names

    @property
    def min_new_host_genera_for_transfer(self) -> int:
        return self.execution.min_new_host_genera_for_transfer

    @property
    def min_new_host_families_for_transfer(self) -> int:
        return self.execution.min_new_host_families_for_transfer

    @property
    def horizon_years(self) -> int:
        return self.execution.horizon_years

    @property
    def clinical_escalation_thresholds(self) -> dict[str, float]:
        return self.evaluation.clinical_escalation_thresholds

    @property
    def forbidden_features(self) -> tuple[str, ...]:
        return self.execution.forbidden_features

    @property
    def label_proxy_caveats(self) -> dict[str, str]:
        return self.evaluation.label_proxy_caveats

    @property
    def eligibility_rules(self) -> dict[str, Any]:
        return self.execution.eligibility_rules

    @property
    def matched_knownness_gap_min(self) -> float:
        return self.evaluation.matched_knownness_gap_min

    @property
    def source_holdout_gap_min(self) -> float:
        return self.evaluation.source_holdout_gap_min

    @property
    def spatial_holdout_gap_min(self) -> float:
        return self.evaluation.spatial_holdout_gap_min

    @property
    def ece_max(self) -> float:
        return self.evaluation.ece_max

    @property
    def selection_adjusted_p_max(self) -> float:
        return self.evaluation.selection_adjusted_p_max

    @property
    def outcome_definition(self) -> dict[str, object]:
        return {
            "outcome_label": "spread_label",
            "split_year": int(self.split_year),
            "min_new_countries_for_spread": int(self.min_new_countries_for_spread),
            "min_new_host_genera_for_transfer": int(self.min_new_host_genera_for_transfer),
            "min_new_host_families_for_transfer": int(self.min_new_host_families_for_transfer),
            "horizon_years": int(self.horizon_years),
        }

    @property
    def benchmark_contract_version(self) -> str:
        return BENCHMARK_CONTRACT_VERSION

    @property
    def benchmark_scope(self) -> dict[str, object]:
        required_assignment_mode = str(
            self.eligibility_rules.get("required_assignment_mode", "training_only")
        )
        require_temporal_metadata = bool(
            self.eligibility_rules.get("require_temporal_metadata", True)
        )
        return {
            "split_year": int(self.split_year),
            "required_assignment_mode": required_assignment_mode,
            "outcome_definition": self.outcome_definition,
            "eligibility_rules": self.eligibility_rules,
            "eligible_cohort": {
                "temporal_split_year": int(self.split_year),
                "min_new_countries_for_spread": int(self.min_new_countries_for_spread),
                "min_new_host_genera_for_transfer": int(self.min_new_host_genera_for_transfer),
                "min_new_host_families_for_transfer": int(self.min_new_host_families_for_transfer),
                "horizon_years": int(self.horizon_years),
                "requires_temporal_metadata": require_temporal_metadata,
                "require_training_only_assignment": bool(
                    self.eligibility_rules.get("require_training_only_assignment", True)
                ),
                "forbidden_features": list(self.forbidden_features),
            },
            "official_model_names": list(self.official_model_names),
            "accepted_audit_gates": list(self.acceptance_thresholds.keys()),
            "calibration_metric_definitions": {
                "ece_max": "Expected calibration error upper bound.",
                "selection_adjusted_p_max": "Selection-adjusted empirical p-value upper bound.",
            },
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
            "matched_knownness_gap_min": float(self.matched_knownness_gap_min),
            "source_holdout_gap_min": float(self.source_holdout_gap_min),
            "spatial_holdout_gap_min": float(self.spatial_holdout_gap_min),
            "ece_max": float(self.ece_max),
            "selection_adjusted_p_max": float(self.selection_adjusted_p_max),
        }

    @property
    def single_model_objective_weights(self) -> dict[str, float]:
        return dict(DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS)

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
        if not self.official_model_names:
            raise ValueError("Scientific protocol must define at least one official model name.")
        if int(self.horizon_years) <= 0:
            raise ValueError("horizon_years must be positive.")
        if not -1.0 <= float(self.matched_knownness_gap_min) <= 1.0:
            raise ValueError("matched_knownness_gap_min must be in the interval [-1, 1].")
        if not 0.0 <= float(self.ece_max) <= 1.0:
            raise ValueError("ece_max must be in the interval [0, 1].")
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
            f"Configured primary model is missing from the provided model surface: {sorted(names)}"
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

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ScientificProtocol":
        models = config.get("models", {}) if isinstance(config, dict) else {}
        pipeline = config.get("pipeline", {}) if isinstance(config, dict) else {}
        if not isinstance(models, dict):
            models = {}
        if not isinstance(pipeline, dict):
            pipeline = {}

        execution = ExecutionProtocol(
            split_year=_coerce_int(pipeline.get("split_year"), default=2015),
            min_new_countries_for_spread=_coerce_int(
                pipeline.get("min_new_countries_for_spread"), default=3
            ),
            min_new_host_genera_for_transfer=_coerce_int(
                pipeline.get("min_new_host_genera_for_transfer"), default=2
            ),
            min_new_host_families_for_transfer=_coerce_int(
                pipeline.get("min_new_host_families_for_transfer"), default=1
            ),
            horizon_years=_coerce_int(pipeline.get("horizon_years"), default=5),
            forbidden_features=_coerce_name_tuple(pipeline.get("forbidden_features")),
            eligibility_rules=_coerce_rules_mapping(
                pipeline.get("eligibility_rules") or _default_eligibility_rules()
            ),
            primary_model_name=str(models.get("primary_model_name", "discovery_boosted")),
            primary_model_fallback=str(
                models.get("primary_model_fallback", "parsimonious_priority")
            ),
            conservative_model_name=str(
                models.get("conservative_model_name", "parsimonious_priority")
            ),
            governance_model_name=str(models.get("governance_model_name", "governance_linear")),
            governance_model_fallback=str(
                models.get("governance_model_fallback", "support_synergy_priority")
            ),
            core_model_names=_coerce_name_tuple(models.get("core_model_names", ())),
            research_model_names=_coerce_name_tuple(models.get("research_model_names", ())),
            ablation_model_names=_coerce_name_tuple(models.get("ablation_model_names", ())),
        )

        evaluation = EvaluationProtocol(
            clinical_escalation_thresholds=_coerce_float_mapping(
                pipeline.get("clinical_escalation_thresholds")
            ),
            label_proxy_caveats=_coerce_string_mapping(pipeline.get("label_proxy_caveats")),
            matched_knownness_gap_min=_coerce_float(
                pipeline.get("matched_knownness_gap_min"), default=DEFAULT_MATCHED_KNOWNNESS_GAP_MIN
            ),
            source_holdout_gap_min=_coerce_float(
                pipeline.get("source_holdout_gap_min"), default=DEFAULT_SOURCE_HOLDOUT_GAP_MIN
            ),
            spatial_holdout_gap_min=_coerce_float(
                pipeline.get("spatial_holdout_gap_min"), default=DEFAULT_SPATIAL_HOLDOUT_GAP_MIN
            ),
            ece_max=_coerce_float(
                pipeline.get("ece_max"), default=DEFAULT_ECE_MAX
            ),
            selection_adjusted_p_max=_coerce_float(
                models.get("selection_adjusted_p_max"), default=DEFAULT_SELECTION_ADJUSTED_P_MAX
            ),
        )

        protocol = cls(execution=execution, evaluation=evaluation)
        protocol.validate()
        return protocol
