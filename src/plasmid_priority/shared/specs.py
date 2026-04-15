"""Generic branch configuration models and loaders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from plasmid_priority.config import build_context


def _normalize_text_sequence(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )


def _normalize_feature_sets(feature_sets: Mapping[str, Sequence[str]] | None, defaults: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not feature_sets:
        return {name: _normalize_text_sequence(columns) for name, columns in defaults.items()}
    normalized: dict[str, tuple[str, ...]] = {}
    for model_name, columns in feature_sets.items():
        normalized[str(model_name).strip()] = _normalize_text_sequence(columns)
    return normalized


def _normalize_fit_config(
    fit_config: Mapping[str, Mapping[str, Any]] | None,
    defaults: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {name: dict(payload) for name, payload in defaults.items()}
    if not fit_config:
        return normalized
    for model_name, payload in fit_config.items():
        normalized[str(model_name).strip()] = dict(payload)
    return normalized


class BranchFitConfig(BaseModel):
    """Model fitting parameters for a branch candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    l2: float = 2.0
    sample_weight_mode: str | None = "source_balanced"
    max_iter: int = 400
    calibration: str = "isotonic"
    model_type: str = "logistic"
    fit_backend: str = "logistic"
    nonlinear_backend: str | None = None
    nonlinear_max_iter: int | None = None
    nonlinear_learning_rate: float | None = None
    nonlinear_max_depth: int | None = None
    nonlinear_min_samples_leaf: int | None = None
    nonlinear_l2: float | None = None
    agreement_review_threshold: float | None = None
    support_knownness_threshold: float | None = None
    ood_knownness_threshold: float | None = None
    confidence_review_threshold: float | None = None
    pairwise_max_pairs: int | None = None
    pairwise_random_state: int | None = None
    preprocess_mode: str | None = None
    preprocess_alpha: float | str | None = None
    preprocess_alpha_grid: tuple[float, ...] | None = None
    preprocess_alpha_grouped: bool | None = None
    preprocess_alpha_T: float | None = None
    preprocess_alpha_H: float | None = None
    preprocess_alpha_A: float | None = None
    pu_negative_min_weight: float | None = None
    pu_negative_power: float | None = None

    @field_validator("calibration")
    @classmethod
    def _validate_calibration(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"none", "platt", "isotonic"}:
            raise ValueError("calibration must be one of: none, platt, isotonic")
        return normalized

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"logistic", "hybrid_stacked", "pairwise_rank_logistic"}:
            raise ValueError("model_type must be one of: logistic, hybrid_stacked, pairwise_rank_logistic")
        return normalized

    @model_validator(mode="after")
    def _validate_fit_config(self) -> "BranchFitConfig":
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.sample_weight_mode is not None and not str(self.sample_weight_mode).strip():
            raise ValueError("sample_weight_mode cannot be empty")
        if self.nonlinear_learning_rate is not None and self.nonlinear_learning_rate <= 0.0:
            raise ValueError("nonlinear_learning_rate must be positive")
        if self.nonlinear_max_depth is not None and self.nonlinear_max_depth < 1:
            raise ValueError("nonlinear_max_depth must be at least 1")
        if self.nonlinear_min_samples_leaf is not None and self.nonlinear_min_samples_leaf < 1:
            raise ValueError("nonlinear_min_samples_leaf must be at least 1")
        if self.pairwise_max_pairs is not None and self.pairwise_max_pairs < 1:
            raise ValueError("pairwise_max_pairs must be at least 1")
        if self.pairwise_random_state is not None and self.pairwise_random_state < 0:
            raise ValueError("pairwise_random_state must be non-negative")
        if self.preprocess_alpha_grid is not None and not self.preprocess_alpha_grid:
            raise ValueError("preprocess_alpha_grid cannot be empty")
        return self


class BranchBenchmarkSpec(BaseModel):
    """Frozen benchmark definition for a branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "branch_v1"
    split_year: int = 2015
    horizon_years: int = 5
    assignment_mode: str = "training_only"
    entity_id_column: str = "backbone_id"
    label_column: str = "spread_label"
    outcome_column: str = "n_new_countries"
    required_columns: tuple[str, ...] = (
        "backbone_id",
        "spread_label",
        "split_year",
        "backbone_assignment_mode",
        "max_resolved_year_train",
        "min_resolved_year_test",
        "training_only_future_unseen_backbone_flag",
    )
    temporal_metadata_columns: tuple[str, ...] = (
        "split_year",
        "backbone_assignment_mode",
        "max_resolved_year_train",
        "min_resolved_year_test",
    )
    future_columns: tuple[str, ...] = ()
    positive_threshold: float = 1.0
    min_positive_conditions: int = 1
    future_label_column: str | None = None

    @model_validator(mode="after")
    def _validate_benchmark(self) -> "BranchBenchmarkSpec":
        if self.horizon_years < 0:
            raise ValueError("horizon_years must be non-negative")
        if self.min_positive_conditions < 1:
            raise ValueError("min_positive_conditions must be at least 1")
        if self.positive_threshold < 0.0:
            raise ValueError("positive_threshold must be non-negative")
        return self


class BranchModelSelectionSpec(BaseModel):
    """Weights for branch model selection scorecards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discrimination_weight: float = 0.40
    calibration_weight: float = 0.20
    stability_weight: float = 0.15
    knownness_weight: float = 0.10
    source_weight: float = 0.10
    explanation_weight: float = 0.05
    ood_penalty_weight: float = 0.10
    disagreement_penalty_weight: float = 0.10


class BranchConfig(BaseModel):
    """Fully resolved branch configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BranchBenchmarkSpec
    primary_model_name: str
    conservative_model_name: str
    fallback_model_name: str | None = None
    research_model_name: str | None = None
    core_model_names: tuple[str, ...] = ()
    research_model_names: tuple[str, ...] = ()
    ablation_model_names: tuple[str, ...] = ()
    feature_sets: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    fit_config: dict[str, BranchFitConfig] = Field(default_factory=dict)
    selection: BranchModelSelectionSpec = Field(default_factory=BranchModelSelectionSpec)

    @model_validator(mode="after")
    def _validate_model_surface(self) -> "BranchConfig":
        required_names = [self.primary_model_name, self.conservative_model_name]
        if self.research_model_name:
            required_names.append(self.research_model_name)
        if self.fallback_model_name:
            required_names.append(self.fallback_model_name)
        missing = sorted(
            {
                str(name)
                for name in required_names
                if str(name).strip() and str(name) not in self.feature_sets
            }
        )
        if missing:
            raise KeyError(
                "Branch configuration is missing feature sets for required model(s): "
                + ", ".join(missing)
            )
        return self

    @property
    def all_model_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    self.primary_model_name,
                    self.conservative_model_name,
                    *(self.core_model_names or ()),
                    *(self.research_model_names or ()),
                    *(self.ablation_model_names or ()),
                ]
            )
        )


def _load_raw_branch_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    return build_context().config


def load_branch_config(
    branch_key: str,
    config: Mapping[str, Any] | None = None,
    *,
    benchmark_defaults: BranchBenchmarkSpec | Mapping[str, Any] | None = None,
    primary_model_name: str,
    conservative_model_name: str,
    fallback_model_name: str | None = None,
    research_model_name: str | None = None,
    core_model_names: Sequence[str] = (),
    research_model_names: Sequence[str] = (),
    ablation_model_names: Sequence[str] = (),
    feature_sets: Mapping[str, Sequence[str]] | None = None,
    fit_config: Mapping[str, Mapping[str, Any]] | None = None,
    selection_defaults: Mapping[str, Any] | None = None,
) -> BranchConfig:
    """Resolve a branch configuration block from the project config."""
    payload = _load_raw_branch_config(config).get(branch_key, {})
    if not isinstance(payload, dict):
        payload = {}

    benchmark_payload = payload.get("benchmark", {})
    if isinstance(benchmark_defaults, BranchBenchmarkSpec):
        benchmark_base = benchmark_defaults.model_dump(mode="python")
    else:
        benchmark_base = dict(benchmark_defaults or {})
    benchmark = BranchBenchmarkSpec.model_validate({**benchmark_base, **dict(benchmark_payload)})

    normalized_feature_sets = _normalize_feature_sets(
        payload.get("feature_sets") if isinstance(payload.get("feature_sets"), Mapping) else None,
        feature_sets or {},
    )
    normalized_fit_config = _normalize_fit_config(
        payload.get("fit_config") if isinstance(payload.get("fit_config"), Mapping) else None,
        fit_config or {},
    )
    selection_payload = payload.get("selection", {}) if isinstance(payload.get("selection"), Mapping) else {}
    selection_base = dict(selection_defaults or {})
    selection = BranchModelSelectionSpec.model_validate({**selection_base, **dict(selection_payload)})

    def _resolve_names(key: str, defaults: Sequence[str]) -> tuple[str, ...]:
        raw = payload.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return _normalize_text_sequence(raw)
        return _normalize_text_sequence(defaults)

    return BranchConfig(
        benchmark=benchmark,
        primary_model_name=str(payload.get("primary_model_name", primary_model_name)).strip(),
        conservative_model_name=str(
            payload.get("conservative_model_name", conservative_model_name)
        ).strip(),
        fallback_model_name=(
            str(payload.get("fallback_model_name", fallback_model_name)).strip()
            if payload.get("fallback_model_name", fallback_model_name) is not None
            else None
        ),
        research_model_name=(
            str(payload.get("research_model_name", research_model_name)).strip()
            if payload.get("research_model_name", research_model_name) is not None
            else None
        ),
        core_model_names=_resolve_names("core_model_names", core_model_names),
        research_model_names=_resolve_names("research_model_names", research_model_names),
        ablation_model_names=_resolve_names("ablation_model_names", ablation_model_names),
        feature_sets=normalized_feature_sets,
        fit_config={name: BranchFitConfig.model_validate(payload) for name, payload in normalized_fit_config.items()},
        selection=selection,
    )


def resolve_branch_model_names(
    config: BranchConfig,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    """Resolve the active model surface for a branch."""
    names = list(config.core_model_names)
    names.extend([config.primary_model_name, config.conservative_model_name])
    if config.fallback_model_name:
        names.append(config.fallback_model_name)
    if include_research:
        names.extend(config.research_model_names)
        if config.research_model_name:
            names.append(config.research_model_name)
    if include_ablation:
        names.extend(config.ablation_model_names)
    return tuple(dict.fromkeys(str(name) for name in names if str(name).strip()))


def resolve_branch_fit_config(
    config: BranchConfig,
    model_name: str,
) -> BranchFitConfig:
    """Return the fit config for a named model."""
    try:
        return config.fit_config[str(model_name)]
    except KeyError as exc:
        raise KeyError(f"Unknown branch model: {model_name}") from exc
