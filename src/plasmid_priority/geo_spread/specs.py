"""Typed specs for the geo spread branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from plasmid_priority.config import build_context
from plasmid_priority.geo_spread.features import validate_geo_spread_feature_set

GEO_SPREAD_COUNTS_BASELINE = "geo_counts_baseline"
GEO_SPREAD_PARSIMONIOUS_PRIORITY = "geo_parsimonious_priority"
GEO_SPREAD_PHYLO_ECOLOGY_PRIORITY = "geo_phylo_ecology_priority"
GEO_SPREAD_SUPPORT_LIGHT_PRIORITY = "geo_support_light_priority"
GEO_SPREAD_CONTEXT_HYBRID_PRIORITY = "geo_context_hybrid_priority"
GEO_SPREAD_RESEARCH_UPPER_BOUND = "geo_research_upper_bound"
GEO_SPREAD_RELIABILITY_BLEND = "geo_reliability_blend"
GEO_SPREAD_ADAPTIVE_PRIORITY = "geo_adaptive_knownness_priority"
GEO_SPREAD_META_PRIORITY = "geo_meta_knownness_priority"
GEO_SPREAD_DERIVED_MODEL_NAMES: tuple[str, ...] = (
    GEO_SPREAD_RELIABILITY_BLEND,
    GEO_SPREAD_ADAPTIVE_PRIORITY,
    GEO_SPREAD_META_PRIORITY,
)

DEFAULT_GEO_SPREAD_CORE_MODEL_NAMES: tuple[str, ...] = (
    GEO_SPREAD_COUNTS_BASELINE,
    GEO_SPREAD_PARSIMONIOUS_PRIORITY,
    GEO_SPREAD_PHYLO_ECOLOGY_PRIORITY,
    GEO_SPREAD_SUPPORT_LIGHT_PRIORITY,
    GEO_SPREAD_CONTEXT_HYBRID_PRIORITY,
)
DEFAULT_GEO_SPREAD_RESEARCH_MODEL_NAMES: tuple[str, ...] = (GEO_SPREAD_RESEARCH_UPPER_BOUND,)
DEFAULT_GEO_SPREAD_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    GEO_SPREAD_COUNTS_BASELINE: (
        "log1p_member_count_train",
        "log1p_n_countries_train",
        "refseq_share_train",
    ),
    GEO_SPREAD_PARSIMONIOUS_PRIORITY: (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "A_eff_norm",
        "coherence_score",
    ),
    GEO_SPREAD_PHYLO_ECOLOGY_PRIORITY: (
        "T_eff_norm",
        "H_phylogenetic_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "host_phylogenetic_dispersion_norm",
        "host_taxon_evenness_norm",
        "ecology_context_diversity_norm",
    ),
    GEO_SPREAD_SUPPORT_LIGHT_PRIORITY: (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "orit_support",
        "H_external_host_range_norm",
    ),
    GEO_SPREAD_CONTEXT_HYBRID_PRIORITY: (
        "T_eff_norm",
        "H_obs_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "orit_support",
        "H_external_host_range_norm",
        "geo_country_entropy_train",
        "geo_macro_region_entropy_train",
        "geo_dominant_region_share_train",
        "geo_country_record_count_train",
    ),
    GEO_SPREAD_RESEARCH_UPPER_BOUND: (
        "T_eff_norm",
        "H_phylogenetic_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "host_phylogenetic_dispersion_norm",
        "host_taxon_evenness_norm",
        "ecology_context_diversity_norm",
        "mash_neighbor_distance_train_norm",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "clinical_context_fraction_norm",
        "pathogenic_context_fraction_norm",
        "support_shrinkage_norm",
        "mash_graph_novelty_score",
        "mash_graph_bridge_fraction",
    ),
}
DEFAULT_GEO_SPREAD_FIT_CONFIG_PAYLOAD: dict[str, dict[str, Any]] = {
    GEO_SPREAD_COUNTS_BASELINE: {
        "l2": 1.0,
        "sample_weight_mode": None,
        "max_iter": 250,
        "calibration": "none",
    },
    GEO_SPREAD_PARSIMONIOUS_PRIORITY: {
        "l2": 2.0,
        "sample_weight_mode": "source_balanced",
        "max_iter": 400,
        "calibration": "isotonic",
    },
    GEO_SPREAD_PHYLO_ECOLOGY_PRIORITY: {
        "l2": 2.5,
        "sample_weight_mode": "source_balanced",
        "max_iter": 400,
        "calibration": "isotonic",
    },
    GEO_SPREAD_SUPPORT_LIGHT_PRIORITY: {
        "l2": 3.0,
        "sample_weight_mode": "source_balanced",
        "max_iter": 500,
        "calibration": "isotonic",
    },
    GEO_SPREAD_CONTEXT_HYBRID_PRIORITY: {
        "l2": 2.0,
        "sample_weight_mode": "source_balanced+knownness_balanced",
        "max_iter": 700,
        "calibration": "isotonic",
        "model_type": "hybrid_stacked",
        "nonlinear_backend": "hist_gbm",
        "nonlinear_max_iter": 300,
        "nonlinear_learning_rate": 0.04,
        "nonlinear_max_depth": 3,
        "nonlinear_min_samples_leaf": 12,
        "agreement_review_threshold": 0.75,
    },
    GEO_SPREAD_RESEARCH_UPPER_BOUND: {
        "l2": 3.5,
        "sample_weight_mode": "class_balanced+knownness_balanced",
        "max_iter": 500,
        "calibration": "platt",
    },
}


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


def _normalize_feature_sets(feature_sets: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    if not feature_sets:
        return {name: tuple(columns) for name, columns in DEFAULT_GEO_SPREAD_FEATURE_SETS.items()}
    normalized: dict[str, tuple[str, ...]] = {}
    for model_name, columns in feature_sets.items():
        normalized[str(model_name).strip()] = _normalize_text_sequence(columns)
    return normalized


def _normalize_fit_config(fit_config: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {
        name: dict(payload) for name, payload in DEFAULT_GEO_SPREAD_FIT_CONFIG_PAYLOAD.items()
    }
    if not fit_config:
        return normalized
    for model_name, payload in fit_config.items():
        normalized[str(model_name).strip()] = dict(payload)
    return normalized


class GeoSpreadFitConfig(BaseModel):
    """Fitting parameters for an individual geo spread model."""

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
    def _validate_fit_config(self) -> "GeoSpreadFitConfig":
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
        return self


class GeoSpreadBenchmarkSpec(BaseModel):
    """Frozen benchmark definition for the geo spread branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "geo_spread_v1"
    split_year: int = 2015
    min_new_countries_for_spread: int = 3
    horizon_years: int = 5
    assignment_mode: str = "training_only"
    entity_id_column: str = "backbone_id"
    label_column: str = "spread_label"
    outcome_column: str = "n_new_countries"
    required_columns: tuple[str, ...] = (
        "backbone_id",
        "spread_label",
        "n_new_countries",
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

    @model_validator(mode="after")
    def _validate_benchmark(self) -> "GeoSpreadBenchmarkSpec":
        if self.split_year < 1900:
            raise ValueError("split_year looks invalid")
        if self.min_new_countries_for_spread < 1:
            raise ValueError("min_new_countries_for_spread must be at least 1")
        if self.horizon_years < 1:
            raise ValueError("horizon_years must be at least 1")
        if self.assignment_mode != "training_only":
            raise ValueError("geo spread benchmark requires training_only assignment")
        if self.entity_id_column.strip() != "backbone_id":
            raise ValueError("geo spread benchmark entity_id_column must be backbone_id")
        if self.label_column.strip() != "spread_label":
            raise ValueError("geo spread benchmark label_column must be spread_label")
        if self.outcome_column.strip() != "n_new_countries":
            raise ValueError("geo spread benchmark outcome_column must be n_new_countries")
        return self


class GeoSpreadModelSelectionSpec(BaseModel):
    """Selection policy for geo spread model families."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_model_name: str = GEO_SPREAD_RELIABILITY_BLEND
    conservative_model_name: str = GEO_SPREAD_COUNTS_BASELINE
    fallback_model_name: str = GEO_SPREAD_CONTEXT_HYBRID_PRIORITY
    research_model_name: str = GEO_SPREAD_RESEARCH_UPPER_BOUND
    core_model_names: tuple[str, ...] = DEFAULT_GEO_SPREAD_CORE_MODEL_NAMES
    research_model_names: tuple[str, ...] = DEFAULT_GEO_SPREAD_RESEARCH_MODEL_NAMES
    ablation_model_names: tuple[str, ...] = ()


class GeoSpreadConfig(BaseModel):
    """Typed geo spread branch configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: GeoSpreadBenchmarkSpec = Field(default_factory=GeoSpreadBenchmarkSpec)
    primary_model_name: str = GEO_SPREAD_RELIABILITY_BLEND
    conservative_model_name: str = GEO_SPREAD_COUNTS_BASELINE
    fallback_model_name: str = GEO_SPREAD_CONTEXT_HYBRID_PRIORITY
    research_model_name: str = GEO_SPREAD_RESEARCH_UPPER_BOUND
    core_model_names: tuple[str, ...] = DEFAULT_GEO_SPREAD_CORE_MODEL_NAMES
    research_model_names: tuple[str, ...] = DEFAULT_GEO_SPREAD_RESEARCH_MODEL_NAMES
    ablation_model_names: tuple[str, ...] = ()
    feature_sets: dict[str, tuple[str, ...]] = Field(
        default_factory=lambda: dict(DEFAULT_GEO_SPREAD_FEATURE_SETS)
    )
    fit_config: dict[str, GeoSpreadFitConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        selection = payload.pop("selection", None)
        if isinstance(selection, dict):
            for key, fallback in (
                ("primary_model_name", None),
                ("conservative_model_name", None),
                ("fallback_model_name", None),
                ("research_model_name", None),
                ("core_model_names", ()),
                ("research_model_names", ()),
                ("ablation_model_names", ()),
            ):
                if key not in payload or payload.get(key) in (None, ""):
                    if key in selection:
                        payload[key] = selection.get(key, fallback)
        payload["feature_sets"] = _normalize_feature_sets(payload.get("feature_sets"))
        payload["fit_config"] = _normalize_fit_config(payload.get("fit_config"))
        for key in (
            "primary_model_name",
            "conservative_model_name",
            "fallback_model_name",
            "research_model_name",
        ):
            if key in payload and payload[key] is not None:
                payload[key] = str(payload[key]).strip()
        for key in ("core_model_names", "research_model_names", "ablation_model_names"):
            if key in payload:
                payload[key] = _normalize_text_sequence(payload.get(key))
        return payload

    @model_validator(mode="after")
    def _validate_config(self) -> "GeoSpreadConfig":
        available = set(self.feature_sets) | set(GEO_SPREAD_DERIVED_MODEL_NAMES)
        selection_names = {
            self.primary_model_name,
            self.conservative_model_name,
            self.fallback_model_name,
            self.research_model_name,
            *self.core_model_names,
            *self.research_model_names,
            *self.ablation_model_names,
        }
        missing = sorted(name for name in selection_names if name not in available)
        if missing:
            raise ValueError(
                "geo spread selection references missing model definitions: "
                + ", ".join(missing)
            )
        missing_fit = sorted(name for name in self.fit_config if name not in available)
        if missing_fit:
            raise ValueError(
                "geo spread fit_config references missing model definitions: "
                + ", ".join(missing_fit)
            )
        selected_fit_names = {
            self.primary_model_name,
            self.conservative_model_name,
            self.fallback_model_name,
            self.research_model_name,
            *self.core_model_names,
            *self.research_model_names,
        } - set(GEO_SPREAD_DERIVED_MODEL_NAMES)
        missing_selected_fit = sorted(selected_fit_names - set(self.fit_config))
        if missing_selected_fit:
            raise ValueError(
                "geo spread fit_config is missing selected model definitions: "
                + ", ".join(missing_selected_fit)
            )
        for model_name, feature_names in self.feature_sets.items():
            validate_geo_spread_feature_set(feature_names, label=f"geo spread model `{model_name}`")
        return self

    @property
    def selection(self) -> GeoSpreadModelSelectionSpec:
        return GeoSpreadModelSelectionSpec(
            primary_model_name=self.primary_model_name,
            conservative_model_name=self.conservative_model_name,
            fallback_model_name=self.fallback_model_name,
            research_model_name=self.research_model_name,
            core_model_names=self.core_model_names,
            research_model_names=self.research_model_names,
            ablation_model_names=self.ablation_model_names,
        )


def resolve_geo_spread_model_names(
    config: Mapping[str, Any] | None = None,
    *,
    include_research: bool = False,
    include_ablation: bool = False,
) -> tuple[str, ...]:
    """Resolve the branch model surface from project config defaults."""
    geo_config = load_geo_spread_config(config)
    names = list(geo_config.core_model_names)
    if include_research:
        names.extend(geo_config.research_model_names)
    if include_ablation:
        names.extend(geo_config.ablation_model_names)
    return tuple(dict.fromkeys(names))


def resolve_geo_spread_fit_config(
    model_name: str,
    config: Mapping[str, Any] | None = None,
) -> GeoSpreadFitConfig:
    """Resolve the typed fit config for a geo spread model."""
    geo_config = load_geo_spread_config(config)
    try:
        return geo_config.fit_config[str(model_name)]
    except KeyError as exc:
        raise KeyError(f"Unknown geo spread model fit config: {model_name}") from exc


def load_geo_spread_config(
    config: Mapping[str, Any] | GeoSpreadConfig | None = None,
) -> GeoSpreadConfig:
    """Load the typed geo spread config from a project config mapping."""
    if isinstance(config, GeoSpreadConfig):
        return config
    project_config = config
    if project_config is None:
        project_config = build_context().config
    branch = project_config.get("geo_spread", {}) if isinstance(project_config, Mapping) else {}
    if not isinstance(branch, dict):
        branch = {}
    return GeoSpreadConfig.model_validate(branch)
