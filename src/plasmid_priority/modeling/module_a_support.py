"""Shared config and low-level fitting helpers for Module A."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from plasmid_priority.config import build_context


def _load_project_config() -> dict:
    return build_context().config


# Thread-local storage for project config - thread-safe
_project_config_local = threading.local()

_DEFAULT_MODEL_CONFIG = {
    "primary_model_name": "bio_clean_priority",
    "primary_model_fallback": "parsimonious_priority",
    "conservative_model_name": "parsimonious_priority",
    "governance_model_name": "phylo_support_fusion_priority",
    "governance_model_fallback": "support_synergy_priority",
    "feature_sets": {},
    "core_model_names": (),
    "research_model_names": (),
    "ablation_model_names": (),
    "fit_config": {},
    "novelty_specialist": {"features": (), "fit_config": {}},
}

_EXACT_COMPLEMENT_FEATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("H_obs_norm", "H_obs_specialization_norm"),
    ("H_breadth_norm", "H_specialization_norm"),
    ("H_phylogenetic_norm", "H_phylogenetic_specialization_norm"),
    ("H_augmented_norm", "H_augmented_specialization_norm"),
    ("H_phylogenetic_augmented_norm", "H_phylogenetic_augmented_specialization_norm"),
)

_BAYESIAN_CI_Z = 1.959963984540054

SINGLE_MODEL_PARETO_PARENT_MODEL_NAMES: tuple[str, ...] = (
    "phylo_support_fusion_priority",
    "sovereign_precision_priority",
    "sovereign_v2_priority",
    "discovery_12f_source",
    "support_synergy_priority",
    "knownness_robust_priority",
    "parsimonious_priority",
)


def _get_model_config() -> dict:
    if not hasattr(_project_config_local, "value") or _project_config_local.value is None:
        _project_config_local.value = _load_project_config()
    models = (
        _project_config_local.value.get("models", {})
        if isinstance(_project_config_local.value, dict)
        else {}
    )
    if not isinstance(models, dict):
        return _DEFAULT_MODEL_CONFIG
    return {
        "primary_model_name": models.get(
            "primary_model_name", _DEFAULT_MODEL_CONFIG["primary_model_name"]
        ),
        "primary_model_fallback": models.get(
            "primary_model_fallback", _DEFAULT_MODEL_CONFIG["primary_model_fallback"]
        ),
        "conservative_model_name": models.get(
            "conservative_model_name", _DEFAULT_MODEL_CONFIG["conservative_model_name"]
        ),
        "governance_model_name": models.get(
            "governance_model_name", _DEFAULT_MODEL_CONFIG["governance_model_name"]
        ),
        "governance_model_fallback": models.get(
            "governance_model_fallback", _DEFAULT_MODEL_CONFIG["governance_model_fallback"]
        ),
        "feature_sets": models.get("feature_sets", _DEFAULT_MODEL_CONFIG["feature_sets"]),
        "core_model_names": tuple(
            models.get("core_model_names", _DEFAULT_MODEL_CONFIG["core_model_names"])
        ),
        "research_model_names": tuple(
            models.get("research_model_names", _DEFAULT_MODEL_CONFIG["research_model_names"])
        ),
        "ablation_model_names": tuple(
            models.get("ablation_model_names", _DEFAULT_MODEL_CONFIG["ablation_model_names"])
        ),
        "fit_config": models.get("fit_config", _DEFAULT_MODEL_CONFIG["fit_config"]),
        "novelty_specialist": models.get(
            "novelty_specialist", _DEFAULT_MODEL_CONFIG["novelty_specialist"]
        ),
    }


def _config_snapshot() -> dict:
    config = _get_model_config()
    novelty = (
        config.get("novelty_specialist", {})
        if isinstance(config.get("novelty_specialist"), dict)
        else {}
    )
    return {
        "PRIMARY_MODEL_NAME": config["primary_model_name"],
        "PRIMARY_MODEL_FALLBACK": config["primary_model_fallback"],
        "CONSERVATIVE_MODEL_NAME": config["conservative_model_name"],
        "GOVERNANCE_MODEL_NAME": config["governance_model_name"],
        "GOVERNANCE_MODEL_FALLBACK": config["governance_model_fallback"],
        "MODULE_A_FEATURE_SETS": config["feature_sets"],
        "CORE_MODEL_NAMES": tuple(config["core_model_names"]),
        "RESEARCH_MODEL_NAMES": tuple(config["research_model_names"]),
        "ABLATION_MODEL_NAMES": tuple(config["ablation_model_names"]),
        "MODEL_FIT_CONFIG": config["fit_config"],
        "NOVELTY_SPECIALIST_FEATURES": novelty.get("features", ()),
        "NOVELTY_SPECIALIST_FIT_CONFIG": novelty.get("fit_config", {}),
    }


@functools.lru_cache(maxsize=1)
def _resolved_model_config() -> dict:
    """Resolve the configured model surface once per interpreter session."""
    return _config_snapshot()


_RESOLVED_MODEL_CONFIG = _resolved_model_config()

PRIMARY_MODEL_NAME: str = cast(str, _RESOLVED_MODEL_CONFIG["PRIMARY_MODEL_NAME"])
PRIMARY_MODEL_FALLBACK: str = cast(str, _RESOLVED_MODEL_CONFIG["PRIMARY_MODEL_FALLBACK"])
CONSERVATIVE_MODEL_NAME: str = cast(str, _RESOLVED_MODEL_CONFIG["CONSERVATIVE_MODEL_NAME"])
GOVERNANCE_MODEL_NAME: str = cast(str, _RESOLVED_MODEL_CONFIG["GOVERNANCE_MODEL_NAME"])
GOVERNANCE_MODEL_FALLBACK: str = cast(str, _RESOLVED_MODEL_CONFIG["GOVERNANCE_MODEL_FALLBACK"])
MODULE_A_FEATURE_SETS: dict[str, list[str]] = cast(
    dict[str, list[str]], _RESOLVED_MODEL_CONFIG["MODULE_A_FEATURE_SETS"]
)
CORE_MODEL_NAMES: tuple[str, ...] = cast(
    tuple[str, ...], _RESOLVED_MODEL_CONFIG["CORE_MODEL_NAMES"]
)
RESEARCH_MODEL_NAMES: tuple[str, ...] = cast(
    tuple[str, ...], _RESOLVED_MODEL_CONFIG["RESEARCH_MODEL_NAMES"]
)
ABLATION_MODEL_NAMES: tuple[str, ...] = cast(
    tuple[str, ...], _RESOLVED_MODEL_CONFIG["ABLATION_MODEL_NAMES"]
)
MODEL_FIT_CONFIG: dict[str, dict[str, object]] = cast(
    dict[str, dict[str, object]], _RESOLVED_MODEL_CONFIG["MODEL_FIT_CONFIG"]
)
NOVELTY_SPECIALIST_FEATURES: tuple[str, ...] = cast(
    tuple[str, ...], _RESOLVED_MODEL_CONFIG["NOVELTY_SPECIALIST_FEATURES"]
)
NOVELTY_SPECIALIST_FIT_CONFIG: dict[str, object] = cast(
    dict[str, object], _RESOLVED_MODEL_CONFIG["NOVELTY_SPECIALIST_FIT_CONFIG"]
)


@dataclass(frozen=True)
class FeatureProvenance:
    track: str
    provenance: str
    note: str = ""


_BASELINE_FEATURES = {
    "log1p_member_count_train",
    "log1p_n_countries_train",
    "refseq_share_train",
}

_DISCOVERY_FEATURES = {
    "A_eff_norm",
    "A_raw_norm",
    "A_recurrence_norm",
    "H_A_synergy_norm",
    "H_breadth_norm",
    "H_obs_norm",
    "H_obs_specialization_norm",
    "H_phylogenetic_specialization_norm",
    "H_specialization_norm",
    "T_A_synergy_norm",
    "T_eff_norm",
    "T_raw_norm",
    "amr_gene_burden_norm",
    "amr_load_density_norm",
    "amr_support_norm",
    "amr_support_norm_residual",
    "assignment_confidence_norm",
    "backbone_purity_norm",
    "clinical_A_synergy_norm",
    "clinical_context_fraction_norm",
    "clinical_context_sparse_penalty_norm",
    "coherence_score",
    "eco_clinical_context_saturation_norm",
    "ecology_context_diversity_norm",
    "evolutionary_jump_score_norm",
    "host_phylogenetic_dispersion_norm",
    "host_taxon_evenness_norm",
    "mash_neighbor_distance_train_norm",
    "mash_graph_bridge_fraction",
    "mash_graph_external_neighbor_count",
    "mash_graph_novelty_score",
    "mean_n_replicon_types_norm",
    "orit_support",
    "pathogenic_context_fraction_norm",
    "replicon_architecture_norm",
    "replicon_multiplicity_saturation_norm",
    "support_shrinkage_norm",
    "amr_agreement_score",
    "mean_amr_uncertainty_score",
    "T_H_obs_synergy_norm",
    "A_H_obs_synergy_norm",
    "T_coherence_synergy_norm",
}

_GOVERNANCE_FEATURES = {
    "H_eff_norm",
    "H_evenness_T_synergy_norm",
    "H_external_host_range_norm",
    "H_external_host_range_support",
    "H_phylogenetic_augmented_specialization_norm",
    "H_support_norm",
    "H_support_norm_residual",
    "amr_burden_saturation_norm",
    "amr_class_richness_norm",
    "amr_clinical_escalation_norm",
    "amr_clinical_threat_norm",
    "amr_mdr_proxy_norm",
    "amr_mechanism_diversity_norm",
    "amr_xdr_proxy_norm",
    "context_support_guard_norm",
    "external_t_synergy_norm",
    "host_range_saturation_norm",
    "last_resort_convergence_norm",
    "metadata_support_depth_norm",
    "monotonic_latent_priority_index",
    "plasmidfinder_complexity_norm",
    "plasmidfinder_support_norm",
    "pmlst_coherence_norm",
    "pmlst_presence_fraction_train",
    "pmlst_presence_norm",
    "priority_index",
    "silent_carrier_risk_norm",
}


def _configured_feature_names() -> tuple[str, ...]:
    configured: set[str] = set()
    for columns in MODULE_A_FEATURE_SETS.values():
        configured.update(str(column) for column in columns)
    configured.update(str(column) for column in NOVELTY_SPECIALIST_FEATURES)
    return tuple(sorted(configured))


def _build_feature_provenance_registry() -> dict[str, FeatureProvenance]:
    registry: dict[str, FeatureProvenance] = {}
    missing: list[str] = []
    for feature_name in _configured_feature_names():
        if feature_name in _BASELINE_FEATURES:
            registry[feature_name] = FeatureProvenance(
                track="baseline",
                provenance="visibility_or_sampling_proxy",
            )
            continue
        if feature_name in _DISCOVERY_FEATURES:
            registry[feature_name] = FeatureProvenance(
                track="discovery",
                provenance="pre_split_observed_or_sequence_intrinsic",
            )
            continue
        if feature_name in _GOVERNANCE_FEATURES:
            registry[feature_name] = FeatureProvenance(
                track="governance",
                provenance="database_or_support_augmented",
            )
            continue
        missing.append(feature_name)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise KeyError(
            "Feature provenance registry is incomplete for configured Module A features: "
            f"{formatted}"
        )
    return registry


def _infer_model_track(columns: list[str]) -> str:
    tracks = {FEATURE_PROVENANCE_REGISTRY[str(column)].track for column in columns}
    if "governance" in tracks:
        return "governance"
    if "discovery" in tracks:
        return "discovery"
    return "baseline"


def _build_model_track_registry() -> dict[str, str]:
    return {
        model_name: _infer_model_track(columns)
        for model_name, columns in MODULE_A_FEATURE_SETS.items()
    }


def _validate_module_a_feature_surface() -> None:
    official_discovery_models = {
        PRIMARY_MODEL_NAME,
        PRIMARY_MODEL_FALLBACK,
        CONSERVATIVE_MODEL_NAME,
    }
    official_governance_models = {
        GOVERNANCE_MODEL_NAME,
        GOVERNANCE_MODEL_FALLBACK,
    }
    unknown_official = sorted(
        (official_discovery_models | official_governance_models) - set(MODULE_A_FEATURE_SETS)
    )
    if unknown_official:
        missing = ", ".join(unknown_official)
        raise KeyError(f"Official Module A model(s) missing from feature sets: {missing}")

    for model_name in official_discovery_models:
        if MODULE_A_MODEL_TRACKS[model_name] != "discovery":
            raise ValueError(
                "Official discovery model "
                f"`{model_name}` is not discovery-safe under the current "
                "feature registry."
            )
    for model_name in official_governance_models:
        if MODULE_A_MODEL_TRACKS[model_name] != "governance":
            raise ValueError(
                "Official governance model "
                f"`{model_name}` must include governance-only support "
                "features."
            )
    governance_columns = MODULE_A_FEATURE_SETS[GOVERNANCE_MODEL_NAME]
    if not any(
        FEATURE_PROVENANCE_REGISTRY[column].track == "governance" for column in governance_columns
    ):
        raise ValueError(
            "Governance model "
            f"`{GOVERNANCE_MODEL_NAME}` does not include any governance-only "
            "features."
        )
    for model_name, columns in MODULE_A_FEATURE_SETS.items():
        feature_names = {str(column) for column in columns}
        conflicting = [
            f"{left}/{right}"
            for left, right in _EXACT_COMPLEMENT_FEATURE_PAIRS
            if left in feature_names and right in feature_names
        ]
        if conflicting:
            raise ValueError(
                "Module A model "
                f"`{model_name}` includes exact-complement feature pairs: {', '.join(conflicting)}"
            )


def _pruned_single_model_features(columns: list[str]) -> tuple[str, ...]:
    """Return a deterministic compact descendant of a parent feature list."""
    if len(columns) <= 3:
        return tuple(columns)
    prune_count = max(1, len(columns) // 4)
    keep_count = max(3, len(columns) - prune_count)
    keep_count = min(keep_count, len(columns) - 1)
    return tuple(columns[:keep_count])


def build_single_model_candidate_family() -> pd.DataFrame:
    """Return the bounded parent-derived family used for single-model screening."""
    rows: list[dict[str, object]] = []
    for parent_model_name in SINGLE_MODEL_PARETO_PARENT_MODEL_NAMES:
        if parent_model_name not in MODULE_A_FEATURE_SETS:
            continue
        feature_set = tuple(str(column) for column in MODULE_A_FEATURE_SETS[parent_model_name])
        rows.append(
            {
                "model_name": parent_model_name,
                "parent_model_name": parent_model_name,
                "feature_set": feature_set,
                "feature_count": int(len(feature_set)),
                "candidate_kind": "parent",
            }
        )
        pruned_feature_set = _pruned_single_model_features(list(feature_set))
        if pruned_feature_set and pruned_feature_set != feature_set:
            rows.append(
                {
                    "model_name": f"{parent_model_name}__pruned",
                    "parent_model_name": parent_model_name,
                    "feature_set": pruned_feature_set,
                    "feature_count": int(len(pruned_feature_set)),
                    "candidate_kind": "pruned",
                }
            )
    family = pd.DataFrame(
        rows,
        columns=[
            "model_name",
            "parent_model_name",
            "feature_set",
            "feature_count",
            "candidate_kind",
        ],
    )
    if family.empty:
        return family
    kind_order = {"parent": 0, "pruned": 1}
    family = family.assign(
        _candidate_kind_order=family["candidate_kind"].map(kind_order).fillna(99)
    )
    return (
        family.sort_values(
            ["parent_model_name", "_candidate_kind_order", "feature_count", "model_name"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop(columns="_candidate_kind_order")
        .reset_index(drop=True)
    )


FEATURE_PROVENANCE_REGISTRY: dict[str, FeatureProvenance] = _build_feature_provenance_registry()
MODULE_A_MODEL_TRACKS: dict[str, str] = _build_model_track_registry()
_validate_module_a_feature_surface()


def assert_all_discovery_safe(feature_names: list[str]) -> None:
    """Raise ValueError if any feature is not in _DISCOVERY_FEATURES ∪ _BASELINE_FEATURES.

    This is a code-level enforcement guard callable from tests and scripts before
    any new discovery feature set is run. It ensures that discovery models only
    use features that are safe for the discovery track.

    Args:
        feature_names: List of feature names to validate.

    Raises:
        ValueError: If any feature is not in _DISCOVERY_FEATURES or _BASELINE_FEATURES.

    Example:
        >>> assert_all_discovery_safe(["T_eff_norm", "H_obs_specialization_norm"])
        # No error raised - all features are discovery-safe
        >>> assert_all_discovery_safe(["H_external_host_range_support"])
        ValueError: Non-discovery features detected: H_external_host_range_support
    """
    allowed = _DISCOVERY_FEATURES | _BASELINE_FEATURES
    invalid = [name for name in feature_names if name not in allowed]
    if invalid:
        raise ValueError(
            f"Non-discovery features detected: {', '.join(sorted(invalid))}. "
            f"These features belong to the governance track and cannot be used in discovery models."
        )


def _fit_kwarg_value(
    fit_kwargs: Mapping[str, object] | None,
    key: str,
    default: object,
) -> object:
    if fit_kwargs is None:
        return default
    value = fit_kwargs.get(key, default)
    return default if value is None else value


def _fit_kwarg_float(
    fit_kwargs: Mapping[str, object] | None,
    key: str,
    default: float,
) -> float:
    value: Any = _fit_kwarg_value(fit_kwargs, key, default)
    return float(value)


def _fit_kwarg_int(
    fit_kwargs: Mapping[str, object] | None,
    key: str,
    default: int,
) -> int:
    value: Any = _fit_kwarg_value(fit_kwargs, key, default)
    return int(value)


def _fit_kwarg_str(
    fit_kwargs: Mapping[str, object] | None,
    key: str,
    default: str,
) -> str:
    value: Any = _fit_kwarg_value(fit_kwargs, key, default)
    return str(value)


def _fit_kwarg_mode(
    fit_kwargs: Mapping[str, object] | None,
    key: str = "sample_weight_mode",
) -> str | None:
    value = _fit_kwarg_value(fit_kwargs, key, None)
    if value in (None, "", "none"):
        return None
    return str(value)


def _ensure_config_loaded() -> None:
    """Retained for backwards compatibility with existing call sites."""
    _resolved_model_config()


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    status: str = "ok"
    error_message: str | None = None


def build_failed_model_result(name: str, error_message: str) -> ModelResult:
    return ModelResult(
        name=name,
        metrics={},
        predictions=pd.DataFrame(),
        status="failed",
        error_message=error_message,
    )


def get_active_model_names(model_metrics: pd.DataFrame | Mapping[str, object]) -> list[str]:
    """Return successful model names, falling back to all available model entries."""
    if isinstance(model_metrics, pd.DataFrame):
        if "model_name" not in model_metrics.columns:
            return []
        frame_all_names = [
            str(name) for name in model_metrics["model_name"].dropna().astype(str).tolist()
        ]
        if "status" not in model_metrics.columns:
            return frame_all_names
        frame_active_names = [
            str(name)
            for name in model_metrics.loc[
                model_metrics["status"].fillna("ok").astype(str).eq("ok"), "model_name"
            ]
            .dropna()
            .astype(str)
            .tolist()
        ]
        return frame_active_names or frame_all_names

    mapping_all_names: list[str] = []
    mapping_active_names: list[str] = []
    for name, metrics in model_metrics.items():
        if not isinstance(metrics, Mapping):
            continue
        model_name = str(name)
        mapping_all_names.append(model_name)
        status = str(metrics.get("status", "ok") or "ok").strip().lower()
        if status == "ok":
            mapping_active_names.append(model_name)
    return mapping_active_names or mapping_all_names


def _top_k_precision_recall(y: np.ndarray, preds: np.ndarray, *, top_k: int) -> tuple[float, float]:
    if len(y) == 0:
        return float("nan"), float("nan")
    k = max(1, min(int(top_k), len(y)))
    order = np.argsort(-preds, kind="mergesort")[:k]
    selected = y[order]
    positives = max(int((y == 1).sum()), 1)
    precision = float(np.mean(selected == 1))
    recall = float(np.sum(selected == 1) / positives)
    return precision, recall


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use robust scaling so long-tailed backbone counts do not dominate optimization.
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = np.asarray(q75 - q25, dtype=float)
    zero_scale_mask = ~np.isfinite(iqr) | (iqr <= 0.0)
    if np.any(zero_scale_mask):
        mad_scale = (
            np.median(
                np.abs(X[:, zero_scale_mask] - median[zero_scale_mask]),
                axis=0,
            )
            * 1.4826
        )
        iqr[zero_scale_mask] = np.asarray(mad_scale, dtype=float)
    zero_scale_mask = ~np.isfinite(iqr) | (iqr <= 0.0)
    if np.any(zero_scale_mask):
        range_scale = np.ptp(X[:, zero_scale_mask], axis=0)
        iqr[zero_scale_mask] = np.asarray(range_scale, dtype=float)
    iqr[~np.isfinite(iqr) | (iqr <= 0.0)] = 1.0
    return (
        np.asarray((X - median) / iqr, dtype=float),
        np.asarray(median, dtype=float),
        np.asarray(iqr, dtype=float),
    )


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray((X - mean) / std, dtype=float)


class _IdentityImputer:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


def _fit_feature_imputer(X_train: np.ndarray) -> tuple[np.ndarray, KNNImputer | _IdentityImputer]:
    """Fit the feature imputer on training rows only so fold holdouts stay disjoint."""
    if not np.isnan(X_train).any():
        return np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0), _IdentityImputer()
    imputer = KNNImputer(
        n_neighbors=5,
        weights="distance",
        keep_empty_features=True,
    )
    X_train_imputed = np.asarray(imputer.fit_transform(X_train), dtype=float)
    return np.nan_to_num(X_train_imputed, nan=0.0), imputer


def _fit_logistic_regression_with_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
    fit_backend: str = "logistic",
) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
    backend = str(fit_backend).strip().lower()
    if backend in {"firthlogist", "firthlogist_sidecar", "firth_external"}:
        return _fit_firthlogist_sidecar_with_diagnostics(
            X,
            y,
            max_iter=max_iter,
            sample_weight=sample_weight,
        )
    if backend in {"firth", "firth_logistic"}:
        return _fit_firth_logistic_regression_with_diagnostics(
            X,
            y,
            l2=l2,
            max_iter=max_iter,
            sample_weight=sample_weight,
        )
    C_val = 1.0 / max(l2, 1e-5)
    base_max_iter = max(int(max_iter), 1000)
    attempts = (
        ("lbfgs", base_max_iter),
        ("lbfgs", max(base_max_iter * 5, 5000)),
        ("liblinear", max(base_max_iter * 5, 5000)),
    )

    last_clf: LogisticRegression | None = None
    last_convergence_warnings: list[warnings.WarningMessage] = []
    last_effective_max_iter = base_max_iter
    last_solver = "lbfgs"

    for solver_name, effective_max_iter in attempts:
        clf = LogisticRegression(
            C=C_val,
            solver=solver_name,
            max_iter=effective_max_iter,
            tol=1e-5,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X, y, sample_weight=sample_weight)

        convergence_warnings = [
            warning_record
            for warning_record in caught
            if issubclass(warning_record.category, ConvergenceWarning)
        ]
        for warning_record in caught:
            if issubclass(warning_record.category, ConvergenceWarning):
                continue
            warnings.warn(
                str(warning_record.message),
                category=warning_record.category,
                stacklevel=2,
            )

        last_clf = clf
        last_convergence_warnings = convergence_warnings
        last_effective_max_iter = effective_max_iter
        last_solver = solver_name
        if not convergence_warnings:
            break

    assert last_clf is not None
    beta = np.asarray(np.concatenate([last_clf.intercept_, last_clf.coef_[0]]), dtype=float)

    diagnostics: dict[str, float | bool | int | str] = {
        "converged": len(last_convergence_warnings) == 0,
        "used_pinv": False,
        "iterations_run": int(np.max(np.asarray(last_clf.n_iter_, dtype=int))),
        "max_abs_delta": 0.0 if len(last_convergence_warnings) == 0 else float("nan"),
        "effective_max_iter": last_effective_max_iter,
        "solver": last_solver,
    }
    return beta, diagnostics


def _default_firthlogist_python() -> Path:
    return build_context().root / ".conda-firthlogist" / "bin" / "python"


def _resolve_firthlogist_python() -> Path | None:
    env_value = os.getenv("PLASMID_PRIORITY_FIRTHLOGIST_PYTHON")
    candidate = Path(env_value).expanduser() if env_value else _default_firthlogist_python()
    if candidate.exists():
        return candidate
    return None


def _fit_firthlogist_sidecar_with_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
    python_path = _resolve_firthlogist_python()
    if python_path is None:
        warnings.warn(
            "firthlogist sidecar environment not found; falling back to internal Firth solver.",
            stacklevel=2,
        )
        return _fit_firth_logistic_regression_with_diagnostics(
            X,
            y,
            l2=0.0,
            max_iter=max_iter,
            sample_weight=sample_weight,
        )
    if sample_weight is not None:
        warnings.warn(
            (
                "firthlogist sidecar does not support sample_weight; "
                "fitting the external Firth model without weights."
            ),
            stacklevel=2,
        )
    project_root = build_context().root
    bridge_script = project_root / "src" / "plasmid_priority" / "modeling" / "firthlogist_bridge.py"
    with tempfile.TemporaryDirectory(prefix="firthlogist_bridge_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.npz"
        output_path = Path(tmp_dir) / "output.json"
        np.savez(
            input_path,
            X=np.asarray(X, dtype=float),
            y=np.asarray(y, dtype=int),
            max_iter=np.asarray([int(max_iter)], dtype=int),
        )
        env = os.environ.copy()
        completed = subprocess.run(
            [str(python_path), str(bridge_script), str(input_path), str(output_path)],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if completed.returncode != 0 or not output_path.exists():
            warnings.warn(
                ("firthlogist sidecar execution failed; falling back to internal Firth solver."),
                stacklevel=2,
            )
            return _fit_firth_logistic_regression_with_diagnostics(
                X,
                y,
                l2=0.0,
                max_iter=max_iter,
                sample_weight=sample_weight,
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    beta = np.asarray(payload.get("beta", []), dtype=float)
    diagnostics: dict[str, float | bool | int | str] = {
        "converged": bool(payload.get("converged", True)),
        "used_pinv": False,
        "iterations_run": int(payload.get("iterations_run", max_iter)),
        "max_abs_delta": 0.0,
        "effective_max_iter": int(max_iter),
        "solver": str(payload.get("solver", "firthlogist_sidecar")),
    }
    return beta, diagnostics


def _firth_penalized_log_likelihood(
    X_aug: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2: float = 0.0,
) -> float:
    probabilities = np.clip(_sigmoid(X_aug @ beta), 1e-8, 1.0 - 1e-8)
    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones(len(y), dtype=float)
    )
    log_likelihood = float(
        np.sum(
            weights
            * (
                (y.astype(float) * np.log(probabilities))
                + ((1.0 - y.astype(float)) * np.log(1.0 - probabilities))
            )
        )
    )
    curvature = np.clip(weights * probabilities * (1.0 - probabilities), 1e-8, None)
    info = X_aug.T @ (X_aug * curvature[:, None])
    if l2 > 0.0:
        penalty = np.eye(X_aug.shape[1], dtype=float) * float(l2)
        penalty[0, 0] = 0.0
        info = np.asarray(info + penalty, dtype=float)
        log_likelihood -= 0.5 * float(l2) * float(np.dot(beta[1:], beta[1:]))
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return float("-inf")
    return float(log_likelihood + (0.5 * logdet))


def _fit_firth_logistic_regression_with_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 0.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    X_aug = np.empty((len(X), X.shape[1] + 1), dtype=float)
    X_aug[:, 0] = 1.0
    X_aug[:, 1:] = X
    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones(len(y), dtype=float)
    )
    beta = np.zeros(X_aug.shape[1], dtype=float)
    warm_start_beta = np.zeros(X_aug.shape[1], dtype=float)
    warm_start_diagnostics: dict[str, float | bool | int | str] | None = None
    if len(np.unique(y)) >= 2:
        try:
            warm_start_beta, warm_start_diagnostics = _fit_logistic_regression_with_diagnostics(
                X,
                y,
                l2=max(float(l2), 1e-4),
                max_iter=min(max(int(max_iter), 200), 1000),
                sample_weight=sample_weight,
                fit_backend="logistic",
            )
            beta = np.asarray(warm_start_beta, dtype=float)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            import warnings

            warnings.warn(f"Warm-start logistic fit failed, using zero initialization: {e}")
            beta = np.zeros(X_aug.shape[1], dtype=float)
    ridge_penalty = np.eye(X_aug.shape[1], dtype=float) * float(max(l2, 0.0))
    ridge_penalty[0, 0] = 0.0
    tol = 1e-6
    max_steps = max(int(max_iter), 200)
    converged = False
    used_pinv = False
    max_abs_delta = float("nan")

    for iteration in range(1, max_steps + 1):
        probabilities = np.clip(_sigmoid(X_aug @ beta), 1e-8, 1.0 - 1e-8)
        curvature = np.clip(weights * probabilities * (1.0 - probabilities), 1e-8, None)
        weighted_X = X_aug * curvature[:, None]
        info = np.asarray(X_aug.T @ weighted_X + ridge_penalty, dtype=float)
        try:
            info_inverse = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            info_inverse = np.linalg.pinv(info)
            used_pinv = True
        sqrt_curvature = np.sqrt(curvature)
        weighted_design = X_aug * sqrt_curvature[:, None]
        leverage = np.einsum(
            "ij,jk,ik->i",
            weighted_design,
            info_inverse,
            weighted_design,
        )
        leverage = np.clip(np.asarray(leverage, dtype=float), 0.0, 1.0)
        adjusted_residual = (weights * (y - probabilities)) + (leverage * (0.5 - probabilities))
        score = np.asarray(X_aug.T @ adjusted_residual - (ridge_penalty @ beta), dtype=float)
        delta = np.asarray(info_inverse @ score, dtype=float)
        if not np.all(np.isfinite(delta)):
            break
        max_abs_delta = float(np.max(np.abs(delta)))
        current_objective = _firth_penalized_log_likelihood(
            X_aug,
            y,
            beta,
            sample_weight=weights,
            l2=l2,
        )
        step = 1.0
        accepted = False
        while step >= 1e-4:
            candidate_beta = np.asarray(beta + (step * delta), dtype=float)
            candidate_objective = _firth_penalized_log_likelihood(
                X_aug,
                y,
                candidate_beta,
                sample_weight=weights,
                l2=l2,
            )
            if candidate_objective >= current_objective:
                beta = candidate_beta
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        if max_abs_delta <= tol:
            converged = True
            break

    diagnostics: dict[str, float | bool | int | str] = {
        "converged": converged,
        "used_pinv": used_pinv,
        "iterations_run": int(iteration),
        "max_abs_delta": float(max_abs_delta),
        "effective_max_iter": int(max_steps),
        "solver": "firth_newton",
    }
    if (
        (not converged)
        or (not np.all(np.isfinite(beta)))
        or (not np.isfinite(max_abs_delta))
        or (abs(float(max_abs_delta)) > 1e4)
    ) and warm_start_diagnostics is not None:
        fallback = dict(warm_start_diagnostics)
        fallback["solver"] = "firth_warm_start_fallback"
        return np.asarray(warm_start_beta, dtype=float), fallback
    return np.asarray(beta, dtype=float), diagnostics


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
    fit_backend: str = "logistic",
) -> np.ndarray:
    beta, diagnostics = _fit_logistic_regression_with_diagnostics(
        X,
        y,
        l2=l2,
        max_iter=max_iter,
        sample_weight=sample_weight,
        fit_backend=fit_backend,
    )
    if not bool(diagnostics["converged"]):
        iterations_run = int(diagnostics["iterations_run"])
        effective_max_iter = int(diagnostics.get("effective_max_iter", max_iter))
        solver_name = str(diagnostics.get("solver", "lbfgs"))
        warnings.warn(
            f"Logistic regression did not converge after {iterations_run} iterations "
            f"(solver: {solver_name}; configured limit: {effective_max_iter}; "
            f"max coefficient change: {float(diagnostics['max_abs_delta']):.2e}; "
            f"pseudo-inverse fallback used: {bool(diagnostics['used_pinv'])})",
            stacklevel=2,
        )
    return beta


def _predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    linear = np.asarray(X @ beta[1:] + beta[0], dtype=float)
    return _sigmoid(linear)


def _predict_calibrated(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(_predict_logistic(X, beta), dtype=float)


def _fit_standardized_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
    fit_backend: str = "logistic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_scaled, mean, std = _standardize_fit(X)
    beta = _fit_logistic_regression(
        X_scaled,
        y,
        l2=l2,
        max_iter=max_iter,
        sample_weight=sample_weight,
        fit_backend=fit_backend,
    )
    return beta, mean, std


def _logistic_posterior_covariance(
    X: np.ndarray,
    beta: np.ndarray,
    *,
    l2: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Laplace covariance approximation around the fitted logistic coefficients."""
    probabilities = np.clip(_predict_logistic(X, beta), 1e-6, 1.0 - 1e-6)
    curvature = probabilities * (1.0 - probabilities)
    if sample_weight is not None:
        curvature = curvature * np.asarray(sample_weight, dtype=float)
    X_aug = np.empty((len(X), X.shape[1] + 1), dtype=float)
    X_aug[:, 0] = 1.0
    X_aug[:, 1:] = X
    weighted_X = X_aug * curvature[:, None]
    hessian = X_aug.T @ weighted_X
    penalty = np.eye(X_aug.shape[1], dtype=float) * float(l2)
    penalty[0, 0] = 0.0
    hessian = np.asarray(hessian + penalty, dtype=float)
    hessian.flat[:: hessian.shape[0] + 1] += 1e-8
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    return np.asarray(0.5 * (covariance + covariance.T), dtype=float)


def _bayesian_coefficient_summary(
    beta: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return {
        "standard_error": np.asarray(standard_error, dtype=float),
        "ci_lower": np.asarray(beta - (_BAYESIAN_CI_Z * standard_error), dtype=float),
        "ci_upper": np.asarray(beta + (_BAYESIAN_CI_Z * standard_error), dtype=float),
    }


def _bayesian_prediction_summary(
    X: np.ndarray,
    beta: np.ndarray,
    covariance: np.ndarray,
    *,
    n_draws: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariance = np.asarray(0.5 * (covariance + covariance.T), dtype=float)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    draws = beta + rng.normal(size=(max(int(n_draws), 1), len(beta))) @ transform.T
    linear = np.asarray(X @ draws[:, 1:].T + draws[:, 0], dtype=float)
    probabilities = _sigmoid(linear)
    return {
        "mean": np.asarray(np.mean(probabilities, axis=1), dtype=float),
        "std": np.asarray(np.std(probabilities, axis=1), dtype=float),
        "q05": np.asarray(np.quantile(probabilities, 0.05, axis=1), dtype=float),
        "q95": np.asarray(np.quantile(probabilities, 0.95, axis=1), dtype=float),
    }


def _stratified_folds(
    y: np.ndarray, *, n_splits: int, n_repeats: int, seed: int
) -> list[list[np.ndarray]]:
    """Build repeated stratified folds while respecting rare-class support."""
    y = np.asarray(y, dtype=int)
    n_repeats = max(int(n_repeats), 1)
    if y.size == 0:
        return [[] for _ in range(n_repeats)]
    _, class_counts = np.unique(y, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Repeated stratified folds require both outcome classes.")
    effective_splits = min(max(int(n_splits), 2), int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("Repeated stratified folds require at least two members in every class.")
    skf = RepeatedStratifiedKFold(n_splits=effective_splits, n_repeats=n_repeats, random_state=seed)
    folds_per_repeat: list[list[np.ndarray]] = [[] for _ in range(n_repeats)]

    all_splits = list(skf.split(np.zeros(len(y), dtype=int), y))
    for i, (_, test_idx) in enumerate(all_splits):
        repeat_idx = i // effective_splits
        folds_per_repeat[repeat_idx].append(test_idx)
    return folds_per_repeat


def _oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
    l2: float = 1.0,
    max_iter: int = 100,
) -> np.ndarray:
    preds = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=float)
    for fold_indices in _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed):
        for test_idx in fold_indices:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            X_train, X_test = X[train_mask], X[test_idx]
            y_train = y[train_mask]
            train_weight = sample_weight[train_mask] if sample_weight is not None else None
            X_train_scaled, mean, std = _standardize_fit(X_train)
            X_test_scaled = _standardize_apply(X_test, mean, std)
            beta = _fit_logistic_regression(
                X_train_scaled,
                y_train,
                l2=l2,
                max_iter=max_iter,
                sample_weight=train_weight,
            )
            preds[test_idx] += _predict_logistic(X_test_scaled, beta)
            counts[test_idx] += 1
    if counts.min() == 0:
        warnings.warn(
            f"{int((counts == 0).sum())} sample(s) never appeared in any test fold",
            stacklevel=2,
        )
        counts[counts == 0] = 1
    return preds / counts


def _model_fit_kwargs(
    model_name: str, overrides: Mapping[str, object] | None = None
) -> dict[str, object]:
    _ensure_config_loaded()
    defaults: dict[str, object] = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    defaults.update(MODEL_FIT_CONFIG.get(model_name, {}))
    if overrides:
        defaults.update(overrides)
    return defaults


def _resolve_parallel_jobs(requested_jobs: int | None, *, max_tasks: int, cap: int = 8) -> int:
    if max_tasks <= 1:
        return 1
    env_cap = os.getenv("PLASMID_PRIORITY_MAX_JOBS")
    if env_cap:
        try:
            cap = max(1, min(cap, int(env_cap)))
        except ValueError:
            pass
    if requested_jobs is None:
        requested = min(cap, os.cpu_count() or 1)
    else:
        requested = int(requested_jobs)
    return max(1, min(requested, max_tasks, cap))
