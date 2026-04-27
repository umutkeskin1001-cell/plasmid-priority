"""Primary retrospective evaluation for the backbone priority score."""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: F401
from typing import Any, Mapping, cast

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from plasmid_priority.modeling.feature_surface import (
    assert_feature_columns_present as _assert_feature_columns_present_impl,
)
from plasmid_priority.modeling.feature_surface import (
    ensure_feature_columns as _ensure_feature_columns_impl,
)
from plasmid_priority.modeling.module_a_support import (
    ABLATION_MODEL_NAMES,
    CONSERVATIVE_MODEL_NAME,
    CORE_MODEL_NAMES,
    FEATURE_PROVENANCE_REGISTRY,
    GOVERNANCE_MODEL_FALLBACK,
    GOVERNANCE_MODEL_NAME,
    MODULE_A_FEATURE_SETS,
    MODULE_A_MODEL_TRACKS,
    PRIMARY_MODEL_FALLBACK,
    PRIMARY_MODEL_NAME,
    RESEARCH_MODEL_NAMES,
    ModelResult,
    _bayesian_coefficient_summary,
    _bayesian_prediction_summary,
    _ensure_config_loaded,
    _fit_feature_imputer,
    _fit_kwarg_float,
    _fit_kwarg_int,
    _fit_kwarg_mode,
    _fit_kwarg_str,
    _fit_kwarg_value,
    _fit_logistic_regression,
    _fit_logistic_regression_with_diagnostics,
    _fit_standardized_model,
    _logistic_posterior_covariance,
    _model_fit_kwargs,
    _predict_calibrated,
    _predict_logistic,
    _resolve_parallel_jobs,
    _standardize_apply,
    _standardize_fit,
    _stratified_folds,
    _top_k_precision_recall,
    build_failed_model_result,
    build_model_folds,
    build_single_model_candidate_family,
)
from plasmid_priority.modeling.module_a_support import (
    NOVELTY_SPECIALIST_FEATURES as _NOVELTY_SPECIALIST_FEATURES,
)
from plasmid_priority.modeling.module_a_support import (
    NOVELTY_SPECIALIST_FIT_CONFIG as _NOVELTY_SPECIALIST_FIT_CONFIG,
)
from plasmid_priority.modeling.single_model_pareto import add_weighted_objective
from plasmid_priority.utils.parallel import limit_native_threads
from plasmid_priority.validation import (
    average_precision,
    average_precision_enrichment,
    average_precision_lift,
    bootstrap_intervals,
    brier_decomposition,
    brier_score,
    decision_utility_summary,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    ndcg_at_k,
    novelty_adjusted_average_precision,
    positive_prevalence,
    roc_auc_score,
    weighted_classification_cost,
)

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:  # pragma: no cover - optional research dependency
    ExplainableBoostingClassifier = None

_lgb: Any
try:
    import lightgbm as _lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - optional research dependency
    _lgb = None
    _HAS_LIGHTGBM = False

NOVELTY_SPECIALIST_FEATURES = _NOVELTY_SPECIALIST_FEATURES
NOVELTY_SPECIALIST_FIT_CONFIG = _NOVELTY_SPECIALIST_FIT_CONFIG


def _fit_backend_name(fit_kwargs: dict[str, object] | None = None) -> str:
    return _fit_kwarg_str(fit_kwargs, "fit_backend", "logistic").strip().lower()


def _model_type_name(fit_kwargs: dict[str, object] | None = None) -> str:
    return _fit_kwarg_str(fit_kwargs, "model_type", "logistic").strip().lower()


def _fit_kwarg_optional_int(
    fit_kwargs: dict[str, object] | None,
    key: str,
    default: int | None = None,
) -> int | None:
    raw = _fit_kwarg_value(fit_kwargs, key, default)
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if not normalized or normalized == "none":
            return None
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(raw, (int, float)):
        return int(raw)
    return default


def get_module_a_model_names(
    *,
    include_research: bool = False,
    include_ablations: bool = False,
) -> tuple[str, ...]:
    _ensure_config_loaded()
    names = list(CORE_MODEL_NAMES)
    if include_research:
        names.extend(RESEARCH_MODEL_NAMES)
    if include_ablations:
        names.extend(ABLATION_MODEL_NAMES)
    return tuple(dict.fromkeys(names))


def _masked_percentile_rank(values: pd.Series, cohort_mask: pd.Series | None = None) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if cohort_mask is None:
        cohort_mask = pd.Series(True, index=values.index, dtype=bool)
    effective_mask = cohort_mask.fillna(False).astype(bool) & values.notna()
    if effective_mask.any():
        result.loc[effective_mask] = values.loc[effective_mask].rank(method="average", pct=True)
    return result


def _knownness_score_series(
    frame: pd.DataFrame, *, cohort_mask: pd.Series | None = None
) -> pd.Series:
    member_rank = _masked_percentile_rank(
        pd.Series(frame.get("log1p_member_count_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    country_rank = _masked_percentile_rank(
        pd.Series(frame.get("log1p_n_countries_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    source_rank = _masked_percentile_rank(
        pd.Series(frame.get("refseq_share_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    return (member_rank + country_rank + source_rank) / 3.0


def _stable_quantile_labels(
    values: pd.Series,
    *,
    q: int,
    label_prefix: str = "q",
) -> tuple[pd.Series, int]:
    labels = pd.Series(np.nan, index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    valid_positions = np.flatnonzero(numeric.notna().to_numpy())
    valid = numeric.loc[numeric.notna()]
    if valid.empty or valid.nunique() < 2:
        return labels, 0
    ranked = valid.rank(method="average")
    try:
        codes, bins = pd.qcut(
            ranked,
            q=min(int(q), int(valid.nunique())),
            labels=False,
            retbins=True,
            duplicates="drop",
        )
    except ValueError:
        return labels, 0
    n_bins = max(int(len(bins) - 1), 0)
    if n_bins <= 0:
        return labels, 0
    mapped = pd.Series(codes, index=valid.index, dtype="Int64").map(
        {idx: f"{label_prefix}{idx + 1}" for idx in range(n_bins)}
    )
    mapped_values = mapped.astype(str).to_numpy(dtype=object)
    if len(valid_positions) != len(mapped_values):
        return labels, n_bins
    label_values = labels.to_numpy(dtype=object, copy=True)
    label_values[valid_positions] = mapped_values
    return pd.Series(label_values, index=values.index, dtype=object), n_bins


def _compute_sample_weight(
    eligible: pd.DataFrame,
    *,
    mode: str | None,
    fit_kwargs: dict[str, object] | None = None,
) -> np.ndarray | None:
    if mode in (None, "", "none"):
        return None
    tokens = [token.strip() for token in str(mode).replace(",", "+").split("+") if token.strip()]
    if not tokens:
        return None
    weights = np.ones(len(eligible), dtype=float)
    for token in tokens:
        if token == "source_balanced":
            dominant_source = np.where(
                eligible["refseq_share_train"].fillna(0.0).to_numpy(dtype=float) >= 0.5,
                "refseq_leaning",
                "insd_leaning",
            )
            counts = pd.Series(dominant_source).value_counts()
            if not counts.empty:
                inverse = (1.0 / counts.clip(lower=1.0)).to_dict()
                weights *= (
                    pd.Series(dominant_source, index=eligible.index)
                    .map(inverse)
                    .fillna(1.0)
                    .to_numpy(dtype=float)
                )
            continue
        if token == "class_balanced":
            labels = eligible["spread_label"].fillna(0).astype(int)
            counts = labels.value_counts()
            if not counts.empty and len(counts) >= 2:
                inverse = (1.0 / counts.clip(lower=1.0)).to_dict()
                weights *= labels.map(inverse).fillna(1.0).to_numpy(dtype=float)
            continue
        if token == "knownness_balanced":
            knownness = _knownness_score_series(eligible)
            quantile_labels, n_bins = _stable_quantile_labels(knownness, q=4)
            if n_bins >= 2:
                counts = quantile_labels.value_counts()
                if not counts.empty:
                    inverse = (1.0 / counts.clip(lower=1.0)).to_dict()
                    weights *= quantile_labels.map(inverse).fillna(1.0).to_numpy(dtype=float)
            continue
        if token == "pu_negative_downweight":
            labels = eligible["spread_label"].fillna(0).astype(int)
            knownness = _knownness_score_series(eligible).fillna(0.0).clip(lower=0.0, upper=1.0)
            min_weight = _fit_kwarg_float(fit_kwargs, "pu_negative_min_weight", 0.25)
            min_weight = float(np.clip(min_weight, 0.0, 1.0))
            power = max(_fit_kwarg_float(fit_kwargs, "pu_negative_power", 1.0), 0.1)
            negative_weight = min_weight + ((1.0 - min_weight) * np.power(knownness, power))
            pu_weight = np.where(
                labels.to_numpy(dtype=int) == 1,
                1.0,
                np.asarray(negative_weight, dtype=float),
            )
            weights *= np.asarray(pu_weight, dtype=float)
            continue
        if token in {"inverse_knownness_weighting", "ipw_balanced"}:
            knownness = _knownness_score_series(eligible).fillna(0.0).clip(0.0, 1.0)
            knownness_values = knownness.to_numpy(dtype=float)
            # Knownness-inverse heuristic: low-knownness backbones get higher weight.
            # Note: this is not propensity-score IPW.
            ipw_weights = 1.0 / (knownness_values + 0.1)
            ipw_weights = np.clip(ipw_weights, 0.2, 5.0)
            weights *= ipw_weights
            continue
        raise ValueError(f"Unsupported sample_weight_mode: {mode}")
    return np.asarray(weights / max(weights.mean(), 1e-6), dtype=float)


def _build_pairwise_rank_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    fit_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    labels = np.asarray(y, dtype=int)
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    if positive_idx.size == 0 or negative_idx.size == 0:
        raise ValueError("Pairwise ranking requires both positive and negative samples.")

    max_pairs = max(_fit_kwarg_int(fit_kwargs, "pairwise_max_pairs", 6000), 1)
    total_pairs = int(positive_idx.size * negative_idx.size)
    rng = np.random.default_rng(_fit_kwarg_int(fit_kwargs, "pairwise_random_state", 42))
    if total_pairs <= max_pairs:
        high_idx = np.repeat(positive_idx, negative_idx.size)
        low_idx = np.tile(negative_idx, positive_idx.size)
    else:
        high_idx = rng.choice(positive_idx, size=max_pairs, replace=True)
        low_idx = rng.choice(negative_idx, size=max_pairs, replace=True)

    forward = np.asarray(X[high_idx] - X[low_idx], dtype=float)
    reverse = np.asarray(X[low_idx] - X[high_idx], dtype=float)
    pair_matrix = np.vstack([forward, reverse])
    pair_labels = np.concatenate(
        [
            np.ones(len(forward), dtype=int),
            np.zeros(len(reverse), dtype=int),
        ]
    )
    if sample_weight is None:
        return pair_matrix, pair_labels, None
    base_weight = 0.5 * (
        np.asarray(sample_weight[high_idx], dtype=float)
        + np.asarray(sample_weight[low_idx], dtype=float)
    )
    pair_weight = np.concatenate([base_weight, base_weight])
    return pair_matrix, pair_labels, np.asarray(pair_weight, dtype=float)


def _fit_pairwise_rank_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fit_kwargs: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_scaled, mean, std = _standardize_fit(X_train)
    pair_X, pair_y, pair_weight = _build_pairwise_rank_dataset(
        X_train_scaled,
        y_train,
        sample_weight=sample_weight,
        fit_kwargs=fit_kwargs,
    )
    beta = _fit_logistic_regression(
        pair_X,
        pair_y,
        l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
        max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
        sample_weight=pair_weight,
        fit_backend=_fit_backend_name(fit_kwargs),
    )
    return beta, mean, std, X_train_scaled


def _knownness_design_matrix(frame: pd.DataFrame) -> np.ndarray:
    member = (
        frame.get("log1p_member_count_train", pd.Series(0.0, index=frame.index))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    country = (
        frame.get("log1p_n_countries_train", pd.Series(0.0, index=frame.index))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    source = (
        frame.get("refseq_share_train", pd.Series(0.0, index=frame.index))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            member,
            country,
            source,
            member * member,
            country * country,
            source * source,
            member * country,
            member * source,
            country * source,
        ]
    )


def _fit_knownness_residualizer(
    train: pd.DataFrame,
    columns: list[str],
    *,
    alpha: float | np.ndarray = 1.0,
    prepared: bool = False,
) -> np.ndarray:
    def _solve_or_warn(lhs: np.ndarray, rhs: np.ndarray, *, label: str) -> np.ndarray:
        try:
            return np.asarray(np.linalg.solve(lhs, rhs), dtype=float)
        except np.linalg.LinAlgError:
            warnings.warn(
                (
                    "Knownness residualizer encountered a singular system "
                    f"({label}); falling back to pseudo-inverse."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return np.asarray(np.linalg.pinv(lhs) @ rhs, dtype=float)

    Z = _knownness_design_matrix(train)
    working = train if prepared else _ensure_feature_columns(train, columns)
    X = working[columns].fillna(0.0).to_numpy(dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 0 or alpha_array.size == 1:
        penalty = np.eye(Z.shape[1], dtype=float) * float(alpha_array.reshape(-1)[0])
        penalty[0, 0] = 0.0
        lhs = Z.T @ Z + penalty
        rhs = Z.T @ X
        return _solve_or_warn(lhs, rhs, label="global-alpha solve")

    if alpha_array.size != len(columns):
        raise ValueError("Grouped preprocess alpha must match the number of feature columns.")
    coefficients = np.zeros((Z.shape[1], X.shape[1]), dtype=float)
    ztz = Z.T @ Z
    base_penalty = np.eye(Z.shape[1], dtype=float)
    base_penalty[0, 0] = 0.0
    for idx, alpha_value in enumerate(alpha_array):
        lhs = ztz + (base_penalty * float(alpha_value))
        rhs = Z.T @ X[:, idx]
        coefficients[:, idx] = _solve_or_warn(lhs, rhs, label=f"grouped-alpha solve idx={idx}")
    return coefficients


def _feature_alpha_group(feature_name: str) -> str:
    name = str(feature_name)
    if "H_" in name or "host_" in name:
        return "H"
    if name.startswith("A_") or "amr_" in name or "resistance" in name:
        return "A"
    if name.startswith("T_") or "orit" in name or "transfer" in name:
        return "T"
    # Governance-track features: PlasmidFinder complexity, AMR class richness,
    # pMLST coherence, host-support composites
    if name in (
        "plasmidfinder_complexity_norm",
        "amr_class_richness_norm",
        "pmlst_coherence_norm",
        "H_support_norm",
    ):
        return "G"
    return "other"


def _resolve_knownness_grouped_alpha(
    columns: list[str],
    *,
    base_alpha: float,
    fit_kwargs: dict[str, object] | None = None,
) -> float | np.ndarray:
    raw_grouped = _fit_kwarg_value(fit_kwargs, "preprocess_alpha_grouped", False)
    if isinstance(raw_grouped, str):
        grouped = raw_grouped.strip().lower() in {"1", "true", "yes", "on"}
    else:
        grouped = bool(raw_grouped)
    has_explicit_group_alpha = any(
        key in (fit_kwargs or {})
        for key in (
            "preprocess_alpha_T",
            "preprocess_alpha_H",
            "preprocess_alpha_A",
            "preprocess_alpha_G",
        )
    )
    if not grouped and not has_explicit_group_alpha:
        return float(base_alpha)
    group_values = {
        "T": _fit_kwarg_float(fit_kwargs, "preprocess_alpha_T", 0.1),
        "H": _fit_kwarg_float(fit_kwargs, "preprocess_alpha_H", 2.0),
        "A": _fit_kwarg_float(fit_kwargs, "preprocess_alpha_A", 1.0),
        "G": _fit_kwarg_float(fit_kwargs, "preprocess_alpha_G", 0.5),
        "other": float(base_alpha),
    }
    return np.asarray(
        [group_values[_feature_alpha_group(feature_name)] for feature_name in columns],
        dtype=float,
    )


def _resolve_knownness_residualizer_alpha(
    train: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
    prepared: bool = False,
) -> float | np.ndarray:
    base_alpha = _select_knownness_residualizer_alpha(
        train,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=prepared,
    )
    return _resolve_knownness_grouped_alpha(columns, base_alpha=base_alpha, fit_kwargs=fit_kwargs)


def _preprocess_alpha_grid(fit_kwargs: dict[str, object] | None) -> tuple[float, ...]:
    raw = _fit_kwarg_value(
        fit_kwargs,
        "preprocess_alpha_grid",
        (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    if isinstance(raw, str):
        candidates: list[object] = [token.strip() for token in raw.split(",") if token.strip()]
    elif isinstance(raw, (list, tuple)):
        candidates = list(raw)
    else:
        candidates = [raw]
    resolved: list[float] = []
    for value in candidates:
        if isinstance(value, (int, float, str)):
            try:
                alpha = float(value)
            except (TypeError, ValueError):
                continue
            if alpha > 0.0:
                resolved.append(alpha)
    if not resolved:
        return (1.0,)
    return tuple(dict.fromkeys(resolved))


def _select_knownness_residualizer_alpha(
    train: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
    prepared: bool = False,
) -> float:
    raw_alpha = _fit_kwarg_value(fit_kwargs, "preprocess_alpha", 1.0)
    if str(raw_alpha).strip().lower() != "auto":
        return _fit_kwarg_float(fit_kwargs, "preprocess_alpha", 1.0)

    working = train if prepared else _ensure_feature_columns(train, columns)
    if "spread_label" not in working.columns:
        return 1.0
    y = working["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return 1.0
    try:
        fold_groups = _stratified_folds(y, n_splits=3, n_repeats=1, seed=17)
    except ValueError:
        return 1.0

    alpha_grid = _preprocess_alpha_grid(fit_kwargs)
    best_alpha = 1.0 if 1.0 in alpha_grid else alpha_grid[0]
    best_score = float("-inf")
    l2_value = _fit_kwarg_float(fit_kwargs, "l2", 1.0)
    max_iter = _fit_kwarg_int(fit_kwargs, "max_iter", 100)
    for alpha in alpha_grid:
        preds = np.zeros(len(working), dtype=float)
        counts = np.zeros(len(working), dtype=float)
        for _, test_idx in fold_groups:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            inner_train = working.loc[train_mask]
            inner_valid = working.iloc[test_idx]
            coefficients = _fit_knownness_residualizer(
                inner_train,
                columns,
                alpha=alpha,
                prepared=True,
            )
            X_train = _apply_knownness_residualizer(
                inner_train,
                columns,
                coefficients,
                prepared=True,
            )
            X_valid = _apply_knownness_residualizer(
                inner_valid,
                columns,
                coefficients,
                prepared=True,
            )
            train_weight = _compute_sample_weight(
                inner_train,
                mode=_fit_kwarg_mode(fit_kwargs),
                fit_kwargs=fit_kwargs,
            )
            X_train_scaled, mean, std = _standardize_fit(X_train)
            X_valid_scaled = _standardize_apply(X_valid, mean, std)
            beta = _fit_logistic_regression(
                X_train_scaled,
                y[train_mask],
                l2=l2_value,
                max_iter=max_iter,
                sample_weight=train_weight,
            )
            preds[test_idx] += _predict_logistic(X_valid_scaled, beta)
            counts[test_idx] += 1.0
        valid_mask = counts > 0
        if not np.any(valid_mask):
            continue
        score = roc_auc_score(y[valid_mask], preds[valid_mask] / counts[valid_mask])
        if np.isnan(score):
            continue
        if score > best_score or (score == best_score and alpha < best_alpha):
            best_alpha = alpha
            best_score = score
    return float(best_alpha)


def _apply_knownness_residualizer(
    frame: pd.DataFrame,
    columns: list[str],
    coefficients: np.ndarray,
    *,
    prepared: bool = False,
) -> np.ndarray:
    working = frame if prepared else _ensure_feature_columns(frame, columns)
    X = working[columns].fillna(0.0).to_numpy(dtype=float)
    Z = _knownness_design_matrix(working)
    return np.asarray(X - (Z @ coefficients), dtype=float)


def _prepare_feature_matrices(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
    prepared: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fit_kwargs = fit_kwargs or {}
    train_working = train if prepared else _ensure_feature_columns(train, columns)
    score_working = score if prepared else _ensure_feature_columns(score, columns)
    preprocess_mode = _fit_kwarg_str(fit_kwargs, "preprocess_mode", "none").strip().lower()
    if preprocess_mode in ("none", ""):
        train_raw = train_working[columns].to_numpy(dtype=float)
        score_raw = score_working[columns].to_numpy(dtype=float)
        train_matrix, imputer = _fit_feature_imputer(train_raw)
        score_matrix = np.nan_to_num(imputer.transform(score_raw), nan=0.0)
        return train_matrix, score_matrix
    if preprocess_mode == "knownness_residualized":
        resolved_alpha = _resolve_knownness_residualizer_alpha(
            train_working,
            columns,
            fit_kwargs=fit_kwargs,
            prepared=True,
        )
        coefficients = _fit_knownness_residualizer(
            train_working,
            columns,
            alpha=resolved_alpha,
            prepared=True,
        )
        return (
            _apply_knownness_residualizer(train_working, columns, coefficients, prepared=True),
            _apply_knownness_residualizer(score_working, columns, coefficients, prepared=True),
        )
    raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")


def _fit_hist_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_kwargs: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingClassifier:
    fit_kwargs = fit_kwargs or {}
    max_depth = _fit_kwarg_optional_int(fit_kwargs, "nonlinear_max_depth", 3)
    clf = HistGradientBoostingClassifier(
        learning_rate=_fit_kwarg_float(fit_kwargs, "nonlinear_learning_rate", 0.05),
        max_iter=_fit_kwarg_int(fit_kwargs, "nonlinear_max_iter", 200),
        max_depth=max_depth,
        min_samples_leaf=_fit_kwarg_int(fit_kwargs, "nonlinear_min_samples_leaf", 5),
        l2_regularization=_fit_kwarg_float(fit_kwargs, "nonlinear_l2", 0.0),
        random_state=_fit_kwarg_int(fit_kwargs, "nonlinear_random_state", 42),
    )
    clf.fit(X, y, sample_weight=sample_weight)
    # LightGBM may set feature_names_in_ even for array inputs, which triggers
    # noisy sklearn warnings when later predictions use ndarray inputs.
    if hasattr(clf, "feature_names_in_"):
        delattr(clf, "feature_names_in_")
    return clf


def _predict_hist_gradient_boosting(
    model: HistGradientBoostingClassifier,
    X: np.ndarray,
) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _resolve_nonlinear_backend_name(fit_kwargs: dict[str, object] | None = None) -> str:
    requested = _fit_kwarg_str(fit_kwargs, "nonlinear_backend", "").strip().lower()
    if requested:
        if requested == "ebm" and ExplainableBoostingClassifier is None:
            warnings.warn(
                (
                    "nonlinear_backend=ebm requested but interpret is unavailable; "
                    "falling back to hist_gbm."
                ),
                stacklevel=2,
            )
            return "hist_gbm"
        return requested
    return "ebm" if ExplainableBoostingClassifier is not None else "hist_gbm"


def _fit_ebm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_kwargs: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> ExplainableBoostingClassifier:
    if ExplainableBoostingClassifier is None:  # pragma: no cover - guarded by resolver
        raise ImportError("interpret is required for nonlinear_backend=ebm")
    fit_kwargs = fit_kwargs or {}
    n_rows = max(int(len(X)), 1)
    interactions = min(10, max(X.shape[1] - 1, 0))
    model = ExplainableBoostingClassifier(
        interactions=_fit_kwarg_int(fit_kwargs, "nonlinear_interactions", interactions),
        outer_bags=_fit_kwarg_int(fit_kwargs, "nonlinear_outer_bags", 8),
        n_jobs=_fit_kwarg_optional_int(fit_kwargs, "nonlinear_n_jobs", 1),
        learning_rate=_fit_kwarg_float(fit_kwargs, "nonlinear_learning_rate", 0.01),
        max_bins=_fit_kwarg_int(fit_kwargs, "nonlinear_max_bins", 64),
        min_samples_leaf=_fit_kwarg_int(
            fit_kwargs,
            "nonlinear_min_samples_leaf",
            max(3, int(np.ceil(0.02 * n_rows))),
        ),
        max_rounds=_fit_kwarg_int(fit_kwargs, "nonlinear_max_rounds", 500),
        early_stopping_rounds=_fit_kwarg_int(fit_kwargs, "nonlinear_early_stopping_rounds", 50),
        random_state=_fit_kwarg_int(fit_kwargs, "nonlinear_random_state", 42),
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _fit_nonlinear_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_kwargs: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[object, str, str]:
    requested = _fit_kwarg_str(fit_kwargs, "nonlinear_backend", "").strip().lower()
    backend = _resolve_nonlinear_backend_name(fit_kwargs)
    if backend == "ebm":
        try:
            return (
                _fit_ebm_classifier(X, y, fit_kwargs=fit_kwargs, sample_weight=sample_weight),
                "ebm",
                "resolved_ebm",
            )
        except (OSError, PermissionError) as exc:
            warnings.warn(
                f"EBM nonlinear fit failed in this environment ({type(exc).__name__}: {exc}); "
                "falling back to hist_gbm for the nonlinear branch.",
                stacklevel=2,
            )
            return (
                _fit_hist_gradient_boosting(
                    X,
                    y,
                    fit_kwargs=fit_kwargs,
                    sample_weight=sample_weight,
                ),
                "hist_gbm",
                "fit_fallback_hist_gbm",
            )
    status = "resolved_hist_gbm"
    if requested == "ebm" and ExplainableBoostingClassifier is None:
        status = "interpret_missing_hist_gbm"
    elif requested and requested != backend:
        status = f"requested_{requested}_resolved_{backend}"
    return (
        _fit_hist_gradient_boosting(X, y, fit_kwargs=fit_kwargs, sample_weight=sample_weight),
        backend,
        status,
    )


def _predict_nonlinear_model(model: object, X: np.ndarray) -> np.ndarray:
    return np.asarray(cast(Any, model).predict_proba(X)[:, 1], dtype=float)


def _fit_lightgbm_classifier(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    *,
    fit_kwargs: dict[str, object] | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit a LightGBM classifier with configurable hyperparameters from fit_kwargs."""
    if not _HAS_LIGHTGBM:
        raise ImportError("lightgbm is required for model_type='lightgbm'")
    import lightgbm as lgb

    fit_kwargs = fit_kwargs or {}
    clf = lgb.LGBMClassifier(
        n_estimators=_fit_kwarg_int(fit_kwargs, "n_estimators", 500),
        max_depth=_fit_kwarg_int(fit_kwargs, "max_depth", 5),
        learning_rate=_fit_kwarg_float(fit_kwargs, "learning_rate", 0.05),
        num_leaves=_fit_kwarg_int(fit_kwargs, "num_leaves", 31),
        min_child_samples=_fit_kwarg_int(fit_kwargs, "min_child_samples", 20),
        subsample=_fit_kwarg_float(fit_kwargs, "subsample", 0.8),
        colsample_bytree=_fit_kwarg_float(fit_kwargs, "colsample_bytree", 0.8),
        reg_lambda=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
        random_state=_fit_kwarg_int(fit_kwargs, "random_state", 42),
        verbose=-1,
        n_jobs=1,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    feature_names = list(getattr(clf, "feature_name_", []))
    original_predict_proba = clf.predict_proba
    original_predict = clf.predict

    def _predict_proba_with_feature_names(data: object) -> Any:
        if isinstance(data, np.ndarray) and feature_names and data.ndim == 2:
            data = pd.DataFrame(data, columns=feature_names)
        return cast(Any, original_predict_proba)(data)

    def _predict_with_feature_names(data: object) -> Any:
        if isinstance(data, np.ndarray) and feature_names and data.ndim == 2:
            data = pd.DataFrame(data, columns=feature_names)
        return cast(Any, original_predict)(data)

    setattr(clf, "predict_proba", _predict_proba_with_feature_names)
    setattr(clf, "predict", _predict_with_feature_names)
    return clf


def _fit_lightgbm_isotonic_calibrator(
    train: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
) -> IsotonicRegression | None:
    """Inner-OOF isotonic calibration for LightGBM (no standardization)."""
    fit_kwargs = fit_kwargs or {}
    y = train["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    try:
        fold_groups = _stratified_folds(
            y,
            n_splits=_fit_kwarg_int(fit_kwargs, "stack_inner_splits", 3),
            n_repeats=1,
            seed=_fit_kwarg_int(fit_kwargs, "stack_seed", 43),
        )
    except ValueError:
        return None
    preds = np.zeros(len(train), dtype=float)
    counts = np.zeros(len(train), dtype=float)
    for _, test_idx in fold_groups:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        inner_train = train.loc[train_mask]
        inner_valid = train.iloc[test_idx]
        # NO standardization for LightGBM — tree-based models are scale-invariant
        X_inner_train, X_inner_valid = _prepare_feature_matrices(
            inner_train, inner_valid, columns, fit_kwargs=fit_kwargs, prepared=True
        )
        X_inner_train_frame = pd.DataFrame(X_inner_train, columns=columns)
        X_inner_valid_frame = pd.DataFrame(X_inner_valid, columns=columns)
        inner_weight = _compute_sample_weight(
            inner_train, mode=_fit_kwarg_mode(fit_kwargs), fit_kwargs=fit_kwargs
        )
        model = _fit_lightgbm_classifier(
            X_inner_train_frame,
            y[train_mask],
            fit_kwargs=fit_kwargs,
            sample_weight=inner_weight,
        )
        preds[test_idx] += model.predict_proba(X_inner_valid_frame)[:, 1]
        counts[test_idx] += 1.0
    if np.any(counts == 0):
        return None
    blend = preds / counts
    sample_weight = _compute_sample_weight(
        train, mode=_fit_kwarg_mode(fit_kwargs), fit_kwargs=fit_kwargs
    )
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(blend, y, sample_weight=sample_weight)
    return calibrator


def _oof_lightgbm_predictions_from_eligible(
    eligible: pd.DataFrame,
    *,
    columns: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    fit_kwargs: dict[str, object] | None = None,
    y_override: np.ndarray | None = None,
    folds_per_repeat: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure LightGBM OOF pipeline: no standardization, optional inner-OOF isotonic calibration."""
    if not _HAS_LIGHTGBM:
        raise ImportError("lightgbm is required for model_type='lightgbm'")
    fit_kwargs = fit_kwargs or {}
    y = (
        np.asarray(y_override, dtype=int)
        if y_override is not None
        else eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    )
    preds = np.zeros(len(eligible), dtype=float)
    counts = np.zeros(len(eligible), dtype=float)
    fold_groups = (
        folds_per_repeat
        if folds_per_repeat is not None
        else _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    )
    for _, test_idx in fold_groups:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train = eligible.loc[train_mask]
        test = eligible.iloc[test_idx]
        # Knownness residualization via _prepare_feature_matrices, NO standard scaling
        X_train, X_test = _prepare_feature_matrices(
            train, test, columns, fit_kwargs=fit_kwargs, prepared=True
        )
        X_train_frame = pd.DataFrame(X_train, columns=columns)
        X_test_frame = pd.DataFrame(X_test, columns=columns)
        train_weight = _compute_sample_weight(
            train, mode=_fit_kwarg_mode(fit_kwargs), fit_kwargs=fit_kwargs
        )
        model = _fit_lightgbm_classifier(
            X_train_frame,
            y[train_mask],
            fit_kwargs=fit_kwargs,
            sample_weight=train_weight,
        )
        raw_preds = model.predict_proba(X_test_frame)[:, 1]
        # Isotonic calibration when requested
        if _fit_kwarg_str(fit_kwargs, "calibration", "").strip().lower() == "isotonic":
            calibrator = _fit_lightgbm_isotonic_calibrator(train, columns, fit_kwargs=fit_kwargs)
            if calibrator is not None:
                raw_preds = calibrator.predict(raw_preds)
        preds[test_idx] += raw_preds
        counts[test_idx] += 1.0
    if counts.min() == 0:
        warnings.warn(
            f"{int((counts == 0).sum())} sample(s) never appeared in any test fold",
            stacklevel=2,
        )
        counts[counts == 0] = 1.0
    return preds / counts, y


def _safe_evaluate_model_name_task(
    scored: pd.DataFrame,
    model_name: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    fold_groups: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[str, ModelResult]:
    try:
        return (
            model_name,
            evaluate_model_name(
                scored,
                model_name=model_name,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                fold_groups=fold_groups,
            ),
        )
    except (ValueError, RuntimeError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            (
                f"Module A model '{model_name}' failed; continuing with remaining models. "
                f"{error_message}"
            ),
            stacklevel=2,
        )
        return model_name, build_failed_model_result(model_name, error_message)


def _fit_hybrid_isotonic_calibrator(
    train: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
) -> IsotonicRegression | None:
    fit_kwargs = fit_kwargs or {}
    y = train["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    try:
        fold_groups = _stratified_folds(
            y,
            n_splits=_fit_kwarg_int(fit_kwargs, "stack_inner_splits", 3),
            n_repeats=1,
            seed=_fit_kwarg_int(fit_kwargs, "stack_seed", 43),
        )
    except ValueError:
        return None
    preds = np.zeros(len(train), dtype=float)
    counts = np.zeros(len(train), dtype=float)
    for _, test_idx in fold_groups:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        inner_train = train.loc[train_mask]
        inner_valid = train.iloc[test_idx]
        X_inner_train, X_inner_valid = _prepare_feature_matrices(
            inner_train,
            inner_valid,
            columns,
            fit_kwargs=fit_kwargs,
            prepared=True,
        )
        inner_weight = _compute_sample_weight(
            inner_train,
            mode=_fit_kwarg_mode(fit_kwargs),
            fit_kwargs=fit_kwargs,
        )
        beta, mean, std = _fit_standardized_model(
            X_inner_train,
            y[train_mask],
            l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
            sample_weight=inner_weight,
            fit_backend=_fit_backend_name(fit_kwargs),
        )
        logistic_valid = _predict_calibrated(_standardize_apply(X_inner_valid, mean, std), beta)
        nonlinear_model, _, _ = _fit_nonlinear_model(
            X_inner_train,
            y[train_mask],
            fit_kwargs=fit_kwargs,
            sample_weight=inner_weight,
        )
        nonlinear_valid = _predict_nonlinear_model(nonlinear_model, X_inner_valid)
        preds[test_idx] += 0.5 * (logistic_valid + nonlinear_valid)
        counts[test_idx] += 1.0
    if np.any(counts == 0):
        return None
    blend = preds / counts
    sample_weight = _compute_sample_weight(
        train,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(blend, y, sample_weight=sample_weight)
    return calibrator


def _fit_predict_hybrid_stacked(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
    l2: float = 1.0,
    max_iter: int = 100,
) -> pd.DataFrame:
    fit_kwargs = fit_kwargs or {}
    y_train = train["spread_label"].astype(int).to_numpy(dtype=int)
    X_train, X_score = _prepare_feature_matrices(
        train,
        score,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=True,
    )
    sample_weight = _compute_sample_weight(
        train,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    beta, mean, std = _fit_standardized_model(
        X_train,
        y_train,
        l2=_fit_kwarg_float(fit_kwargs, "l2", l2),
        max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", max_iter),
        sample_weight=sample_weight,
        fit_backend=_fit_backend_name(fit_kwargs),
    )
    X_score_scaled = _standardize_apply(X_score, mean, std)
    logistic_prediction = _predict_calibrated(X_score_scaled, beta)
    nonlinear_model, resolved_backend, backend_status = _fit_nonlinear_model(
        X_train,
        y_train,
        fit_kwargs=fit_kwargs,
        sample_weight=sample_weight,
    )
    nonlinear_prediction = _predict_nonlinear_model(nonlinear_model, X_score)
    blend_prediction = 0.5 * (logistic_prediction + nonlinear_prediction)
    calibrator = _fit_hybrid_isotonic_calibrator(train, columns, fit_kwargs=fit_kwargs)
    final_prediction = (
        np.asarray(calibrator.predict(blend_prediction), dtype=float)
        if calibrator is not None
        else np.asarray(blend_prediction, dtype=float)
    )
    agreement_score = np.clip(1.0 - np.abs(logistic_prediction - nonlinear_prediction), 0.0, 1.0)
    review_threshold = _fit_kwarg_float(fit_kwargs, "agreement_review_threshold", 0.80)
    result = pd.DataFrame(
        {
            "backbone_id": score["backbone_id"].astype(str).tolist(),
            "prediction": final_prediction.tolist(),
            "logistic_base_prediction": logistic_prediction.tolist(),
            "nonlinear_base_prediction": nonlinear_prediction.tolist(),
            "agreement_score": agreement_score.tolist(),
            "agreement_review_flag": (agreement_score < review_threshold).tolist(),
            "nonlinear_backend_requested": _fit_kwarg_str(fit_kwargs, "nonlinear_backend", "")
            .strip()
            .lower()
            or ("ebm" if ExplainableBoostingClassifier is not None else "hist_gbm"),
            "nonlinear_backend_resolved": resolved_backend,
            "nonlinear_backend_resolution_status": backend_status,
        }
    )
    if "spread_label" in score.columns:
        result["spread_label"] = score["spread_label"].tolist()
    return result


def _oof_hybrid_predictions_from_eligible(
    eligible: pd.DataFrame,
    *,
    columns: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    fit_kwargs: dict[str, object] | None = None,
    y_override: np.ndarray | None = None,
    folds_per_repeat: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fit_kwargs = fit_kwargs or {}
    y = (
        np.asarray(y_override, dtype=int)
        if y_override is not None
        else eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    )
    preds = np.zeros(len(eligible), dtype=float)
    logistic_base = np.zeros(len(eligible), dtype=float)
    nonlinear_base = np.zeros(len(eligible), dtype=float)
    agreement = np.zeros(len(eligible), dtype=float)
    review = np.zeros(len(eligible), dtype=float)
    nonlinear_backend_requested = np.full(len(eligible), "", dtype=object)
    nonlinear_backend_resolved = np.full(len(eligible), "", dtype=object)
    nonlinear_backend_resolution_status = np.full(len(eligible), "", dtype=object)
    counts = np.zeros(len(eligible), dtype=float)
    fold_groups = (
        folds_per_repeat
        if folds_per_repeat is not None
        else _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    )
    for _, test_idx in fold_groups:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train = eligible.loc[train_mask]
        test = eligible.iloc[test_idx]
        fold_predictions = _fit_predict_hybrid_stacked(
            train,
            test,
            columns,
            fit_kwargs=fit_kwargs,
            l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
        )
        preds[test_idx] += fold_predictions["prediction"].to_numpy(dtype=float)
        logistic_base[test_idx] += fold_predictions["logistic_base_prediction"].to_numpy(
            dtype=float
        )
        nonlinear_base[test_idx] += fold_predictions["nonlinear_base_prediction"].to_numpy(
            dtype=float
        )
        agreement[test_idx] += fold_predictions["agreement_score"].to_numpy(dtype=float)
        review[test_idx] += (
            fold_predictions["agreement_review_flag"].astype(float).to_numpy(dtype=float)
        )
        if "nonlinear_backend_requested" in fold_predictions.columns:
            nonlinear_backend_requested[test_idx] = str(
                fold_predictions["nonlinear_backend_requested"].iloc[0]
            )
        if "nonlinear_backend_resolved" in fold_predictions.columns:
            nonlinear_backend_resolved[test_idx] = str(
                fold_predictions["nonlinear_backend_resolved"].iloc[0]
            )
        if "nonlinear_backend_resolution_status" in fold_predictions.columns:
            nonlinear_backend_resolution_status[test_idx] = str(
                fold_predictions["nonlinear_backend_resolution_status"].iloc[0]
            )
        counts[test_idx] += 1.0
    if counts.min() == 0:
        warnings.warn(
            f"{int((counts == 0).sum())} sample(s) never appeared in any test fold",
            stacklevel=2,
        )
        counts[counts == 0] = 1.0
    detail = pd.DataFrame(
        {
            "logistic_base_prediction": logistic_base / counts,
            "nonlinear_base_prediction": nonlinear_base / counts,
            "agreement_score": agreement / counts,
            "agreement_review_flag": (review / counts) >= 0.5,
            "nonlinear_backend_requested": nonlinear_backend_requested,
            "nonlinear_backend_resolved": nonlinear_backend_resolved,
            "nonlinear_backend_resolution_status": nonlinear_backend_resolution_status,
        },
        index=eligible.index,
    )
    return preds / counts, y, detail


def _oof_predictions_from_eligible(
    eligible: pd.DataFrame,
    *,
    columns: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    fit_kwargs: dict[str, object] | None = None,
    y_override: np.ndarray | None = None,
    folds_per_repeat: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_kwargs = fit_kwargs or {}
    model_type = _model_type_name(fit_kwargs)
    if model_type == "lightgbm":
        return _oof_lightgbm_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
            y_override=y_override,
            folds_per_repeat=folds_per_repeat,
        )
    if model_type == "hybrid_stacked":
        hybrid_preds, hybrid_y, _ = _oof_hybrid_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
            y_override=y_override,
            folds_per_repeat=folds_per_repeat,
        )
        return hybrid_preds, hybrid_y
    y = (
        np.asarray(y_override, dtype=int)
        if y_override is not None
        else eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    )
    preds = np.zeros(len(eligible), dtype=float)
    counts = np.zeros(len(eligible), dtype=float)
    fold_groups = (
        folds_per_repeat
        if folds_per_repeat is not None
        else _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    )
    for _, test_idx in fold_groups:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train = eligible.loc[train_mask]
        test = eligible.iloc[test_idx]
        X_train, X_test = _prepare_feature_matrices(
            train,
            test,
            columns,
            fit_kwargs=fit_kwargs,
            prepared=True,
        )
        train_weight = _compute_sample_weight(
            train,
            mode=_fit_kwarg_mode(fit_kwargs),
            fit_kwargs=fit_kwargs,
        )
        if model_type == "pairwise_rank_logistic":
            beta, mean, std, _ = _fit_pairwise_rank_model(
                X_train,
                y[train_mask],
                fit_kwargs=fit_kwargs,
                sample_weight=train_weight,
            )
            X_test_scaled = _standardize_apply(X_test, mean, std)
        else:
            X_train_scaled, mean, std = _standardize_fit(X_train)
            X_test_scaled = _standardize_apply(X_test, mean, std)
            beta = _fit_logistic_regression(
                X_train_scaled,
                y[train_mask],
                l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
                max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
                sample_weight=train_weight,
                fit_backend=_fit_backend_name(fit_kwargs),
            )
        preds[test_idx] += _predict_logistic(X_test_scaled, beta)
        counts[test_idx] += 1
    if counts.min() == 0:
        warnings.warn(
            f"{int((counts == 0).sum())} sample(s) never appeared in any test fold",
            stacklevel=2,
        )
        counts[counts == 0] = 1
    return preds / counts, y


def _oof_predictions_with_detail_from_eligible(
    eligible: pd.DataFrame,
    *,
    columns: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    fit_kwargs: dict[str, object] | None = None,
    y_override: np.ndarray | None = None,
    folds_per_repeat: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    fit_kwargs = fit_kwargs or {}
    if _model_type_name(fit_kwargs) == "hybrid_stacked":
        return _oof_hybrid_predictions_from_eligible(
            eligible,
            columns=columns,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
            y_override=y_override,
            folds_per_repeat=folds_per_repeat,
        )
    preds, y = _oof_predictions_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
        y_override=y_override,
        folds_per_repeat=folds_per_repeat,
    )
    return preds, y, None


def _evaluate_model_name_task(
    scored: pd.DataFrame,
    model_name: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    fold_groups: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[str, ModelResult]:
    return _safe_evaluate_model_name_task(
        scored, model_name, n_splits, n_repeats, seed, fold_groups
    )


def _build_dropout_feature_row(
    eligible: pd.DataFrame,
    *,
    model_name: str,
    columns: list[str],
    feature_name: str,
    feature_rank: int,
    y: np.ndarray,
    fold_groups: list[tuple[np.ndarray, np.ndarray]],
    fit_kwargs: dict[str, object],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> dict[str, object]:
    reduced_columns = [column for column in columns if column != feature_name]
    reduced_preds, _ = _oof_predictions_from_eligible(
        eligible,
        columns=reduced_columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
        folds_per_repeat=fold_groups,
    )
    reduced_auc = roc_auc_score(y, reduced_preds)
    return {
        "model_name": model_name,
        "feature_name": feature_name,
        "feature_rank": feature_rank,
        "roc_auc_without_feature": reduced_auc,
        "average_precision_without_feature": average_precision(y, reduced_preds),
        "brier_without_feature": brier_score(y, reduced_preds),
        "n_backbones": int(len(eligible)),
    }


def _evaluate_prediction_set(
    name: str,
    y: np.ndarray,
    preds: np.ndarray,
    index: pd.Index | pd.Series,
    *,
    include_ci: bool = True,
    knownness_score: np.ndarray | None = None,
    extra_metrics: Mapping[str, float | int | bool | str] | None = None,
    prediction_detail: pd.DataFrame | None = None,
) -> ModelResult:
    if knownness_score is not None:
        assert len(knownness_score) == len(y), (
            "knownness_score must align with y array for post-hoc novelty stratification."
        )
    prevalence = positive_prevalence(y)
    ap = average_precision(y, preds)
    precision_at_10, recall_at_10 = _top_k_precision_recall(y, preds, top_k=10)
    precision_at_25, recall_at_25 = _top_k_precision_recall(y, preds, top_k=25)
    brier_parts = brier_decomposition(y, preds)
    utility_summary = decision_utility_summary(y, preds)
    metrics: dict[str, float | int | bool | str] = {
        "roc_auc": roc_auc_score(y, preds),
        "average_precision": ap,
        "positive_prevalence": prevalence,
        "average_precision_lift": average_precision_lift(y, preds),
        "average_precision_enrichment": average_precision_enrichment(y, preds),
        "brier_score": brier_score(y, preds),
        "log_loss": log_loss(y, preds),
        "expected_calibration_error": expected_calibration_error(y, preds),
        "max_calibration_error": max_calibration_error(y, preds),
        "ndcg_at_10": ndcg_at_k(y, preds, k=10),
        "ndcg_at_25": ndcg_at_k(y, preds, k=25),
        "brier_reliability": brier_parts["reliability"],
        "brier_resolution": brier_parts["resolution"],
        "brier_uncertainty": brier_parts["uncertainty"],
        "precision_at_top_10": precision_at_10,
        "recall_at_top_10": recall_at_10,
        "precision_at_top_25": precision_at_25,
        "recall_at_top_25": recall_at_25,
        "optimal_decision_threshold": utility_summary["optimal_threshold"],
        "decision_utility_score": utility_summary["optimal_threshold_utility_per_sample"],
        "decision_utility_cost": utility_summary["optimal_threshold_cost_per_sample"],
        "decision_utility_precision": utility_summary["optimal_threshold_precision"],
        "decision_utility_recall": utility_summary["optimal_threshold_recall"],
        "decision_utility_positive_rate": utility_summary["optimal_threshold_positive_rate"],
        "decision_utility_grid_size": utility_summary["utility_grid_size"],
        "n_backbones": int(len(y)),
        "n_positive": int((np.asarray(y, dtype=int) == 1).sum()),
        "weighted_classification_cost": weighted_classification_cost(y, preds),
    }
    if knownness_score is not None:
        metrics["novelty_adjusted_average_precision"] = novelty_adjusted_average_precision(
            y,
            preds,
            knownness_score,
        )
    if extra_metrics:
        metrics.update(extra_metrics)
    if include_ci:
        intervals = bootstrap_intervals(
            y,
            preds,
            {
                "roc_auc": roc_auc_score,
                "average_precision": average_precision,
                "brier_score": brier_score,
                "log_loss": log_loss,
            },
        )
        metrics["roc_auc_ci_lower"] = intervals["roc_auc"]["lower"]
        metrics["roc_auc_ci_upper"] = intervals["roc_auc"]["upper"]
        metrics["average_precision_ci_lower"] = intervals["average_precision"]["lower"]
        metrics["average_precision_ci_upper"] = intervals["average_precision"]["upper"]
        metrics["brier_score_ci_lower"] = intervals["brier_score"]["lower"]
        metrics["brier_score_ci_upper"] = intervals["brier_score"]["upper"]
        metrics["log_loss_ci_lower"] = intervals["log_loss"]["lower"]
        metrics["log_loss_ci_upper"] = intervals["log_loss"]["upper"]
    predictions = pd.DataFrame(
        {
            "backbone_id": pd.Series(index).astype(str).to_numpy(dtype=str),
            "oof_prediction": preds,
            "spread_label": y,
            "visibility_expansion_label": y,  # kept for backward compat; always equals spread_label
        }
    )
    if prediction_detail is not None and not prediction_detail.empty:
        predictions = pd.concat(
            [predictions.reset_index(drop=True), prediction_detail.reset_index(drop=True)],
            axis=1,
        )
    return ModelResult(name=name, metrics=metrics, predictions=predictions)


def _discrete_entropy(codes: np.ndarray) -> float:
    valid = np.asarray(codes, dtype=int)
    valid = valid[valid >= 0]
    if valid.size == 0:
        return 0.0
    _, counts = np.unique(valid, return_counts=True)
    probabilities = counts.astype(float) / float(valid.size)
    return float(-(probabilities * np.log2(np.clip(probabilities, 1e-15, 1.0))).sum())


def _mutual_information_discrete(feature_codes: np.ndarray, target_codes: np.ndarray) -> float:
    feature_codes = np.asarray(feature_codes, dtype=int)
    target_codes = np.asarray(target_codes, dtype=int)
    valid_mask = (feature_codes >= 0) & (target_codes >= 0)
    if not np.any(valid_mask):
        return 0.0
    x = feature_codes[valid_mask]
    y = target_codes[valid_mask]
    h_x = _discrete_entropy(x)
    h_y = _discrete_entropy(y)
    joint_codes = np.column_stack([x, y])
    _, joint_counts = np.unique(joint_codes, axis=0, return_counts=True)
    joint_probabilities = joint_counts.astype(float) / float(len(x))
    h_xy = float(-(joint_probabilities * np.log2(np.clip(joint_probabilities, 1e-15, 1.0))).sum())
    return float(max(h_x + h_y - h_xy, 0.0))


def _conditional_mutual_information_discrete(
    feature_codes: np.ndarray,
    target_codes: np.ndarray,
    condition_codes: np.ndarray,
) -> float:
    feature_codes = np.asarray(feature_codes, dtype=int)
    target_codes = np.asarray(target_codes, dtype=int)
    condition_codes = np.asarray(condition_codes, dtype=int)
    valid_mask = (feature_codes >= 0) & (target_codes >= 0) & (condition_codes >= 0)
    if not np.any(valid_mask):
        return 0.0
    x = feature_codes[valid_mask]
    y = target_codes[valid_mask]
    z = condition_codes[valid_mask]
    total = float(len(x))
    score = 0.0
    for code in np.unique(z):
        z_mask = z == code
        if int(z_mask.sum()) < 2:
            continue
        score += float(z_mask.sum() / total) * _mutual_information_discrete(x[z_mask], y[z_mask])
    return float(max(score, 0.0))


def _quantile_binned_codes(values: pd.Series, *, n_bins: int) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    result = np.full(len(numeric), -1, dtype=int)
    valid_mask = numeric.notna().to_numpy()
    if not np.any(valid_mask):
        return result
    valid = numeric.loc[numeric.notna()]
    if valid.nunique() < 2:
        result[valid_mask] = 0
        return result
    if int(valid.nunique()) <= max(int(n_bins), 2):
        unique_values = sorted(valid.astype(float).unique().tolist())
        category_map = {float(value): idx for idx, value in enumerate(unique_values)}
        result[valid_mask] = (
            valid.astype(float).map(category_map).fillna(0).astype(int).to_numpy(dtype=int)
        )
        return result
    ranked = valid.rank(method="average")
    try:
        codes = pd.qcut(
            ranked,
            q=min(max(int(n_bins), 2), int(valid.nunique())),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        result[valid_mask] = 0
        return result
    result[valid_mask] = (
        pd.Series(codes, index=valid.index).fillna(0).astype(int).to_numpy(dtype=int)
    )
    return result


def select_cmim_features(
    scored: pd.DataFrame,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    top_n: int = 12,
    n_bins: int = 5,
) -> list[str]:
    return build_cmim_feature_selection_table(
        scored,
        columns=columns,
        top_n=top_n,
        n_bins=n_bins,
    )["feature_name"].astype(str).tolist()


def build_cmim_feature_selection_table(
    scored: pd.DataFrame,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    top_n: int = 12,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Greedy CMIM ranking over candidate features for the spread label."""
    candidate_features = {
        feature for feature_set in MODULE_A_FEATURE_SETS.values() for feature in feature_set
    }
    candidate_columns = (
        list(dict.fromkeys(str(column) for column in columns))
        if columns is not None
        else sorted(candidate_features)
    )
    if not candidate_columns:
        return pd.DataFrame(
            columns=[
                "rank",
                "feature_name",
                "mutual_information",
                "cmim_score",
                "min_conditional_mutual_information",
                "n_unique_bins",
            ]
        )
    eligible = (
        _ensure_feature_columns(scored, candidate_columns)
        .loc[scored["spread_label"].notna()]
        .copy()
    )
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "feature_name",
                "mutual_information",
                "cmim_score",
                "min_conditional_mutual_information",
                "n_unique_bins",
            ]
        )
    y_codes = eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    feature_codes = {
        feature_name: _quantile_binned_codes(eligible[feature_name], n_bins=n_bins)
        for feature_name in candidate_columns
    }
    mutual_information = {
        feature_name: _mutual_information_discrete(codes, y_codes)
        for feature_name, codes in feature_codes.items()
    }
    remaining = list(candidate_columns)
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for rank in range(1, min(max(int(top_n), 1), len(remaining)) + 1):
        best_feature: str | None = None
        best_score = float("-inf")
        best_min_conditional = float("nan")
        best_mi = float("-inf")
        for feature_name in remaining:
            base_mi = float(mutual_information[feature_name])
            if selected:
                conditional_values = [
                    _conditional_mutual_information_discrete(
                        feature_codes[feature_name],
                        y_codes,
                        feature_codes[selected_feature],
                    )
                    for selected_feature in selected
                ]
                min_conditional = float(min(conditional_values)) if conditional_values else base_mi
                cmim_score = min(base_mi, min_conditional)
            else:
                min_conditional = float("nan")
                cmim_score = base_mi
            if (
                cmim_score > best_score
                or (cmim_score == best_score and base_mi > best_mi)
                or (
                    cmim_score == best_score
                    and base_mi == best_mi
                    and (best_feature is None or feature_name < best_feature)
                )
            ):
                best_feature = feature_name
                best_score = cmim_score
                best_min_conditional = min_conditional
                best_mi = base_mi
        if best_feature is None:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        codes = feature_codes[best_feature]
        rows.append(
            {
                "rank": int(rank),
                "feature_name": best_feature,
                "mutual_information": float(mutual_information[best_feature]),
                "cmim_score": float(best_score),
                "min_conditional_mutual_information": float(best_min_conditional)
                if np.isfinite(best_min_conditional)
                else float("nan"),
                "n_unique_bins": int(len(np.unique(codes[codes >= 0]))),
            }
        )
    return pd.DataFrame(rows)


def get_primary_model_name(model_names: list[str] | pd.Series | set[str] | tuple[str, ...]) -> str:
    """Resolve the preferred benchmark model while preserving backward compatibility."""
    names = {str(name) for name in model_names}
    if PRIMARY_MODEL_NAME in names:
        return PRIMARY_MODEL_NAME
    if PRIMARY_MODEL_FALLBACK in names:
        return PRIMARY_MODEL_FALLBACK
    return sorted(names)[0]


def get_conservative_model_name(
    model_names: list[str] | pd.Series | set[str] | tuple[str, ...],
) -> str:
    """Resolve the most conservative feature-light benchmark model."""
    names = {str(name) for name in model_names}
    if CONSERVATIVE_MODEL_NAME in names:
        return CONSERVATIVE_MODEL_NAME
    if "T_plus_H_plus_A" in names:
        return "T_plus_H_plus_A"
    return get_primary_model_name(names)


def get_governance_model_name(
    model_names: list[str] | pd.Series | set[str] | tuple[str, ...],
) -> str:
    """Resolve the governance benchmark while preserving backward compatibility."""
    names = {str(name) for name in model_names}
    if GOVERNANCE_MODEL_NAME in names:
        return GOVERNANCE_MODEL_NAME
    if GOVERNANCE_MODEL_FALLBACK in names:
        return GOVERNANCE_MODEL_FALLBACK
    if PRIMARY_MODEL_FALLBACK in names:
        return PRIMARY_MODEL_FALLBACK
    return get_primary_model_name(names)


def get_official_model_names(
    model_names: list[str] | pd.Series | set[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the jury-facing official model surface: discovery, governance, baseline."""
    names = {str(name) for name in model_names}
    official: list[str] = []
    if names:
        official.append(get_primary_model_name(names))
        official.append(get_governance_model_name(names))
        if "baseline_both" in names:
            official.append("baseline_both")
    return tuple(dict.fromkeys(name for name in official if name))


def get_model_track(model_name: str) -> str:
    """Return the admissibility track for a configured Module A model."""
    return MODULE_A_MODEL_TRACKS[str(model_name)]


def _default_split_strategy(model_name: str, fit_kwargs: dict[str, object] | None) -> str:
    explicit = _fit_kwarg_str(fit_kwargs, "split_strategy", "")
    if explicit:
        return explicit
    if get_model_track(model_name) == "governance":
        return "temporal_group"
    return "stratified_repeated"


def _resolve_split_strategy_for_frame(frame: pd.DataFrame, strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if normalized == "temporal_group" and "resolved_year" not in frame.columns:
        warnings.warn(
            "Falling back to stratified_repeated because resolved_year is missing "
            "for temporal_group.",
            stacklevel=2,
        )
        return "stratified_repeated"
    return normalized


def get_feature_track(feature_name: str) -> str:
    """Return the provenance track for a configured Module A feature."""
    return FEATURE_PROVENANCE_REGISTRY[str(feature_name)].track


def get_research_models_by_track(track: str) -> tuple[str, ...]:
    """Filter RESEARCH_MODEL_NAMES by inferred track.

    Args:
        track: The track to filter by ('discovery' | 'governance' | 'baseline')

    Returns:
        Tuple of model names belonging to the specified track.

    Usage note: Discovery scripts must call get_research_models_by_track("discovery")
    rather than iterating over raw RESEARCH_MODEL_NAMES to avoid mixing governance
    candidates into discovery sweeps.
    """
    _ensure_config_loaded()
    return tuple(
        name for name in RESEARCH_MODEL_NAMES if MODULE_A_MODEL_TRACKS.get(str(name)) == str(track)
    )


def _ensure_feature_columns(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return _ensure_feature_columns_impl(scored, columns)


def assert_feature_columns_present(
    scored: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | set[str],
    *,
    label: str,
) -> None:
    """Fail loudly when engineered score columns are stale or missing."""
    _assert_feature_columns_present_impl(scored, columns, label=label)


def annotate_knownness_metadata(scored: pd.DataFrame) -> pd.DataFrame:
    """Add training-visibility proxy metadata used in novelty and bias audits."""
    working = scored.copy()
    for column in ("log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"):
        if column not in working.columns:
            working[column] = 0.0
    cohort_mask = (
        working["spread_label"].notna()
        if "spread_label" in working.columns
        else pd.Series(True, index=working.index, dtype=bool)
    )
    working["member_rank_norm"] = _masked_percentile_rank(
        working["log1p_member_count_train"], cohort_mask=cohort_mask
    )
    working["country_rank_norm"] = _masked_percentile_rank(
        working["log1p_n_countries_train"], cohort_mask=cohort_mask
    )
    working["source_rank_norm"] = _masked_percentile_rank(
        working["refseq_share_train"], cohort_mask=cohort_mask
    )
    working["knownness_score"] = _knownness_score_series(working, cohort_mask=cohort_mask)
    if not working.empty:
        working["knownness_half"] = pd.Series("out_of_scope", index=working.index, dtype=object)
        knownness_values = pd.to_numeric(working["knownness_score"], errors="coerce")
        valid_mask = cohort_mask.fillna(False).astype(bool) & knownness_values.notna()
        if valid_mask.any():
            median_knownness = float(knownness_values.loc[valid_mask].median())
            working.loc[valid_mask, "knownness_half"] = np.where(
                knownness_values.loc[valid_mask] <= median_knownness,
                "lower_half",
                "upper_half",
            )
        working["knownness_quartile"] = pd.Series(np.nan, index=working.index, dtype=object)
        working["knownness_quartile_supported"] = False
        if valid_mask.any():
            quartile_labels, n_bins = _stable_quantile_labels(knownness_values.loc[valid_mask], q=4)
            if n_bins == 4:
                quartile_labels = quartile_labels.replace(
                    {"q1": "q1_lowest", "q2": "q2", "q3": "q3", "q4": "q4_highest"}
                )
                working.loc[valid_mask, "knownness_quartile"] = quartile_labels.astype(str)
                working.loc[valid_mask, "knownness_quartile_supported"] = True
    else:
        working["knownness_half"] = pd.Series(dtype=object)
        working["knownness_quartile"] = pd.Series(dtype=object)
        working["knownness_quartile_supported"] = pd.Series(dtype=bool)
    working["member_count_band"] = pd.cut(
        np.expm1(working["log1p_member_count_train"].fillna(0.0)),
        bins=[-np.inf, 1, 2, 4, 9, np.inf],
        labels=["1", "2", "3_4", "5_9", "10_plus"],
    ).astype(str)
    working["country_count_band"] = pd.cut(
        np.expm1(working["log1p_n_countries_train"].fillna(0.0)),
        bins=[-np.inf, 0, 1, 2, 4, np.inf],
        labels=["0", "1", "2", "3_4", "5_plus"],
    ).astype(str)
    working["source_band"] = np.where(
        working["refseq_share_train"].fillna(0.0) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    return working


def _eligible_xy(
    scored: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    working = _ensure_feature_columns(scored, columns)
    eligible = working.loc[working["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    X = eligible[columns].fillna(0.0).to_numpy(dtype=float)
    y = eligible["spread_label"].to_numpy(dtype=int)
    return eligible, X, y


def fit_predict_model_holdout(
    scored_train: pd.DataFrame,
    scored_test: pd.DataFrame,
    *,
    model_name: str,
    l2: float = 1.0,
    max_iter: int = 100,
    fit_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fit a named model on one cohort and predict on a disjoint holdout cohort."""
    _ensure_config_loaded()
    columns = MODULE_A_FEATURE_SETS[model_name]
    train = (
        _ensure_feature_columns(scored_train, columns)
        .loc[scored_train["spread_label"].notna()]
        .copy()
    )
    test = (
        _ensure_feature_columns(scored_test, columns)
        .loc[scored_test["spread_label"].notna()]
        .copy()
    )
    if train.empty or test.empty:
        return pd.DataFrame(columns=["backbone_id", "spread_label", "prediction"])

    train["spread_label"] = train["spread_label"].astype(int)
    test["spread_label"] = test["spread_label"].astype(int)
    if train["spread_label"].nunique() < 2:
        return pd.DataFrame(columns=["backbone_id", "spread_label", "prediction"])

    y_train = train["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    model_type = _model_type_name(fit_kwargs)
    if model_type == "hybrid_stacked":
        return _fit_predict_hybrid_stacked(
            train,
            test,
            columns,
            fit_kwargs=fit_kwargs,
            l2=l2,
            max_iter=max_iter,
        )
    X_train, X_test = _prepare_feature_matrices(
        train,
        test,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=True,
    )
    sample_weight = _compute_sample_weight(
        train,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    if model_type == "pairwise_rank_logistic":
        beta, mean, std, X_train_scaled = _fit_pairwise_rank_model(
            X_train,
            y_train,
            fit_kwargs=fit_kwargs,
            sample_weight=sample_weight,
        )
    else:
        beta, mean, std = _fit_standardized_model(
            X_train,
            y_train,
            l2=_fit_kwarg_float(fit_kwargs, "l2", l2),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", max_iter),
            sample_weight=sample_weight,
            fit_backend=_fit_backend_name(fit_kwargs),
        )
        X_train_scaled = _standardize_apply(X_train, mean, std)
    effective_l2 = _fit_kwarg_float(fit_kwargs, "l2", l2)
    X_test_scaled = _standardize_apply(X_test, mean, std)
    posterior_covariance = _logistic_posterior_covariance(
        X_train_scaled,
        beta,
        l2=effective_l2,
        sample_weight=sample_weight,
    )
    posterior_summary = _bayesian_prediction_summary(X_test_scaled, beta, posterior_covariance)
    preds = _predict_calibrated(X_test_scaled, beta)
    return pd.DataFrame(
        {
            "backbone_id": test["backbone_id"].astype(str).tolist(),
            "spread_label": test["spread_label"].astype(int).tolist(),
            "prediction": preds.tolist(),
            "prediction_posterior_mean": posterior_summary["mean"].tolist(),
            "prediction_std": posterior_summary["std"].tolist(),
            "prediction_ci_lower": posterior_summary["q05"].tolist(),
            "prediction_ci_upper": posterior_summary["q95"].tolist(),
        }
    )


def _component_rank_score(values: pd.Series, *, ascending: bool) -> pd.Series:
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return scores
    ranks = values.loc[valid].rank(method="average", ascending=ascending)
    if int(valid.sum()) == 1:
        scores.loc[valid] = 1.0
        return scores
    scores.loc[valid] = 1.0 - ((ranks - 1.0) / (int(valid.sum()) - 1.0))
    return scores


def _knownness_matched_weighted_roc_auc(
    scored: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> float:
    eligible = annotate_knownness_metadata(scored.loc[scored["spread_label"].notna()].copy())
    merged = eligible[
        [
            "backbone_id",
            "spread_label",
            "member_count_band",
            "country_count_band",
            "source_band",
        ]
    ].merge(
        prediction_frame[["backbone_id", "oof_prediction"]],
        on="backbone_id",
        how="inner",
        validate="1:1",
    )
    if merged.empty:
        return float("nan")
    rows: list[dict[str, float | int | str]] = []
    for keys, frame in merged.groupby(
        ["member_count_band", "country_count_band", "source_band"],
        sort=False,
        observed=True,
    ):
        if len(frame) < 20 or frame["spread_label"].astype(int).nunique() < 2:
            continue
        rows.append(
            {
                "member_count_band": str(keys[0]),
                "country_count_band": str(keys[1]),
                "source_band": str(keys[2]),
                "n_backbones": int(len(frame)),
                "roc_auc": float(
                    roc_auc_score(
                        frame["spread_label"].astype(int).to_numpy(),
                        frame["oof_prediction"].astype(float).to_numpy(),
                    )
                ),
            }
        )
    if not rows:
        return float("nan")
    strata = pd.DataFrame(rows)
    weights = pd.to_numeric(strata["n_backbones"], errors="coerce").fillna(0.0)
    aucs = pd.to_numeric(strata["roc_auc"], errors="coerce")
    if float(weights.sum()) <= 0.0:
        return float(aucs.mean())
    return float(np.average(aucs, weights=weights))


def _grouped_prediction_weighted_roc_auc(
    scored: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    min_group_size: int,
    max_groups_per_column: int,
) -> float:
    if prediction_frame.empty:
        return float("nan")
    merged = scored.loc[scored["spread_label"].notna()].copy()
    merged = merged.merge(
        prediction_frame[["backbone_id", "oof_prediction"]],
        on="backbone_id",
        how="inner",
        validate="1:1",
    )
    if merged.empty:
        return float("nan")

    rows: list[dict[str, float | int]] = []
    for group_column in group_columns:
        if group_column not in merged.columns:
            continue
        working = merged.copy()
        working[group_column] = working[group_column].fillna("unknown").astype(str)
        counts = working[group_column].value_counts()
        if group_column == "dominant_source":
            selected_groups = counts.index.tolist()
        else:
            selected_groups = (
                counts.loc[counts >= min_group_size].head(max_groups_per_column).index.tolist()
            )
        for group_value in selected_groups:
            frame = working.loc[working[group_column] == group_value].copy()
            if len(frame) < min_group_size or frame["spread_label"].astype(int).nunique() < 2:
                continue
            rows.append(
                {
                    "n_backbones": int(len(frame)),
                    "roc_auc": float(
                        roc_auc_score(
                            frame["spread_label"].astype(int).to_numpy(),
                            frame["oof_prediction"].astype(float).to_numpy(),
                        )
                    ),
                }
            )
    if not rows:
        return float("nan")
    summary = pd.DataFrame(rows)
    weights = pd.to_numeric(summary["n_backbones"], errors="coerce").fillna(0.0)
    aucs = pd.to_numeric(summary["roc_auc"], errors="coerce")
    if float(weights.sum()) <= 0.0:
        return float(aucs.mean())
    return float(np.average(aucs, weights=weights))


def build_single_model_pareto_screen(
    scored: pd.DataFrame,
    *,
    family: pd.DataFrame | None = None,
    n_splits: int = 3,
    n_repeats: int = 2,
    seed: int = 42,
    min_group_size: int = 25,
    max_groups_per_column: int = 8,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Build the Stage A screen for bounded single-model Pareto candidates."""
    candidate_family = (
        family.copy() if family is not None else build_single_model_candidate_family()
    )
    if candidate_family.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "parent_model_name",
                "feature_count",
                "roc_auc",
                "average_precision",
                "knownness_matched_gap",
                "source_holdout_gap",
                "blocked_holdout_weighted_roc_auc",
                "screen_fit_seconds",
                "predictive_power_score",
                "reliability_score",
                "compute_efficiency_score",
                "weighted_objective_score",
            ]
        )

    def _screen_candidate(candidate: dict[str, object]) -> dict[str, object]:
        model_name = str(candidate["model_name"])
        parent_model_name = str(candidate["parent_model_name"])
        feature_set = [str(column) for column in candidate["feature_set"]]  # type: ignore[attr-defined]
        feature_count = int(candidate["feature_count"])  # type: ignore[call-overload]
        candidate_kind = str(candidate["candidate_kind"])
        fit_config = _model_fit_kwargs(parent_model_name)

        started_at = time.perf_counter()
        if candidate_kind == "parent" and model_name in MODULE_A_FEATURE_SETS:
            result = evaluate_model_name(
                scored,
                model_name=model_name,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                include_ci=False,
            )
        else:
            result = evaluate_feature_columns(
                scored,
                columns=feature_set,
                label=model_name,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                fit_config=fit_config,
                include_ci=False,
            )
        screen_fit_seconds = float(max(time.perf_counter() - started_at, 0.0))

        prediction_frame = result.predictions.copy()
        if not prediction_frame.empty:
            if "spread_label" not in prediction_frame.columns:
                prediction_frame = prediction_frame.merge(
                    scored.loc[scored["spread_label"].notna(), ["backbone_id", "spread_label"]],
                    on="backbone_id",
                    how="left",
                    validate="1:1",
                )
            prediction_frame["model_name"] = model_name

        roc_auc_value = float(result.metrics.get("roc_auc", np.nan))
        average_precision_value = float(result.metrics.get("average_precision", np.nan))
        ece_value = float(
            result.metrics.get(
                "ece",
                result.metrics.get("expected_calibration_error", np.nan),
            )
        )
        if prediction_frame.empty:
            matched_knownness_weighted_roc_auc = float("nan")
            source_holdout_weighted_roc_auc = float("nan")
            blocked_holdout_weighted_roc_auc = float("nan")
            spatial_holdout_weighted_roc_auc = float("nan")
        else:
            merged_for_score = scored.loc[scored["spread_label"].notna()].copy()
            merged_for_score = merged_for_score.merge(
                prediction_frame[["backbone_id", "oof_prediction"]],
                on="backbone_id",
                how="inner",
                validate="1:1",
            )
            matched_knownness_weighted_roc_auc = _knownness_matched_weighted_roc_auc(
                scored,
                prediction_frame,
            )

            def local_grouped_auc(group_cols: tuple[str, ...]) -> float:
                rows = []
                for group_col in group_cols:
                    if group_col not in merged_for_score.columns:
                        continue
                    working = merged_for_score.copy()
                    working[group_col] = working[group_col].fillna("unknown").astype(str)
                    counts = working[group_col].value_counts()
                    selected_groups = (
                        counts.index.tolist()
                        if group_col == "dominant_source"
                        else counts.loc[counts >= min_group_size]
                        .head(max_groups_per_column)
                        .index.tolist()
                    )
                    for group_value in selected_groups:
                        frame = working.loc[working[group_col] == group_value]
                        if (
                            len(frame) < min_group_size
                            or frame["spread_label"].astype(int).nunique() < 2
                        ):
                            continue
                        from sklearn.metrics import roc_auc_score

                        rows.append(
                            {
                                "n_backbones": int(len(frame)),
                                "roc_auc": float(
                                    roc_auc_score(
                                        frame["spread_label"].astype(int).to_numpy(),
                                        frame["oof_prediction"].astype(float).to_numpy(),
                                    )
                                ),
                            }
                        )
                if not rows:
                    return float("nan")
                summary = pd.DataFrame(rows)
                weights = pd.to_numeric(summary["n_backbones"], errors="coerce").fillna(0.0)
                aucs = pd.to_numeric(summary["roc_auc"], errors="coerce")
                return (
                    float(np.average(aucs, weights=weights))
                    if float(weights.sum()) > 0.0
                    else float(aucs.mean())
                )

            source_holdout_weighted_roc_auc = local_grouped_auc(("dominant_source",))
            blocked_holdout_weighted_roc_auc = local_grouped_auc(
                ("dominant_source", "dominant_region_train")
            )
            spatial_holdout_weighted_roc_auc = local_grouped_auc(("dominant_region_train",))
        screen_fit_seconds = float(max(time.perf_counter() - started_at, 0.0))
        return {
            "model_name": model_name,
            "parent_model_name": parent_model_name,
            "feature_set": tuple(feature_set),
            "feature_count": feature_count,
            "candidate_kind": candidate_kind,
            "status": result.status,
            "roc_auc": roc_auc_value,
            "average_precision": average_precision_value,
            "matched_knownness_weighted_roc_auc": matched_knownness_weighted_roc_auc,
            "knownness_matched_gap": matched_knownness_weighted_roc_auc - roc_auc_value
            if pd.notna(matched_knownness_weighted_roc_auc)
            else np.nan,
            "source_holdout_weighted_roc_auc": source_holdout_weighted_roc_auc,
            "source_holdout_gap": source_holdout_weighted_roc_auc - roc_auc_value
            if pd.notna(source_holdout_weighted_roc_auc)
            else np.nan,
            "spatial_holdout_weighted_roc_auc": spatial_holdout_weighted_roc_auc,
            "spatial_holdout_gap": spatial_holdout_weighted_roc_auc - roc_auc_value
            if pd.notna(spatial_holdout_weighted_roc_auc)
            else np.nan,
            "blocked_holdout_weighted_roc_auc": blocked_holdout_weighted_roc_auc,
            "ece": ece_value,
            "screen_fit_seconds": screen_fit_seconds,
        }

    candidate_rows = cast(list[dict[str, object]], candidate_family.to_dict("records"))
    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(candidate_rows), cap=8)
    if jobs > 1 and candidate_rows:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                rows = list(executor.map(_screen_candidate, candidate_rows))
    else:
        rows = [_screen_candidate(candidate) for candidate in candidate_rows]

    screen = pd.DataFrame(rows)
    if screen.empty:
        return screen

    screen["roc_auc_component_score"] = _component_rank_score(
        pd.to_numeric(screen["roc_auc"], errors="coerce"),
        ascending=False,
    )
    screen["average_precision_component_score"] = _component_rank_score(
        pd.to_numeric(screen["average_precision"], errors="coerce"),
        ascending=False,
    )
    screen["knownness_component_score"] = _component_rank_score(
        pd.to_numeric(screen["matched_knownness_weighted_roc_auc"], errors="coerce"),
        ascending=False,
    )
    screen["source_holdout_component_score"] = _component_rank_score(
        pd.to_numeric(screen["source_holdout_weighted_roc_auc"], errors="coerce"),
        ascending=False,
    )
    screen["blocked_holdout_component_score"] = _component_rank_score(
        pd.to_numeric(screen["blocked_holdout_weighted_roc_auc"], errors="coerce"),
        ascending=False,
    )
    screen["feature_count_component_score"] = _component_rank_score(
        pd.to_numeric(screen["feature_count"], errors="coerce"),
        ascending=True,
    )
    screen["fit_seconds_component_score"] = _component_rank_score(
        pd.to_numeric(screen["screen_fit_seconds"], errors="coerce"),
        ascending=True,
    )
    screen["predictive_power_score"] = screen[
        ["roc_auc_component_score", "average_precision_component_score"]
    ].mean(axis=1)
    screen["reliability_score"] = screen[
        [
            "knownness_component_score",
            "source_holdout_component_score",
            "blocked_holdout_component_score",
        ]
    ].mean(axis=1)
    screen["compute_efficiency_score"] = screen[
        ["feature_count_component_score", "fit_seconds_component_score"]
    ].mean(axis=1)
    screen = add_weighted_objective(screen)
    return screen.sort_values(
        ["weighted_objective_score", "reliability_score", "model_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def fit_full_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    l2: float = 1.0,
    max_iter: int = 100,
    fit_config: dict[str, object] | None = None,
    include_posterior_uncertainty: bool = True,
) -> pd.DataFrame:
    """Fit a named model on the full eligible cohort and score all backbones.

    These scores are for candidate contextualization only. They must not be used
    as replacement evaluation metrics because they are not out-of-fold.
    """
    _ensure_config_loaded()
    columns = MODULE_A_FEATURE_SETS[model_name]
    working = _ensure_feature_columns(scored, columns)
    train = working.loc[working["spread_label"].notna()].copy()
    all_rows = working.copy()
    if train.empty or all_rows.empty:
        return pd.DataFrame(columns=["backbone_id", "prediction"])

    train["spread_label"] = train["spread_label"].astype(int)
    if train["spread_label"].nunique() < 2:
        return pd.DataFrame(columns=["backbone_id", "prediction"])

    y_train = train["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    model_type = _model_type_name(fit_kwargs)
    if model_type == "hybrid_stacked":
        return _fit_predict_hybrid_stacked(
            train,
            all_rows,
            columns,
            fit_kwargs=fit_kwargs,
            l2=l2,
            max_iter=max_iter,
        )
    X_train, X_all = _prepare_feature_matrices(
        train,
        all_rows,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=True,
    )
    sample_weight = _compute_sample_weight(
        train,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    if model_type == "pairwise_rank_logistic":
        beta, mean, std, X_train_scaled = _fit_pairwise_rank_model(
            X_train,
            y_train,
            fit_kwargs=fit_kwargs,
            sample_weight=sample_weight,
        )
    else:
        beta, mean, std = _fit_standardized_model(
            X_train,
            y_train,
            l2=_fit_kwarg_float(fit_kwargs, "l2", l2),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", max_iter),
            sample_weight=sample_weight,
            fit_backend=_fit_backend_name(fit_kwargs),
        )
        X_train_scaled = _standardize_apply(X_train, mean, std)
    X_all_scaled = _standardize_apply(X_all, mean, std)
    preds = _predict_calibrated(X_all_scaled, beta)
    if not include_posterior_uncertainty:
        return pd.DataFrame(
            {
                "backbone_id": all_rows["backbone_id"].astype(str).tolist(),
                "prediction": preds.tolist(),
            }
        )
    effective_l2 = _fit_kwarg_float(fit_kwargs, "l2", l2)
    posterior_covariance = _logistic_posterior_covariance(
        X_train_scaled,
        beta,
        l2=effective_l2,
        sample_weight=sample_weight,
    )
    posterior_summary = _bayesian_prediction_summary(X_all_scaled, beta, posterior_covariance)
    return pd.DataFrame(
        {
            "backbone_id": all_rows["backbone_id"].astype(str).tolist(),
            "prediction": preds.tolist(),
            "prediction_posterior_mean": posterior_summary["mean"].tolist(),
            "prediction_std": posterior_summary["std"].tolist(),
            "prediction_ci_lower": posterior_summary["q05"].tolist(),
            "prediction_ci_upper": posterior_summary["q95"].tolist(),
        }
    )


def evaluate_model_name(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    fit_config: dict[str, object] | None = None,
    include_ci: bool = True,
    fold_groups: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> ModelResult:
    """Evaluate a single named model on the eligible cohort using OOF predictions."""
    _ensure_config_loaded()
    columns = MODULE_A_FEATURE_SETS[model_name]
    assert "knownness_score" not in columns, (
        "knownness_score is a stratification post-hoc metric, not a training feature."
    )
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    from plasmid_priority.modeling.module_a import (
        _knownness_score_series,
        annotate_knownness_metadata,
    )

    knownness_score = _knownness_score_series(annotate_knownness_metadata(eligible)).to_numpy(
        dtype=float
    )
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    split_strategy = _resolve_split_strategy_for_frame(
        eligible,
        _default_split_strategy(model_name, fit_kwargs),
    )
    resolved_fold_groups = fold_groups
    if resolved_fold_groups is None:
        y_for_folds = eligible["spread_label"].astype(int).to_numpy(dtype=int)
        resolved_fold_groups = build_model_folds(
            eligible,
            y_for_folds,
            strategy=split_strategy,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
        )
    preds, y, prediction_detail = _oof_predictions_with_detail_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
        folds_per_repeat=resolved_fold_groups,
    )
    extra_metrics: dict[str, float | int | bool | str] = {"split_strategy": split_strategy}
    if prediction_detail is not None and not prediction_detail.empty:
        agreement_delta = np.abs(
            prediction_detail["logistic_base_prediction"].to_numpy(dtype=float)
            - prediction_detail["nonlinear_base_prediction"].to_numpy(dtype=float)
        )
        extra_metrics.update(
            {
                "mean_agreement_score": float(
                    prediction_detail["agreement_score"].astype(float).mean()
                ),
                "review_fraction": float(
                    prediction_detail["agreement_review_flag"].astype(float).mean()
                ),
                "mean_logistic_nonlinear_delta": float(np.mean(agreement_delta)),
            }
        )
    return _evaluate_prediction_set(
        model_name,
        y,
        preds,
        eligible["backbone_id"],
        include_ci=include_ci,
        knownness_score=knownness_score,
        extra_metrics=extra_metrics,
        prediction_detail=prediction_detail,
    )


def evaluate_feature_columns(
    scored: pd.DataFrame,
    *,
    columns: list[str],
    label: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    fit_config: dict[str, object] | None = None,
    include_ci: bool = True,
) -> ModelResult:
    """Evaluate an arbitrary feature set without registering a named pipeline model."""
    fit_kwargs: dict[str, object] = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    if fit_config:
        fit_kwargs.update(fit_config)
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    split_strategy = _resolve_split_strategy_for_frame(
        eligible,
        _fit_kwarg_str(fit_kwargs, "split_strategy", "stratified_repeated"),
    )
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    from plasmid_priority.modeling.module_a import (
        _knownness_score_series,
        annotate_knownness_metadata,
    )

    knownness_score = _knownness_score_series(annotate_knownness_metadata(eligible)).to_numpy(
        dtype=float
    )
    y_for_folds = eligible["spread_label"].astype(int).to_numpy(dtype=int)
    fold_groups = build_model_folds(
        eligible,
        y_for_folds,
        strategy=split_strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )
    preds, y, prediction_detail = _oof_predictions_with_detail_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
        folds_per_repeat=fold_groups,
    )
    extra_metrics: dict[str, float | int | bool | str] = {"split_strategy": split_strategy}
    if prediction_detail is not None and not prediction_detail.empty:
        agreement_delta = np.abs(
            prediction_detail["logistic_base_prediction"].to_numpy(dtype=float)
            - prediction_detail["nonlinear_base_prediction"].to_numpy(dtype=float)
        )
        extra_metrics.update(
            {
                "mean_agreement_score": float(
                    prediction_detail["agreement_score"].astype(float).mean()
                ),
                "review_fraction": float(
                    prediction_detail["agreement_review_flag"].astype(float).mean()
                ),
                "mean_logistic_nonlinear_delta": float(np.mean(agreement_delta)),
            }
        )
    return _evaluate_prediction_set(
        label,
        y,
        preds,
        eligible["backbone_id"],
        include_ci=include_ci,
        knownness_score=knownness_score,
        extra_metrics=extra_metrics,
        prediction_detail=prediction_detail,
    )


def fit_feature_columns_predictions(
    scored_train: pd.DataFrame,
    scored_score: pd.DataFrame,
    *,
    columns: list[str],
    fit_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fit an arbitrary feature set on one cohort and score another cohort."""
    train = (
        _ensure_feature_columns(scored_train, columns)
        .loc[scored_train["spread_label"].notna()]
        .copy()
    )
    score = _ensure_feature_columns(scored_score, columns).copy()
    if train.empty or score.empty:
        return pd.DataFrame(columns=["backbone_id", "prediction"])
    train["spread_label"] = train["spread_label"].astype(int)
    if train["spread_label"].nunique() < 2:
        return pd.DataFrame(columns=["backbone_id", "prediction"])

    y_train = train["spread_label"].to_numpy(dtype=int)
    fit_kwargs: dict[str, object] = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    if fit_config:
        fit_kwargs.update(fit_config)
    model_type = _model_type_name(fit_kwargs)
    if model_type == "hybrid_stacked":
        return _fit_predict_hybrid_stacked(train, score, columns, fit_kwargs=fit_kwargs)
    X_train, X_score = _prepare_feature_matrices(
        train,
        score,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=True,
    )
    sample_weight = _compute_sample_weight(
        train,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    if model_type == "pairwise_rank_logistic":
        beta, mean, std, X_train_scaled = _fit_pairwise_rank_model(
            X_train,
            y_train,
            fit_kwargs=fit_kwargs,
            sample_weight=sample_weight,
        )
    else:
        beta, mean, std = _fit_standardized_model(
            X_train,
            y_train,
            l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
            sample_weight=sample_weight,
            fit_backend=_fit_backend_name(fit_kwargs),
        )
        X_train_scaled = _standardize_apply(X_train, mean, std)
    effective_l2 = _fit_kwarg_float(fit_kwargs, "l2", 1.0)
    X_score_scaled = _standardize_apply(X_score, mean, std)
    posterior_covariance = _logistic_posterior_covariance(
        X_train_scaled,
        beta,
        l2=effective_l2,
        sample_weight=sample_weight,
    )
    posterior_summary = _bayesian_prediction_summary(X_score_scaled, beta, posterior_covariance)
    preds = _predict_calibrated(X_score_scaled, beta)
    return pd.DataFrame(
        {
            "backbone_id": score["backbone_id"].astype(str).tolist(),
            "prediction": preds.tolist(),
            "prediction_posterior_mean": posterior_summary["mean"].tolist(),
            "prediction_std": posterior_summary["std"].tolist(),
            "prediction_ci_lower": posterior_summary["q05"].tolist(),
            "prediction_ci_upper": posterior_summary["q95"].tolist(),
        }
    )


def build_feature_dropout_audit(
    scored: pd.DataFrame,
    *,
    model_name: str,
    columns: list[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    fit_config: dict[str, object] | None = None,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Measure how much OOF AUC drops when each feature is removed from a model."""
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    fold_groups = _stratified_folds(
        eligible["spread_label"].to_numpy(dtype=int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )
    full_preds, y = _oof_predictions_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
        folds_per_repeat=fold_groups,
    )
    full_auc = roc_auc_score(y, full_preds)

    rows = [
        {
            "model_name": model_name,
            "feature_name": "__full_model__",
            "feature_rank": 0,
            "roc_auc_without_feature": full_auc,
            "roc_auc_drop_vs_full": 0.0,
            "average_precision_without_feature": average_precision(y, full_preds),
            "brier_without_feature": brier_score(y, full_preds),
            "n_backbones": int(len(eligible)),
        }
    ]

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(columns))
    feature_payload = [
        (
            feature_rank,
            feature_name,
        )
        for feature_rank, feature_name in enumerate(columns, start=1)
    ]
    if jobs > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                feature_rows = list(
                    executor.map(
                        lambda payload: _build_dropout_feature_row(
                            eligible,
                            model_name=model_name,
                            columns=columns,
                            feature_name=payload[1],
                            feature_rank=payload[0],
                            y=y,
                            fold_groups=fold_groups,
                            fit_kwargs=fit_kwargs,
                            n_splits=n_splits,
                            n_repeats=n_repeats,
                            seed=seed,
                        ),
                        feature_payload,
                    )
                )
    else:
        feature_rows = [
            _build_dropout_feature_row(
                eligible,
                model_name=model_name,
                columns=columns,
                feature_name=feature_name,
                feature_rank=feature_rank,
                y=y,
                fold_groups=fold_groups,
                fit_kwargs=fit_kwargs,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
            )
            for feature_rank, feature_name in feature_payload
        ]
    for feature_row in feature_rows:
        reduced_auc = float(cast(float, feature_row["roc_auc_without_feature"]))
        rows.append(
            {
                "model_name": model_name,
                "feature_name": feature_row["feature_name"],
                "feature_rank": feature_row["feature_rank"],
                "roc_auc_without_feature": reduced_auc,
                "roc_auc_drop_vs_full": full_auc - reduced_auc,
                "average_precision_without_feature": feature_row[
                    "average_precision_without_feature"
                ],
                "brier_without_feature": feature_row["brier_without_feature"],
                "n_backbones": feature_row["n_backbones"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["roc_auc_drop_vs_full", "feature_name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_standardized_coefficient_table(
    scored: pd.DataFrame,
    *,
    model_name: str,
    columns: list[str],
    l2: float = 1.0,
    max_iter: int = 100,
    fit_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fit the model on the full eligible cohort and expose standardized coefficients."""
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    y = eligible["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    X, _ = _prepare_feature_matrices(
        eligible,
        eligible,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=True,
    )
    sample_weight = _compute_sample_weight(
        eligible,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    if _model_type_name(fit_kwargs) == "pairwise_rank_logistic":
        beta, _, _, X_scaled = _fit_pairwise_rank_model(
            X,
            y,
            fit_kwargs=fit_kwargs,
            sample_weight=sample_weight,
        )
    else:
        beta, _, _ = _fit_standardized_model(
            X,
            y,
            l2=_fit_kwarg_float(fit_kwargs, "l2", l2),
            max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", max_iter),
            sample_weight=sample_weight,
            fit_backend=_fit_backend_name(fit_kwargs),
        )
        X_scaled, _, _ = _standardize_fit(X)
    effective_l2 = _fit_kwarg_float(fit_kwargs, "l2", l2)
    posterior_covariance = _logistic_posterior_covariance(
        X_scaled,
        beta,
        l2=effective_l2,
        sample_weight=sample_weight,
    )
    coefficient_summary = _bayesian_coefficient_summary(beta, posterior_covariance)
    coefficients = beta[1:]
    frame = pd.DataFrame(
        {
            "model_name": model_name,
            "feature_name": columns,
            "coefficient": coefficients,
            "standard_error": coefficient_summary["standard_error"][1:],
            "coefficient_ci_lower": coefficient_summary["ci_lower"][1:],
            "coefficient_ci_upper": coefficient_summary["ci_upper"][1:],
            "abs_coefficient": np.abs(coefficients),
            "direction": np.where(coefficients >= 0, "positive", "negative"),
            "n_backbones": int(len(eligible)),
        }
    )
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def build_coefficient_stability_table(
    scored: pd.DataFrame,
    *,
    model_name: str,
    columns: list[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    l2: float = 1.0,
    max_iter: int = 100,
    fit_config: dict[str, object] | None = None,
    n_jobs: int | None = None,
    include_fold_coefficients: bool = False,
) -> pd.DataFrame:
    """Estimate coefficient stability across repeated stratified training folds."""
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    y = eligible["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    sample_weight = _compute_sample_weight(
        eligible,
        mode=_fit_kwarg_mode(fit_kwargs),
        fit_kwargs=fit_kwargs,
    )
    _, class_counts = np.unique(y, return_counts=True)
    effective_splits = min(max(int(n_splits), 2), int(class_counts.min()))
    tasks: list[tuple[int, int, np.ndarray]] = []
    for split_index, (_, test_idx) in enumerate(
        _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    ):
        repeat_index = split_index // effective_splits + 1
        fold_index = split_index % effective_splits + 1
        tasks.append((repeat_index, fold_index, test_idx))

    def _coefficient_rows(task: tuple[int, int, np.ndarray]) -> list[dict[str, object]]:
        repeat_index, fold_index, test_idx = task
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_weight = sample_weight[train_mask] if sample_weight is not None else None
        train = eligible.loc[train_mask]
        X_train, _ = _prepare_feature_matrices(
            train,
            train,
            columns,
            fit_kwargs=fit_kwargs,
            prepared=True,
        )
        if _model_type_name(fit_kwargs) == "pairwise_rank_logistic":
            beta, _, _, _ = _fit_pairwise_rank_model(
                X_train,
                y[train_mask],
                fit_kwargs=fit_kwargs,
                sample_weight=train_weight,
            )
        else:
            beta, _, _ = _fit_standardized_model(
                X_train,
                y[train_mask],
                l2=_fit_kwarg_float(fit_kwargs, "l2", l2),
                max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", max_iter),
                sample_weight=train_weight,
                fit_backend=_fit_backend_name(fit_kwargs),
            )
        return [
            {
                "model_name": model_name,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "feature_name": feature_name,
                "coefficient": float(coefficient),
            }
            for feature_name, coefficient in zip(columns, beta[1:])
        ]

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(tasks))
    if jobs > 1 and tasks:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                row_groups = list(executor.map(_coefficient_rows, tasks))
        rows = [row for group in row_groups for row in group]
    else:
        rows = [row for task in tasks for row in _coefficient_rows(task)]
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model_name", "feature_name"], as_index=False).agg(
        mean_coefficient=("coefficient", "mean"),
        std_coefficient=("coefficient", "std"),
        min_coefficient=("coefficient", "min"),
        max_coefficient=("coefficient", "max"),
        n_fits=("coefficient", "size"),
        positive_fraction=("coefficient", lambda values: float((values > 0).mean())),
    )
    summary["sign_stability"] = summary["positive_fraction"].map(
        lambda value: "positive" if value >= 0.9 else ("negative" if value <= 0.1 else "mixed")
    )
    summary["abs_mean_coefficient"] = summary["mean_coefficient"].abs()
    if include_fold_coefficients and n_repeats == 1 and not frame.empty:
        fold_wide = (
            frame.assign(
                fold_label=frame["fold_index"].map(lambda value: f"fold_{int(value)}_coef")
            )
            .pivot_table(
                index=["model_name", "feature_name"],
                columns="fold_label",
                values="coefficient",
                aggfunc="first",
            )
            .reset_index()
        )
        fold_columns = [column for column in fold_wide.columns if str(column).startswith("fold_")]
        if fold_columns:
            ordered_fold_columns = sorted(
                fold_columns,
                key=lambda value: int(str(value).split("_", 2)[1]),
            )
            fold_wide = fold_wide[["model_name", "feature_name", *ordered_fold_columns]]
            summary = summary.merge(fold_wide, on=["model_name", "feature_name"], how="left")
            summary["cv_of_coef"] = np.where(
                summary["mean_coefficient"].abs() > 1e-12,
                summary["std_coefficient"] / summary["mean_coefficient"].abs(),
                np.nan,
            )
            summary["sign_stable"] = summary["sign_stability"].isin({"positive", "negative"})
            summary = summary.rename(
                columns={
                    "feature_name": "feature",
                    "mean_coefficient": "mean_coef",
                    "std_coefficient": "std_coef",
                }
            )
            preferred = [
                "model_name",
                "feature",
                "mean_coef",
                "std_coef",
                "cv_of_coef",
                "sign_stable",
                *ordered_fold_columns,
                "min_coefficient",
                "max_coefficient",
                "n_fits",
                "positive_fraction",
                "sign_stability",
                "abs_mean_coefficient",
            ]
            available = [column for column in preferred if column in summary.columns]
            return (
                summary[available]
                .sort_values("abs_mean_coefficient", ascending=False)
                .reset_index(drop=True)
            )
    return summary.sort_values("abs_mean_coefficient", ascending=False).reset_index(drop=True)


def build_logistic_convergence_audit(
    scored: pd.DataFrame,
    *,
    model_names: list[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    min_rows: int = 20,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Record convergence diagnostics for repeated CV fits across selected models."""
    _ensure_config_loaded()

    def _audit_model(model_name: str) -> list[dict[str, object]]:
        columns = MODULE_A_FEATURE_SETS[model_name]
        eligible = (
            _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
        )
        eligible["spread_label"] = eligible["spread_label"].astype(int)
        y = eligible["spread_label"].to_numpy(dtype=int)
        if len(eligible) < min_rows or len(np.unique(y)) < 2:
            return [
                {
                    "model_name": model_name,
                    "repeat_index": np.nan,
                    "fold_index": np.nan,
                    "n_train_backbones": int(len(eligible)),
                    "n_train_positive": int(np.sum(y == 1)),
                    "converged": np.nan,
                    "used_pinv": np.nan,
                    "iterations_run": np.nan,
                    "max_abs_delta": np.nan,
                    "status": "skipped_insufficient_label_variation",
                }
            ]

        fit_kwargs = _model_fit_kwargs(model_name)
        sample_weight = _compute_sample_weight(
            eligible,
            mode=_fit_kwarg_mode(fit_kwargs),
            fit_kwargs=fit_kwargs,
        )
        model_rows: list[dict[str, object]] = []
        _, class_counts = np.unique(y, return_counts=True)
        effective_splits = min(max(int(n_splits), 2), int(class_counts.min()))
        for split_index, (_, test_idx) in enumerate(
            _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
        ):
            repeat_index = split_index // effective_splits + 1
            fold_index = split_index % effective_splits + 1
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_weight = sample_weight[train_mask] if sample_weight is not None else None
            train = eligible.loc[train_mask]
            X_train, _ = _prepare_feature_matrices(
                train,
                train,
                columns,
                fit_kwargs=fit_kwargs,
                prepared=True,
            )
            if _model_type_name(fit_kwargs) == "pairwise_rank_logistic":
                X_train_scaled, _, _ = _standardize_fit(X_train)
                pair_X, pair_y, pair_weight = _build_pairwise_rank_dataset(
                    X_train_scaled,
                    y[train_mask],
                    sample_weight=train_weight,
                    fit_kwargs=fit_kwargs,
                )
                _, diagnostics = _fit_logistic_regression_with_diagnostics(
                    pair_X,
                    pair_y,
                    l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
                    max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
                    sample_weight=pair_weight,
                    fit_backend=_fit_backend_name(fit_kwargs),
                )
            else:
                X_train_scaled, _, _ = _standardize_fit(X_train)
                _, diagnostics = _fit_logistic_regression_with_diagnostics(
                    X_train_scaled,
                    y[train_mask],
                    l2=_fit_kwarg_float(fit_kwargs, "l2", 1.0),
                    max_iter=_fit_kwarg_int(fit_kwargs, "max_iter", 100),
                    sample_weight=train_weight,
                    fit_backend=_fit_backend_name(fit_kwargs),
                )
            model_rows.append(
                {
                    "model_name": model_name,
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "n_train_backbones": int(train_mask.sum()),
                    "n_train_positive": int(y[train_mask].sum()),
                    "converged": bool(diagnostics["converged"]),
                    "used_pinv": bool(diagnostics["used_pinv"]),
                    "iterations_run": int(diagnostics["iterations_run"]),
                    "max_abs_delta": float(diagnostics["max_abs_delta"]),
                    "status": "ok",
                }
            )
        return model_rows

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(model_names))
    if jobs > 1 and model_names:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                row_groups = list(executor.map(_audit_model, model_names))
        rows = [row for group in row_groups for row in group]
    else:
        rows = [row for model_name in model_names for row in _audit_model(model_name)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_module_a(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    n_jobs: int | None = 1,
) -> dict[str, ModelResult]:
    """Run the main retrospective modeling stack on the eligible backbone cohort."""
    _ensure_config_loaded()
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)

    results: dict[str, ModelResult] = {}
    y = eligible["spread_label"].to_numpy(dtype=int)
    selected_model_names = (
        list(get_module_a_model_names())
        if model_names is None
        else [str(name) for name in model_names]
    )
    missing_model_names = sorted(set(selected_model_names) - set(MODULE_A_FEATURE_SETS))
    if missing_model_names:
        missing = ", ".join(missing_model_names)
        raise KeyError(f"Unknown Module A model(s): {missing}")

    jobs = _resolve_parallel_jobs(n_jobs, max_tasks=len(selected_model_names))
    fold_groups = _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    if jobs > 1:
        completed: dict[str, ModelResult] = {}
        try:
            with limit_native_threads(1):
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    future_to_name = {
                        executor.submit(
                            _evaluate_model_name_task,
                            scored,
                            name,
                            n_splits,
                            n_repeats,
                            seed,
                            fold_groups,
                        ): name
                        for name in selected_model_names
                    }
                for future, name in future_to_name.items():
                    try:
                        resolved_name, result = future.result()
                    except (
                        ValueError,
                        RuntimeError,
                        KeyError,
                        TypeError,
                        np.linalg.LinAlgError,
                        OSError,
                        MemoryError,
                    ) as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        warnings.warn(
                            (
                                f"Module A model '{name}' failed in the worker path; "
                                "continuing with remaining models."
                            ),
                            stacklevel=2,
                        )
                        resolved_name, result = (
                            name,
                            build_failed_model_result(
                                name,
                                error_message,
                            ),
                        )
                    completed[resolved_name] = result
        except (OSError, PermissionError, RuntimeError):
            warnings.warn(
                (
                    "ThreadPoolExecutor unavailable in this environment; "
                    "falling back to sequential model evaluation."
                ),
                stacklevel=2,
            )
            for name in selected_model_names:
                resolved_name, result = _safe_evaluate_model_name_task(
                    scored,
                    name,
                    n_splits,
                    n_repeats,
                    seed,
                    fold_groups,
                )
                completed[resolved_name] = result
        for name in selected_model_names:
            results[name] = completed[name]
    else:
        for name in selected_model_names:
            resolved_name, result = _safe_evaluate_model_name_task(
                scored,
                name,
                n_splits,
                n_repeats,
                seed,
                fold_groups,
            )
            results[resolved_name] = result

    # Label-permutation null: use the primary model's full feature set and
    # identical fit parameters (L2, sample weights) so the null distribution
    # is comparable to the actual evaluation.
    rng = np.random.default_rng(seed)
    primary_name = get_primary_model_name(list(MODULE_A_FEATURE_SETS.keys()))
    primary_columns = MODULE_A_FEATURE_SETS[primary_name]
    primary_fit_kwargs = _model_fit_kwargs(primary_name)
    primary_eligible = (
        _ensure_feature_columns(scored, primary_columns).loc[scored["spread_label"].notna()].copy()
    )
    primary_eligible["spread_label"] = primary_eligible["spread_label"].astype(int)
    permuted_y = rng.permutation(y)
    permuted_preds, _ = _oof_predictions_from_eligible(
        primary_eligible,
        columns=primary_columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=primary_fit_kwargs,
        y_override=permuted_y,
    )
    results["label_permutation"] = _evaluate_prediction_set(
        "label_permutation",
        permuted_y,
        permuted_preds,
        primary_eligible["backbone_id"],
    )

    random_preds = rng.random(len(y))
    results["random_score_control"] = _evaluate_prediction_set(
        "random_score_control",
        y,
        random_preds,
        eligible["backbone_id"],
    )

    return results
