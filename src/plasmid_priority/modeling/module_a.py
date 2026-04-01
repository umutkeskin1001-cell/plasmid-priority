"""Primary retrospective evaluation for the backbone priority score."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from plasmid_priority.config import build_context


def _load_project_config() -> dict:
    return build_context().config


_project_config = None

import numpy as np

import pandas as pd

from plasmid_priority.validation import (
    average_precision,
    average_precision_enrichment,
    average_precision_lift,
    brier_score,
    bootstrap_intervals,
    positive_prevalence,
    roc_auc_score,
)

_DEFAULT_MODEL_CONFIG = {
    "primary_model_name": "support_synergy_priority",
    "primary_model_fallback": "support_calibrated_priority",
    "conservative_model_name": "bio_clean_priority",
    "feature_sets": {},
    "core_model_names": (),
    "research_model_names": (),
    "ablation_model_names": (),
    "fit_config": {},
    "novelty_specialist": {"features": (), "fit_config": {}},
}


def _get_model_config() -> dict:
    global _project_config
    if _project_config is None:
        _project_config = _load_project_config()
    models = _project_config.get("models", {}) if isinstance(_project_config, dict) else {}
    if not isinstance(models, dict):
        return _DEFAULT_MODEL_CONFIG
    return {
        "primary_model_name": models.get("primary_model_name", _DEFAULT_MODEL_CONFIG["primary_model_name"]),
        "primary_model_fallback": models.get("primary_model_fallback", _DEFAULT_MODEL_CONFIG["primary_model_fallback"]),
        "conservative_model_name": models.get("conservative_model_name", _DEFAULT_MODEL_CONFIG["conservative_model_name"]),
        "feature_sets": models.get("feature_sets", _DEFAULT_MODEL_CONFIG["feature_sets"]),
        "core_model_names": tuple(models.get("core_model_names", _DEFAULT_MODEL_CONFIG["core_model_names"])),
        "research_model_names": tuple(models.get("research_model_names", _DEFAULT_MODEL_CONFIG["research_model_names"])),
        "ablation_model_names": tuple(models.get("ablation_model_names", _DEFAULT_MODEL_CONFIG["ablation_model_names"])),
        "fit_config": models.get("fit_config", _DEFAULT_MODEL_CONFIG["fit_config"]),
        "novelty_specialist": models.get("novelty_specialist", _DEFAULT_MODEL_CONFIG["novelty_specialist"]),
    }


def _config_snapshot() -> dict:
    config = _get_model_config()
    novelty = config.get("novelty_specialist", {}) if isinstance(config.get("novelty_specialist"), dict) else {}
    return {
        "PRIMARY_MODEL_NAME": config["primary_model_name"],
        "PRIMARY_MODEL_FALLBACK": config["primary_model_fallback"],
        "CONSERVATIVE_MODEL_NAME": config["conservative_model_name"],
        "MODULE_A_FEATURE_SETS": config["feature_sets"],
        "CORE_MODEL_NAMES": tuple(config["core_model_names"]),
        "RESEARCH_MODEL_NAMES": tuple(config["research_model_names"]),
        "ABLATION_MODEL_NAMES": tuple(config["ablation_model_names"]),
        "MODEL_FIT_CONFIG": config["fit_config"],
        "NOVELTY_SPECIALIST_FEATURES": novelty.get("features", ()),
        "NOVELTY_SPECIALIST_FIT_CONFIG": novelty.get("fit_config", {}),
    }


_CONFIG_LOADED = False

PRIMARY_MODEL_NAME = _DEFAULT_MODEL_CONFIG["primary_model_name"]
PRIMARY_MODEL_FALLBACK = _DEFAULT_MODEL_CONFIG["primary_model_fallback"]
CONSERVATIVE_MODEL_NAME = _DEFAULT_MODEL_CONFIG["conservative_model_name"]
MODULE_A_FEATURE_SETS = _DEFAULT_MODEL_CONFIG["feature_sets"]
CORE_MODEL_NAMES = _DEFAULT_MODEL_CONFIG["core_model_names"]
RESEARCH_MODEL_NAMES = _DEFAULT_MODEL_CONFIG["research_model_names"]
ABLATION_MODEL_NAMES = _DEFAULT_MODEL_CONFIG["ablation_model_names"]
MODEL_FIT_CONFIG = _DEFAULT_MODEL_CONFIG["fit_config"]
NOVELTY_SPECIALIST_FEATURES = _DEFAULT_MODEL_CONFIG["novelty_specialist"]["features"]
NOVELTY_SPECIALIST_FIT_CONFIG = _DEFAULT_MODEL_CONFIG["novelty_specialist"]["fit_config"]


def _ensure_config_loaded() -> None:
    global _CONFIG_LOADED
    global PRIMARY_MODEL_NAME
    global PRIMARY_MODEL_FALLBACK
    global CONSERVATIVE_MODEL_NAME
    global MODULE_A_FEATURE_SETS
    global CORE_MODEL_NAMES
    global RESEARCH_MODEL_NAMES
    global ABLATION_MODEL_NAMES
    global MODEL_FIT_CONFIG
    global NOVELTY_SPECIALIST_FEATURES
    global NOVELTY_SPECIALIST_FIT_CONFIG

    if _CONFIG_LOADED:
        return
    config = _config_snapshot()
    PRIMARY_MODEL_NAME = config["PRIMARY_MODEL_NAME"]
    PRIMARY_MODEL_FALLBACK = config["PRIMARY_MODEL_FALLBACK"]
    CONSERVATIVE_MODEL_NAME = config["CONSERVATIVE_MODEL_NAME"]
    MODULE_A_FEATURE_SETS = config["MODULE_A_FEATURE_SETS"]
    CORE_MODEL_NAMES = config["CORE_MODEL_NAMES"]
    RESEARCH_MODEL_NAMES = config["RESEARCH_MODEL_NAMES"]
    ABLATION_MODEL_NAMES = config["ABLATION_MODEL_NAMES"]
    MODEL_FIT_CONFIG = config["MODEL_FIT_CONFIG"]
    NOVELTY_SPECIALIST_FEATURES = config["NOVELTY_SPECIALIST_FEATURES"]
    NOVELTY_SPECIALIST_FIT_CONFIG = config["NOVELTY_SPECIALIST_FIT_CONFIG"]
    _CONFIG_LOADED = True


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame


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
    return 1.0 / (1.0 + np.exp(-values))


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use robust scaling so long-tailed backbone counts do not dominate optimization.
    median = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (X - median) / iqr, median, iqr


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def _fit_logistic_regression_with_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    # l2 parameter in this codebase maps to alpha in ridge regression.
    # sklearn LogisticRegression uses C = 1 / alpha
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
    beta = np.concatenate([last_clf.intercept_, last_clf.coef_[0]])

    diagnostics = {
        "converged": len(last_convergence_warnings) == 0,
        "used_pinv": False,
        "iterations_run": int(np.max(last_clf.n_iter_)),
        "max_abs_delta": 0.0 if len(last_convergence_warnings) == 0 else float("nan"),
        "effective_max_iter": last_effective_max_iter,
        "solver": last_solver,
    }
    return beta, diagnostics


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    beta, diagnostics = _fit_logistic_regression_with_diagnostics(
        X,
        y,
        l2=l2,
        max_iter=max_iter,
        sample_weight=sample_weight,
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


def _predict_calibrated(X: np.ndarray, beta: np.ndarray, predictor: _LinearPredictor) -> np.ndarray:
    return predictor.predict_proba(X, beta)

def _predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    return _sigmoid(X @ beta)


class _LinearPredictor:
    """Keep prediction, auditing, and coefficient tables aligned to one logistic model."""

    def __init__(self):
        pass

    def fit(self, X_raw, beta, y, sample_weight=None):
        return self

    def predict_proba(self, X_raw, beta):
        return _predict_logistic(X_raw, beta)

def _fit_standardized_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _LinearPredictor]:
    X_scaled, mean, std = _standardize_fit(X)
    beta = _fit_logistic_regression(
        X_scaled,
        y,
        l2=l2,
        max_iter=max_iter,
        sample_weight=sample_weight,
    )

    predictor = _LinearPredictor().fit(X_scaled, beta, y, sample_weight=sample_weight)

    return beta, mean, std, predictor

from sklearn.model_selection import RepeatedStratifiedKFold

def _stratified_folds(y: np.ndarray, *, n_splits: int, n_repeats: int, seed: int) -> list[list[np.ndarray]]:
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


def _model_fit_kwargs(model_name: str, overrides: dict[str, object] | None = None) -> dict[str, object]:
    _ensure_config_loaded()
    defaults = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    defaults.update(MODEL_FIT_CONFIG.get(model_name, {}))
    if overrides:
        defaults.update(overrides)
    return defaults


def _masked_percentile_rank(values: pd.Series, cohort_mask: pd.Series | None = None) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if cohort_mask is None:
        cohort_mask = pd.Series(True, index=values.index, dtype=bool)
    effective_mask = cohort_mask.fillna(False).astype(bool) & values.notna()
    if effective_mask.any():
        result.loc[effective_mask] = values.loc[effective_mask].rank(method="average", pct=True)
    return result


def _knownness_score_series(frame: pd.DataFrame, *, cohort_mask: pd.Series | None = None) -> pd.Series:
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
    return (
        member_rank
        + country_rank
        + source_rank
    ) / 3.0


def _stable_quantile_labels(
    values: pd.Series,
    *,
    q: int,
    label_prefix: str = "q",
) -> tuple[pd.Series, int]:
    labels = pd.Series(pd.NA, index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric.notna().to_numpy()
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
    labels.iloc[np.flatnonzero(valid_mask)] = mapped.astype(str).to_numpy(dtype=object)
    return labels, n_bins


def _compute_sample_weight(eligible: pd.DataFrame, *, mode: str | None) -> np.ndarray | None:
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
                weights *= np.asarray(
                    [1.0 / max(float(counts.get(source, 1.0)), 1.0) for source in dominant_source],
                    dtype=float,
                )
            continue
        if token == "class_balanced":
            labels = eligible["spread_label"].fillna(0).astype(int)
            counts = labels.value_counts()
            if not counts.empty and len(counts) >= 2:
                weights *= np.asarray(
                    [1.0 / max(float(counts.get(label, 1.0)), 1.0) for label in labels],
                    dtype=float,
                )
            continue
        if token == "magnitude_weighted":
            magnitude = eligible.get("n_new_countries", pd.Series(1.0, index=eligible.index)).fillna(1.0).to_numpy(dtype=float)
            weights *= np.log1p(magnitude + 1.0)
            continue
        if token == "knownness_balanced":
            knownness = _knownness_score_series(eligible)
            quantile_labels, n_bins = _stable_quantile_labels(knownness, q=4)
            if n_bins >= 2:
                counts = quantile_labels.value_counts()
                if not counts.empty:
                    weights *= np.asarray(
                        [1.0 / max(float(counts.get(label, 1.0)), 1.0) for label in quantile_labels],
                        dtype=float,
                    )
            continue
        raise ValueError(f"Unsupported sample_weight_mode: {mode}")
    return weights / max(weights.mean(), 1e-6)


def _knownness_design_matrix(frame: pd.DataFrame) -> np.ndarray:
    member = frame.get("log1p_member_count_train", pd.Series(0.0, index=frame.index)).fillna(0.0).to_numpy(dtype=float)
    country = frame.get("log1p_n_countries_train", pd.Series(0.0, index=frame.index)).fillna(0.0).to_numpy(dtype=float)
    source = frame.get("refseq_share_train", pd.Series(0.0, index=frame.index)).fillna(0.0).to_numpy(dtype=float)
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
    alpha: float = 1.0,
) -> np.ndarray:
    Z = _knownness_design_matrix(train)
    X = _ensure_feature_columns(train, columns)[columns].fillna(0.0).to_numpy(dtype=float)
    penalty = np.eye(Z.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = Z.T @ Z + penalty
    rhs = Z.T @ X
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _apply_knownness_residualizer(
    frame: pd.DataFrame,
    columns: list[str],
    coefficients: np.ndarray,
) -> np.ndarray:
    working = _ensure_feature_columns(frame, columns)
    X = working[columns].fillna(0.0).to_numpy(dtype=float)
    Z = _knownness_design_matrix(working)
    return X - (Z @ coefficients)


def _prepare_feature_matrices(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_kwargs = fit_kwargs or {}
    preprocess_mode = str(fit_kwargs.get("preprocess_mode", "none") or "none").strip().lower()
    if preprocess_mode in ("none", ""):
        train_raw = _ensure_feature_columns(train, columns)[columns].to_numpy(dtype=float)
        score_raw = _ensure_feature_columns(score, columns)[columns].to_numpy(dtype=float)

        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_matrix = imputer.fit_transform(train_raw)
        score_matrix = imputer.transform(score_raw)
        train_matrix = np.nan_to_num(train_matrix, nan=0.0)
        score_matrix = np.nan_to_num(score_matrix, nan=0.0)
        return train_matrix, score_matrix
    if preprocess_mode == "knownness_residualized":
        coefficients = _fit_knownness_residualizer(
            train,
            columns,
            alpha=float(fit_kwargs.get("preprocess_alpha", 1.0)),
        )
        return (
            _apply_knownness_residualizer(train, columns, coefficients),
            _apply_knownness_residualizer(score, columns, coefficients),
        )
    raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")


def _oof_predictions_from_eligible(
    eligible: pd.DataFrame,
    *,
    columns: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    fit_kwargs: dict[str, object] | None = None,
    y_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fit_kwargs = fit_kwargs or {}
    y = (
        np.asarray(y_override, dtype=int)
        if y_override is not None
        else eligible["spread_label"].fillna(0).astype(int).to_numpy(dtype=int)
    )
    preds = np.zeros(len(eligible), dtype=float)
    counts = np.zeros(len(eligible), dtype=float)
    for fold_indices in _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed):
        for test_idx in fold_indices:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train = eligible.loc[train_mask].copy()
            test = eligible.iloc[test_idx].copy()
            X_train, X_test = _prepare_feature_matrices(
                train,
                test,
                columns,
                fit_kwargs=fit_kwargs,
            )
            train_weight = _compute_sample_weight(train, mode=fit_kwargs.get("sample_weight_mode"))
            X_train_scaled, mean, std = _standardize_fit(X_train)
            X_test_scaled = _standardize_apply(X_test, mean, std)
            beta = _fit_logistic_regression(
                X_train_scaled,
                y[train_mask],
                l2=float(fit_kwargs.get("l2", 1.0)),
                max_iter=int(fit_kwargs.get("max_iter", 100)),
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
    return preds / counts, y


def _evaluate_prediction_set(
    name: str,
    y: np.ndarray,
    preds: np.ndarray,
    index: pd.Index,
    *,
    include_ci: bool = True,
) -> ModelResult:
    prevalence = positive_prevalence(y)
    ap = average_precision(y, preds)
    precision_at_10, recall_at_10 = _top_k_precision_recall(y, preds, top_k=10)
    precision_at_25, recall_at_25 = _top_k_precision_recall(y, preds, top_k=25)
    metrics = {
        "roc_auc": roc_auc_score(y, preds),
        "average_precision": ap,
        "positive_prevalence": prevalence,
        "average_precision_lift": average_precision_lift(y, preds),
        "average_precision_enrichment": average_precision_enrichment(y, preds),
        "brier_score": brier_score(y, preds),
        "precision_at_top_10": precision_at_10,
        "recall_at_top_10": recall_at_10,
        "precision_at_top_25": precision_at_25,
        "recall_at_top_25": recall_at_25,
        "n_backbones": int(len(y)),
        "n_positive": int((np.asarray(y, dtype=int) == 1).sum()),
    }
    if include_ci:
        intervals = bootstrap_intervals(
            y,
            preds,
            {
                "roc_auc": roc_auc_score,
                "average_precision": average_precision,
                "brier_score": brier_score,
            },
        )
        metrics["roc_auc_ci_lower"] = intervals["roc_auc"]["lower"]
        metrics["roc_auc_ci_upper"] = intervals["roc_auc"]["upper"]
        metrics["average_precision_ci_lower"] = intervals["average_precision"]["lower"]
        metrics["average_precision_ci_upper"] = intervals["average_precision"]["upper"]
        metrics["brier_score_ci_lower"] = intervals["brier_score"]["lower"]
        metrics["brier_score_ci_upper"] = intervals["brier_score"]["upper"]
    predictions = pd.DataFrame(
        {
            "backbone_id": index.astype(str),
            "oof_prediction": preds,
            "spread_label": y,
            "visibility_expansion_label": y,
        }
    )
    return ModelResult(name=name, metrics=metrics, predictions=predictions)


def get_primary_model_name(model_names: list[str] | pd.Series | set[str] | tuple[str, ...]) -> str:
    """Resolve the preferred benchmark model while preserving backward compatibility."""
    names = {str(name) for name in model_names}
    if PRIMARY_MODEL_NAME in names:
        return PRIMARY_MODEL_NAME
    if PRIMARY_MODEL_FALLBACK in names:
        return PRIMARY_MODEL_FALLBACK
    return sorted(names)[0]


def get_conservative_model_name(model_names: list[str] | pd.Series | set[str] | tuple[str, ...]) -> str:
    """Resolve the most conservative feature-light benchmark model."""
    names = {str(name) for name in model_names}
    if CONSERVATIVE_MODEL_NAME in names:
        return CONSERVATIVE_MODEL_NAME
    if "T_plus_H_plus_A" in names:
        return "T_plus_H_plus_A"
    return get_primary_model_name(names)


def _ensure_feature_columns(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    working = scored.copy()
    if "H_specialization_norm" in columns and "H_specialization_norm" not in working.columns:
        if "H_breadth_norm" in working.columns:
            working["H_specialization_norm"] = (
                1.0 - pd.to_numeric(working["H_breadth_norm"], errors="coerce").fillna(0.0)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_specialization_norm"] = 0.0
    if "H_augmented_specialization_norm" in columns and "H_augmented_specialization_norm" not in working.columns:
        if "H_augmented_norm" in working.columns:
            working["H_augmented_specialization_norm"] = (
                1.0 - pd.to_numeric(working["H_augmented_norm"], errors="coerce").fillna(0.0)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_augmented_specialization_norm"] = 0.0
    if "H_phylogenetic_specialization_norm" in columns and "H_phylogenetic_specialization_norm" not in working.columns:
        if "H_phylogenetic_norm" in working.columns:
            working["H_phylogenetic_specialization_norm"] = (
                1.0 - pd.to_numeric(working["H_phylogenetic_norm"], errors="coerce").fillna(0.0)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_phylogenetic_specialization_norm"] = 0.0
    for column in columns:
        if column not in working.columns:
            working[column] = 0.0
    return working


def assert_feature_columns_present(
    scored: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | set[str],
    *,
    label: str,
) -> None:
    """Fail loudly when engineered score columns are stale or missing."""
    available = set(scored.columns.astype(str))
    derivable_sources = {
        "H_specialization_norm": "H_breadth_norm",
        "H_augmented_specialization_norm": "H_augmented_norm",
        "H_phylogenetic_specialization_norm": "H_phylogenetic_norm",
    }
    missing: list[str] = []
    for column in dict.fromkeys(str(column) for column in columns):
        if column in available:
            continue
        source = derivable_sources.get(column)
        if source is not None and source in available:
            continue
        missing.append(column)
    if missing:
        formatted = ", ".join(f"`{column}`" for column in missing)
        raise ValueError(
            f"{label} is missing required scored feature columns: {formatted}. "
            "Rerun `python3 scripts/15_normalize_and_score.py` before downstream modeling."
        )


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
    working["member_rank_norm"] = _masked_percentile_rank(working["log1p_member_count_train"], cohort_mask=cohort_mask)
    working["country_rank_norm"] = _masked_percentile_rank(working["log1p_n_countries_train"], cohort_mask=cohort_mask)
    working["source_rank_norm"] = _masked_percentile_rank(working["refseq_share_train"], cohort_mask=cohort_mask)
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
        working["knownness_quartile"] = pd.Series(pd.NA, index=working.index, dtype=object)
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
    train = _ensure_feature_columns(scored_train, columns).loc[scored_train["spread_label"].notna()].copy()
    test = _ensure_feature_columns(scored_test, columns).loc[scored_test["spread_label"].notna()].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["backbone_id", "spread_label", "prediction"])

    train["spread_label"] = train["spread_label"].astype(int)
    test["spread_label"] = test["spread_label"].astype(int)
    if train["spread_label"].nunique() < 2:
        return pd.DataFrame(columns=["backbone_id", "spread_label", "prediction"])

    y_train = train["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    X_train, X_test = _prepare_feature_matrices(train, test, columns, fit_kwargs=fit_kwargs)
    sample_weight = _compute_sample_weight(train, mode=fit_kwargs.get("sample_weight_mode"))
    beta, mean, std, calibrator = _fit_standardized_model(
        X_train,
        y_train,
        l2=float(fit_kwargs.get("l2", l2)),
        max_iter=int(fit_kwargs.get("max_iter", max_iter)),
        sample_weight=sample_weight,
    )
    preds = _predict_calibrated(_standardize_apply(X_test, mean, std), beta, calibrator)
    return pd.DataFrame(
        {
            "backbone_id": test["backbone_id"].astype(str).tolist(),
            "spread_label": test["spread_label"].astype(int).tolist(),
            "prediction": preds.tolist(),
        }
    )


def fit_full_model_predictions(
    scored: pd.DataFrame,
    *,
    model_name: str,
    l2: float = 1.0,
    max_iter: int = 100,
    fit_config: dict[str, object] | None = None,
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
    X_train, X_all = _prepare_feature_matrices(train, all_rows, columns, fit_kwargs=fit_kwargs)
    sample_weight = _compute_sample_weight(train, mode=fit_kwargs.get("sample_weight_mode"))
    beta, mean, std, calibrator = _fit_standardized_model(
        X_train,
        y_train,
        l2=float(fit_kwargs.get("l2", l2)),
        max_iter=int(fit_kwargs.get("max_iter", max_iter)),
        sample_weight=sample_weight,
    )
    preds = _predict_calibrated(_standardize_apply(X_all, mean, std), beta, calibrator)
    return pd.DataFrame(
        {
            "backbone_id": all_rows["backbone_id"].astype(str).tolist(),
            "prediction": preds.tolist(),
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
) -> ModelResult:
    """Evaluate a single named model on the eligible cohort using OOF predictions."""
    _ensure_config_loaded()
    columns = MODULE_A_FEATURE_SETS[model_name]
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    preds, y = _oof_predictions_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
    )
    return _evaluate_prediction_set(
        model_name,
        y,
        preds,
        eligible["backbone_id"],
        include_ci=include_ci,
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
    fit_kwargs = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    if fit_config:
        fit_kwargs.update(fit_config)
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    preds, y = _oof_predictions_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
    )
    return _evaluate_prediction_set(
        label,
        y,
        preds,
        eligible["backbone_id"],
        include_ci=include_ci,
    )


def fit_feature_columns_predictions(
    scored_train: pd.DataFrame,
    scored_score: pd.DataFrame,
    *,
    columns: list[str],
    fit_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fit an arbitrary feature set on one cohort and score another cohort."""
    train = _ensure_feature_columns(scored_train, columns).loc[scored_train["spread_label"].notna()].copy()
    score = _ensure_feature_columns(scored_score, columns).copy()
    if train.empty or score.empty:
        return pd.DataFrame(columns=["backbone_id", "prediction"])
    train["spread_label"] = train["spread_label"].astype(int)
    if train["spread_label"].nunique() < 2:
        return pd.DataFrame(columns=["backbone_id", "prediction"])

    y_train = train["spread_label"].to_numpy(dtype=int)
    fit_kwargs = {"l2": 1.0, "max_iter": 100, "sample_weight_mode": None}
    if fit_config:
        fit_kwargs.update(fit_config)
    X_train, X_score = _prepare_feature_matrices(train, score, columns, fit_kwargs=fit_kwargs)
    sample_weight = _compute_sample_weight(train, mode=fit_kwargs.get("sample_weight_mode"))
    beta, mean, std, calibrator = _fit_standardized_model(
        X_train,
        y_train,
        l2=float(fit_kwargs.get("l2", 1.0)),
        max_iter=int(fit_kwargs.get("max_iter", 100)),
        sample_weight=sample_weight,
    )
    preds = _predict_calibrated(_standardize_apply(X_score, mean, std), beta, calibrator)
    return pd.DataFrame(
        {
            "backbone_id": score["backbone_id"].astype(str).tolist(),
            "prediction": preds.tolist(),
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
) -> pd.DataFrame:
    """Measure how much OOF AUC drops when each feature is removed from a model."""
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    full_preds, y = _oof_predictions_from_eligible(
        eligible,
        columns=columns,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        fit_kwargs=fit_kwargs,
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

    for feature_rank, feature_name in enumerate(columns, start=1):
        reduced_columns = [column for column in columns if column != feature_name]
        reduced_preds, _ = _oof_predictions_from_eligible(
            eligible,
            columns=reduced_columns,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            fit_kwargs=fit_kwargs,
        )
        reduced_auc = roc_auc_score(y, reduced_preds)
        rows.append(
            {
                "model_name": model_name,
                "feature_name": feature_name,
                "feature_rank": feature_rank,
                "roc_auc_without_feature": reduced_auc,
                "roc_auc_drop_vs_full": full_auc - reduced_auc,
                "average_precision_without_feature": average_precision(y, reduced_preds),
                "brier_without_feature": brier_score(y, reduced_preds),
                "n_backbones": int(len(eligible)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["roc_auc_drop_vs_full", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


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
    X, _ = _prepare_feature_matrices(eligible, eligible, columns, fit_kwargs=fit_kwargs)
    sample_weight = _compute_sample_weight(eligible, mode=fit_kwargs.get("sample_weight_mode"))
    beta, _, _ = _fit_standardized_model(
        X,
        y,
        l2=float(fit_kwargs.get("l2", l2)),
        max_iter=int(fit_kwargs.get("max_iter", max_iter)),
        sample_weight=sample_weight,
    )
    coefficients = beta[1:]
    frame = pd.DataFrame(
        {
            "model_name": model_name,
            "feature_name": columns,
            "coefficient": coefficients,
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
) -> pd.DataFrame:
    """Estimate coefficient stability across repeated stratified training folds."""
    eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
    eligible["spread_label"] = eligible["spread_label"].astype(int)
    y = eligible["spread_label"].to_numpy(dtype=int)
    fit_kwargs = _model_fit_kwargs(model_name, fit_config)
    sample_weight = _compute_sample_weight(eligible, mode=fit_kwargs.get("sample_weight_mode"))
    rows = []
    for repeat_index, fold_indices in enumerate(
        _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed),
        start=1,
    ):
        for fold_index, test_idx in enumerate(fold_indices, start=1):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_weight = sample_weight[train_mask] if sample_weight is not None else None
            train = eligible.loc[train_mask].copy()
            X_train, _ = _prepare_feature_matrices(train, train, columns, fit_kwargs=fit_kwargs)
            beta, _, _ = _fit_standardized_model(
                X_train,
                y[train_mask],
                l2=float(fit_kwargs.get("l2", l2)),
                max_iter=int(fit_kwargs.get("max_iter", max_iter)),
                sample_weight=train_weight,
            )
            for feature_name, coefficient in zip(columns, beta[1:]):
                rows.append(
                    {
                        "model_name": model_name,
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                        "feature_name": feature_name,
                        "coefficient": float(coefficient),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model_name", "feature_name"], as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
            n_fits=("coefficient", "size"),
            positive_fraction=("coefficient", lambda values: float((values > 0).mean())),
        )
    )
    summary["sign_stability"] = summary["positive_fraction"].map(
        lambda value: "positive" if value >= 0.9 else ("negative" if value <= 0.1 else "mixed")
    )
    summary["abs_mean_coefficient"] = summary["mean_coefficient"].abs()
    return summary.sort_values("abs_mean_coefficient", ascending=False).reset_index(drop=True)


def build_logistic_convergence_audit(
    scored: pd.DataFrame,
    *,
    model_names: list[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Record convergence diagnostics for repeated CV fits across selected models."""
    _ensure_config_loaded()
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        columns = MODULE_A_FEATURE_SETS[model_name]
        eligible = _ensure_feature_columns(scored, columns).loc[scored["spread_label"].notna()].copy()
        eligible["spread_label"] = eligible["spread_label"].astype(int)
        y = eligible["spread_label"].to_numpy(dtype=int)
        if len(eligible) < min_rows or len(np.unique(y)) < 2:
            rows.append(
                {
                    "model_name": model_name,
                    "repeat_index": pd.NA,
                    "fold_index": pd.NA,
                    "n_train_backbones": int(len(eligible)),
                    "n_train_positive": int(np.sum(y == 1)),
                    "converged": pd.NA,
                    "used_pinv": pd.NA,
                    "iterations_run": pd.NA,
                    "max_abs_delta": pd.NA,
                    "status": "skipped_insufficient_label_variation",
                }
            )
            continue

        fit_kwargs = _model_fit_kwargs(model_name)
        sample_weight = _compute_sample_weight(eligible, mode=fit_kwargs.get("sample_weight_mode"))
        for repeat_index, fold_indices in enumerate(
            _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed),
            start=1,
        ):
            for fold_index, test_idx in enumerate(fold_indices, start=1):
                train_mask = np.ones(len(y), dtype=bool)
                train_mask[test_idx] = False
                train_weight = sample_weight[train_mask] if sample_weight is not None else None
                train = eligible.loc[train_mask].copy()
                X_train, _ = _prepare_feature_matrices(train, train, columns, fit_kwargs=fit_kwargs)
                X_train_scaled, _, _ = _standardize_fit(X_train)
                _, diagnostics = _fit_logistic_regression_with_diagnostics(
                    X_train_scaled,
                    y[train_mask],
                    l2=float(fit_kwargs.get("l2", 1.0)),
                    max_iter=int(fit_kwargs.get("max_iter", 100)),
                    sample_weight=train_weight,
                )
                rows.append(
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

    for name in selected_model_names:
        results[name] = evaluate_model_name(
            scored,
            model_name=name,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
        )

    # Label-permutation null: use the primary model's full feature set and
    # identical fit parameters (L2, sample weights) so the null distribution
    # is comparable to the actual evaluation.
    rng = np.random.default_rng(seed)
    primary_name = get_primary_model_name(MODULE_A_FEATURE_SETS.keys())
    primary_columns = MODULE_A_FEATURE_SETS[primary_name]
    primary_fit_kwargs = _model_fit_kwargs(primary_name)
    primary_eligible = _ensure_feature_columns(scored, primary_columns).loc[scored["spread_label"].notna()].copy()
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
