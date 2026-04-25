"""Nested cross-validation with bootstrap CI and Bayesian model comparison.

The evaluator performs outer-fold performance estimation. Inside each outer
train split it can run inner-fold parameter selection when a candidate
parameter grid is available. If no tunable grid exists for the estimator,
the evaluator records explicit no-tuning metadata rather than implying
full nested tuning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_log = logging.getLogger(__name__)


class NestedCVEvaluator:
    """Nested CV evaluator with explicit inner-selection tracing.

    Parameters
    ----------
    outer_cv : int
        Outer folds for performance estimation.
    inner_cv : int
        Inner folds for model selection / HPO.
    scoring : Callable
        Metric to optimize.
    random_state : int
    candidate_param_grid : Sequence[dict[str, object]] | None
        Optional explicit parameter candidates for inner selection.
    """

    def __init__(
        self,
        *,
        outer_cv: int = 5,
        inner_cv: int = 3,
        scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
        random_state: int = 42,
        candidate_param_grid: Sequence[dict[str, object]] | None = None,
    ) -> None:
        self.outer_cv = int(outer_cv)
        self.inner_cv = int(inner_cv)
        self.scoring = scoring or self._default_auc
        self.random_state = int(random_state)
        self.candidate_param_grid = (
            [dict(cfg) for cfg in candidate_param_grid]
            if candidate_param_grid is not None
            else None
        )

    @staticmethod
    def _default_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X), dtype=float)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.reshape(-1)
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    def _resolve_candidate_params(self, base_model: Any) -> list[dict[str, object]]:
        if self.candidate_param_grid is not None:
            return [dict(cfg) for cfg in self.candidate_param_grid] or [{}]
        if not hasattr(base_model, "get_params"):
            return [{}]
        params = base_model.get_params(deep=False)
        if "C" in params:
            return [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
        if "alpha" in params:
            return [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1.0}]
        if "l2" in params:
            return [{"l2": 0.1}, {"l2": 1.0}, {"l2": 10.0}]
        return [{}]

    def _select_inner_params(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model: Any,
        candidate_params: Sequence[dict[str, object]],
        fold_seed: int,
    ) -> tuple[dict[str, object], float, int, str]:
        if self.inner_cv < 2 or len(candidate_params) <= 1:
            return {}, float("nan"), 1, "explicit_no_tuning"

        inner = StratifiedKFold(
            n_splits=self.inner_cv,
            shuffle=True,
            random_state=fold_seed,
        )
        best_params: dict[str, object] = {}
        best_score = float("-inf")
        evaluated_candidates = 0

        for candidate in candidate_params:
            inner_scores: list[float] = []
            try:
                for inner_train_idx, inner_val_idx in inner.split(X_train, y_train):
                    model = clone(base_model)
                    if candidate:
                        model.set_params(**candidate)
                    model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                    y_val_score = self._predict_scores(model, X_train[inner_val_idx])
                    inner_scores.append(self.scoring(y_train[inner_val_idx], y_val_score))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                _log.warning("Inner CV candidate failed (%s): %s", candidate, exc)
                continue
            if not inner_scores:
                continue
            evaluated_candidates += 1
            mean_inner_score = float(np.mean(inner_scores))
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = dict(candidate)

        if evaluated_candidates == 0:
            return {}, float("nan"), 0, "inner_cv_unavailable"
        return best_params, best_score, evaluated_candidates, "inner_cv_tuning"

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Any],
    ) -> dict[str, object]:
        """Run nested CV and return performance + inner-selection metadata."""
        outer = StratifiedKFold(
            n_splits=self.outer_cv, shuffle=True, random_state=self.random_state
        )
        outer_scores: list[float] = []
        all_y_true: list[np.ndarray] = []
        all_y_score: list[np.ndarray] = []
        selection_trace: list[dict[str, object]] = []
        inner_selection_performed = False

        for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            base_model = clone(model_factory())
            candidate_params = self._resolve_candidate_params(base_model)
            (
                selected_params,
                best_inner_score,
                evaluated_candidates,
                selection_mode,
            ) = self._select_inner_params(
                X_train=X_train,
                y_train=y_train,
                base_model=base_model,
                candidate_params=candidate_params,
                fold_seed=self.random_state + fold_idx,
            )
            if selection_mode == "inner_cv_tuning":
                inner_selection_performed = True

            model = clone(base_model)
            if selected_params:
                model.set_params(**selected_params)
            model.fit(X_train, y_train)
            y_score = self._predict_scores(model, X_test)

            score = self.scoring(y_test, y_score)
            outer_scores.append(score)
            all_y_true.append(y_test)
            all_y_score.append(y_score)
            selection_trace.append(
                {
                    "fold_idx": fold_idx,
                    "selection_mode": selection_mode,
                    "selected_params": selected_params,
                    "best_inner_score": best_inner_score,
                    "candidate_count": len(candidate_params),
                    "n_candidates_evaluated": evaluated_candidates,
                },
            )
            _log.info("NestedCV outer fold %d: score=%.4f", fold_idx + 1, score)

        if not outer_scores:
            return {
                "mean_score": float("nan"),
                "std_score": float("nan"),
                "fold_scores": [],
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_samples": 0,
                "selection_mode": "explicit_no_tuning",
                "inner_selection_performed": False,
                "selection_trace": selection_trace,
            }

        # Bootstrap CI on pooled predictions
        y_true_pooled = np.concatenate(all_y_true)
        y_score_pooled = np.concatenate(all_y_score)
        ci = bootstrap_ci(
            y_true_pooled,
            y_score_pooled,
            self.scoring,
            n_bootstrap=1000,
            random_state=self.random_state,
        )
        selection_mode = "inner_cv_tuning" if inner_selection_performed else "explicit_no_tuning"

        return {
            "mean_score": float(np.mean(outer_scores)),
            "std_score": float(np.std(outer_scores)),
            "fold_scores": [float(s) for s in outer_scores],
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "n_samples": len(y_true_pooled),
            "selection_mode": selection_mode,
            "inner_selection_performed": inner_selection_performed,
            "selection_trace": selection_trace,
        }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric(y_true[idx], y_score[idx])
        except ValueError:
            s = 0.5
        scores.append(s)
    arr = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1 - alpha)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def bayesian_model_comparison(
    scores_dict: dict[str, list[float]],
) -> dict[str, object]:
    """Approximate Bayesian model comparison using WAIC-like criterion.

    Each key is a model name, value is a list of fold scores (log-likelihood proxy).
    Returns posterior model probabilities assuming uniform prior.
    """
    # Convert scores to pseudo-log-likelihoods (higher score = better)
    # Use a simple Gaussian approximation per model
    results: dict[str, dict[str, float]] = {}
    loo_scores: dict[str, float] = {}

    for model_name, scores in scores_dict.items():
        arr = np.array(scores)
        n = len(arr)
        mean = arr.mean()
        var = arr.var(ddof=1) if n > 1 else 1e-6
        # WAIC approximation: log p(y | theta_hat) - p_WAIC
        log_lik = n * mean
        p_waic = n * var / 2.0  # effective parameters
        waic = -2.0 * (log_lik - p_waic)
        loo_scores[model_name] = waic
        results[model_name] = {
            "mean_score": float(mean),
            "waic": float(waic),
            "p_waic": float(p_waic),
        }

    # Convert WAIC to weights (lower WAIC = better; use delta WAIC)
    min_waic = min(loo_scores.values())
    delta = {m: w - min_waic for m, w in loo_scores.items()}
    weights = {m: np.exp(-0.5 * d) for m, d in delta.items()}
    total = sum(weights.values())
    for m in weights:
        weights[m] /= total
        results[m]["posterior_prob"] = float(weights[m])

    _log.info(
        "Bayesian model comparison: %s", {m: f"{results[m]['posterior_prob']:.3f}" for m in results}
    )
    return dict(results)
