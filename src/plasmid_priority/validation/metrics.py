"""Validation metrics implemented without third-party ML dependencies."""

from __future__ import annotations

import math

import numpy as np


def positive_prevalence(y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == 1).mean())


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average_ranks = first_idx + (counts + 1.0) / 2.0
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.repeat(average_ranks, counts)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    positives = max(int((y_true == 1).sum()), 1)
    return float((precision * (y_sorted == 1)).sum() / positives)


def average_precision_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prevalence = positive_prevalence(y_true)
    if np.isnan(prevalence):
        return float("nan")
    return float(average_precision(y_true, y_score) - prevalence)


def average_precision_enrichment(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prevalence = positive_prevalence(y_true)
    if np.isnan(prevalence) or prevalence <= 0.0:
        return float("nan")
    return float(average_precision(y_true, y_score) / prevalence)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_true - y_score) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(y_score)
    bins = np.array_split(order, min(n_bins, len(y_true)))
    total = len(y_true)
    error = 0.0
    for indices in bins:
        if len(indices) == 0:
            continue
        observed = float(y_true[indices].mean())
        predicted = float(y_score[indices].mean())
        error += abs(observed - predicted) * (len(indices) / total)
    return float(error)


def bootstrap_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    metrics = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        score = metric_fn(y_true[indices], y_score[indices])
        if not np.isnan(score):
            metrics.append(float(score))
    if not metrics:
        return {"lower": float("nan"), "upper": float("nan")}
    return {
        "lower": float(np.quantile(metrics, 0.025)),
        "upper": float(np.quantile(metrics, 0.975)),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fns: dict[str, object],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate confidence intervals for multiple metrics in one bootstrap pass."""
    metric_items = [(str(name), fn) for name, fn in metric_fns.items()]
    if not metric_items:
        return {}
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected: dict[str, list[float]] = {name: [] for name, _ in metric_items}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y = y_true[indices]
        sample_score = y_score[indices]
        for name, metric_fn in metric_items:
            score = metric_fn(sample_y, sample_score)
            if not np.isnan(score):
                collected[name].append(float(score))
    intervals: dict[str, dict[str, float]] = {}
    for name, scores in collected.items():
        if not scores:
            intervals[name] = {"lower": float("nan"), "upper": float("nan")}
            continue
        intervals[name] = {
            "lower": float(np.quantile(scores, 0.025)),
            "upper": float(np.quantile(scores, 0.975)),
        }
    return intervals


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        metric_a = metric_fn(y_true[indices], y_score_a[indices])
        metric_b = metric_fn(y_true[indices], y_score_b[indices])
        if np.isnan(metric_a) or np.isnan(metric_b):
            continue
        deltas.append(float(metric_a - metric_b))
    if not deltas:
        return {"delta": float("nan"), "lower": float("nan"), "upper": float("nan")}
    return {
        "delta": float(np.mean(deltas)),
        "lower": float(np.quantile(deltas, 0.025)),
        "upper": float(np.quantile(deltas, 0.975)),
    }


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fns: dict[str, object],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate paired metric deltas for multiple metrics in one bootstrap pass."""
    metric_items = [(str(name), fn) for name, fn in metric_fns.items()]
    if not metric_items:
        return {}
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas: dict[str, list[float]] = {name: [] for name, _ in metric_items}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y = y_true[indices]
        sample_a = y_score_a[indices]
        sample_b = y_score_b[indices]
        for name, metric_fn in metric_items:
            metric_a = metric_fn(sample_y, sample_a)
            metric_b = metric_fn(sample_y, sample_b)
            if np.isnan(metric_a) or np.isnan(metric_b):
                continue
            deltas[name].append(float(metric_a - metric_b))
    summary: dict[str, dict[str, float]] = {}
    for name, values in deltas.items():
        if not values:
            summary[name] = {"delta": float("nan"), "lower": float("nan"), "upper": float("nan")}
            continue
        summary[name] = {
            "delta": float(np.mean(values)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
    return summary


def paired_auc_delong(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict[str, float]:
    """Compute a paired DeLong-style ROC AUC comparison on the same cases.

    This implementation follows the standard U-statistic decomposition used in
    DeLong's covariance estimate while staying dependency-light.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "delta_auc": float("nan"),
            "var_delta": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
        }

    pos_a = y_score_a[pos_mask]
    neg_a = y_score_a[neg_mask]
    pos_b = y_score_b[pos_mask]
    neg_b = y_score_b[neg_mask]

    def _kernel(pos_scores: np.ndarray, neg_scores: np.ndarray) -> np.ndarray:
        diff = pos_scores[:, None] - neg_scores[None, :]
        return (diff > 0).astype(float) + 0.5 * (diff == 0)

    kernel_a = _kernel(pos_a, neg_a)
    kernel_b = _kernel(pos_b, neg_b)
    auc_a = float(kernel_a.mean())
    auc_b = float(kernel_b.mean())

    v10_a = kernel_a.mean(axis=1)
    v10_b = kernel_b.mean(axis=1)
    v01_a = kernel_a.mean(axis=0)
    v01_b = kernel_b.mean(axis=0)

    if n_pos > 1:
        cov_pos = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    else:
        cov_pos = np.zeros((2, 2), dtype=float)
    if n_neg > 1:
        cov_neg = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    else:
        cov_neg = np.zeros((2, 2), dtype=float)

    covariance = cov_pos / max(n_pos, 1) + cov_neg / max(n_neg, 1)
    var_delta = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if not np.isfinite(var_delta) or var_delta <= 0.0:
        z_score = float("nan")
        p_value = float("nan")
    else:
        z_score = float((auc_a - auc_b) / math.sqrt(var_delta))
        p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": float(auc_a - auc_b),
        "var_delta": var_delta,
        "z_score": z_score,
        "p_value": p_value,
    }
