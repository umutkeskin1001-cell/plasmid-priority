"""Validation metrics implemented without third-party ML dependencies."""

from __future__ import annotations

import math

import numpy as np


def _calibration_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return []
    order = np.argsort(y_score)
    return [
        indices for indices in np.array_split(order, min(n_bins, len(y_true))) if len(indices) > 0
    ]


def _bootstrap_index_batches(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_bootstrap: int,
    batch_size: int = 128,
):
    remaining = max(int(n_bootstrap), 0)
    while remaining > 0:
        current = min(batch_size, remaining)
        yield rng.integers(0, n_samples, size=(current, n_samples))
        remaining -= current


def _batched_metric_scores(
    sample_y: np.ndarray,
    sample_score: np.ndarray,
    metric_fn,
) -> np.ndarray:
    if metric_fn is brier_score:
        return np.asarray(
            np.mean((sample_y.astype(float) - sample_score.astype(float)) ** 2, axis=1),
            dtype=float,
        )
    if metric_fn is positive_prevalence:
        return np.asarray(np.mean(sample_y.astype(int) == 1, axis=1), dtype=float)
    scores = np.empty(sample_y.shape[0], dtype=float)
    for idx in range(sample_y.shape[0]):
        scores[idx] = float(metric_fn(sample_y[idx], sample_score[idx]))
    return scores


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
    """Compute tie-invariant average precision using score-threshold groups."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return 0.0

    positives = int((y_true == 1).sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    y_score_sorted = np.where(np.isnan(y_score[order]), -np.inf, y_score[order])

    group_starts = np.r_[0, np.flatnonzero(y_score_sorted[1:] != y_score_sorted[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(y_sorted)]

    ap = 0.0
    cumulative_true_positives = 0
    for start, end in zip(group_starts, group_ends, strict=True):
        group = y_sorted[start:end]
        group_true_positives = int((group == 1).sum())
        if group_true_positives == 0:
            continue
        cumulative_true_positives += group_true_positives
        precision = cumulative_true_positives / end
        ap += precision * (group_true_positives / positives)
    return float(ap)


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


def log_loss(y_true: np.ndarray, y_score: np.ndarray, *, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    if len(y_true) == 0:
        return float("nan")
    loss = -(y_true * np.log(y_score) + (1.0 - y_true) * np.log(1.0 - y_score))
    return float(np.mean(loss))


def calibration_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rows: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(_calibration_bins(y_true, y_score, n_bins=n_bins), start=1):
        rows.append(
            {
                "bin_index": int(bin_index),
                "n_obs": int(len(indices)),
                "predicted_mean": float(np.mean(y_score[indices])),
                "observed_rate": float(np.mean(y_true[indices])),
                "score_min": float(np.min(y_score[indices])),
                "score_max": float(np.max(y_score[indices])),
            }
        )
    return rows


def brier_decomposition(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
        }
    bins = _calibration_bins(y_true, y_score, n_bins=n_bins)
    base_rate = float(np.mean(y_true))
    reliability = 0.0
    resolution = 0.0
    total = max(len(y_true), 1)
    for indices in bins:
        observed = float(np.mean(y_true[indices]))
        predicted = float(np.mean(y_score[indices]))
        frac = len(indices) / total
        reliability += frac * (predicted - observed) ** 2
        resolution += frac * (observed - base_rate) ** 2
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(base_rate * (1.0 - base_rate)),
    }


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, *, k: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    k = max(1, min(int(k), len(y_true)))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    ranked = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float(np.sum(ranked * discounts))
    ideal = np.sort(y_true)[::-1][:k]
    ideal_dcg = float(np.sum(ideal * discounts))
    if ideal_dcg <= 0.0:
        return 0.0
    return float(dcg / ideal_dcg)


def novelty_adjusted_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    knownness_score: np.ndarray,
    *,
    gamma: float = 2.0,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    knownness_score = np.clip(np.asarray(knownness_score, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return 0.0
    positive_mask = y_true == 1
    if not np.any(positive_mask):
        return 0.0

    novelty_weight = np.asarray((1.0 - knownness_score) ** float(gamma), dtype=float)
    weighted_total = float(np.sum(novelty_weight[positive_mask]))
    if weighted_total <= 0.0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = np.where(np.isnan(y_score[order]), -np.inf, y_score[order])
    weight_sorted = novelty_weight[order]

    group_starts = np.r_[0, np.flatnonzero(score_sorted[1:] != score_sorted[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(y_sorted)]

    naap = 0.0
    cumulative_weighted_tp = 0.0
    for start, end in zip(group_starts, group_ends, strict=True):
        group_positive = y_sorted[start:end] == 1
        group_weight = float(np.sum(weight_sorted[start:end][group_positive]))
        if group_weight <= 0.0:
            continue
        cumulative_weighted_tp += group_weight
        precision = cumulative_weighted_tp / end
        naap += precision * (group_weight / weighted_total)
    return float(naap)


def decision_utility_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    thresholds: np.ndarray | list[float] | tuple[float, ...] | None = None,
    true_positive_reward: float = 1.0,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
    true_negative_reward: float = 0.0,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {
            "optimal_threshold": float("nan"),
            "optimal_threshold_cost_per_sample": float("nan"),
            "optimal_threshold_utility_per_sample": float("nan"),
            "optimal_threshold_utility_total": float("nan"),
            "optimal_threshold_true_positive": float("nan"),
            "optimal_threshold_false_positive": float("nan"),
            "optimal_threshold_false_negative": float("nan"),
            "optimal_threshold_true_negative": float("nan"),
            "optimal_threshold_precision": float("nan"),
            "optimal_threshold_recall": float("nan"),
            "optimal_threshold_positive_rate": float("nan"),
            "utility_grid_size": 0.0,
            "utility_grid_min_threshold": float("nan"),
            "utility_grid_max_threshold": float("nan"),
        }

    clean_scores = np.nan_to_num(y_score, nan=-np.inf)
    if thresholds is None:
        finite_scores = np.unique(np.clip(y_score[np.isfinite(y_score)], 0.0, 1.0))
        grid = np.linspace(0.0, 1.0, 101, dtype=float)
        if finite_scores.size:
            thresholds_array = np.unique(np.concatenate([grid, finite_scores]))
        else:
            thresholds_array = grid
    else:
        thresholds_array = np.unique(np.asarray(thresholds, dtype=float))
    thresholds_array = thresholds_array[np.isfinite(thresholds_array)]
    if thresholds_array.size == 0:
        return {
            "optimal_threshold": float("nan"),
            "optimal_threshold_cost_per_sample": float("nan"),
            "optimal_threshold_utility_per_sample": float("nan"),
            "optimal_threshold_utility_total": float("nan"),
            "optimal_threshold_true_positive": float("nan"),
            "optimal_threshold_false_positive": float("nan"),
            "optimal_threshold_false_negative": float("nan"),
            "optimal_threshold_true_negative": float("nan"),
            "optimal_threshold_precision": float("nan"),
            "optimal_threshold_recall": float("nan"),
            "optimal_threshold_positive_rate": float("nan"),
            "utility_grid_size": 0.0,
            "utility_grid_min_threshold": float("nan"),
            "utility_grid_max_threshold": float("nan"),
        }

    y_true_bool = y_true == 1
    n_samples = len(y_true_bool)
    best: dict[str, float] | None = None
    best_rank: tuple[float, float, float] | None = None
    for threshold in thresholds_array:
        predicted_positive = clean_scores >= float(threshold)
        tp = int(np.sum(y_true_bool & predicted_positive))
        fp = int(np.sum((~y_true_bool) & predicted_positive))
        fn = int(np.sum(y_true_bool & (~predicted_positive)))
        tn = int(np.sum((~y_true_bool) & (~predicted_positive)))
        utility_total = (
            float(true_positive_reward) * tp
            + float(true_negative_reward) * tn
            - float(false_positive_cost) * fp
            - float(false_negative_cost) * fn
        )
        utility_per_sample = utility_total / max(n_samples, 1)
        cost_per_sample = -utility_per_sample
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        positive_rate = float(predicted_positive.mean()) if n_samples > 0 else float("nan")
        candidate = {
            "optimal_threshold": float(threshold),
            "optimal_threshold_cost_per_sample": float(cost_per_sample),
            "optimal_threshold_utility_per_sample": float(utility_per_sample),
            "optimal_threshold_utility_total": float(utility_total),
            "optimal_threshold_true_positive": float(tp),
            "optimal_threshold_false_positive": float(fp),
            "optimal_threshold_false_negative": float(fn),
            "optimal_threshold_true_negative": float(tn),
            "optimal_threshold_precision": float(precision),
            "optimal_threshold_recall": float(recall),
            "optimal_threshold_positive_rate": float(positive_rate),
            "utility_grid_size": float(len(thresholds_array)),
            "utility_grid_min_threshold": float(thresholds_array.min()),
            "utility_grid_max_threshold": float(thresholds_array.max()),
        }
        rank = (utility_per_sample, recall if np.isfinite(recall) else -np.inf, -float(threshold))
        if best is None or best_rank is None or rank > best_rank:
            best = candidate
            best_rank = rank
    return best if best is not None else {}


def weighted_classification_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    target_positive_rate: float | None = None,
) -> float:
    """Cost of a threshold chosen by the best utility grid under asymmetric errors."""
    summary = decision_utility_summary(
        y_true,
        y_score,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    return float(summary.get("optimal_threshold_cost_per_sample", float("nan")))


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
    bins = _calibration_bins(y_true, y_score, n_bins=n_bins)
    total = max(len(y_true), 1)
    error = 0.0
    for indices in bins:
        observed = float(y_true[indices].mean())
        predicted = float(y_score[indices].mean())
        error += abs(observed - predicted) * (len(indices) / total)
    return float(error)


def max_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    bins = _calibration_bins(y_true, y_score, n_bins=n_bins)
    if not bins:
        return float("nan")
    return float(
        max(abs(float(y_true[indices].mean()) - float(y_score[indices].mean())) for indices in bins)
    )


def permutation_pvalue(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_permutations: int = 1000,
    metric_fn=roc_auc_score,
    rng_seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Return the empirical permutation p-value and null metric distribution."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan"), np.array([], dtype=float)
    observed = float(metric_fn(y_true, y_score))
    if np.isnan(observed):
        return float("nan"), np.array([], dtype=float)
    rng = np.random.default_rng(rng_seed)
    null = np.array(
        [
            float(metric_fn(rng.permutation(y_true), y_score))
            for _ in range(max(int(n_permutations), 0))
        ],
        dtype=float,
    )
    valid = null[~np.isnan(null)]
    if valid.size == 0:
        return float("nan"), null
    p = float((1 + np.sum(valid >= observed)) / (valid.size + 1))
    return p, null


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return {
            "statistic": float("nan"),
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        observed = float("nan")
    else:
        observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(x)
    for _ in range(max(int(n_bootstrap), 0)):
        indices = rng.integers(0, n, size=n)
        sample_x = x[indices]
        sample_y = y[indices]
        if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
            continue
        statistic = float(spearmanr(sample_x, sample_y).statistic)
        if np.isfinite(statistic):
            values.append(statistic)
    if not values:
        return {
            "statistic": observed,
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "statistic": observed,
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, 0.025)),
        "upper": float(np.quantile(arr, 0.975)),
    }


def bootstrap_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    metrics = []
    n = len(y_true)
    for indices in _bootstrap_index_batches(rng, n_samples=n, n_bootstrap=n_bootstrap):
        batch_scores = _batched_metric_scores(y_true[indices], y_score[indices], metric_fn)
        valid_scores = batch_scores[~np.isnan(batch_scores)]
        if valid_scores.size:
            metrics.extend(valid_scores.astype(float).tolist())
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
    for indices in _bootstrap_index_batches(rng, n_samples=n, n_bootstrap=n_bootstrap):
        sample_y = y_true[indices]
        sample_score = y_score[indices]
        for name, metric_fn in metric_items:
            batch_scores = _batched_metric_scores(sample_y, sample_score, metric_fn)
            valid_scores = batch_scores[~np.isnan(batch_scores)]
            if valid_scores.size:
                collected[name].extend(valid_scores.astype(float).tolist())
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
    for indices in _bootstrap_index_batches(rng, n_samples=n, n_bootstrap=n_bootstrap):
        sample_y = y_true[indices]
        sample_a = y_score_a[indices]
        sample_b = y_score_b[indices]
        metrics_a = _batched_metric_scores(sample_y, sample_a, metric_fn)
        metrics_b = _batched_metric_scores(sample_y, sample_b, metric_fn)
        batch_deltas = metrics_a - metrics_b
        valid = ~np.isnan(batch_deltas)
        if np.any(valid):
            deltas.extend(batch_deltas[valid].astype(float).tolist())
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
    for indices in _bootstrap_index_batches(rng, n_samples=n, n_bootstrap=n_bootstrap):
        sample_y = y_true[indices]
        sample_a = y_score_a[indices]
        sample_b = y_score_b[indices]
        for name, metric_fn in metric_items:
            metrics_a = _batched_metric_scores(sample_y, sample_a, metric_fn)
            metrics_b = _batched_metric_scores(sample_y, sample_b, metric_fn)
            batch_deltas = metrics_a - metrics_b
            valid = ~np.isnan(batch_deltas)
            if np.any(valid):
                deltas[name].extend(batch_deltas[valid].astype(float).tolist())
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
        return np.asarray((diff > 0).astype(float) + 0.5 * (diff == 0), dtype=float)

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
