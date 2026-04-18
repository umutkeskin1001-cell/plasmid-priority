"""Falsification tests and negative-control audits.

This module provides lightweight falsification checks that verify the model
is not merely fitting arbitrary artifacts. These are audit findings, not
causal claims.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from plasmid_priority.validation.metrics import average_precision, roc_auc_score

ModelEvaluator = Callable[[pd.DataFrame, str], dict[str, float]]


def _default_model_evaluator(_scored: pd.DataFrame, _model_name: str) -> dict[str, float]:
    raise RuntimeError(
        "No model evaluator provided. Pass `model_evaluator=` from the caller "
        "(for example via plasmid_priority.modeling.evaluate_model_name wrapper)."
    )


def _default_known_models() -> set[str]:
    return set()


def build_outcome_permutation_falsification(
    scored: pd.DataFrame,
    model_name: str,
    *,
    n_permutations: int = 100,
    seed: int = 42,
    model_evaluator: ModelEvaluator | None = None,
    known_model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Falsification test: Verify performance collapses when outcome is permuted.

    This test permutes the outcome labels and verifies that model performance
    collapses to near-random levels. If performance remains high under permutation,
    the model may be exploiting leakage or artifacts rather than true signal.

    Args:
        scored: DataFrame with backbone features and spread_label
        model_name: Name of the model to evaluate
        n_permutations: Number of permutation iterations
        seed: Random seed for reproducibility

    Returns:
        DataFrame with falsification test results
    """
    evaluator = model_evaluator or _default_model_evaluator
    model_set = set(known_model_names or _default_known_models())
    if model_set and model_name not in model_set:
        return pd.DataFrame(
            {
                "test_name": ["outcome_permutation_falsification"],
                "model_name": [model_name],
                "status": ["skipped_unknown_model"],
                "n_permutations": [0],
            }
        )

    # Get eligible records
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty or eligible["spread_label"].nunique() < 2:
        return pd.DataFrame(
            {
                "test_name": ["outcome_permutation_falsification"],
                "model_name": [model_name],
                "status": ["skipped_insufficient_labels"],
                "n_permutations": [0],
            }
        )

    # Evaluate on true labels
    try:
        true_metrics = evaluator(eligible, model_name)
    except RuntimeError as exc:
        return pd.DataFrame(
            {
                "test_name": ["outcome_permutation_falsification"],
                "model_name": [model_name],
                "status": ["skipped_missing_model_evaluator"],
                "error_message": [str(exc)],
                "n_permutations": [0],
            }
        )
    true_auc = float(true_metrics.get("roc_auc", float("nan")))
    true_ap = float(true_metrics.get("average_precision", float("nan")))

    # Run permutations
    rng = np.random.default_rng(seed)
    permuted_aucs = []
    permuted_aps = []

    for _i in range(n_permutations):
        permuted = eligible.copy()
        permuted["spread_label"] = rng.permutation(permuted["spread_label"].to_numpy())

        # Skip if permutation resulted in single class
        if permuted["spread_label"].nunique() < 2:
            continue

        try:
            perm_metrics = evaluator(permuted, model_name)
        except RuntimeError:
            continue
        permuted_aucs.append(float(perm_metrics.get("roc_auc", float("nan"))))
        permuted_aps.append(float(perm_metrics.get("average_precision", float("nan"))))

    if not permuted_aucs:
        return pd.DataFrame(
            {
                "test_name": ["outcome_permutation_falsification"],
                "model_name": [model_name],
                "status": ["failed_no_valid_permutations"],
                "n_permutations": [n_permutations],
            }
        )

    permuted_aucs_arr = np.asarray([x for x in permuted_aucs if not np.isnan(x)], dtype=float)
    permuted_aps_arr = np.asarray([x for x in permuted_aps if not np.isnan(x)], dtype=float)

    # Compute statistics
    mean_perm_auc = (
        float(np.mean(permuted_aucs_arr)) if len(permuted_aucs_arr) > 0 else float("nan")
    )
    std_perm_auc = float(np.std(permuted_aucs_arr)) if len(permuted_aucs_arr) > 0 else float("nan")
    mean_perm_ap = float(np.mean(permuted_aps_arr)) if len(permuted_aps_arr) > 0 else float("nan")

    # Collapse metric: how much did AUC drop?
    auc_collapse = (
        true_auc - mean_perm_auc
        if not np.isnan(true_auc) and not np.isnan(mean_perm_auc)
        else float("nan")
    )

    # Empirical p-value: fraction of permutations that achieved >= true AUC
    empirical_p = (
        np.mean(permuted_aucs_arr >= true_auc)
        if not np.isnan(true_auc) and len(permuted_aucs_arr) > 0
        else float("nan")
    )

    # Interpretation
    if np.isnan(auc_collapse):
        interpretation = "inconclusive"
    elif auc_collapse < 0.05:
        interpretation = "warning_minimal_collapse"
    elif empirical_p > 0.05:
        interpretation = "warning_high_perm_overlap"
    else:
        interpretation = "pass_collapses_under_falsification"

    return pd.DataFrame(
        {
            "test_name": ["outcome_permutation_falsification"],
            "model_name": [model_name],
            "status": ["completed"],
            "interpretation": [interpretation],
            "n_permutations": [n_permutations],
            "true_roc_auc": [true_auc],
            "permuted_roc_auc_mean": [mean_perm_auc],
            "permuted_roc_auc_std": [std_perm_auc],
            "auc_collapse_delta": [auc_collapse],
            "empirical_p_value": [empirical_p],
            "true_average_precision": [true_ap],
            "permuted_average_precision_mean": [mean_perm_ap],
        }
    )


def build_label_shuffle_falsification(
    predictions: pd.DataFrame,
    model_names: list[str],
    *,
    n_shuffles: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Falsification test: Compare model to shuffled-label baseline.

    Shuffles labels and compares observed performance to shuffled distribution.

    Args:
        predictions: DataFrame with model predictions
        model_names: Models to test
        n_shuffles: Number of shuffles per model
        seed: Random seed

    Returns:
        DataFrame with shuffle test results
    """
    rows = []
    rng = np.random.default_rng(seed)

    for model_name in model_names:
        model_preds = predictions.loc[predictions["model_name"] == model_name].copy()
        if model_preds.empty or "spread_label" not in model_preds.columns:
            continue

        y_true = model_preds["spread_label"].to_numpy(dtype=int)
        y_scores = model_preds["oof_prediction"].to_numpy(dtype=float)

        if len(np.unique(y_true)) < 2:
            continue

        observed_auc = roc_auc_score(y_true, y_scores)
        observed_ap = average_precision(y_true, y_scores)

        # Shuffle test
        shuffled_aucs = []
        shuffled_aps = []

        for _ in range(n_shuffles):
            y_shuffled = rng.permutation(y_true)
            if len(np.unique(y_shuffled)) < 2:
                continue
            shuffled_aucs.append(roc_auc_score(y_shuffled, y_scores))
            shuffled_aps.append(average_precision(y_shuffled, y_scores))

        if not shuffled_aucs:
            continue

        shuffled_aucs_arr = np.asarray(shuffled_aucs, dtype=float)
        shuffled_aps_arr = np.asarray(shuffled_aps, dtype=float)

        # Empirical p-value
        p_value = float(np.mean(shuffled_aucs_arr >= observed_auc))

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(shuffled_aucs_arr) ** 2 + 0) / 2)  # vs 0 for single obs
        cohens_d = (observed_auc - np.mean(shuffled_aucs_arr)) / (pooled_std + 1e-10)

        rows.append(
            {
                "test_name": "label_shuffle_falsification",
                "model_name": model_name,
                "n_shuffles": n_shuffles,
                "observed_roc_auc": observed_auc,
                "shuffled_roc_auc_mean": np.mean(shuffled_aucs_arr),
                "shuffled_roc_auc_std": np.std(shuffled_aucs_arr),
                "shuffled_roc_auc_q95": np.quantile(shuffled_aucs_arr, 0.95),
                "observed_ap": observed_ap,
                "shuffled_ap_mean": np.mean(shuffled_aps_arr),
                "empirical_p_value": p_value,
                "cohens_d_vs_shuffle": cohens_d,
                "status": "completed",
                "interpretation": (
                    "pass_strong_signal"
                    if p_value < 0.01 and cohens_d > 1.0
                    else "moderate_signal"
                    if p_value < 0.05
                    else "weak_signal_warning"
                ),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def summarize_falsification_findings(
    permutation_result: pd.DataFrame | None,
    shuffle_result: pd.DataFrame | None,
) -> dict[str, object]:
    """Summarize falsification test findings.

    Returns a compact interpretation suitable for audit reports.
    """
    findings: dict[str, object] = {
        "overall_assessment": "unknown",
        "concerns": [],
        "details": {},
    }

    if permutation_result is not None and not permutation_result.empty:
        interp = permutation_result.get("interpretation", pd.Series(["unknown"])).iloc[0]
        auc_collapse_series = permutation_result.get(
            "auc_collapse_delta",
            pd.Series([float("nan")]),
        )
        details = findings["details"]
        if isinstance(details, dict):
            details["outcome_permutation"] = {
                "interpretation": interp,
                "auc_collapse": float(auc_collapse_series.iloc[0]),
            }
        if "warning" in str(interp):
            concerns = findings["concerns"]
            if isinstance(concerns, list):
                concerns.append(f"outcome_permutation: {interp}")

    if shuffle_result is not None and not shuffle_result.empty:
        weak_signals = shuffle_result[
            shuffle_result["interpretation"].str.contains("weak", na=False)
        ]
        if not weak_signals.empty:
            concerns = findings["concerns"]
            if isinstance(concerns, list):
                concerns.append(f"label_shuffle_weak_signal: {len(weak_signals)} models")
        details = findings["details"]
        if isinstance(details, dict):
            details["label_shuffle"] = {
                "n_models_tested": len(shuffle_result),
                "n_passed": int((shuffle_result["interpretation"] == "pass_strong_signal").sum()),
            }

    concerns = findings["concerns"]
    concern_count = len(concerns) if isinstance(concerns, list) else 0
    if concern_count == 0:
        findings["overall_assessment"] = "no_concerns"
    elif concern_count <= 1:
        findings["overall_assessment"] = "minor_concerns"
    else:
        findings["overall_assessment"] = "review_recommended"

    return findings
