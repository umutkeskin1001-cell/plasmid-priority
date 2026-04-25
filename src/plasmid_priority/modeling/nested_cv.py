"""Nested cross-validation utilities for module-A style model evaluation.

``nested_cross_validate`` runs an outer stratified CV loop for unbiased
performance estimation. Within each outer-train fold, it can run an inner
CV model-selection step over candidate ``fit_config`` values. The selected
config is then fit once on the full outer-train fold and scored on the
outer-test fold.

If no meaningful tuning grid is available (for example only one candidate
config), the function returns explicit metadata marking that it ran in a
no-tuning compatibility mode.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_log = logging.getLogger(__name__)
_DEFAULT_INNER_L2_GRID = (0.1, 1.0, 10.0)


def nested_cross_validate(
    scored: pd.DataFrame,
    *,
    model_name: str,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    inner_fit_configs: Sequence[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Run nested CV for one model name with explicit inner-selection metadata.

    The outer loop estimates generalization by scoring each held-out fold.
    The inner loop evaluates candidate ``fit_config`` values on the outer
    training fold via ``evaluate_model_name`` and picks the best candidate
    by inner ROC-AUC.

    When ``inner_fit_configs`` effectively contains a single candidate
    (or ``n_inner_splits < 2``), the function runs in explicit no-tuning
    mode and reports this in ``selection_mode``.

    Args:
        scored: Scored backbone DataFrame with label and feature columns.
        model_name: Model identifier (e.g. ``"geo_spread_baseline"``).
        n_outer_splits: Number of outer CV folds.
        n_inner_splits: Number of inner CV folds.
        n_repeats: Number of repeated outer CV runs.
        seed: Random seed for reproducibility.
        inner_fit_configs: Optional candidate fit configs for inner selection.
            If omitted, a small L2 grid is used.

    Returns:
        Dict with keys:
            - ``outer_aucs``, ``mean_auc``, ``std_auc``, ``n_folds``
            - ``selection_mode`` (``inner_cv_tuning`` or ``explicit_no_tuning``)
            - ``inner_cv_participated`` (bool)
            - ``fold_selection_summary`` (per-fold tuning/selection trace)
    """

    label_col = "spread_label"
    outer_repeats = max(1, int(n_repeats))
    fit_config_grid: list[dict[str, object]] = (
        [dict(cfg) for cfg in inner_fit_configs]
        if inner_fit_configs is not None
        else [{"l2": float(l2)} for l2 in _DEFAULT_INNER_L2_GRID]
    )
    if not fit_config_grid:
        fit_config_grid = [{}]

    def _empty_result() -> dict[str, Any]:
        return {
            "outer_aucs": [],
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "n_folds": 0,
            "selection_mode": "explicit_no_tuning",
            "inner_cv_participated": False,
            "outer_repeats": outer_repeats,
            "outer_splits": int(n_outer_splits),
            "inner_splits": int(n_inner_splits),
            "inner_fit_config_count": len(fit_config_grid),
            "fold_selection_summary": [],
        }

    if label_col not in scored.columns:
        _log.warning("Label column %s not found; cannot run nested CV", label_col)
        return _empty_result()

    labels = scored[label_col].dropna()
    if labels.nunique() < 2:
        _log.warning("Only one class present; cannot run nested CV")
        return _empty_result()

    from plasmid_priority.modeling.module_a import evaluate_model_name, fit_predict_model_holdout
    from plasmid_priority.validation.metrics import roc_auc_score

    outer_aucs: list[float] = []
    fold_selection_summary: list[dict[str, Any]] = []
    inner_cv_participated = False

    valid_idx = labels.index
    X_valid = scored.loc[valid_idx]
    y_valid = labels.loc[valid_idx]

    for repeat_idx in range(outer_repeats):
        outer_cv = StratifiedKFold(
            n_splits=n_outer_splits,
            shuffle=True,
            random_state=seed + repeat_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_valid, y_valid)):
            train_df = X_valid.iloc[train_idx]
            test_df = X_valid.iloc[test_idx]

            selected_fit_config: dict[str, object] | None = None
            best_inner_auc = float("nan")
            evaluated_candidates = 0
            selection_mode = "explicit_no_tuning"

            if n_inner_splits >= 2 and len(fit_config_grid) > 1:
                selection_mode = "inner_cv_unavailable"
                for candidate_idx, candidate in enumerate(fit_config_grid):
                    candidate_cfg = candidate.copy()
                    inner_seed = (
                        seed
                        + (repeat_idx * 10_000)
                        + (fold_idx * 101)
                        + candidate_idx
                    )
                    try:
                        inner_result = evaluate_model_name(
                            train_df,
                            model_name=model_name,
                            n_splits=n_inner_splits,
                            n_repeats=1,
                            seed=inner_seed,
                            fit_config=candidate_cfg,
                            include_ci=False,
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        _log.warning(
                            "Inner CV failed (repeat=%d fold=%d cfg=%d): %s",
                            repeat_idx,
                            fold_idx,
                            candidate_idx,
                            exc,
                        )
                        continue
                    inner_auc = float(
                        getattr(inner_result, "metrics", {}).get("roc_auc", float("nan")),
                    )
                    if not np.isfinite(inner_auc):
                        continue
                    evaluated_candidates += 1
                    if selected_fit_config is None or inner_auc > best_inner_auc:
                        selected_fit_config = candidate_cfg
                        best_inner_auc = inner_auc
                        selection_mode = "inner_cv_tuning"

                if selection_mode == "inner_cv_tuning":
                    inner_cv_participated = True

            fold_status = "ok"
            outer_auc = float("nan")
            try:
                preds = fit_predict_model_holdout(
                    train_df,
                    test_df,
                    model_name=model_name,
                    fit_config=selected_fit_config,
                )
                if not preds.empty and "prediction" in preds.columns:
                    y_true = test_df[label_col].to_numpy(dtype=int)
                    y_score = np.asarray(preds["prediction"].to_numpy(), dtype=float)
                    outer_auc = float(roc_auc_score(y_true, y_score))
                    outer_aucs.append(outer_auc)
            except (ValueError, KeyError, TypeError) as exc:
                _log.warning(
                    "Nested CV outer fold failed (repeat=%d fold=%d): %s",
                    repeat_idx,
                    fold_idx,
                    exc,
                )
                fold_status = "failed"

            fold_selection_summary.append(
                {
                    "repeat_idx": repeat_idx,
                    "fold_idx": fold_idx,
                    "status": fold_status,
                    "selection_mode": selection_mode,
                    "selected_fit_config": selected_fit_config,
                    "best_inner_auc": best_inner_auc,
                    "evaluated_candidates": evaluated_candidates,
                    "outer_auc": outer_auc,
                },
            )

    mean_auc = float(np.mean(outer_aucs)) if outer_aucs else float("nan")
    std_auc = float(np.std(outer_aucs)) if len(outer_aucs) > 1 else float("nan")
    selection_mode = "inner_cv_tuning" if inner_cv_participated else "explicit_no_tuning"

    return {
        "outer_aucs": outer_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_folds": len(outer_aucs),
        "selection_mode": selection_mode,
        "inner_cv_participated": inner_cv_participated,
        "outer_repeats": outer_repeats,
        "outer_splits": int(n_outer_splits),
        "inner_splits": int(n_inner_splits),
        "inner_fit_config_count": len(fit_config_grid),
        "fold_selection_summary": fold_selection_summary,
    }
