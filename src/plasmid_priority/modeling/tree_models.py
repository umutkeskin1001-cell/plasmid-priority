"""Tree-based model backends (LightGBM / XGBoost) for Plasmid Priority.

Provides ``evaluate_lightgbm`` and ``evaluate_xgboost`` which follow the
same interface as ``evaluate_model_name`` for drop-in comparison.

Both libraries are optional dependencies — if not installed, the functions
return a ``ModelResult`` with ``status="skipped"``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from plasmid_priority.modeling.module_a_support import ModelResult, build_failed_model_result

_log = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _extract_xy(
    scored: pd.DataFrame,
    label_col: str = "spread_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from scored table.

    Drops non-feature columns (labels, IDs, metadata) so that only
    genuine numeric features remain in X.
    """
    if label_col not in scored.columns:
        raise KeyError(f"Label column '{label_col}' not found")
    y = scored[label_col].astype(float)
    # Non-feature columns that must be excluded even though they are numeric.
    # ⚠ FRAGILE: new outcome/metadata columns added elsewhere must be added here
    #   too.  Prefer deriving features from the configured feature_sets when
    #   the model name is available.
    _NON_FEATURE_COLUMNS = frozenset(
        {
            label_col,
            "backbone_id",
            "knownness_score",
            "member_count_band",
            "country_count_band",
            "source_band",
            "knownness_half",
            "knownness_quartile",
            "knownness_quartile_supported",
            "split_year",
            "n_new_countries",
            "n_new_countries_future",
            "future_new_host_genera_count",
            "future_new_host_families_count",
            "clinical_fraction_future",
            "last_resort_fraction_future",
            "mdr_proxy_fraction_future",
            "pd_clinical_support_future",
            "training_only_future_unseen_backbone_flag",
        }
    )
    drop_cols = [c for c in scored.columns if c in _NON_FEATURE_COLUMNS]
    X = scored.drop(columns=drop_cols).select_dtypes(include="number")
    return X, y


def evaluate_lightgbm(
    scored: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> ModelResult:
    """Evaluate LightGBM with stratified repeated K-Fold CV.

    Returns a ``ModelResult`` with ``status="skipped"`` if LightGBM is
    not installed.

    Uses out-of-fold predictions to avoid data leakage — the reported
    metrics reflect generalisation performance, not training-set fit.
    """
    if not LIGHTGBM_AVAILABLE:
        return ModelResult(
            name="lightgbm",
            metrics={},
            predictions=pd.DataFrame(),
            status="skipped",
            error_message="lightgbm not installed",
        )

    from sklearn.model_selection import (
        RepeatedStratifiedKFold,
        StratifiedKFold,
        cross_validate,
    )

    try:
        X, y = _extract_xy(scored)
        default_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": seed,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
        }
        default_params.update(params or {})

        clf = lgb.LGBMClassifier(**default_params)

        # Repeated CV for stable metric estimates
        cv_repeated = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        cv_results = cross_validate(
            clf,
            X,
            y,
            cv=cv_repeated,
            scoring={"roc_auc": "roc_auc", "ap": "average_precision"},
            return_estimator=False,
        )
        aucs = cv_results["test_roc_auc"]
        aps = cv_results["test_ap"]

        # Single-split CV for out-of-fold predictions (each sample appears
        # exactly once in a test fold, avoiding the averaging artefact that
        # cross_val_predict introduces with repeated CV).
        cv_single = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        from sklearn.model_selection import cross_val_predict as _cvp

        preds = _cvp(clf, X, y, cv=cv_single, method="predict_proba")[:, 1]

        return ModelResult(
            name="lightgbm",
            metrics={
                "roc_auc": float(aucs.mean()),
                "average_precision": float(aps.mean()),
                "roc_auc_std": float(aucs.std()),
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": scored.get("backbone_id", pd.RangeIndex(len(preds))),
                    "lightgbm": preds,
                }
            ),
            status="ok",
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        _log.warning("LightGBM evaluation failed: %s", exc)
        return build_failed_model_result("lightgbm", str(exc))


def evaluate_xgboost(
    scored: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> ModelResult:
    """Evaluate XGBoost with stratified repeated K-Fold CV.

    Returns a ``ModelResult`` with ``status="skipped"`` if XGBoost is
    not installed.

    Uses out-of-fold predictions to avoid data leakage — the reported
    metrics reflect generalisation performance, not training-set fit.
    """
    if not XGBOOST_AVAILABLE:
        return ModelResult(
            name="xgboost",
            metrics={},
            predictions=pd.DataFrame(),
            status="skipped",
            error_message="xgboost not installed",
        )

    from sklearn.model_selection import (
        RepeatedStratifiedKFold,
        StratifiedKFold,
        cross_validate,
    )

    try:
        X, y = _extract_xy(scored)
        default_params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "seed": seed,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
        }
        default_params.update(params or {})

        clf = xgb.XGBClassifier(**default_params)

        # Repeated CV for stable metric estimates
        cv_repeated = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        cv_results = cross_validate(
            clf,
            X,
            y,
            cv=cv_repeated,
            scoring={"roc_auc": "roc_auc", "ap": "average_precision"},
            return_estimator=False,
        )
        aucs = cv_results["test_roc_auc"]
        aps = cv_results["test_ap"]

        # Single-split CV for out-of-fold predictions (each sample appears
        # exactly once in a test fold, avoiding the averaging artefact that
        # cross_val_predict introduces with repeated CV).
        cv_single = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        from sklearn.model_selection import cross_val_predict as _cvp

        preds = _cvp(clf, X, y, cv=cv_single, method="predict_proba")[:, 1]

        return ModelResult(
            name="xgboost",
            metrics={
                "roc_auc": float(aucs.mean()),
                "average_precision": float(aps.mean()),
                "roc_auc_std": float(aucs.std()),
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": scored.get("backbone_id", pd.RangeIndex(len(preds))),
                    "xgboost": preds,
                }
            ),
            status="ok",
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        _log.warning("XGBoost evaluation failed: %s", exc)
        return build_failed_model_result("xgboost", str(exc))
