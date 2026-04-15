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

from plasmid_priority.modeling.module_a import ModelResult
from plasmid_priority.modeling.module_a_support import build_failed_model_result

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
    """Extract feature matrix X and label vector y from scored table."""
    if label_col not in scored.columns:
        raise KeyError(f"Label column '{label_col}' not found")
    y = scored[label_col].astype(float)
    drop_cols = [c for c in scored.columns if c in (label_col, "backbone_id", "knownness_score")]
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
    """
    if not LIGHTGBM_AVAILABLE:
        return ModelResult(
            name="lightgbm",
            metrics={},
            predictions=pd.DataFrame(),
            status="skipped",
            error_message="lightgbm not installed",
        )

    from sklearn.model_selection import StratifiedKFold, cross_val_score

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
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        aps = cross_val_score(clf, X, y, cv=cv, scoring="average_precision")

        clf.fit(X, y)
        preds = clf.predict_proba(X)[:, 1]

        return ModelResult(
            name="lightgbm",
            metrics={
                "roc_auc": float(aucs.mean()),
                "average_precision": float(aps.mean()),
                "roc_auc_std": float(aucs.std()),
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": scored.get(
                        "backbone_id", pd.RangeIndex(len(preds))
                    ),
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
    """
    if not XGBOOST_AVAILABLE:
        return ModelResult(
            name="xgboost",
            metrics={},
            predictions=pd.DataFrame(),
            status="skipped",
            error_message="xgboost not installed",
        )

    from sklearn.model_selection import StratifiedKFold, cross_val_score

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
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        aps = cross_val_score(clf, X, y, cv=cv, scoring="average_precision")

        clf.fit(X, y)
        preds = clf.predict_proba(X)[:, 1]

        return ModelResult(
            name="xgboost",
            metrics={
                "roc_auc": float(aucs.mean()),
                "average_precision": float(aps.mean()),
                "roc_auc_std": float(aucs.std()),
            },
            predictions=pd.DataFrame(
                {
                    "backbone_id": scored.get(
                        "backbone_id", pd.RangeIndex(len(preds))
                    ),
                    "xgboost": preds,
                }
            ),
            status="ok",
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        _log.warning("XGBoost evaluation failed: %s", exc)
        return build_failed_model_result("xgboost", str(exc))
