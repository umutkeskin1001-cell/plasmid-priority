"""Tests for model configuration integrity and backend availability.

Guards against:
- LightGBM not being installed (primary model silently degraded)
- Config vs feature set mismatches
- Stratified fold statistical properties
"""

from __future__ import annotations

import numpy as np
import pytest

# ─── Test: _HAS_LIGHTGBM ─────────────────────────────────────────────────────


def test_has_lightgbm_flag_is_true() -> None:
    """_HAS_LIGHTGBM must be True in CI for primary model to use LightGBM.

    If this test fails, discovery_boosted silently uses HistGradientBoostingClassifier.
    Fix: pip install -e '.[analysis,dev,tree-models]'
    """
    from plasmid_priority.modeling.module_a import _HAS_LIGHTGBM  # noqa: PLC0415

    assert _HAS_LIGHTGBM, (
        "_HAS_LIGHTGBM is False — LightGBM not installed. "
        "Primary model will use hist_gbm fallback (different from production). "
        "Fix: pip install -e '.[analysis,dev,tree-models]'"
    )


def test_lightgbm_importable() -> None:
    """LightGBM must be importable."""
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"LightGBM not importable: {exc}. Fix: pip install -e '.[analysis,dev,tree-models]'"
        )


# ─── Test: Primary model config consistency ──────────────────────────────────


def test_primary_model_in_feature_sets() -> None:
    """Primary model declared in config must have a feature set definition."""
    from plasmid_priority.modeling.module_a import (  # noqa: PLC0415
        MODULE_A_FEATURE_SETS,
        get_primary_model_name,
    )

    primary = get_primary_model_name(MODULE_A_FEATURE_SETS.keys())
    assert primary in MODULE_A_FEATURE_SETS, (
        f"Primary model '{primary}' not in MODULE_A_FEATURE_SETS. "
        "Check config.yaml: models.primary_model_name"
    )


def test_primary_model_fallback_in_feature_sets() -> None:
    """Primary model fallback must exist in feature sets."""
    from plasmid_priority.modeling.module_a_support import (  # noqa: PLC0415
        MODULE_A_FEATURE_SETS,
        PRIMARY_MODEL_FALLBACK,
    )

    assert PRIMARY_MODEL_FALLBACK in MODULE_A_FEATURE_SETS, (
        f"Primary model fallback '{PRIMARY_MODEL_FALLBACK}' not in MODULE_A_FEATURE_SETS."
    )


def test_governance_model_in_feature_sets() -> None:
    """Governance model must have a feature set."""
    from plasmid_priority.modeling.module_a import MODULE_A_FEATURE_SETS  # noqa: PLC0415
    from plasmid_priority.modeling.module_a_support import GOVERNANCE_MODEL_NAME  # noqa: PLC0415

    assert GOVERNANCE_MODEL_NAME in MODULE_A_FEATURE_SETS, (
        f"Governance model '{GOVERNANCE_MODEL_NAME}' missing from MODULE_A_FEATURE_SETS."
    )


# ─── Test: Feature set completeness ──────────────────────────────────────────


def test_all_model_feature_sets_non_empty() -> None:
    """Every model in MODULE_A_FEATURE_SETS must have at least one feature."""
    from plasmid_priority.modeling.module_a import MODULE_A_FEATURE_SETS  # noqa: PLC0415

    empty_models = [name for name, features in MODULE_A_FEATURE_SETS.items() if not features]
    assert not empty_models, f"Models with empty feature sets: {empty_models}"


def test_no_duplicate_features_within_model() -> None:
    """No feature should appear twice in the same model's feature set."""
    from plasmid_priority.modeling.module_a import MODULE_A_FEATURE_SETS  # noqa: PLC0415

    problems: dict[str, list[str]] = {}
    for model_name, features in MODULE_A_FEATURE_SETS.items():
        seen: set[str] = set()
        dupes = [f for f in features if f in seen or seen.add(f)]  # type: ignore[func-returns-value]
        if dupes:
            problems[model_name] = dupes

    assert not problems, f"Models with duplicate features: {problems}"


# ─── Test: Stratified folds ───────────────────────────────────────────────────


def test_stratified_folds_respect_class_balance() -> None:
    """Stratified folds must keep consistent positive rate across test folds."""
    from plasmid_priority.modeling.module_a_support import _stratified_folds  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 200
    y = (rng.uniform(0, 1, n) < 0.2).astype(int)  # ~20% positive
    overall_rate = y.mean()

    fold_groups = _stratified_folds(y, n_splits=5, n_repeats=2, seed=42)
    for fold_idx, (_, test_idx) in enumerate(fold_groups):
        test_rate = y[test_idx].mean()
        assert abs(test_rate - overall_rate) < 0.15, (
            f"Fold {fold_idx}: test positive rate {test_rate:.3f} deviates "
            f"from overall {overall_rate:.3f} — stratification may be broken."
        )


# ─── Test: KNN Imputer isolation ─────────────────────────────────────────────


def test_knn_imputer_is_deterministic_on_same_input() -> None:
    """KNN imputer must produce identical results on same input data."""
    from plasmid_priority.modeling.module_a_support import _fit_feature_imputer  # noqa: PLC0415

    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, (100, 4))
    arr[5, 2] = np.nan
    arr[10, 0] = np.nan

    imputed1, _ = _fit_feature_imputer(arr)
    imputed2, _ = _fit_feature_imputer(arr)

    np.testing.assert_array_almost_equal(
        imputed1,
        imputed2,
        decimal=10,
        err_msg="KNN imputer non-deterministic on same input — possible hidden state",
    )


def test_knn_imputer_state_not_modified_by_transform() -> None:
    """Applying transform to test data must not change imputer's train-fitted state."""
    from plasmid_priority.modeling.module_a_support import _fit_feature_imputer  # noqa: PLC0415

    rng = np.random.default_rng(42)
    train_arr = rng.uniform(0, 1, (100, 4))
    test_arr = rng.uniform(10, 20, (30, 4))  # far from train — would corrupt imputer if leaked
    train_arr[5, 2] = np.nan

    train_imputed, imputer = _fit_feature_imputer(train_arr)
    # Transform test data (should not affect imputer)
    _ = np.nan_to_num(imputer.transform(test_arr), nan=0.0)
    # Re-transform train — must be identical
    train_after = imputer.transform(train_arr)

    np.testing.assert_array_almost_equal(
        train_imputed,
        train_after,
        decimal=10,
        err_msg="Imputer state was modified by test transform — potential data leak",
    )
