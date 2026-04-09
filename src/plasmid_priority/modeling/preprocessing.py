"""Public wrappers for Module A preprocessing primitives."""

from __future__ import annotations

from typing import Any

import pandas as pd

from plasmid_priority.modeling.design_matrix_cache import (
    DesignMatrixCache,
    DesignMatrixCacheKey,
)


def build_design_matrix_cache_key(
    *,
    protocol_hash: str,
    feature_set: list[str] | tuple[str, ...],
    preprocess_mode: str,
    fold_plan_id: str,
) -> DesignMatrixCacheKey:
    return DesignMatrixCacheKey(
        protocol_hash=str(protocol_hash),
        feature_set=tuple(str(column) for column in feature_set),
        preprocess_mode=str(preprocess_mode),
        fold_plan_id=str(fold_plan_id),
    )


def prepare_feature_matrices(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
    *,
    fit_kwargs: dict[str, object] | None = None,
    prepared: bool = False,
    cache: DesignMatrixCache | None = None,
    cache_key: DesignMatrixCacheKey | None = None,
) -> tuple[Any, Any]:
    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    from plasmid_priority.modeling import module_a as module_a_impl

    matrices = module_a_impl._prepare_feature_matrices(
        train,
        score,
        columns,
        fit_kwargs=fit_kwargs,
        prepared=prepared,
    )
    if cache is not None and cache_key is not None:
        cache.set(cache_key, matrices)
    return matrices
