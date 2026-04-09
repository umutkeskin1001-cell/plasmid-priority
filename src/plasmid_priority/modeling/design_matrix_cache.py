"""Small in-memory cache for repeated design-matrix preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DesignMatrixCacheKey:
    protocol_hash: str
    feature_set: tuple[str, ...]
    preprocess_mode: str
    fold_plan_id: str


@dataclass
class DesignMatrixCache:
    _store: dict[DesignMatrixCacheKey, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    def get(
        self, key: DesignMatrixCacheKey
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return self._store.get(key)

    def set(self, key: DesignMatrixCacheKey, matrices: tuple[np.ndarray, np.ndarray]) -> None:
        self._store[key] = matrices

    def get_or_set(
        self,
        key: DesignMatrixCacheKey,
        factory: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self.get(key)
        if cached is not None:
            return cached
        matrices = factory()
        self.set(key, matrices)
        return matrices
