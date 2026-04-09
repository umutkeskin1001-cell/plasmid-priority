"""Reusable repeated-fold planning for Module A style cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def _effective_splits(y: np.ndarray, n_splits: int) -> int:
    y = np.asarray(y, dtype=int)
    if y.size == 0:
        return 0
    _, class_counts = np.unique(y, return_counts=True)
    if len(class_counts) < 2:
        return 0
    return min(max(int(n_splits), 2), int(class_counts.min()))


@dataclass(frozen=True)
class FoldPlan:
    """Immutable repeated stratified fold plan."""

    effective_splits: int
    n_repeats: int
    seed: int
    fold_groups: tuple[tuple[np.ndarray, ...], ...]
    fold_plan_id: str

    @classmethod
    def from_labels(
        cls,
        y: Iterable[int] | np.ndarray,
        *,
        n_splits: int,
        n_repeats: int,
        seed: int,
    ) -> "FoldPlan":
        labels = np.asarray(list(y) if not isinstance(y, np.ndarray) else y, dtype=int)
        effective_splits = _effective_splits(labels, n_splits)
        n_repeats = max(int(n_repeats), 1)
        if labels.size == 0:
            groups = tuple(() for _ in range(n_repeats))
            return cls(
                effective_splits=0,
                n_repeats=n_repeats,
                seed=int(seed),
                fold_groups=groups,
                fold_plan_id=f"s0-r{n_repeats}-seed{int(seed)}",
            )
        _, class_counts = np.unique(labels, return_counts=True)
        if len(class_counts) < 2:
            groups = tuple(() for _ in range(n_repeats))
            return cls(
                effective_splits=0,
                n_repeats=n_repeats,
                seed=int(seed),
                fold_groups=groups,
                fold_plan_id=f"s0-r{n_repeats}-seed{int(seed)}",
            )
        effective_splits = min(max(int(n_splits), 2), int(class_counts.min()))
        if effective_splits < 2:
            groups = tuple(() for _ in range(n_repeats))
            return cls(
                effective_splits=0,
                n_repeats=n_repeats,
                seed=int(seed),
                fold_groups=groups,
                fold_plan_id=f"s0-r{n_repeats}-seed{int(seed)}",
            )
        skf = RepeatedStratifiedKFold(
            n_splits=effective_splits,
            n_repeats=n_repeats,
            random_state=int(seed),
        )
        fold_groups: list[list[np.ndarray]] = [[] for _ in range(n_repeats)]
        all_splits = list(skf.split(np.zeros(len(labels), dtype=int), labels))
        for i, (_, test_idx) in enumerate(all_splits):
            repeat_idx = i // effective_splits
            fold_groups[repeat_idx].append(np.asarray(test_idx, dtype=int))
        groups = tuple(tuple(group) for group in fold_groups)
        return cls(
            effective_splits=effective_splits,
            n_repeats=n_repeats,
            seed=int(seed),
            fold_groups=groups,
            fold_plan_id=f"s{effective_splits}-r{n_repeats}-seed{int(seed)}",
        )
