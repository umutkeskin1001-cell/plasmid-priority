"""Feature-surface matrix cache metadata utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.cache.cache_key import stable_hash
from plasmid_priority.utils.files import ensure_directory


def _series_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).astype("uint64").tolist()
    return stable_hash(hashed)


@dataclass(frozen=True)
class MatrixCache:
    cache_root: Path

    def build_key(
        self,
        *,
        scored: pd.DataFrame,
        columns: list[str],
        preprocessing_config: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        required_columns = ["backbone_id", "spread_label", *columns]
        working = scored.copy()
        for column in required_columns:
            if column in working.columns:
                continue
            # Missing engineered features are treated as neutral so cache-key
            # generation stays aligned with downstream 0-fill behavior.
            if column == "backbone_id":
                working[column] = pd.Series("", index=working.index, dtype=object)
            elif column == "spread_label":
                working[column] = pd.Series(pd.NA, index=working.index, dtype="Float64")
            else:
                working[column] = 0.0
        eligible = working.loc[working["spread_label"].notna(), required_columns].copy()
        payload = {
            "feature_surface_hash": _series_hash(eligible[["backbone_id", "spread_label"]]),
            "column_set_hash": stable_hash(sorted(columns)),
            "preprocessing_config_hash": stable_hash(preprocessing_config),
            "n_rows": int(len(eligible)),
        }
        key = stable_hash(payload)
        return key, payload

    def save(self, key: str, payload: dict[str, Any]) -> None:
        path = ensure_directory(self.cache_root / "matrix_cache") / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, key: str) -> dict[str, Any] | None:
        path = ensure_directory(self.cache_root / "matrix_cache") / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
