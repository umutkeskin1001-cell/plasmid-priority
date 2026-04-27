"""Persistent variant cache for sensitivity pipelines."""


import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.cache import stable_hash
from plasmid_priority.utils.files import ensure_directory

LOGGER = logging.getLogger(__name__)

_COMPONENT_FRAME_KEYS: tuple[str, ...] = (
    "records",
    "feature_t",
    "feature_h",
    "feature_a",
    "backbone_base",
    "future_country_first_year",
)


@dataclass
class VariantCacheManager:
    cache_root: Path
    source_signatures: dict[str, str]

    def __post_init__(self) -> None:
        # Keep legacy path/field names for compatibility with older callers/tests.
        self.cache_dir = ensure_directory(self.cache_root / "sensitivity_cache")
        self._memory: dict[str, pd.DataFrame] = {}
        self._memory_scored = self._memory

    def _build_key(self, category: str, payload: dict[str, Any]) -> str:
        wrapped = {
            "category": category,
            "payload": payload,
            "source_signatures": self.source_signatures,
            "schema_version": "variant-cache-v1",
        }
        return stable_hash(wrapped)[:20]

    def _component_key(self, split_year: int, **flags: Any) -> str:
        payload = {"split_year": int(split_year), **{k: flags[k] for k in sorted(flags)}}
        return self._build_key("components", payload)

    # Backward-compatible alias used by legacy tests/callers.
    def _key(self, split_year: int, **flags: Any) -> str:
        return self._component_key(split_year, **flags)

    def _variant_key(
        self,
        *,
        split_year: int,
        test_year_end: int,
        horizon_years: int,
        assignment_mode: str,
        normalization_mode: str,
        thresholds: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "split_year": int(split_year),
            "test_year_end": int(test_year_end),
            "horizon_years": int(horizon_years),
            "assignment_mode": str(assignment_mode),
            "normalization_mode": str(normalization_mode),
            "thresholds": dict(thresholds or {}),
        }
        return self._build_key("variant", payload)

    def _read_meta(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("source_signatures") != self.source_signatures:
            return None
        return payload

    def get_components(
        self,
        split_year: int,
        **flags: Any,
    ) -> dict[str, object] | pd.DataFrame | None:
        key = self._component_key(split_year, **flags)
        component_dir = self.cache_dir / f"components_{key}"
        meta = self._read_meta(component_dir / "meta.json")
        if meta is None:
            return None
        if str(meta.get("kind", "")).strip().lower() == "dataframe":
            frame_path = component_dir / "components.parquet"
            if not frame_path.exists():
                return None
            try:
                return pd.read_parquet(frame_path)
            except Exception as exc:
                LOGGER.warning(
                    "Caught suppressed exception: %s",
                    exc,
                    exc_info=True,
                )
                return None
        frames: dict[str, pd.DataFrame] = {}
        for frame_key in _COMPONENT_FRAME_KEYS:
            frame_path = component_dir / f"{frame_key}.parquet"
            if not frame_path.exists():
                return None
            try:
                frames[frame_key] = pd.read_parquet(frame_path)
            except Exception as exc:
                LOGGER.warning(
                    "Caught suppressed exception: %s",
                    exc,
                    exc_info=True,
                )
                return None
        return {
            "cache_key": tuple(meta.get("cache_key_tuple", [])),
            "split_year": int(meta.get("split_year", split_year)),
            **frames,
        }

    def put_components(
        self,
        split_year: int,
        components: dict[str, object] | pd.DataFrame,
        **flags: Any,
    ) -> None:
        key = self._component_key(split_year, **flags)
        component_dir = ensure_directory(self.cache_dir / f"components_{key}")
        if isinstance(components, pd.DataFrame):
            components.to_parquet(component_dir / "components.parquet", index=False)
            meta = {
                "split_year": int(split_year),
                "flags": {k: flags[k] for k in sorted(flags)},
                "kind": "dataframe",
                "source_signatures": self.source_signatures,
            }
            (component_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return
        for frame_key in _COMPONENT_FRAME_KEYS:
            frame_value = components.get(frame_key)
            if not isinstance(frame_value, pd.DataFrame):
                continue
            frame_value.to_parquet(component_dir / f"{frame_key}.parquet", index=False)
        cache_key = components.get("cache_key", ())
        meta = {
            "split_year": int(split_year),
            "flags": {k: flags[k] for k in sorted(flags)},
            "cache_key_tuple": list(cache_key) if isinstance(cache_key, (list, tuple)) else [],
            "kind": "components",
            "source_signatures": self.source_signatures,
        }
        (component_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def get_variant(
        self,
        *,
        split_year: int,
        test_year_end: int,
        horizon_years: int,
        assignment_mode: str,
        normalization_mode: str,
        thresholds: dict[str, Any] | None = None,
    ) -> pd.DataFrame | None:
        key = self._variant_key(
            split_year=split_year,
            test_year_end=test_year_end,
            horizon_years=horizon_years,
            assignment_mode=assignment_mode,
            normalization_mode=normalization_mode,
            thresholds=thresholds,
        )
        if key in self._memory_scored:
            return self._memory_scored[key].copy()
        variant_dir = self.cache_dir / f"variant_{key}"
        meta = self._read_meta(variant_dir / "meta.json")
        table_path = variant_dir / "scored.parquet"
        if meta is None or not table_path.exists():
            return None
        try:
            frame = pd.read_parquet(table_path)
        except Exception as exc:
            LOGGER.warning(
                "Caught suppressed exception: %s",
                exc,
                exc_info=True,
            )
            return None
        self._memory_scored[key] = frame.copy()
        return frame

    def put_variant(
        self,
        *,
        split_year: int,
        test_year_end: int,
        horizon_years: int,
        assignment_mode: str,
        normalization_mode: str,
        scored: pd.DataFrame,
        thresholds: dict[str, Any] | None = None,
    ) -> None:
        key = self._variant_key(
            split_year=split_year,
            test_year_end=test_year_end,
            horizon_years=horizon_years,
            assignment_mode=assignment_mode,
            normalization_mode=normalization_mode,
            thresholds=thresholds,
        )
        variant_dir = ensure_directory(self.cache_dir / f"variant_{key}")
        scored.to_parquet(variant_dir / "scored.parquet", index=False)
        meta = {
            "split_year": int(split_year),
            "test_year_end": int(test_year_end),
            "horizon_years": int(horizon_years),
            "assignment_mode": str(assignment_mode),
            "normalization_mode": str(normalization_mode),
            "thresholds": dict(thresholds or {}),
            "source_signatures": self.source_signatures,
        }
        (variant_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._memory_scored[key] = scored.copy()

    # Backward-compat wrappers
    def get_scored(self, split_year: int, test_year_end: int, **flags: Any) -> pd.DataFrame | None:
        key = self._build_key(
            "scored-legacy",
            {
                "split_year": int(split_year),
                "test_year_end": int(test_year_end),
                **{k: flags[k] for k in sorted(flags)},
            },
        )
        cached = self._memory_scored.get(key)
        return None if cached is None else cached.copy()

    def put_scored(
        self, split_year: int, test_year_end: int, scored: pd.DataFrame, **flags: Any
    ) -> None:
        key = self._build_key(
            "scored-legacy",
            {
                "split_year": int(split_year),
                "test_year_end": int(test_year_end),
                **{k: flags[k] for k in sorted(flags)},
            },
        )
        self._memory_scored[key] = scored.copy()

    def clear_memory(self) -> None:
        self._memory_scored.clear()
