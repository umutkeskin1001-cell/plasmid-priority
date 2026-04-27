"""Persistent cache for repeated stratified fold assignments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plasmid_priority.cache.cache_key import stable_hash
from plasmid_priority.utils.files import ensure_directory


@dataclass(frozen=True)
class FoldCache:
    cache_root: Path

    def _key(
        self,
        *,
        y: np.ndarray,
        split_strategy: str,
        n_splits: int,
        n_repeats: int,
        seed: int,
    ) -> str:
        payload = {
            "split_strategy": split_strategy,
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "seed": int(seed),
            "label_hash": stable_hash(np.asarray(y, dtype=int).tolist()),
        }
        return stable_hash(payload)

    def _path_for_key(self, key: str) -> Path:
        return ensure_directory(self.cache_root / "fold_cache") / f"{key}.json"

    def load(self, key: str) -> list[tuple[np.ndarray, np.ndarray]] | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        folds = payload.get("folds", [])
        if not isinstance(folds, list):
            return None
        parsed: list[tuple[np.ndarray, np.ndarray]] = []
        for item in folds:
            if not isinstance(item, dict):
                return None
            train = item.get("train_idx", [])
            test = item.get("test_idx", [])
            if not isinstance(train, list) or not isinstance(test, list):
                return None
            parsed.append((np.asarray(train, dtype=int), np.asarray(test, dtype=int)))
        return parsed

    def save(self, key: str, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
        path = self._path_for_key(key)
        payload = {
            "key": key,
            "n_folds": len(folds),
            "folds": [
                {
                    "train_idx": train_idx.astype(int).tolist(),
                    "test_idx": test_idx.astype(int).tolist(),
                }
                for train_idx, test_idx in folds
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_or_create(
        self,
        *,
        y: np.ndarray,
        split_strategy: str,
        n_splits: int,
        n_repeats: int,
        seed: int,
        factory: Callable[[], list[tuple[np.ndarray, np.ndarray]]],
    ) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], str]:
        key = self._key(
            y=np.asarray(y, dtype=int),
            split_strategy=split_strategy,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
        )
        cached = self.load(key)
        if cached is not None:
            return key, cached, "hit"
        folds = factory()
        self.save(key, folds)
        return key, folds, "miss"


class UniversalModelCache:
    """Persistent cache for trained models to avoid re-training the same folds.

    Uses joblib for efficient serialization and SHA256 for key generation
    based on model parameters and data state.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib

            self.joblib = joblib
        except ImportError:
            self.joblib = None

    def _get_key(
        self,
        model_name: str,
        features: tuple[str, ...],
        fold_idx: int,
        data_hash: str,
        *,
        fit_config_hash: str = "default",
        protocol_hash: str = "default",
        software_hash: str = "default",
    ) -> str:
        key_payload = self._key_payload(
            model_name,
            features,
            fold_idx,
            data_hash,
            fit_config_hash=fit_config_hash,
            protocol_hash=protocol_hash,
            software_hash=software_hash,
        )
        key_content = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(key_content.encode("utf-8")).hexdigest()[:32]

    def _key_payload(
        self,
        model_name: str,
        features: tuple[str, ...],
        fold_idx: int,
        data_hash: str,
        *,
        fit_config_hash: str = "default",
        protocol_hash: str = "default",
        software_hash: str = "default",
    ) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "features": list(features),
            "fold_idx": int(fold_idx),
            "data_hash": data_hash,
            "fit_config_hash": fit_config_hash,
            "protocol_hash": protocol_hash,
            "software_hash": software_hash,
        }

    def load(
        self,
        model_name: str,
        features: tuple[str, ...],
        fold_idx: int,
        data_hash: str,
        *,
        fit_config_hash: str = "default",
        protocol_hash: str = "default",
        software_hash: str = "default",
    ) -> Any | None:
        if self.joblib is None:
            return None

        key = self._get_key(
            model_name,
            features,
            fold_idx,
            data_hash,
            fit_config_hash=fit_config_hash,
            protocol_hash=protocol_hash,
            software_hash=software_hash,
        )
        path = self.cache_dir / f"{key}.joblib"
        manifest_path = path.with_suffix(".manifest.json")
        if path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected = self._key_payload(
                    model_name,
                    features,
                    fold_idx,
                    data_hash,
                    fit_config_hash=fit_config_hash,
                    protocol_hash=protocol_hash,
                    software_hash=software_hash,
                )
                if manifest != expected:
                    return None
                return self.joblib.load(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, Exception):
                return None
        return None

    def save(
        self,
        model: Any,
        model_name: str,
        features: tuple[str, ...],
        fold_idx: int,
        data_hash: str,
        *,
        fit_config_hash: str = "default",
        protocol_hash: str = "default",
        software_hash: str = "default",
    ) -> None:
        if self.joblib is None:
            return

        key = self._get_key(
            model_name,
            features,
            fold_idx,
            data_hash,
            fit_config_hash=fit_config_hash,
            protocol_hash=protocol_hash,
            software_hash=software_hash,
        )
        path = self.cache_dir / f"{key}.joblib"
        manifest_path = path.with_suffix(".manifest.json")
        try:
            self.joblib.dump(model, path, compress=3)
            manifest = self._key_payload(
                model_name,
                features,
                fold_idx,
                data_hash,
                fit_config_hash=fit_config_hash,
                protocol_hash=protocol_hash,
                software_hash=software_hash,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            # Silently fail if save fails to avoid blocking the pipeline
            import logging
            logging.getLogger(__name__).warning(
                "Caught suppressed exception: %s", exc, exc_info=True
            )
            pass
