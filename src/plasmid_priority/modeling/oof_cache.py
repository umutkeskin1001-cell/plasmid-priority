"""Persistent OOF prediction cache for model evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.cache.cache_key import stable_hash
from plasmid_priority.utils.files import ensure_directory


@dataclass(frozen=True)
class OOFCache:
    cache_root: Path

    def _entry_dir(self, key: str) -> Path:
        return ensure_directory(self.cache_root / "oof_cache" / key)

    def build_key(
        self,
        *,
        matrix_key: str,
        fold_key: str,
        model_name: str,
        model_config: dict[str, Any],
        model_code_hash: str,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "matrix_key": matrix_key,
            "fold_key": fold_key,
            "model_name": model_name,
            "model_config_hash": stable_hash(model_config),
            "model_code_hash": model_code_hash,
        }
        key = stable_hash(payload)
        return key, payload

    def load(self, key: str) -> dict[str, Any] | None:
        entry = self._entry_dir(key)
        metadata_path = entry / "metadata.json"
        predictions_path = entry / "predictions.parquet"
        if not metadata_path.exists() or not predictions_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(metadata, dict):
            return None
        try:
            predictions = pd.read_parquet(predictions_path)
        except Exception:
            return None
        return {"metadata": metadata, "predictions": predictions}

    def save(
        self,
        key: str,
        *,
        key_payload: dict[str, Any],
        model_name: str,
        metrics: dict[str, Any],
        predictions: pd.DataFrame,
        status: str,
        error_message: str | None,
    ) -> None:
        entry = self._entry_dir(key)
        metadata = {
            "cache_key": key,
            "cache_key_payload": key_payload,
            "model_name": model_name,
            "metrics": metrics,
            "status": status,
            "error_message": error_message,
            "n_predictions": int(len(predictions)),
        }
        (entry / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        predictions.to_parquet(entry / "predictions.parquet", index=False)
