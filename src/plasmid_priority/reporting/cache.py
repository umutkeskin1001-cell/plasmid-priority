"""Report and figure cache helpers for incremental report rebuilds."""


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from plasmid_priority.cache import stable_hash
from plasmid_priority.utils.files import ensure_directory, path_signature


def frame_fingerprint(frame: pd.DataFrame, *, sample_rows: int = 128) -> str:
    """Compute a stable, bounded-cost fingerprint for a dataframe."""
    if frame.empty:
        payload: dict[str, Any] = {
            "rows": 0,
            "columns": [str(col) for col in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "sample_rows": 0,
        }
        return stable_hash(payload)

    head = frame.head(sample_rows)
    tail = frame.tail(sample_rows) if len(frame) > sample_rows else frame.iloc[0:0]
    payload = {
        "rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "head": head.to_dict(orient="list"),
        "tail": tail.to_dict(orient="list"),
        "sample_rows": int(sample_rows),
    }
    return stable_hash(payload)


@dataclass
class ReportCache:
    """Persistent JSON cache index for report-level and figure-level keys."""

    cache_root: Path

    def __post_init__(self) -> None:
        self.cache_root = ensure_directory(self.cache_root)
        self._index_path = self.cache_root / "report_cache_index.json"
        self._index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"reports": {}, "figures": {}}
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reports": {}, "figures": {}}
        if not isinstance(payload, dict):
            return {"reports": {}, "figures": {}}
        reports = payload.get("reports", {})
        figures = payload.get("figures", {})
        return {
            "reports": reports if isinstance(reports, dict) else {},
            "figures": figures if isinstance(figures, dict) else {},
        }

    def _save(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def build_report_key(
        self,
        *,
        report_name: str,
        input_hashes: list[dict[str, object]],
        config_hash: str,
        protocol_hash: str,
        code_hash: str,
        mode: str,
    ) -> str:
        return stable_hash(
            {
                "report_name": report_name,
                "input_hashes": input_hashes,
                "config_hash": config_hash,
                "protocol_hash": protocol_hash,
                "code_hash": code_hash,
                "mode": mode,
            },
        )

    def is_report_current(self, *, report_name: str, report_key: str, outputs: list[Path]) -> bool:
        entry = self._index.get("reports", {}).get(report_name)
        if not isinstance(entry, dict):
            return False
        if str(entry.get("report_key", "")) != report_key:
            return False
        for output in outputs:
            if not output.exists():
                return False
        return True

    def put_report(self, *, report_name: str, report_key: str, outputs: list[Path]) -> None:
        self._index.setdefault("reports", {})
        self._index["reports"][report_name] = {
            "report_key": report_key,
            "outputs": [str(path.resolve()) for path in outputs if path.exists()],
        }
        self._save()

    def build_figure_key(
        self,
        *,
        figure_name: str,
        data_fingerprint: str,
        function_hash: str,
        mode: str,
    ) -> str:
        return stable_hash(
            {
                "figure_name": figure_name,
                "data_fingerprint": data_fingerprint,
                "function_hash": function_hash,
                "mode": mode,
            },
        )

    def is_figure_current(self, *, figure_name: str, figure_key: str, output_path: Path) -> bool:
        entry = self._index.get("figures", {}).get(figure_name)
        if not isinstance(entry, dict):
            return False
        if str(entry.get("figure_key", "")) != figure_key:
            return False
        cached_path = Path(str(entry.get("output_path", output_path)))
        if not cached_path.exists():
            return False
        return path_signature(cached_path) == entry.get("output_signature")

    def put_figure(self, *, figure_name: str, figure_key: str, output_path: Path) -> None:
        self._index.setdefault("figures", {})
        self._index["figures"][figure_name] = {
            "figure_key": figure_key,
            "output_path": str(output_path.resolve()),
            "output_signature": path_signature(output_path) if output_path.exists() else {},
        }
        self._save()
