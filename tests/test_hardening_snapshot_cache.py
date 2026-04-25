from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "scripts" / "generate_hardening_snapshot.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def __init__(self, root: Path, reports_dir: Path, data_root: Path) -> None:
        self.root = root
        self.reports_dir = reports_dir
        self._data_root = data_root

    def resolve_path(self, path_str: str | Path) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "data":
            return (self._data_root / Path(*candidate.parts[1:])).resolve()
        return (self.root / candidate).resolve()


def _seed_data_files(data_root: Path) -> None:
    (data_root / "features").mkdir(parents=True, exist_ok=True)
    (data_root / "scores").mkdir(parents=True, exist_ok=True)
    (data_root / "harmonized").mkdir(parents=True, exist_ok=True)
    (data_root / "deduplicated").mkdir(parents=True, exist_ok=True)
    (data_root / "features" / "backbone_table.tsv").write_text(
        "backbone_id\nbb1\n", encoding="utf-8"
    )
    (data_root / "scores" / "backbone_scored.tsv").write_text(
        "backbone_id\tspread_label\nbb1\t1\n", encoding="utf-8"
    )
    (data_root / "harmonized" / "harmonized_plasmids.tsv").write_text(
        "sequence_accession\tbackbone_id\nx\tbb1\n", encoding="utf-8"
    )
    (data_root / "deduplicated" / "deduplicated_plasmids.tsv").write_text(
        "sequence_accession\tbackbone_id\nx\tbb1\n", encoding="utf-8"
    )


def test_snapshot_reuses_cached_audit_results_when_signatures_unchanged() -> None:
    module = _load_script("generate_hardening_snapshot_cache_test")
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        reports_dir = root / "reports"
        data_root = root / "external-data"
        reports_dir.mkdir(parents=True, exist_ok=True)
        _seed_data_files(data_root)
        fake_context = _FakeContext(root, reports_dir, data_root)

        summary_payload = {
            "models": {"epv": {"status": "ok", "n_models_evaluated": 1}},
            "lead_time_bias": {"status": "ok", "overall_concern_level": "low"},
            "missingness": {"overall_status": "ok", "high_missingness_columns_total": 0},
            "schema_validation": {
                "overall_status": "ok",
                "pandera_available": True,
                "tables_validated": [],
            },
        }

        with (
            mock.patch.object(module, "build_context", return_value=fake_context),
            mock.patch(
                "plasmid_priority.reporting.hardening_summary.build_hardening_audit_summary",
                return_value=summary_payload,
            ),
            mock.patch("pandas.read_csv", wraps=pd.read_csv) as read_csv_mock,
        ):
            first = module.run_data_audits_if_available()
            first_calls = read_csv_mock.call_count
            second = module.run_data_audits_if_available()
            second_calls = read_csv_mock.call_count

        assert first == second
        assert first_calls > 0
        # Second run should hit cache and not trigger additional table reads.
        assert second_calls == first_calls
        cache_path = reports_dir / module.SNAPSHOT_AUDIT_CACHE_NAME
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload.get("results", {}) == first
