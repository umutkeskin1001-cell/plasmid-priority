from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, script_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "scripts" / script_name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def __init__(self, root: Path, data_root: Path) -> None:
        self.root = root
        self._data_root = data_root
        self.reports_dir = root / "reports"

    def resolve_path(self, path_str: str | Path) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "data":
            return (self._data_root / Path(*candidate.parts[1:])).resolve()
        return (self.root / candidate).resolve()


def _seed_hardening_tables(data_root: Path) -> None:
    (data_root / "features").mkdir(parents=True, exist_ok=True)
    (data_root / "scores").mkdir(parents=True, exist_ok=True)
    (data_root / "harmonized").mkdir(parents=True, exist_ok=True)
    (data_root / "deduplicated").mkdir(parents=True, exist_ok=True)
    (data_root / "features" / "backbone_table.tsv").write_text(
        "backbone_id\tspread_label\nbb1\t1\n", encoding="utf-8"
    )
    (data_root / "scores" / "backbone_scored.tsv").write_text(
        "backbone_id\tspread_label\nbb1\t1\n", encoding="utf-8"
    )
    (data_root / "harmonized" / "harmonized_plasmids.tsv").write_text(
        "plasmid_id\np1\n", encoding="utf-8"
    )
    (data_root / "deduplicated" / "deduplicated_plasmids.tsv").write_text(
        "plasmid_id\np1\n", encoding="utf-8"
    )


def test_run_schema_validation_resolves_defaults_via_context_data_root() -> None:
    module = _load_script("run_schema_validation_script", "run_schema_validation.py")
    with tempfile.TemporaryDirectory() as tmp_dir:
        external = Path(tmp_dir) / "external-data"
        _seed_hardening_tables(external)
        fake_context = _FakeContext(PROJECT_ROOT, external)
        expected_backbone = str((external / "features" / "backbone_table.tsv").resolve())
        expected_scored = str((external / "scores" / "backbone_scored.tsv").resolve())

        with (
            mock.patch.object(module, "build_context", return_value=fake_context),
            mock.patch.object(
                module,
                "validate_tables_from_paths",
                return_value={"_summary": {"overall_status": "skipped"}},
            ) as validate_mock,
            mock.patch.object(module, "print_validation_report"),
        ):
            rc = module.main(["--quiet"])

        assert rc == 0
        assert validate_mock.call_args.kwargs["backbones_path"] == expected_backbone
        assert validate_mock.call_args.kwargs["scored_path"] == expected_scored


def test_run_missingness_audit_resolves_defaults_via_context_data_root() -> None:
    module = _load_script("run_missingness_audit_script", "run_missingness_audit.py")
    with tempfile.TemporaryDirectory() as tmp_dir:
        external = Path(tmp_dir) / "external-data"
        _seed_hardening_tables(external)
        fake_context = _FakeContext(PROJECT_ROOT, external)
        expected_backbone = str((external / "features" / "backbone_table.tsv").resolve())
        expected_scored = str((external / "scores" / "backbone_scored.tsv").resolve())
        output_dir = Path(tmp_dir) / "audit-output"

        with (
            mock.patch.object(module, "build_context", return_value=fake_context),
            mock.patch.object(
                module, "audit_backbone_tables", return_value={"overall_status": "ok"}
            ),
            mock.patch.object(module, "format_missingness_report", return_value="ok"),
            mock.patch.object(module, "print_backbone_audit_report"),
            mock.patch.object(module.pd, "read_csv", wraps=module.pd.read_csv) as read_csv_mock,
        ):
            rc = module.main(["--json-only", "--output-dir", str(output_dir)])

        assert rc == 0
        read_paths = [str(call.args[0]) for call in read_csv_mock.call_args_list[:2]]
        assert expected_backbone in read_paths
        assert expected_scored in read_paths


def test_run_hardening_summary_resolves_defaults_via_context_data_root() -> None:
    module = _load_script("run_hardening_summary_script", "run_hardening_summary.py")
    with tempfile.TemporaryDirectory() as tmp_dir:
        external = Path(tmp_dir) / "external-data"
        _seed_hardening_tables(external)
        fake_context = _FakeContext(PROJECT_ROOT, external)
        expected_backbone = str((external / "features" / "backbone_table.tsv").resolve())
        expected_scored = str((external / "scores" / "backbone_scored.tsv").resolve())
        summary_payload = {"overall_status": "ok"}

        with (
            mock.patch.object(module, "build_context", return_value=fake_context),
            mock.patch.object(
                module, "build_hardening_audit_summary", return_value=summary_payload
            ),
            mock.patch.object(module.pd, "read_csv", wraps=module.pd.read_csv) as read_csv_mock,
        ):
            rc = module.main([])

        assert rc == 0
        read_paths = [str(call.args[0]) for call in read_csv_mock.call_args_list[:2]]
        assert expected_backbone in read_paths
        assert expected_scored in read_paths


def test_generate_hardening_snapshot_uses_context_data_root_for_presence_check() -> None:
    module = _load_script("generate_hardening_snapshot_script", "generate_hardening_snapshot.py")
    with tempfile.TemporaryDirectory() as tmp_dir:
        external = Path(tmp_dir) / "external-data"
        _seed_hardening_tables(external)
        fake_context = _FakeContext(PROJECT_ROOT, external)

        with mock.patch.object(module, "build_context", return_value=fake_context):
            result = module.check_data_dependent_audits()

        assert result["any_data_available"] is True
        assert all(result["data_files"].values())
