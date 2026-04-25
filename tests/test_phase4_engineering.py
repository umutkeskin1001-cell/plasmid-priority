from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from plasmid_priority.pipeline.prefect_flow import (
    PREFECT_AVAILABLE,
    PrefectStage,
    build_phase4_stage_plan,
    render_phase4_stage_plan,
    resolve_phase4_runtime_options,
    run_phase4_prefect_flow,
    run_workflow_mode,
)
from plasmid_priority.settings import AppSettings


def test_settings_resolve_data_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLASMID_PRIORITY_DATA_ROOT", str(tmp_path / "data_override"))
    settings = AppSettings()
    assert settings.data_root is not None
    assert settings.data_root.name == "data_override"


def test_phase4_stage_plan_has_expected_dependencies() -> None:
    stages = build_phase4_stage_plan(include_fetch=True, run_release=True)
    by_name = {stage.name: stage for stage in stages}
    assert isinstance(by_name["core_refresh"], PrefectStage)
    assert by_name["core_refresh"].deps == ("fetch_external",)
    assert by_name["support_refresh"].deps == ("core_refresh",)
    assert by_name["reports_only"].deps == ("core_refresh", "support_refresh")
    assert by_name["release_bundle"].deps == ("reports_only",)


def test_phase4_stage_plan_render_has_dependency_text() -> None:
    stages = build_phase4_stage_plan(include_fetch=True, run_release=True)
    rendered = render_phase4_stage_plan(stages)
    assert rendered[0].startswith("fetch_external: mode=fetch-external")
    assert any("deps=core_refresh,support_refresh" in line for line in rendered)
    assert rendered[-1].endswith("deps=reports_only")


@given(include_fetch=st.booleans(), run_release=st.booleans())
def test_phase4_stage_plan_is_topologically_valid(include_fetch: bool, run_release: bool) -> None:
    stages = build_phase4_stage_plan(include_fetch=include_fetch, run_release=run_release)
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names))
    seen: set[str] = set()
    for stage in stages:
        for dep in stage.deps:
            assert dep in names
            assert dep in seen
        seen.add(stage.name)


def test_resolve_phase4_runtime_options_uses_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSettings:
        max_jobs = 9
        data_root = Path("data/override")

        @staticmethod
        def resolved_data_root(project_root: Path | None = None) -> Path:
            if project_root is None:
                return Path("/tmp/fallback")
            return (project_root / "data" / "override").resolve()

    monkeypatch.setattr(
        "plasmid_priority.pipeline.prefect_flow.get_settings",
        lambda: _FakeSettings(),
    )
    workers, data_root = resolve_phase4_runtime_options(max_workers=None, data_root=None)
    assert workers == 9
    assert isinstance(data_root, str)
    assert data_root.endswith("/data/override")


def test_run_workflow_mode_invokes_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(  # type: ignore[no-untyped-def]
        command,
        cwd=None,
        env=None,
        check=False,
        timeout=None,
    ):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout

        class _Completed:
            returncode = 0

        return _Completed()

    monkeypatch.setattr("plasmid_priority.pipeline.prefect_flow.subprocess.run", _fake_run)
    exit_code = run_workflow_mode("core-refresh", max_workers=3, data_root="/tmp/plasmid-priority-data")
    assert exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "scripts/run_workflow.py" in " ".join(str(part) for part in command)
    assert "core-refresh" in command
    assert "--max-workers" in command
    assert "--dry-run" not in command
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("PLASMID_PRIORITY_DATA_ROOT") == "/tmp/plasmid-priority-data"
    timeout_value = captured["timeout"]
    assert isinstance(timeout_value, int)
    assert timeout_value >= 1


def test_run_workflow_mode_appends_dry_run_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(  # type: ignore[no-untyped-def]
        command,
        cwd=None,
        env=None,
        check=False,
        timeout=None,
    ):
        captured["command"] = command
        captured["timeout"] = timeout

        class _Completed:
            returncode = 0

        return _Completed()

    monkeypatch.setattr("plasmid_priority.pipeline.prefect_flow.subprocess.run", _fake_run)
    monkeypatch.setenv("PLASMID_PRIORITY_WORKFLOW_TIMEOUT_SECONDS", "321")
    exit_code = run_workflow_mode("reports-only", max_workers=1, dry_run=True)
    assert exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "--dry-run" in command
    assert captured["timeout"] == 321


def test_phase4_artifacts_exist_and_are_well_formed() -> None:
    root = Path(__file__).resolve().parents[1]
    dvc_path = root / "dvc.yaml"
    just_path = root / "Justfile"
    flake_path = root / "flake.nix"
    assert dvc_path.exists()
    assert just_path.exists()
    assert flake_path.exists()

    dvc_payload = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
    assert isinstance(dvc_payload, dict)
    assert "stages" in dvc_payload
    assert "fetch_external" in dvc_payload["stages"]
    assert "core_refresh" in dvc_payload["stages"]
    assert "reports_only" in dvc_payload["stages"]
    assert "release_bundle" in dvc_payload["stages"]

    just_content = just_path.read_text(encoding="utf-8")
    assert "workflow" in just_content
    assert "prefect-plan" in just_content
    assert "prefect" in just_content
    assert "--extra engineering" in just_content
    assert "mutation" in just_content
    assert "--extra dev" in just_content

    flake_content = flake_path.read_text(encoding="utf-8")
    assert "devShells" in flake_content
    assert "dvc" in flake_content
    assert "just" in flake_content


def test_prefect_flow_unavailable_raises() -> None:
    if PREFECT_AVAILABLE:
        pytest.skip("Prefect is available in this environment.")
    with pytest.raises(RuntimeError, match="Prefect is not installed"):
        run_phase4_prefect_flow()
