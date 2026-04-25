from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_module_a_script",
    PROJECT_ROOT / "scripts/16_run_module_A.py",
)
assert SPEC is not None and SPEC.loader is not None
run_module_a_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_module_a_script
SPEC.loader.exec_module(run_module_a_script)


def test_load_compute_tiers_uses_defaults_for_missing_file() -> None:
    missing = PROJECT_ROOT / "config" / "__missing_model_compute_tiers__.yaml"
    tiers = run_module_a_script._load_compute_tiers(missing)
    assert tiers["smoke"]["n_splits"] == 2
    assert tiers["release-full"]["n_repeats"] == 5


def test_load_compute_tiers_merges_custom_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        custom = Path(tmp_dir) / "model_compute_tiers.yaml"
        custom.write_text(
            yaml.safe_dump(
                {
                    "tiers": {
                        "dev": {"n_splits": 4, "n_repeats": 2},
                    },
                },
            ),
            encoding="utf-8",
        )
        tiers = run_module_a_script._load_compute_tiers(custom)
        assert tiers["dev"]["n_splits"] == 4
        assert tiers["dev"]["n_repeats"] == 2
        assert tiers["smoke"]["n_splits"] == 2
