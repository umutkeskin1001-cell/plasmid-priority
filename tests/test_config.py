from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib import reload
from pathlib import Path
from unittest import mock

import plasmid_priority.config as config_module
from plasmid_priority.config import (
    DATA_ROOT_ENV_VAR,
    build_context,
    find_project_root,
    load_data_contract,
    load_project_config,
    resolve_data_root,
)


class ConfigTests(unittest.TestCase):
    def test_find_project_root_from_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir).resolve()
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [],
                    }
                ),
                encoding="utf-8",
            )
            nested = root / "src/plasmid_priority"
            nested.mkdir(parents=True)
            self.assertEqual(find_project_root(nested), root.resolve())

    def test_load_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "notes": ["hello"],
                        "assets": [
                            {
                                "key": "x",
                                "relative_path": "data/x.txt",
                                "kind": "file",
                                "stage": "core",
                                "required": True,
                                "description": "x",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            contract = load_data_contract(root)
            context = build_context(root)
            self.assertEqual(contract.assets[0].key, "x")
            self.assertEqual(context.asset_path("x"), (root / "data/x.txt").resolve())

    def test_project_config_declares_governance_track(self) -> None:
        context = build_context()
        models = context.config["models"]
        self.assertEqual(models["primary_model_name"], "discovery_12f_source")
        self.assertEqual(models["conservative_model_name"], "parsimonious_priority")
        self.assertEqual(models["governance_model_name"], "phylo_support_fusion_priority")
        self.assertEqual(models["governance_model_fallback"], "support_synergy_priority")
        self.assertEqual(
            models["fit_config"]["regime_stability_priority"]["preprocess_alpha"], "auto"
        )
        self.assertEqual(context.pipeline_settings.host_evenness_bias_power, 0.5)
        self.assertAlmostEqual(context.pipeline_settings.host_phylo_breadth_weight, 0.65)
        self.assertAlmostEqual(context.pipeline_settings.host_phylo_dispersion_weight, 0.35)
        self.assertTrue(
            bool(models["fit_config"]["regime_stability_priority"]["preprocess_alpha_grouped"])
        )

    def test_load_project_config_merges_layered_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "config.yaml").write_text(
                "pipeline:\n  split_year: 2014\nmodels:\n  primary_model_name: root_model\n",
                encoding="utf-8",
            )
            (root / "config").mkdir(parents=True)
            (root / "config" / "10_models.yaml").write_text(
                "models:\n  primary_model_name: layered_model\n  governance_model_name: layered_governance\n",
                encoding="utf-8",
            )
            (root / "config" / "20_pipeline.yaml").write_text(
                "pipeline:\n  split_year: 2017\n  host_evenness_bias_power: 0.25\n",
                encoding="utf-8",
            )
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [],
                    }
                ),
                encoding="utf-8",
            )
            project_config = load_project_config(root)
            context = build_context(root)
            self.assertEqual(project_config.pipeline.split_year, 2017)
            self.assertEqual(project_config.pipeline.host_evenness_bias_power, 0.25)
            self.assertEqual(context.config["models"]["primary_model_name"], "layered_model")
            self.assertEqual(
                context.config["models"]["governance_model_name"], "layered_governance"
            )
            self.assertEqual(
                context.config_paths,
                (
                    (root / "config.yaml").resolve(),
                    (root / "config" / "10_models.yaml").resolve(),
                    (root / "config" / "20_pipeline.yaml").resolve(),
                ),
            )

    def test_context_uses_explicit_external_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            external_data = root / "usb-data"
            external_data.mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [
                            {
                                "key": "x",
                                "relative_path": "data/x.txt",
                                "kind": "file",
                                "stage": "core",
                                "required": True,
                                "description": "x",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = build_context(root, data_root=external_data)
            self.assertEqual(context.data_dir, external_data.resolve())
            self.assertEqual(context.asset_path("x"), (external_data / "x.txt").resolve())

    def test_resolve_data_root_reads_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            external = root / "external-data"
            external.mkdir()
            with mock.patch.dict(os.environ, {DATA_ROOT_ENV_VAR: str(external)}, clear=False):
                self.assertEqual(resolve_data_root(root), external.resolve())

    def test_project_context_config_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "config.yaml").write_text(
                "models:\n  primary_model_name: test\n", encoding="utf-8"
            )
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "created_on": "2026-03-22",
                        "download_date": "2026-03-22",
                        "assets": [],
                    }
                ),
                encoding="utf-8",
            )
            # Ensure fresh module state for the cached_property test.
            reload(config_module)
            context = config_module.build_context(root)
            with mock.patch("yaml.safe_load", wraps=__import__("yaml").safe_load) as safe_load_mock:
                first = context.config
                second = context.config
            self.assertEqual(first, second)
            self.assertEqual(safe_load_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
