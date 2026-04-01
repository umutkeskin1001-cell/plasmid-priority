from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest


from plasmid_priority.config import build_context, find_project_root, load_data_contract


class ConfigTests(unittest.TestCase):
    def test_find_project_root_from_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "data/manifests").mkdir(parents=True)
            (root / "data/manifests/data_contract.json").write_text(
                json.dumps({"version": 1, "created_on": "2026-03-22", "download_date": "2026-03-22", "assets": []}),
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
                                "description": "x"
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


if __name__ == "__main__":
    unittest.main()
