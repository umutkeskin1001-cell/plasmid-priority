"""Inventory helpers for the geo spread branch."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


def _iter_project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__", ".mypy_cache"} for part in path.parts):
            continue
        files.append(path.relative_to(root))
    return sorted(files)


def _python_module_map(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in root.rglob("*.py"):
        if any(part in {".git", ".pytest_cache", "__pycache__", ".mypy_cache"} for part in path.parts):
            continue
        rel = path.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        if module.startswith("src."):
            module = module[len("src.") :]
        mapping[module] = rel
    return mapping


def _geo_spread_test_paths(root: Path) -> list[Path]:
    candidates = list((root / "tests").glob("test_geo_spread*.py"))
    candidates.extend((root / "tests" / "geo_spread").rglob("*.py"))
    return sorted(path.relative_to(root) for path in candidates if path.is_file())


def _static_used_code_paths(root: Path) -> set[Path]:
    module_map = _python_module_map(root)
    queue: list[Path] = []
    queue.extend((root / "src" / "plasmid_priority" / "geo_spread").rglob("*.py"))
    queue.extend(path for path in _geo_spread_test_paths(root))
    for script_path in (
        root / "scripts" / "run_geo_spread_branch.py",
        root / "scripts" / "geo_spread" / "run_branch.py",
    ):
        if script_path.exists():
            queue.append(script_path)
    used: set[Path] = set()
    seen: set[Path] = set()
    while queue:
        current = Path(queue.pop())
        current_abs = current if current.is_absolute() else (root / current)
        if current_abs in seen or not current_abs.exists():
            continue
        seen.add(current_abs)
        rel = current_abs.relative_to(root)
        used.add(rel)
        try:
            tree = ast.parse(current_abs.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]
            else:
                continue
            for module_name in module_names:
                candidates = [module_name, f"plasmid_priority.{module_name}"]
                for candidate in candidates:
                    target = module_map.get(candidate)
                    if target is not None:
                        queue.append(target)
    return used


def build_geo_spread_inventory(
    root: Path,
    *,
    branch_data_root: Path,
    legacy_input_path: Path | None = None,
    legacy_records_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build explicit used and unused path inventories for the geo spread branch."""
    all_files = _iter_project_files(root)
    used_paths = _static_used_code_paths(root)
    explicit_paths = {
        Path("config.yaml"),
    }
    if legacy_input_path is not None and legacy_input_path.exists():
        explicit_paths.add(legacy_input_path.relative_to(root))
    if legacy_records_path is not None and legacy_records_path.exists():
        explicit_paths.add(legacy_records_path.relative_to(root))
    if branch_data_root.exists():
        for path in branch_data_root.rglob("*"):
            if path.is_file():
                explicit_paths.add(path.relative_to(root))
    used_paths.update(explicit_paths)

    def _category(path: Path) -> str:
        parts = path.parts
        if not parts:
            return "other"
        if parts[0] == "src":
            return "code"
        if parts[0] == "scripts":
            return "script"
        if parts[0] == "tests":
            return "test"
        if parts[0] == "data":
            return "data"
        if parts[0] == "config.yaml":
            return "config"
        return "other"

    used_rows = [{"path": str(path), "category": _category(path)} for path in sorted(used_paths)]
    unused_rows = [
        {"path": str(path), "category": _category(path)}
        for path in all_files
        if path not in used_paths
    ]
    summary = {
        "used_file_count": int(len(used_rows)),
        "unused_file_count": int(len(unused_rows)),
        "used_data_file_count": int(sum(row["category"] == "data" for row in used_rows)),
        "unused_data_file_count": int(sum(row["category"] == "data" for row in unused_rows)),
        "branch_data_root": str(branch_data_root.relative_to(root)) if branch_data_root.exists() else str(branch_data_root),
    }
    return pd.DataFrame(used_rows), pd.DataFrame(unused_rows), summary
