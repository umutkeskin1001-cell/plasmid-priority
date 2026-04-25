#!/usr/bin/env python3
"""Lightweight import contract gate for layered architecture."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "plasmid_priority"
ALLOWLIST_PATH = PROJECT_ROOT / "config" / "import_contract_allowlist.txt"

LAYER_ORDER = [
    "utils",
    "exceptions",
    "schemas",
    "protocol",
    "config",
    "io",
    "features",
    "qc",
    "scoring",
    "validation",
    "modeling",
    "shared",
    "geo_spread",
    "bio_transfer",
    "clinical_hazard",
    "consensus",
    "reporting",
    "api",
    "pipeline",
]

# Validation is allowed to depend on scoring but not modeling.
FORBIDDEN_IMPORT_PREFIXES = {
    "plasmid_priority.validation": ("plasmid_priority.modeling",),
}

LAYER_INDEX = {name: i for i, name in enumerate(LAYER_ORDER)}


@dataclass(frozen=True)
class Violation:
    file: Path
    line: int
    message: str


def _module_layer(module_name: str) -> str | None:
    prefix = "plasmid_priority."
    if not module_name.startswith(prefix):
        return None
    tail = module_name[len(prefix) :]
    top = tail.split(".", 1)[0]
    return top if top in LAYER_INDEX else None


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _extract_imports(tree: ast.AST) -> list[tuple[str, int]]:
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
    return imports


def check_import_contracts() -> list[Violation]:
    allowlist = _load_allowlist(ALLOWLIST_PATH)
    violations: list[Violation] = []
    for file_path in _iter_python_files(SRC_ROOT):
        module_name = "plasmid_priority." + str(file_path.relative_to(SRC_ROOT)).replace(
            "/",
            ".",
        ).removesuffix(".py")
        source_layer = _module_layer(module_name)
        if source_layer is None:
            continue

        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for imported_module, lineno in _extract_imports(tree):
            target_layer = _module_layer(imported_module)
            if target_layer is not None and LAYER_INDEX[target_layer] > LAYER_INDEX[source_layer]:
                if not _is_allowlisted(allowlist, module_name, imported_module):
                    violations.append(
                        Violation(
                            file=file_path,
                            line=lineno,
                            message=(
                                f"layer breach: `{source_layer}` imported higher layer "
                                f"`{target_layer}`"
                            ),
                        ),
                    )

            for forbidden_prefix in FORBIDDEN_IMPORT_PREFIXES.get(module_name, ()):
                if imported_module == forbidden_prefix or imported_module.startswith(
                    f"{forbidden_prefix}.",
                ):
                    if not _is_allowlisted(allowlist, module_name, imported_module):
                        violations.append(
                            Violation(
                                file=file_path,
                                line=lineno,
                                message=(
                                    f"forbidden import: `{module_name}` -> `{imported_module}`"
                                ),
                            ),
                        )

            for module_prefix, blocked in FORBIDDEN_IMPORT_PREFIXES.items():
                if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
                    for blocked_prefix in blocked:
                        if imported_module == blocked_prefix or imported_module.startswith(
                            f"{blocked_prefix}.",
                        ):
                            if not _is_allowlisted(allowlist, module_name, imported_module):
                                violations.append(
                                    Violation(
                                        file=file_path,
                                        line=lineno,
                                        message=(
                                            f"forbidden import: `{module_name}` -> "
                                            f"`{imported_module}`"
                                        ),
                                    ),
                                )
    return violations


def _load_allowlist(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    allowlist: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "->" not in stripped:
            continue
        source, target = [part.strip() for part in stripped.split("->", 1)]
        if source and target:
            allowlist.add((source, target))
    return allowlist


def _is_allowlisted(
    allowlist: set[tuple[str, str]],
    source_module: str,
    imported_module: str,
) -> bool:
    for source_prefix, import_prefix in allowlist:
        source_match = source_module == source_prefix or source_module.startswith(
            f"{source_prefix}.",
        )
        target_match = imported_module == import_prefix or imported_module.startswith(
            f"{import_prefix}.",
        )
        if source_match and target_match:
            return True
    return False


def main() -> int:
    violations = check_import_contracts()
    if not violations:
        print("Import contract check passed.")
        return 0
    for violation in violations:
        print(f"{violation.file}:{violation.line}: {violation.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
