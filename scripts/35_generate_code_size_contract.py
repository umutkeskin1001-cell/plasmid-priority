#!/usr/bin/env python3
"""Generate code-size contract report for files and top-level functions."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("src/plasmid_priority", "scripts")


@dataclass(frozen=True)
class Thresholds:
    max_file_lines: int = 1500
    max_function_lines: int = 200
    max_main_lines: int = 80


@dataclass(frozen=True)
class FileViolation:
    path: str
    line_count: int
    limit: int


@dataclass(frozen=True)
class FunctionViolation:
    path: str
    function: str
    start_line: int
    end_line: int
    line_count: int
    limit: int


def _iter_python_files(target_roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for root in target_roots:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        files.extend(sorted(path for path in base.rglob("*.py") if path.is_file()))
    return files


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _function_violations(path: Path, thresholds: Thresholds) -> list[FunctionViolation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[FunctionViolation] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        end_line = getattr(node, "end_lineno", node.lineno)
        length = int(end_line - node.lineno + 1)
        limit = thresholds.max_main_lines if node.name == "main" else thresholds.max_function_lines
        if length > limit:
            violations.append(
                FunctionViolation(
                    path=str(path.relative_to(PROJECT_ROOT)),
                    function=node.name,
                    start_line=int(node.lineno),
                    end_line=int(end_line),
                    line_count=length,
                    limit=limit,
                )
            )
    return violations


def _build_report(target_roots: tuple[str, ...], thresholds: Thresholds) -> dict[str, object]:
    files = _iter_python_files(target_roots)
    file_violations: list[FileViolation] = []
    function_violations: list[FunctionViolation] = []
    for path in files:
        count = _line_count(path)
        if count > thresholds.max_file_lines:
            file_violations.append(
                FileViolation(
                    path=str(path.relative_to(PROJECT_ROOT)),
                    line_count=count,
                    limit=thresholds.max_file_lines,
                )
            )
        function_violations.extend(_function_violations(path, thresholds))
    return {
        "thresholds": asdict(thresholds),
        "scanned_file_count": len(files),
        "file_violations": [asdict(item) for item in file_violations],
        "function_violations": [asdict(item) for item in function_violations],
        "status": "pass" if not file_violations and not function_violations else "fail",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "quality" / "code_size_contract.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit with non-zero code when contract violations exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _build_report(DEFAULT_TARGETS, Thresholds())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote code-size contract report to {args.output}")
    if args.fail_on_violations and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
