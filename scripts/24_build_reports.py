#!/usr/bin/env python3
"""Entrypoint shim for report build workflow.

Full implementation lives in `plasmid_priority.reporting.build_reports_script`.
"""

import json
import os
from pathlib import Path
from shutil import copyfile

from plasmid_priority.reporting import build_reports_script as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})


def _load_report_summary(project_root: Path) -> dict[str, object] | None:
    summary_path = project_root / "data/tmp/logs/24_build_reports_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _iter_analysis_backed_exports(project_root: Path) -> list[tuple[Path, Path]]:
    summary = _load_report_summary(project_root)
    if not summary:
        return []
    analysis_dir = project_root / "data/analysis"
    exports: list[tuple[Path, Path]] = []
    for output in summary.get("output_files_written", []):
        destination = Path(output)
        if not destination.is_absolute():
            destination = project_root / destination
        try:
            relative = destination.relative_to(project_root / "reports")
        except ValueError:
            continue
        source = analysis_dir / destination.name
        if source.exists() and relative.parts:
            exports.append((source, destination))
    return exports


def _sync_analysis_backed_exports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for source, destination in _iter_analysis_backed_exports(project_root):
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
        os.utime(destination, None)


def main() -> int:
    status = int(_impl.main())
    if status == 0:
        _sync_analysis_backed_exports()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
