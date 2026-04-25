#!/usr/bin/env python3
"""Entrypoint shim for report build workflow.

Full implementation lives in `plasmid_priority.reporting.build_reports_script_impl`.
"""

from pathlib import Path
from shutil import copy2

from plasmid_priority.reporting import build_reports_script_impl as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})


def _sync_validation_matrix_exports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    exports = {
        project_root
        / "data/analysis/rolling_temporal_validation.tsv": project_root
        / "reports/diagnostic_tables/rolling_temporal_validation.tsv",
    }
    for source, destination in exports.items():
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)


def main() -> int:
    status = int(_impl.main())
    if status == 0:
        _sync_validation_matrix_exports()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
