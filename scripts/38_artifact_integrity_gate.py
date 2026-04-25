#!/usr/bin/env python3
"""Validate release artifact integrity and provenance links."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from plasmid_priority.validation.artifact_integrity import validate_release_artifact_integrity


def _list_value(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return list(value) if isinstance(value, list) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    result = validate_release_artifact_integrity(args.project_root.resolve())
    errors = [str(item) for item in _list_value(result, "errors")]

    if errors:
        print("artifact-integrity gate failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("artifact-integrity gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
