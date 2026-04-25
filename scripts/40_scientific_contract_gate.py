#!/usr/bin/env python3
"""Run scientific contract gate checks for release readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from plasmid_priority.validation.scientific_contract import validate_release_scientific_contract


def _list_value(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return list(value) if isinstance(value, list) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    result = validate_release_scientific_contract(args.project_root.resolve())
    if result.get("status") != "pass":
        print("scientific-contract gate failed:")
        for err in _list_value(result, "errors"):
            print(f"  - {err}")
        return 1
    print("scientific-contract gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
