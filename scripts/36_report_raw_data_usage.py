#!/usr/bin/env python3
"""Audit raw-data usage and report potentially unused files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasmid_priority.config import build_context
from plasmid_priority.utils.files import ensure_directory


def _collect_recent_raw_inputs(logs_dir: Path, project_root: Path, data_dir: Path) -> set[Path]:
    used: set[Path] = set()
    if not logs_dir.exists():
        return used
    for summary_path in logs_dir.glob("*_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        manifest = payload.get("input_manifest", {})
        if not isinstance(manifest, dict):
            continue
        for entry in manifest.values():
            if not isinstance(entry, dict):
                continue
            path_value = entry.get("path")
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.is_absolute():
                if str(path).startswith("data/"):
                    path = data_dir / Path(*path.parts[1:])
                else:
                    path = project_root / path
            resolved = path.resolve()
            if "/data/raw/" in str(resolved):
                used.add(resolved)
    return used


def _load_code_corpus(project_root: Path) -> str:
    chunks: list[str] = []
    for root_name in ("scripts", "src", "config"):
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt"}:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    context = build_context(args.project_root)
    raw_root = context.data_dir / "raw"
    logs_dir = context.data_dir / "tmp" / "logs"
    report_dir = ensure_directory(context.root / "reports" / "audits")
    md_report_path = report_dir / "raw_data_usage_report.md"
    json_report_path = report_dir / "raw_data_usage_report.json"

    if not raw_root.exists():
        payload = {"status": "no_raw_dir", "raw_root": str(raw_root)}
        json_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_report_path.write_text(
            "# Raw Data Usage Report\n\nNo raw directory found.\n", encoding="utf-8"
        )
        return 0

    all_raw_files = sorted(
        path.resolve()
        for path in raw_root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    recent_used = _collect_recent_raw_inputs(logs_dir, context.root, context.data_dir)
    code_corpus = _load_code_corpus(context.root)

    context.contract.asset_map()
    declared_raw_paths: set[Path] = set()
    for asset in context.contract.assets:
        if not asset.relative_path.startswith("data/raw/"):
            continue
        resolved = asset.resolved_path(context.root, context.data_dir).resolve()
        if asset.kind.value == "file":
            declared_raw_paths.add(resolved)
        elif asset.kind.value == "directory" and resolved.exists():
            declared_raw_paths.update(
                path.resolve() for path in resolved.rglob("*") if path.is_file()
            )

    records: list[dict[str, object]] = []
    for path in all_raw_files:
        rel = (
            path.relative_to(context.root).as_posix()
            if str(path).startswith(str(context.root))
            else str(path)
        )
        name = path.name
        referenced_in_code = (rel in code_corpus) or (name in code_corpus)
        used_recently = path in recent_used
        declared = path in declared_raw_paths
        records.append(
            {
                "path": rel,
                "declared_in_contract": declared,
                "referenced_in_code": referenced_in_code,
                "used_in_recent_step_inputs": used_recently,
            },
        )

    inactive_recent_candidates = [
        record
        for record in records
        if not record["referenced_in_code"] and not record["used_in_recent_step_inputs"]
    ]
    likely_unused_candidates = [
        record
        for record in records
        if not record["declared_in_contract"]
        and not record["referenced_in_code"]
        and not record["used_in_recent_step_inputs"]
    ]
    undeclared_files = [record for record in records if not record["declared_in_contract"]]

    payload = {
        "raw_file_count": len(records),  # type: ignore
        "inactive_recent_candidate_count": len(inactive_recent_candidates),  # type: ignore
        "likely_unused_candidate_count": len(likely_unused_candidates),  # type: ignore
        "undeclared_raw_file_count": len(undeclared_files),  # type: ignore
        "inactive_recent_candidates": inactive_recent_candidates,  # type: ignore
        "likely_unused_candidates": likely_unused_candidates,  # type: ignore
        "undeclared_raw_files": undeclared_files,  # type: ignore
    }
    json_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Raw Data Usage Report",
        "",
        f"- raw_file_count: `{len(records)}`",
        f"- inactive_recent_candidate_count: `{len(inactive_recent_candidates)}`",
        f"- likely_unused_candidate_count: `{len(likely_unused_candidates)}`",
        f"- undeclared_raw_file_count: `{len(undeclared_files)}`",
        "",
        "## Likely Unused Candidates",
        "",
    ]
    if likely_unused_candidates:
        for item in likely_unused_candidates:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Inactive In Recent Runs", ""])
    if inactive_recent_candidates:
        for item in inactive_recent_candidates:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Undeclared Raw Files", ""])
    if undeclared_files:
        for item in undeclared_files:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("- none")
    lines.append("")
    md_report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {md_report_path}")
    print(f"Wrote {json_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
