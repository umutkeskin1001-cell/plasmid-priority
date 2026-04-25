#!/usr/bin/env python3
"""Generate disposition ledger for legacy/archive/stub surfaces."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tree_model_usage() -> list[str]:
    output = _run_capture(
        [
            "rg",
            "-n",
            "plasmid_priority.modeling.tree_models|from plasmid_priority.modeling import .*tree|tree_models",
            "src",
            "scripts",
            "tests",
            "-S",
        ],
    )
    if not output:
        return []
    lines = []
    for line in output.splitlines():
        if line.startswith("scripts/34_generate_disposition_ledger.py:"):
            continue
        lines.append(line)
    return lines


def _archive_entries() -> list[str]:
    archive_dir = PROJECT_ROOT / "scripts" / "archive"
    if not archive_dir.exists():
        return []
    return sorted(path.name for path in archive_dir.glob("*.py"))


def _render() -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tree_usage = _tree_model_usage()
    archive_entries = _archive_entries()

    lines: list[str] = [
        "# Disposition Ledger",
        "",
        f"- generated_at_utc: `{generated_at}`",
        "- principle: keep / absorb / replace / delete",
        "",
        "## Decisions",
        "",
        "| Surface | Disposition | Status | Rationale | Next Action |",
        "|---|---|---|---|---|",
        "| `scripts/run_geo_spread_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |",
        "| `scripts/run_bio_transfer_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |",
        "| `scripts/run_clinical_hazard_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |",
        "| `scripts/run_consensus_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |",
        "| `scripts/archive/*` | delete | inactive | Legacy scripts are outside canonical workflow. | Remove after provenance parity check and one release cycle. |",
        "| `src/plasmid_priority/reporting/model_audit_*.py` re-export stubs | delete | completed | Stub-only re-export surfaces removed. | Keep APIs served from `model_audit.py` until real module split lands. |",
        "| `src/plasmid_priority/modeling/tree_models.py` | replace_or_delete | unclear usage | Tree backends should be canonical backend or removed. | If no runtime dependency remains, remove; else integrate into official model surface. |",
        "",
        "## Evidence",
        "",
        "### tree_models usage graph",
    ]
    if tree_usage:
        lines.append("")
        lines.append("```text")
        lines.extend(tree_usage)
        lines.append("```")
    else:
        lines.append("")
        lines.append("No references found.")

    lines.extend(
        [
            "",
            "### scripts/archive inventory",
            "",
        ],
    )
    if archive_entries:
        lines.append("```text")
        lines.extend(archive_entries)
        lines.append("```")
    else:
        lines.append("No archive scripts found.")

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "No new legacy wrapper, archive script, or re-export stub should be added",
            "without an explicit ledger entry and planned deletion path.",
            "",
        ],
    )
    return "\n".join(lines)


def main() -> int:
    output_path = PROJECT_ROOT / "docs" / "disposition_ledger.md"
    output_path.write_text(_render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
