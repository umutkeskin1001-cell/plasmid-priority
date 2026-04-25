#!/usr/bin/env python3
"""Generate current hardening state snapshot.

This script creates a compact artifact summarizing the current hardening state,
covering both code-level hardening (always available) and data-dependent audits
(when data is present).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context

SNAPSHOT_AUDIT_CACHE_NAME = "current_hardening_snapshot_audit_cache.json"


def check_precommit_status() -> dict[str, Any]:
    """Check if pre-commit hooks are configured and installed."""
    config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
    git_hooks_path = PROJECT_ROOT / ".git" / "hooks" / "pre-commit"

    result = {
        "config_exists": config_path.exists(),
        "hooks_installed": git_hooks_path.exists(),
        "status": "not_installed",
    }

    if result["config_exists"] and result["hooks_installed"]:
        result["status"] = "installed"
    elif result["config_exists"]:
        result["status"] = "config_only"

    return result


def check_hardening_modules() -> dict[str, Any]:
    """Check which hardening modules are available in the codebase."""
    src_path = PROJECT_ROOT / "src" / "plasmid_priority"

    modules = {
        "country_macro_region_mapping": (src_path / "utils" / "geography.py").exists(),
        "clean_text_deduplication": (src_path / "utils" / "dataframe.py").exists(),
        "dominant_share_util": (src_path / "utils" / "dataframe.py").exists(),
        "average_precision_metric": (src_path / "validation" / "metrics.py").exists(),
        "epv_audit": (src_path / "reporting" / "epv_audit.py").exists(),
        "lead_time_bias_audit": (src_path / "reporting" / "lead_time_bias_audit.py").exists(),
        "missingness_audit": (src_path / "validation" / "missingness.py").exists(),
        "schema_validation": (src_path / "validation" / "schemas.py").exists(),
        "consolidated_hardening_summary": (
            src_path / "reporting" / "hardening_summary.py"
        ).exists(),
    }

    return {
        "modules_available": modules,
        "total_modules": len(modules),
        "available_count": sum(modules.values()),
        "status": "available" if all(modules.values()) else "partial",
    }


def check_hardening_scripts() -> dict[str, Any]:
    """Check which hardening scripts are available."""
    scripts_path = PROJECT_ROOT / "scripts"

    scripts = {
        "missingness_audit": (scripts_path / "run_missingness_audit.py").exists(),
        "schema_validation": (scripts_path / "run_schema_validation.py").exists(),
        "consolidated_hardening_summary": (scripts_path / "run_hardening_summary.py").exists(),
        "advanced_audits": (scripts_path / "27_run_advanced_audits.py").exists(),
    }

    return {
        "scripts_available": scripts,
        "total_scripts": len(scripts),
        "available_count": sum(scripts.values()),
        "status": "available" if all(scripts.values()) else "partial",
    }


def _data_audit_paths() -> dict[str, Path]:
    context = build_context(PROJECT_ROOT)
    return {
        "backbone_table": context.resolve_path("data/features/backbone_table.tsv"),
        "scored_backbone": context.resolve_path("data/scores/backbone_scored.tsv"),
        "harmonized_plasmids": context.resolve_path("data/harmonized/harmonized_plasmids.tsv"),
        "deduplicated_plasmids": context.resolve_path(
            "data/deduplicated/deduplicated_plasmids.tsv",
        ),
    }


def _audit_cache_path() -> Path:
    context = build_context(PROJECT_ROOT)
    return context.reports_dir / SNAPSHOT_AUDIT_CACHE_NAME


def _data_file_signatures(data_paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    signatures: dict[str, dict[str, object]] = {}
    for key, path in data_paths.items():
        if path.exists():
            stat = path.stat()
            signatures[key] = {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        else:
            signatures[key] = {"path": str(path.resolve()), "missing": True}
    return signatures


def _load_cached_audit_results(signatures: dict[str, dict[str, object]]) -> dict[str, Any] | None:
    cache_path = _audit_cache_path()
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("signatures") != signatures:
        return None
    results = payload.get("results")
    return results if isinstance(results, dict) else None


def _write_cached_audit_results(
    signatures: dict[str, dict[str, object]],
    results: dict[str, Any],
) -> None:
    cache_path = _audit_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cached_at": datetime.now().isoformat(),
        "signatures": signatures,
        "results": results,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def check_data_dependent_audits() -> dict[str, Any]:
    """Check if data exists for data-dependent audits."""
    data_paths = _data_audit_paths()

    available = {k: v.exists() for k, v in data_paths.items()}
    any_data = any(available.values())

    return {
        "data_files": available,
        "any_data_available": any_data,
        "status": "available" if any_data else "no_data",
    }


def run_data_audits_if_available() -> dict[str, Any]:
    """Run data-dependent audits if data is available."""
    data_check = check_data_dependent_audits()

    if not data_check["any_data_available"]:
        return {
            "epv_audit": {"status": "skipped_no_data"},
            "lead_time_bias_audit": {"status": "skipped_no_data"},
            "missingness_audit": {"status": "skipped_no_data"},
            "schema_validation": {"status": "skipped_no_data"},
        }

    data_paths = _data_audit_paths()
    signatures = _data_file_signatures(data_paths)
    cached_results = _load_cached_audit_results(signatures)
    if cached_results is not None:
        return cached_results

    results = {}

    # Try to import and run the consolidated hardening summary
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import pandas as pd

        from plasmid_priority.reporting.hardening_summary import (
            build_hardening_audit_summary,
        )

        # Load available tables
        backbone_table = None
        scored_table = None
        harmonized_table = None
        deduplicated_table = None

        if data_check["data_files"].get("backbone_table"):
            backbone_table = pd.read_csv(data_paths["backbone_table"], sep="\t")
        if data_check["data_files"].get("scored_backbone"):
            scored_table = pd.read_csv(data_paths["scored_backbone"], sep="\t")
        if data_check["data_files"].get("harmonized_plasmids"):
            harmonized_table = pd.read_csv(data_paths["harmonized_plasmids"], sep="\t")
        if data_check["data_files"].get("deduplicated_plasmids"):
            deduplicated_table = pd.read_csv(data_paths["deduplicated_plasmids"], sep="\t")

        summary = build_hardening_audit_summary(
            backbone_table=backbone_table,
            scored_backbone_table=scored_table,
            harmonized_plasmids=harmonized_table,
            deduplicated_plasmids=deduplicated_table,
        )

        results["epv_audit"] = {
            "status": summary.get("models", {}).get("epv", {}).get("status", "unknown"),
            "models_evaluated": summary.get("models", {})
            .get("epv", {})
            .get("n_models_evaluated", 0),
        }
        results["lead_time_bias_audit"] = {
            "status": summary.get("lead_time_bias", {}).get("status", "unknown"),
            "concern_level": summary.get("lead_time_bias", {}).get(
                "overall_concern_level",
                "unknown",
            ),
        }
        results["missingness_audit"] = {
            "status": summary.get("missingness", {}).get("overall_status", "unknown"),
            "high_missingness_columns": summary.get("missingness", {}).get(
                "high_missingness_columns_total",
                0,
            ),
        }
        results["schema_validation"] = {
            "status": summary.get("schema_validation", {}).get("overall_status", "unknown"),
            "pandera_available": summary.get("schema_validation", {}).get(
                "pandera_available",
                False,
            ),
            "tables_validated": len(
                summary.get("schema_validation", {}).get("tables_validated", []),
            ),
        }

    except Exception as e:
        results = {
            "epv_audit": {"status": "error", "error": str(e)},
            "lead_time_bias_audit": {"status": "error", "error": str(e)},
            "missingness_audit": {"status": "error", "error": str(e)},
            "schema_validation": {"status": "error", "error": str(e)},
        }

    _write_cached_audit_results(signatures, results)
    return results


def build_hardening_snapshot() -> dict[str, Any]:
    """Build the complete hardening snapshot."""
    snapshot = {
        "snapshot_timestamp": datetime.now().isoformat(),
        "snapshot_version": "1.0.0",
        "overall_status": "unknown",
    }

    # Code-level hardening (always check)
    snapshot["code_hardening"] = {  # type: ignore
        "pre_commit": check_precommit_status(),
        "modules": check_hardening_modules(),
        "scripts": check_hardening_scripts(),
    }

    # Data-dependent audits (check when data available)
    snapshot["data_audits"] = run_data_audits_if_available()  # type: ignore
    snapshot["data_availability"] = check_data_dependent_audits()  # type: ignore

    # Compute overall status
    code_status = (
        "available"
        if (
            snapshot["code_hardening"]["modules"]["status"] == "available"  # type: ignore
            and snapshot["code_hardening"]["scripts"]["status"] == "available"  # type: ignore
        )
        else "partial"
    )

    if snapshot["data_availability"]["any_data_available"]:  # type: ignore
        # If data exists, overall status depends on audit results
        # Priority order: error > fail > concern > incomplete > ok
        audit_statuses = [
            snapshot["data_audits"]["epv_audit"]["status"],  # type: ignore
            snapshot["data_audits"]["lead_time_bias_audit"]["status"],  # type: ignore
            snapshot["data_audits"]["missingness_audit"]["status"],  # type: ignore
            snapshot["data_audits"]["schema_validation"]["status"],  # type: ignore
        ]
        if any(s == "error" for s in audit_statuses):
            snapshot["overall_status"] = "error"
        elif any(s == "fail" for s in audit_statuses):
            snapshot["overall_status"] = "fail"
        elif any(s == "concern" for s in audit_statuses):
            snapshot["overall_status"] = "concern"
        elif any(str(s).startswith("skipped") or s == "unknown" for s in audit_statuses):
            snapshot["overall_status"] = "incomplete"
        else:
            snapshot["overall_status"] = "ok"
    else:
        # No data - status is code-only
        snapshot["overall_status"] = code_status

    return snapshot


def format_snapshot_markdown(snapshot: dict[str, Any]) -> str:
    """Format snapshot as Markdown."""
    lines = [
        "# Current Hardening State Snapshot",
        "",
        f"**Generated:** {snapshot.get('snapshot_timestamp', 'unknown')}",
        f"**Overall Status:** {snapshot.get('overall_status', 'unknown').upper()}",
        "",
        "## Summary",
        "",
        "This snapshot captures the current hardening state of the plasmid-priority repository.",
        "Hardening includes code-level improvements, pre-commit hooks, and data quality audits.",
        "",
        "## Code-Level Hardening",
        "",
    ]

    # Pre-commit section
    precommit = snapshot.get("code_hardening", {}).get("pre_commit", {})
    lines.extend(
        [
            "### Pre-commit Hooks",
            "",
            f"- **Config exists:** {precommit.get('config_exists', False)}",
            f"- **Hooks installed:** {precommit.get('hooks_installed', False)}",
            f"- **Status:** {precommit.get('status', 'unknown')}",
            "",
        ],
    )

    if precommit.get("status") == "config_only":
        lines.extend(
            [
                "*To install: `pip install pre-commit && pre-commit install`*",
                "",
            ],
        )

    # Modules section
    modules = snapshot.get("code_hardening", {}).get("modules", {})
    lines.extend(
        [
            "### Hardening Modules",
            "",
            f"- **Total modules:** {modules.get('total_modules', 0)}",
            f"- **Available:** {modules.get('available_count', 0)}",
            f"- **Status:** {modules.get('status', 'unknown')}",
            "",
        ],
    )

    if modules.get("modules_available"):
        lines.append("| Module | Status |")
        lines.append("|--------|--------|")
        for name, available in sorted(modules["modules_available"].items()):
            status = "available" if available else "missing"
            lines.append(f"| {name} | {status} |")
        lines.append("")

    # Scripts section
    scripts = snapshot.get("code_hardening", {}).get("scripts", {})
    lines.extend(
        [
            "### Hardening Scripts",
            "",
            f"- **Total scripts:** {scripts.get('total_scripts', 0)}",
            f"- **Available:** {scripts.get('available_count', 0)}",
            f"- **Status:** {scripts.get('status', 'unknown')}",
            "",
        ],
    )

    if scripts.get("scripts_available"):
        lines.append("| Script | Status |")
        lines.append("|--------|--------|")
        for name, available in sorted(scripts["scripts_available"].items()):
            status = "available" if available else "missing"
            lines.append(f"| {name} | {status} |")
        lines.append("")

    # Data-dependent section
    data_avail = snapshot.get("data_availability", {})
    lines.extend(
        [
            "## Data-Dependent Audits",
            "",
        ],
    )

    if data_avail.get("any_data_available"):
        lines.extend(
            [
                "Data files are available. Audit results:",
                "",
            ],
        )

        audits = snapshot.get("data_audits", {})
        lines.append("| Audit | Status | Details |")
        lines.append("|-------|--------|---------|")

        epv = audits.get("epv_audit", {})
        lines.append(
            f"| EPV Audit | {epv.get('status', 'unknown')} | {epv.get('models_evaluated', 'N/A')} models |",
        )

        lt = audits.get("lead_time_bias_audit", {})
        lines.append(
            f"| Lead-time Bias | {lt.get('status', 'unknown')} | concern={lt.get('concern_level', 'N/A')} |",
        )

        miss = audits.get("missingness_audit", {})
        lines.append(
            f"| Missingness | {miss.get('status', 'unknown')} | {miss.get('high_missingness_columns', 0)} high-missing cols |",
        )

        schema = audits.get("schema_validation", {})
        lines.append(
            f"| Schema Validation | {schema.get('status', 'unknown')} | {schema.get('tables_validated', 0)} tables |",
        )
        lines.append("")
    else:
        lines.extend(
            [
                "**No data files available.** Data-dependent audits are skipped.",
                "",
                "Expected data files:",
                "- `data/features/backbone_table.tsv`",
                "- `data/scores/backbone_scored.tsv`",
                "- `data/harmonized/harmonized_plasmids.tsv`",
                "- `data/deduplicated/deduplicated_plasmids.tsv`",
                "",
                "*Run the pipeline to generate data, then re-run this snapshot.*",
                "",
            ],
        )

    lines.extend(
        [
            "---",
            "",
            "*This snapshot is generated by `scripts/generate_hardening_snapshot.py`*",
            "",
        ],
    )

    return "\n".join(lines)


def main() -> int:
    """Generate hardening snapshot files."""
    context = build_context(PROJECT_ROOT)
    reports_dir = context.reports_dir
    snapshot = build_hardening_snapshot()

    # Write JSON
    json_path = reports_dir / "current_hardening_snapshot.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"Wrote JSON: {json_path}")

    # Write Markdown
    md_path = reports_dir / "current_hardening_snapshot.md"
    with open(md_path, "w") as f:
        f.write(format_snapshot_markdown(snapshot))
    print(f"Wrote Markdown: {md_path}")

    print(f"\nOverall status: {snapshot['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
