"""Pandera schema validation for critical intermediate and derived tables.

This module provides lightweight content validation for high-value tables
in the data pipeline. Schemas validate:
- Required columns exist
- Key identifiers are non-null
- Year columns are within sensible bounds
- Normalized columns are in [0, 1] where appropriate
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd

# Pandera import with fallback surface for restricted runtime environments
try:
    import pandera.pandas as pa
    from pandera import Check
    from pandera.pandas import Column, DataFrameSchema

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None  # type: ignore

_log = logging.getLogger(__name__)

# Sensible bounds for year columns
MIN_YEAR = 1950
MAX_YEAR = 2050


def _build_schema(
    columns: dict[str, Any],
    description: str = "",
    strict: bool | Literal["filter"] = "filter",
) -> "DataFrameSchema | None":
    """Build a DataFrameSchema if pandera is available."""
    if not PANDERA_AVAILABLE:
        return None
    return DataFrameSchema(columns, description=description, strict=strict)


# =============================================================================
# SCHEMA DEFINITIONS (built lazily or conditionally)
# =============================================================================

# Harmonized plasmid table schema (output of 04_harmonize_metadata.py)
HARMONIZED_PLASMID_SCHEMA = (
    _build_schema(
        {
            "sequence_accession": Column(
                str,
                nullable=False,
                description="Primary sequence accession (NUCCORE_ACC)",
            ),
            "backbone_id": Column(
                str,
                nullable=False,
                description="Backbone identifier",
            ),
            "resolved_year": Column(
                float,
                checks=[
                    Check.greater_than(MIN_YEAR),
                    Check.less_than(MAX_YEAR),
                ],
                nullable=True,
                required=False,
                description="Resolved year of sequence",
            ),
            "country": Column(
                str,
                nullable=True,
                required=False,
                description="Resolved country from location",
            ),
            "predicted_mobility": Column(
                str,
                nullable=True,
                required=False,
                description="Mobility prediction from MobSuite",
            ),
            "gc_content": Column(
                float,
                checks=Check.in_range(0.0, 100.0),
                nullable=True,
                required=False,
                description="GC content percentage",
            ),
            "size": Column(
                float,
                checks=Check.greater_than(0),
                nullable=True,
                required=False,
                description="Plasmid size in bp",
            ),
        },
        strict="filter",
        description="Harmonized plasmid metadata table",
    )
    if PANDERA_AVAILABLE
    else None
)


# Backbone table schema (output of backbone assignment)
BACKBONE_TABLE_SCHEMA = (
    _build_schema(
        {
            "backbone_id": Column(
                str,
                nullable=False,
                description="Unique backbone identifier",
            ),
            "primary_cluster_id": Column(
                str,
                nullable=True,
                required=False,
                description="Primary MOB-cluster assignment",
            ),
            "predicted_mobility": Column(
                str,
                nullable=True,
                required=False,
                description="Dominant mobility class",
            ),
            "mpf_type": Column(
                str,
                nullable=True,
                required=False,
                description="Dominant MPF type",
            ),
            "primary_replicon": Column(
                str,
                nullable=True,
                required=False,
                description="Dominant replicon type",
            ),
            "backbone_assignment_rule": Column(
                str,
                nullable=True,
                required=False,
                description="Rule used for backbone assignment",
            ),
            "backbone_seen_in_training": Column(
                object,  # pandas uses object dtype for nullable booleans
                nullable=True,
                required=False,
                description="Whether backbone appeared in training period",
            ),
        },
        strict="filter",
        description="Backbone assignment table",
    )
    if PANDERA_AVAILABLE
    else None
)


# Scored backbone table schema (critical model input)
SCORED_BACKBONE_SCHEMA = (
    _build_schema(
        {
            "backbone_id": Column(
                str,
                nullable=False,
                description="Unique backbone identifier",
            ),
            "spread_label": Column(
                float,
                checks=Check.isin([0.0, 1.0, float("nan")]),
                nullable=True,
                required=False,
                description="Binary spread outcome (1=spread, 0=no spread)",
            ),
            "priority_index": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Overall priority score",
            ),
            "bio_priority_index": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Biological component score",
            ),
            "T_eff_norm": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Normalized transfer efficiency",
            ),
            "H_eff_norm": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Normalized host range efficiency",
            ),
            "A_eff_norm": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Normalized AMR efficiency",
            ),
            "log1p_member_count_train": Column(
                float,
                checks=Check.greater_than_or_equal_to(0),
                nullable=True,
                required=False,
                description="Log-transformed training member count",
            ),
            "log1p_n_countries_train": Column(
                float,
                checks=Check.greater_than_or_equal_to(0),
                nullable=True,
                required=False,
                description="Log-transformed training country count",
            ),
            "refseq_share_train": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Share of RefSeq sources in training",
            ),
            "coherence_score": Column(
                float,
                checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True),
                nullable=True,
                required=False,
                description="Backbone coherence score",
            ),
        },
        strict="filter",
        description="Scored backbone table with model features",
    )
    if PANDERA_AVAILABLE
    else None
)


# Deduplicated plasmid table schema
DEDUPLICATED_PLASMID_SCHEMA = (
    _build_schema(
        {
            "sequence_accession": Column(
                str,
                nullable=False,
                description="Primary sequence accession",
            ),
            "backbone_id": Column(
                str,
                nullable=False,
                description="Assigned backbone",
            ),
            "is_canonical_representative": Column(
                bool,
                nullable=True,
                required=False,
                description="Whether this is the canonical sequence for its group",
            ),
            "dedup_representative_group": Column(
                str,
                nullable=True,
                required=False,
                description="Deduplication group identifier",
            ),
        },
        strict="filter",
        description="Deduplicated plasmid table",
    )
    if PANDERA_AVAILABLE
    else None
)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _validate_table(
    df: pd.DataFrame,
    schema: "DataFrameSchema | None",
    table_name: str,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Generic schema validation helper (DRY).

    Validates *df* against *schema* and returns a structured result dict.
    Handles pandera-not-available, schema errors, and unexpected exceptions.
    """
    if not PANDERA_AVAILABLE or schema is None:
        return {
            "status": "skipped",
            "reason": "pandera_not_installed",
            "table": table_name,
            "n_rows": int(len(df)),
            "errors": [],
        }

    try:
        validated = schema.validate(df, lazy=lazy)
        return {
            "status": "pass",
            "table": table_name,
            "n_rows": int(len(validated)),
            "errors": [],
        }
    except Exception as e:  # pragma: no cover - depends on optional pandera internals
        schema_errors_type: type[Exception] | None = None
        if pa is not None:
            errors_module = getattr(pa, "errors", None)
            schema_errors_type = getattr(errors_module, "SchemaErrors", None)
        if schema_errors_type is not None and isinstance(e, schema_errors_type):
            failure_cases = getattr(e, "failure_cases", None)
            failure_rows = (
                failure_cases.to_dict("records")
                if failure_cases is not None and hasattr(failure_cases, "to_dict")
                else []
            )
            _log.warning("Schema validation failed for %s", table_name)
            return {
                "status": "fail",
                "table": table_name,
                "n_rows": int(len(df)),
                "errors": failure_rows,
            }
        _log.error("Unexpected error validating %s: %s", table_name, e)
        return {
            "status": "error",
            "table": table_name,
            "n_rows": int(len(df)),
            "errors": [{"message": str(e)}],
        }


def validate_harmonized_plasmids(
    df: pd.DataFrame,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Validate harmonized plasmid table."""
    return _validate_table(df, HARMONIZED_PLASMID_SCHEMA, "harmonized_plasmids", lazy=lazy)


def validate_backbone_table(
    df: pd.DataFrame,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Validate backbone assignment table."""
    return _validate_table(df, BACKBONE_TABLE_SCHEMA, "backbone_table", lazy=lazy)


def validate_scored_backbones(
    df: pd.DataFrame,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Validate scored backbone table (model input)."""
    return _validate_table(df, SCORED_BACKBONE_SCHEMA, "scored_backbones", lazy=lazy)


def validate_deduplicated_plasmids(
    df: pd.DataFrame,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Validate deduplicated plasmid table."""
    return _validate_table(df, DEDUPLICATED_PLASMID_SCHEMA, "deduplicated_plasmids", lazy=lazy)


def run_all_validations(
    harmonized: pd.DataFrame | None = None,
    backbones: pd.DataFrame | None = None,
    scored: pd.DataFrame | None = None,
    deduplicated: pd.DataFrame | None = None,
    *,
    lazy: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run all schema validations and return summary.

    Args:
        harmonized: Harmonized plasmid table (optional)
        backbones: Backbone table (optional)
        scored: Scored backbone table (optional)
        deduplicated: Deduplicated plasmid table (optional)
        lazy: If True, collect all errors per table

    Returns:
        Dict mapping table name to validation result
    """
    results: dict[str, dict[str, Any]] = {}

    if harmonized is not None:
        results["harmonized_plasmids"] = validate_harmonized_plasmids(harmonized, lazy=lazy)
    if backbones is not None:
        results["backbone_table"] = validate_backbone_table(backbones, lazy=lazy)
    if scored is not None:
        results["scored_backbones"] = validate_scored_backbones(scored, lazy=lazy)
    if deduplicated is not None:
        results["deduplicated_plasmids"] = validate_deduplicated_plasmids(deduplicated, lazy=lazy)

    # Add summary
    total_tables = len(results)
    passed = sum(1 for r in results.values() if r.get("status") == "pass")
    failed = sum(1 for r in results.values() if r.get("status") in ("fail", "error"))
    skipped = sum(1 for r in results.values() if r.get("status") == "skipped")

    results["_summary"] = {
        "total_tables": total_tables,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "overall_status": (
            "pass" if failed == 0 and passed > 0 else "fail" if failed > 0 else "skipped"
        ),
    }

    return results


def format_validation_report(results: dict[str, dict[str, Any]]) -> str:
    """Format validation results as a clear human-readable report.

    Args:
        results: Result dict from run_all_validations()

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "Schema Validation Report",
        "=" * 60,
    ]

    summary = results.get("_summary", {})
    overall = summary.get("overall_status", "unknown")

    # Clear overall status indicator
    if overall == "pass":
        status_indicator = "PASS"
    elif overall == "fail":
        status_indicator = "FAIL"
    elif overall == "skipped":
        status_indicator = "SKIPPED (pandera not installed)"
    else:
        status_indicator = overall.upper()

    lines.append(f"Overall Status: {status_indicator}")
    lines.append("")

    # Pandera availability message
    if not PANDERA_AVAILABLE:
        lines.append(
            "Note: Pandera is unavailable in this environment; "
            "schema checks are reported as skipped."
        )
        lines.append("Validation was skipped - no schema checks were performed.")
        lines.append("")

    # Summary stats
    total = summary.get("total_tables", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)

    lines.append(f"Tables evaluated: {total}")
    if passed:
        lines.append(f"  - Passed: {passed}")
    if failed:
        lines.append(f"  - Failed: {failed}")
    if skipped:
        lines.append(f"  - Skipped: {skipped}")
    lines.append("")

    # Per-table details
    for table_name, result in results.items():
        if table_name.startswith("_"):
            continue

        status = result.get("status", "unknown")
        n_rows = result.get("n_rows", 0)
        errors = result.get("errors", [])

        # Status indicator with emoji-style markers (using text)
        if status == "pass":
            marker = "[OK]"
        elif status == "fail":
            marker = "[FAIL]"
        elif status == "skipped":
            marker = "[SKIPPED]"
        elif status == "error":
            marker = "[ERROR]"
        else:
            marker = f"[{status.upper()}]"

        lines.append(f"{marker} {table_name}: {n_rows} rows")

        if status == "skipped" and result.get("reason"):
            lines.append(f"       Reason: {result['reason']}")

        if errors:
            lines.append(f"       Errors: {len(errors)}")
            for i, err in enumerate(errors[:3], 1):  # Show first 3 errors
                msg = err.get("message", str(err))
                lines.append(f"         {i}. {msg[:80]}")
            if len(errors) > 3:
                lines.append(f"         ... and {len(errors) - 3} more")

        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def print_validation_report(results: dict[str, dict[str, Any]]) -> None:
    """Log validation results via the logging framework.

    This is a convenience function for CLI usage that clearly
    indicates whether validation actually ran.

    Args:
        results: Result dict from run_all_validations()
    """
    _log.info("\n%s", format_validation_report(results))


def validate_tables_from_paths(
    harmonized_path: str | None = None,
    backbones_path: str | None = None,
    scored_path: str | None = None,
    deduplicated_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate tables from file paths - convenience for CLI usage.

    This function loads data from paths and runs all validations.
    It provides clear feedback about what was validated.

    Args:
        harmonized_path: Path to harmonized plasmid TSV
        backbones_path: Path to backbone table TSV
        scored_path: Path to scored backbone TSV
        deduplicated_path: Path to deduplicated plasmid TSV

    Returns:
        Validation results dict
    """
    harmonized = pd.read_csv(harmonized_path, sep="\t") if harmonized_path else None
    backbones = pd.read_csv(backbones_path, sep="\t") if backbones_path else None
    scored = pd.read_csv(scored_path, sep="\t") if scored_path else None
    deduplicated = pd.read_csv(deduplicated_path, sep="\t") if deduplicated_path else None

    return run_all_validations(
        harmonized=harmonized,
        backbones=backbones,
        scored=scored,
        deduplicated=deduplicated,
        lazy=True,
    )
