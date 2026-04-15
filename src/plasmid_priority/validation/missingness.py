"""Missingness/NaN propagation audit for critical tables.

This module provides lightweight missingness audits that make
missingness visible in critical derived/model-input tables without
being a hard global failure gate.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def audit_missingness(
    df: pd.DataFrame,
    table_name: str,
    *,
    split_column: str | None = None,
    high_missingness_threshold: float = 0.5,
) -> dict[str, Any]:
    """Audit missingness in a DataFrame and return structured report.

    Args:
        df: DataFrame to audit
        table_name: Name of the table for reporting
        split_column: Optional column to split analysis by (e.g., "split" or "model_track")
        high_missingness_threshold: Fraction above which to flag high missingness

    Returns:
        Dict with audit results including per-column missingness and flags
    """
    if df.empty:
        return {
            "table_name": table_name,
            "n_rows": 0,
            "n_columns": 0,
            "columns": [],
            "high_missingness_count": 0,
            "high_missingness_threshold": high_missingness_threshold,
            "status": "empty_table",
        }

    n_rows = len(df)
    columns_audit: list[dict[str, Any]] = []
    high_missingness_count = 0

    # Determine splits if requested
    split_values: list[str | None] = [None]
    if split_column and split_column in df.columns:
        split_values = df[split_column].dropna().unique().tolist()

    for col in df.columns:
        col_info: dict[str, Any] = {
            "column": col,
            "dtype": str(df[col].dtype),
        }

        # Overall missingness
        missing_count = int(df[col].isna().sum())
        missing_fraction = missing_count / n_rows if n_rows > 0 else 0.0
        col_info["missing_count"] = missing_count
        col_info["missing_fraction"] = round(missing_fraction, 4)

        # Per-split missingness if applicable
        if split_column and split_column in df.columns and len(split_values) > 1:
            col_info["by_split"] = {}
            for split_val in split_values:
                split_df = df[df[split_column] == split_val]
                split_missing = int(split_df[col].isna().sum())
                split_total = len(split_df)
                col_info["by_split"][str(split_val)] = {
                    "missing_count": split_missing,
                    "missing_fraction": (
                        round(split_missing / split_total, 4) if split_total > 0 else 0.0
                    ),
                    "n_rows": split_total,
                }

        # Flag high missingness
        if missing_fraction > high_missingness_threshold:
            col_info["high_missingness_flag"] = True
            high_missingness_count += 1
        else:
            col_info["high_missingness_flag"] = False

        columns_audit.append(col_info)

    # Sort by missing fraction descending
    columns_audit.sort(key=lambda x: x["missing_fraction"], reverse=True)

    return {
        "table_name": table_name,
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "high_missingness_count": high_missingness_count,
        "high_missingness_threshold": high_missingness_threshold,
        "status": "ok" if high_missingness_count == 0 else "concern",
        "columns": columns_audit,
    }


def format_missingness_report(audit_result: dict[str, Any]) -> str:
    """Format missingness audit result as human-readable string.

    Args:
        audit_result: Result dict from audit_missingness()

    Returns:
        Formatted report string
    """
    lines = [
        f"Missingness Audit: {audit_result['table_name']}",
        "=" * 60,
        f"Rows: {audit_result['n_rows']}, Columns: {audit_result['n_columns']}",
        f"High missingness threshold: {audit_result['high_missingness_threshold']}",
        f"Status: {audit_result['status']}",
        "",
        "Column Missingness (sorted by fraction):",
        "-" * 60,
    ]

    for col in audit_result["columns"]:
        flag = " [HIGH]" if col.get("high_missingness_flag") else ""
        lines.append(
            f"  {col['column']}: {col['missing_count']}/{audit_result['n_rows']} "
            f"({col['missing_fraction']:.1%}){flag}"
        )

    if audit_result["high_missingness_count"] > 0:
        lines.extend(
            [
                "",
                f"⚠️  {audit_result['high_missingness_count']} columns exceed missingness threshold",
            ]
        )

    return "\n".join(lines)


def audit_backbone_tables(
    backbone_table: pd.DataFrame | None = None,
    scored_backbone_table: pd.DataFrame | None = None,
    *,
    high_missingness_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run missingness audit on backbone and scored backbone tables.

    This is the main entry point for the NaN propagation audit task.
    Audits both tables and returns combined results.

    Args:
        backbone_table: Optional backbone table to audit
        scored_backbone_table: Optional scored backbone table to audit
        high_missingness_threshold: Fraction above which to flag concern

    Returns:
        Dict with results for both tables
    """
    results: dict[str, Any] = {
        "tables_audited": [],
        "overall_status": "ok",
        "high_missingness_columns_total": 0,
    }

    if backbone_table is not None and not backbone_table.empty:
        backbone_audit = audit_missingness(
            backbone_table,
            "backbone_table",
            high_missingness_threshold=high_missingness_threshold,
        )
        results["backbone_table"] = backbone_audit
        results["tables_audited"].append("backbone_table")
        if backbone_audit["status"] == "concern":
            results["overall_status"] = "concern"
            results["high_missingness_columns_total"] += backbone_audit["high_missingness_count"]

    if scored_backbone_table is not None and not scored_backbone_table.empty:
        # For scored table, also check by eligibility if spread_label exists
        split_col = None
        if "spread_label" in scored_backbone_table.columns:
            # Create a synthetic split for eligible vs non-eligible
            scored_backbone_table = scored_backbone_table.copy()
            scored_backbone_table["_eligibility_split"] = (
                scored_backbone_table["spread_label"]
                .notna()
                .map({True: "eligible", False: "ineligible"})
            )
            split_col = "_eligibility_split"

        scored_audit = audit_missingness(
            scored_backbone_table,
            "scored_backbone_table",
            split_column=split_col,
            high_missingness_threshold=high_missingness_threshold,
        )
        results["scored_backbone_table"] = scored_audit
        results["tables_audited"].append("scored_backbone_table")
        if scored_audit["status"] == "concern":
            results["overall_status"] = "concern"
            results["high_missingness_columns_total"] += scored_audit["high_missingness_count"]

        # Clean up temporary column from the copy
        if "_eligibility_split" in scored_backbone_table.columns:
            del scored_backbone_table["_eligibility_split"]

    return results


def print_backbone_audit_report(audit_results: dict[str, Any]) -> None:
    """Print formatted backbone audit report to stdout.

    Args:
        audit_results: Results from audit_backbone_tables()
    """
    print("\n" + "=" * 70)
    print("NaN PROPAGATION AUDIT REPORT")
    print("=" * 70)
    print(f"Overall Status: {audit_results['overall_status'].upper()}")
    print(f"Tables Audited: {', '.join(audit_results['tables_audited'])}")
    print(f"Total High-Missingness Columns: {audit_results['high_missingness_columns_total']}")
    print()

    if "backbone_table" in audit_results:
        print(format_missingness_report(audit_results["backbone_table"]))
        print()

    if "scored_backbone_table" in audit_results:
        print(format_missingness_report(audit_results["scored_backbone_table"]))
        print()

    print("=" * 70)
    print("End of NaN Propagation Audit Report")
    print("=" * 70 + "\n")
