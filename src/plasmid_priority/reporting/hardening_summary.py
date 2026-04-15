"""Consolidated hardening audit summary builder.

This module provides a compact, unified surface for all hardening audits:
- EPV (events-per-variable)
- Lead-time bias
- Missingness
- Schema validation status

It reuses existing audit functions and produces both machine-readable (JSON)
and human-readable (Markdown) summaries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from plasmid_priority.modeling.module_a_support import (
    GOVERNANCE_MODEL_NAME,
    PRIMARY_MODEL_NAME,
)
from plasmid_priority.reporting.epv_audit import build_epv_audit_table, summarize_epv_concerns
from plasmid_priority.reporting.lead_time_bias_audit import (
    build_lead_time_bias_audit,
    summarize_lead_time_bias_findings,
)
from plasmid_priority.validation.missingness import audit_backbone_tables
from plasmid_priority.validation.schemas import PANDERA_AVAILABLE, run_all_validations


def build_hardening_audit_summary(
    backbone_table: pd.DataFrame | None = None,
    scored_backbone_table: pd.DataFrame | None = None,
    harmonized_plasmids: pd.DataFrame | None = None,
    deduplicated_plasmids: pd.DataFrame | None = None,
    *,
    include_schema_validation: bool = True,
    include_epv: bool = True,
    include_lead_time_bias: bool = True,
    include_missingness: bool = True,
) -> dict[str, Any]:
    """Build a consolidated hardening audit summary.

    This is a lightweight, non-invasive summary that reuses existing audit
    functions to provide a compact view of data quality and model health.

    Args:
        backbone_table: Optional backbone table for missingness/schema
        scored_backbone_table: Optional scored table for EPV/lead-time/missingness
        harmonized_plasmids: Optional harmonized table for schema validation
        deduplicated_plasmids: Optional deduplicated table for schema validation
        include_schema_validation: Whether to run schema validation
        include_epv: Whether to include EPV audit
        include_lead_time_bias: Whether to include lead-time bias audit
        include_missingness: Whether to include missingness audit

    Returns:
        Dict with consolidated summary sections
    """
    summary: dict[str, Any] = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "tables_audited": [],
        "models": {},
        "lead_time_bias": {},
        "missingness": {},
        "schema_validation": {},
        "overall_status": "ok",
    }

    # Track which tables were provided
    if backbone_table is not None:
        summary["tables_audited"].append("backbone_table")
    if scored_backbone_table is not None:
        summary["tables_audited"].append("scored_backbone_table")
    if harmonized_plasmids is not None:
        summary["tables_audited"].append("harmonized_plasmids")
    if deduplicated_plasmids is not None:
        summary["tables_audited"].append("deduplicated_plasmids")

    # EPV Audit (requires scored table)
    if include_epv and scored_backbone_table is not None:
        epv_table = build_epv_audit_table(scored_backbone_table)
        epv_summary = summarize_epv_concerns(epv_table)
        low_epv_count = epv_summary.get("n_models_low_epv", 0)
        if not isinstance(low_epv_count, (int, float)):
            low_epv_count = 0

        summary["models"]["epv"] = {
            "primary_model": PRIMARY_MODEL_NAME,
            "governance_model": GOVERNANCE_MODEL_NAME,
            "n_models_evaluated": epv_summary["n_models_evaluated"],
            "n_models_low_epv": epv_summary["n_models_low_epv"],
            "n_models_very_low_epv": epv_summary["n_models_very_low_epv"],
            "models_requiring_review": epv_summary["models_requiring_review"],
            "status": "concern" if low_epv_count > 0 else "ok",
        }

        # Add per-model EPV if available
        if not epv_table.empty:
            summary["models"]["epv_details"] = epv_table.to_dict("records")

    # Lead-time bias audit (requires scored table)
    if include_lead_time_bias and scored_backbone_table is not None:
        lt_summary_df, lt_decile_df = build_lead_time_bias_audit(scored_backbone_table)
        lt_findings = summarize_lead_time_bias_findings(lt_summary_df, lt_decile_df)

        summary["lead_time_bias"] = {
            "overall_concern_level": lt_findings["overall_concern_level"],
            "n_metrics_evaluated": lt_findings["n_metrics_evaluated"],
            "n_high_concern": lt_findings["n_high_concern"],
            "n_moderate_concern": lt_findings["n_moderate_concern"],
            "trend_direction": lt_findings["trend_direction"],
            "interpretation": lt_findings["interpretation"],
            "status": (
                "concern" if lt_findings["overall_concern_level"] in ("high", "moderate") else "ok"
            ),
        }

    # Missingness audit
    if include_missingness:
        missingness_result = audit_backbone_tables(
            backbone_table=backbone_table,
            scored_backbone_table=scored_backbone_table,
        )

        summary["missingness"] = {
            "tables_audited": missingness_result.get("tables_audited", []),
            "overall_status": missingness_result.get("overall_status", "unknown"),
            "high_missingness_columns_total": missingness_result.get(
                "high_missingness_columns_total", 0
            ),
        }

        # Include top high-missingness columns per table
        for table_key in ["backbone_table", "scored_backbone_table"]:
            if table_key in missingness_result:
                table_audit = missingness_result[table_key]
                high_missing_cols = [
                    col
                    for col in table_audit.get("columns", [])
                    if col.get("high_missingness_flag")
                ][:5]  # Top 5
                summary["missingness"][f"{table_key}_high_missingness_top5"] = [
                    {"column": c["column"], "missing_fraction": c["missing_fraction"]}
                    for c in high_missing_cols
                ]

    # Schema validation
    if include_schema_validation:
        validation_results = run_all_validations(
            harmonized=harmonized_plasmids,
            backbones=backbone_table,
            scored=scored_backbone_table,
            deduplicated=deduplicated_plasmids,
            lazy=True,
        )

        # Remove summary key for detailed output
        validation_detail = {k: v for k, v in validation_results.items() if not k.startswith("_")}

        summary["schema_validation"] = {
            "pandera_available": PANDERA_AVAILABLE,
            "tables_validated": list(validation_detail.keys()),
            "results": {
                k: {
                    "status": v.get("status"),
                    "n_rows": v.get("n_rows"),
                    "reason": v.get("reason"),  # For skipped status
                }
                for k, v in validation_detail.items()
            },
            "overall_status": validation_results.get("_summary", {}).get(
                "overall_status", "unknown"
            ),
        }

    # Compute overall status (max severity)
    statuses = []
    if "epv" in summary["models"]:
        statuses.append(summary["models"]["epv"].get("status", "ok"))
    if summary["lead_time_bias"]:
        statuses.append(summary["lead_time_bias"].get("status", "ok"))
    if summary["missingness"]:
        statuses.append(summary["missingness"].get("overall_status", "ok"))
    if summary["schema_validation"]:
        statuses.append(summary["schema_validation"].get("overall_status", "ok"))

    if "fail" in statuses:
        summary["overall_status"] = "fail"
    elif "concern" in statuses:
        summary["overall_status"] = "concern"
    elif "skipped" in statuses and "ok" not in statuses:
        summary["overall_status"] = "skipped"
    else:
        summary["overall_status"] = "ok"

    return summary


def format_hardening_summary_markdown(summary: dict[str, Any]) -> str:
    """Format hardening audit summary as Markdown.

    Args:
        summary: Result from build_hardening_audit_summary()

    Returns:
        Markdown-formatted summary string
    """
    lines = [
        "# Hardening Audit Summary",
        "",
        f"**Generated:** {summary.get('audit_timestamp', 'unknown')}",
        f"**Overall Status:** {summary.get('overall_status', 'unknown').upper()}",
        "",
        f"**Tables Audited:** {', '.join(summary.get('tables_audited', [])) or 'none'}",
        "",
        "## Schema Validation",
        "",
    ]

    schema = summary.get("schema_validation", {})
    lines.append(f"- **Pandera Available:** {schema.get('pandera_available', False)}")
    lines.append(f"- **Overall Status:** {schema.get('overall_status', 'unknown')}")
    lines.append("")

    if schema.get("results"):
        lines.append("| Table | Status | Rows |")
        lines.append("|-------|--------|------|")
        for table, result in schema["results"].items():
            status = result.get("status", "unknown")
            rows = result.get("n_rows", "N/A")
            lines.append(f"| {table} | {status} | {rows} |")
        lines.append("")

    # EPV Section
    lines.append("## EPV (Events-Per-Variable) Audit")
    lines.append("")

    epv = summary.get("models", {}).get("epv", {})
    if epv:
        lines.append(f"- **Primary Model:** {epv.get('primary_model', 'N/A')}")
        lines.append(f"- **Governance Model:** {epv.get('governance_model', 'N/A')}")
        lines.append(f"- **Models Evaluated:** {epv.get('n_models_evaluated', 0)}")
        lines.append(f"- **Models with Low EPV (<10):** {epv.get('n_models_low_epv', 0)}")
        lines.append(f"- **Models with Very Low EPV (<5):** {epv.get('n_models_very_low_epv', 0)}")
        if epv.get("models_requiring_review"):
            review_models = ", ".join(epv["models_requiring_review"])
            lines.append(f"- **Models Requiring Review:** {review_models}")
        lines.append(f"- **Status:** {epv.get('status', 'unknown')}")
    else:
        lines.append("*EPV audit not run (no scored table provided)*")
    lines.append("")

    # Lead-time Bias Section
    lines.append("## Lead-Time Bias Audit")
    lines.append("")

    lt = summary.get("lead_time_bias", {})
    if lt:
        lines.append(f"- **Overall Concern Level:** {lt.get('overall_concern_level', 'unknown')}")
        lines.append(f"- **Metrics Evaluated:** {lt.get('n_metrics_evaluated', 0)}")
        lines.append(f"- **High Concern Metrics:** {lt.get('n_high_concern', 0)}")
        lines.append(f"- **Moderate Concern Metrics:** {lt.get('n_moderate_concern', 0)}")
        lines.append(f"- **Trend Direction:** {lt.get('trend_direction', 'unknown')}")
        lines.append(f"- **Status:** {lt.get('status', 'unknown')}")
        lines.append("")
        lines.append(f"**Interpretation:** {lt.get('interpretation', 'N/A')}")
    else:
        lines.append("*Lead-time bias audit not run (no scored table provided)*")
    lines.append("")

    # Missingness Section
    lines.append("## Missingness Audit")
    lines.append("")

    miss = summary.get("missingness", {})
    if miss:
        lines.append(f"- **Tables Audited:** {', '.join(miss.get('tables_audited', []))}")
        lines.append(f"- **Overall Status:** {miss.get('overall_status', 'unknown')}")
        lines.append(
            f"- **Total High-Missingness Columns:** {miss.get('high_missingness_columns_total', 0)}"
        )
        lines.append("")

        # Show top high-missingness columns per table
        for key in miss:
            if key.endswith("_high_missingness_top5"):
                table_name = key.replace("_high_missingness_top5", "")
                cols = miss[key]
                if cols:
                    lines.append(f"**{table_name} - High Missingness Columns:**")
                    for col in cols:
                        lines.append(f"- {col['column']}: {col['missing_fraction']:.1%}")
                    lines.append("")
    else:
        lines.append("*Missingness audit not run*")
    lines.append("")

    lines.append("---")
    lines.append(
        "*This is a lightweight hardening summary. "
        "For detailed diagnostics, see individual audit reports.*"
    )

    return "\n".join(lines)
