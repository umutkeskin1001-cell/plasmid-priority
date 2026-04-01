#!/usr/bin/env python3
"""Export simplified summary metrics for the final TÜBİTAK report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD, build_context
from plasmid_priority.modeling import get_primary_model_name
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import ensure_directory


def _clean_metric(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> int:
    context = build_context(PROJECT_ROOT)
    audit_json = context.root / "data/analysis/module_a_metrics.json"
    text_output_path = context.root / "reports/tubitak_final_metrics.txt"
    ensure_directory(text_output_path.parent)

    with ManagedScriptRun(context, "25_export_tubitak_summary") as run:
        run.record_input(audit_json)
        run.record_output(text_output_path)

        with audit_json.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)

        primary_model = get_primary_model_name(list(audit.keys()))
        if primary_model not in audit:
            run.warn(f"Primary model '{primary_model}' not found in audit JSON.")
            return 1

        primary_metrics = audit[primary_model]

        tubitak_payload = {
            "model_adi": primary_model,
            "roc_auc": _clean_metric(primary_metrics.get("roc_auc")),
            "roc_auc_ci_lower": _clean_metric(primary_metrics.get("roc_auc_ci_lower")),
            "roc_auc_ci_upper": _clean_metric(primary_metrics.get("roc_auc_ci_upper")),
            "pr_auc": _clean_metric(primary_metrics.get("average_precision")),
            "pr_auc_ci_lower": _clean_metric(primary_metrics.get("average_precision_ci_lower")),
            "pr_auc_ci_upper": _clean_metric(primary_metrics.get("average_precision_ci_upper")),
            "brier_score": _clean_metric(primary_metrics.get("brier_score")),
            "brier_score_ci_lower": _clean_metric(primary_metrics.get("brier_score_ci_lower")),
            "brier_score_ci_upper": _clean_metric(primary_metrics.get("brier_score_ci_upper")),
            "prevalence": _clean_metric(primary_metrics.get("positive_prevalence")),
        }
        ap_lift = (
            None
            if tubitak_payload["pr_auc"] is None or tubitak_payload["prevalence"] is None
            else tubitak_payload["pr_auc"] - tubitak_payload["prevalence"]
        )

        lines = [
            "=================================================================",
            "  TÜBİTAK RAPORU İÇİN İSTATİSTİKSEL ÖZET",
            "=================================================================",
            f"Ana Model           : {primary_model}",
            f"Ana Outcome Eşiği   : Sonradan >= {int(DEFAULT_MIN_NEW_COUNTRIES_FOR_SPREAD)} yeni ülke",
            "",
            "1. AYIRICILIK GÜCÜ (ROC AUC)",
            f"   Değer            : {_format_metric(tubitak_payload['roc_auc'])}",
            f"   %95 Güven Aralığı: [{_format_metric(tubitak_payload['roc_auc_ci_lower'])} - {_format_metric(tubitak_payload['roc_auc_ci_upper'])}]",
            "",
            "2. PRECISION-RECALL ÖZETİ (Average Precision)",
            f"   Değer            : {_format_metric(tubitak_payload['pr_auc'])}",
            f"   Temel Oran       : {_format_metric(tubitak_payload['prevalence'])}",
            f"   Lift (Kazanç)    : {_format_metric(ap_lift)}",
            "",
            "3. KALİBRASYON (Brier Score)",
            f"   Değer            : {_format_metric(tubitak_payload['brier_score'])}",
            "=================================================================",
        ]

        with text_output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
