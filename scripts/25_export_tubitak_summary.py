#!/usr/bin/env python3
"""Export simplified summary metrics for the final TÜBİTAK report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import get_primary_model_name
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_cached_manifest(
    manifest_path: Path,
    *,
    input_paths: list[Path],
    source_paths: list[Path],
    output_path: Path,
    pipeline_settings: dict[str, object],
) -> bool:
    if not manifest_path.exists() or not output_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("pipeline_settings") != pipeline_settings:
        return False
    if payload.get("input_signatures") != [_path_signature(path) for path in input_paths]:
        return False
    if payload.get("source_signatures") != [_path_signature(path) for path in source_paths]:
        return False
    if payload.get("output_signature") != _path_signature(output_path):
        return False
    return True


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


def _format_pvalue(value: float | None) -> str:
    if value is None:
        return "NA"
    if value < 0.001:
        return "< 0.001"
    return f"= {value:.4f}"


def main() -> int:
    context = build_context(PROJECT_ROOT)
    audit_json = context.data_dir / "analysis/module_a_metrics.json"
    permutation_summary_path = context.data_dir / "analysis/permutation_null_summary.tsv"
    selection_adjusted_permutation_summary_path = (
        context.data_dir / "analysis/selection_adjusted_permutation_null_summary.tsv"
    )
    count_outcome_audit_path = context.data_dir / "analysis/new_country_count_audit.tsv"
    text_output_path = context.reports_dir / "tubitak_final_metrics.txt"
    manifest_path = text_output_path.with_suffix(text_output_path.suffix + ".manifest.json")
    ensure_directory(text_output_path.parent)
    config_paths = context_config_paths(context)
    source_paths = [
        PROJECT_ROOT / "scripts/25_export_tubitak_summary.py",
        *sorted((context.root / "src/plasmid_priority").rglob("*.py")),
    ]

    with ManagedScriptRun(context, "25_export_tubitak_summary") as run:
        input_paths = [audit_json]
        input_paths.extend(config_paths)
        run.record_input(audit_json)
        for path in config_paths:
            run.record_input(path)
        if permutation_summary_path.exists():
            input_paths.append(permutation_summary_path)
            run.record_input(permutation_summary_path)
        if selection_adjusted_permutation_summary_path.exists():
            input_paths.append(selection_adjusted_permutation_summary_path)
            run.record_input(selection_adjusted_permutation_summary_path)
        if count_outcome_audit_path.exists():
            input_paths.append(count_outcome_audit_path)
            run.record_input(count_outcome_audit_path)
        run.record_output(text_output_path)

        pipeline_settings = {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
        if _load_cached_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            output_path=text_output_path,
            pipeline_settings=pipeline_settings,
        ):
            run.note("Inputs and source files unchanged; reusing cached TÜBİTAK summary.")
            run.set_metric("cache_hit", True)
            return 0

        with audit_json.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        pipeline = context.pipeline_settings

        primary_model = get_primary_model_name(list(audit.keys()))
        if primary_model not in audit:
            run.warn(f"Primary model '{primary_model}' not found in audit JSON.")
            return 1

        primary_metrics = audit[primary_model]
        permutation_summary = (
            pd.read_csv(permutation_summary_path, sep="\t")
            if permutation_summary_path.exists()
            else pd.DataFrame()
        )
        selection_adjusted_permutation_summary = (
            pd.read_csv(selection_adjusted_permutation_summary_path, sep="\t")
            if selection_adjusted_permutation_summary_path.exists()
            else pd.DataFrame()
        )
        count_outcome_audit = (
            pd.read_csv(count_outcome_audit_path, sep="\t")
            if count_outcome_audit_path.exists()
            else pd.DataFrame()
        )

        primary_permutation = permutation_summary.loc[
            permutation_summary.get("model_name", pd.Series(dtype=str)).astype(str)
            == str(primary_model)
        ].head(1)
        primary_selection_adjusted = selection_adjusted_permutation_summary.loc[
            selection_adjusted_permutation_summary.get("model_name", pd.Series(dtype=str)).astype(
                str
            )
            == str(primary_model)
        ].head(1)
        best_count_alignment = (
            count_outcome_audit.loc[
                count_outcome_audit.get("status", pd.Series(dtype=str)).astype(str) == "ok"
            ]
            .sort_values("spearman_corr", ascending=False)
            .head(1)
        )

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
            "permutation_p_roc_auc": _clean_metric(
                primary_permutation.iloc[0]["empirical_p_roc_auc"]
            )
            if not primary_permutation.empty
            else None,
            "selection_adjusted_permutation_p_roc_auc": _clean_metric(
                primary_selection_adjusted.iloc[0]["selection_adjusted_empirical_p_roc_auc"]
            )
            if not primary_selection_adjusted.empty
            else None,
            "n_permutations": int(primary_permutation.iloc[0]["n_permutations"])
            if not primary_permutation.empty
            else None,
            "best_spearman": _clean_metric(best_count_alignment.iloc[0]["spearman_corr"])
            if not best_count_alignment.empty
            else None,
            "best_spearman_ci_lower": _clean_metric(
                best_count_alignment.iloc[0].get("spearman_ci_lower")
            )
            if not best_count_alignment.empty
            else None,
            "best_spearman_ci_upper": _clean_metric(
                best_count_alignment.iloc[0].get("spearman_ci_upper")
            )
            if not best_count_alignment.empty
            else None,
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
            f"Ana Outcome Eşiği   : Sonradan >= {int(pipeline.min_new_countries_for_spread)} yeni ülke",
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
            "",
            "4. FORMAL ANLAMLILIK (Permütasyon Testi)",
            f"   Seçim-düzeltilmiş ROC AUC p-değeri : {_format_pvalue(tubitak_payload['selection_adjusted_permutation_p_roc_auc'])}",
            f"   Sabit-skor ROC AUC p-değeri      : {_format_pvalue(tubitak_payload['permutation_p_roc_auc'])}",
            (
                f"   Permütasyon sayısı: {tubitak_payload['n_permutations']}"
                if tubitak_payload["n_permutations"] is not None
                else "   Permütasyon sayısı: NA"
            ),
            "",
            "5. SIRALAMA UYUMU (Ham yeni ülke sayısı)",
            (
                f"   Spearman ρ       : {_format_metric(tubitak_payload['best_spearman'])} "
                f"[{_format_metric(tubitak_payload['best_spearman_ci_lower'])} - {_format_metric(tubitak_payload['best_spearman_ci_upper'])}]"
            ),
            "=================================================================",
        ]

        with text_output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        atomic_write_json(
            manifest_path,
            {
                "pipeline_settings": pipeline_settings,
                "input_signatures": [_path_signature(path) for path in input_paths],
                "source_signatures": [_path_signature(path) for path in source_paths],
                "output_signature": _path_signature(text_output_path),
            },
        )
        run.set_metric("cache_hit", False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
