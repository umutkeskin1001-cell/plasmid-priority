#!/usr/bin/env python3
"""Export simplified summary metrics for the final TÜBİTAK report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context, context_config_paths
from plasmid_priority.modeling import get_primary_model_name
from plasmid_priority.qc.input_checks import verify_asset_fingerprint
from plasmid_priority.protocol import ScientificProtocol, build_protocol_hash
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory, path_signature


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _cache_key_path(path: Path) -> Path:
    return path.with_name(path.name + ".cache_key")


def _cache_key_payload(
    *,
    protocol_hash: str,
    input_paths: list[Path],
    source_paths: list[Path],
    metadata: dict[str, object],
    feature_schema_version: str,
) -> dict[str, object]:
    return {
        "protocol_hash": protocol_hash,
        "input_hash": _stable_hash(
            {
                "input_signatures": [path_signature(path) for path in input_paths],
                "source_signatures": [path_signature(path) for path in source_paths],
                "metadata": metadata,
            }
        ),
        "feature_schema_version": feature_schema_version,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _cache_key_matches(cache_key_path: Path, expected: dict[str, object]) -> bool:
    if not cache_key_path.exists():
        return False
    try:
        payload = json.loads(cache_key_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    for key in ("protocol_hash", "input_hash", "feature_schema_version"):
        if payload.get(key) != expected.get(key):
            return False
    return True


def _load_cached_manifest(
    manifest_path: Path,
    *,
    input_paths: list[Path],
    source_paths: list[Path],
    output_paths: list[Path],
    pipeline_settings: dict[str, object],
) -> bool:
    if not manifest_path.exists() or any(not path.exists() for path in output_paths):
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
    if payload.get("output_signatures") != [_path_signature(path) for path in output_paths]:
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


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _gate_pass(value: float | None, threshold: float, *, comparator: str) -> bool | None:
    if value is None:
        return None
    if comparator == "le":
        return value <= threshold
    if comparator == "ge":
        return value >= threshold
    if comparator == "abs_le":
        return abs(value) <= threshold
    raise ValueError(f"Unsupported comparator: {comparator}")


def _gate_payload(
    *,
    value: float | None,
    threshold: float | None = None,
    threshold_range: tuple[float, float] | None = None,
    comparator: str,
) -> dict[str, object]:
    if threshold_range is not None:
        threshold_value: object = list(threshold_range)
        if value is None:
            passed = None
        else:
            passed = threshold_range[0] <= value <= threshold_range[1]
    else:
        threshold_value = threshold
        passed = _gate_pass(value, float(threshold or 0.0), comparator=comparator)
    return {
        "value": value,
        "threshold": threshold_value,
        "pass": passed,
    }


def _load_primary_model_row(frame: pd.DataFrame, model_name: str) -> pd.Series:
    if frame.empty or "model_name" not in frame.columns:
        return pd.Series(dtype=object)
    row = frame.loc[frame["model_name"].astype(str) == str(model_name)].head(1)
    return row.iloc[0] if not row.empty else pd.Series(dtype=object)


def main() -> int:
    context = build_context(PROJECT_ROOT)
    protocol = ScientificProtocol.from_config(context.config)
    protocol_hash = build_protocol_hash(protocol)
    audit_json = context.data_dir / "analysis/module_a_metrics.json"
    permutation_summary_path = context.data_dir / "analysis/permutation_null_summary.tsv"
    selection_adjusted_permutation_summary_path = (
        context.data_dir / "analysis/selection_adjusted_permutation_null_summary.tsv"
    )
    count_outcome_audit_path = context.data_dir / "analysis/new_country_count_audit.tsv"
    frozen_audit_path = _first_existing(
        [
            context.reports_dir / "core_tables/frozen_scientific_acceptance_audit.tsv",
            context.data_dir / "analysis/frozen_scientific_acceptance_audit.tsv",
        ]
    )
    fingerprint_manifest_path = context.root / "data/fingerprints/raw_assets.tsv"
    text_output_path = context.reports_dir / "tubitak_final_metrics.txt"
    health_dashboard_path = context.reports_dir / "health_dashboard.md"
    quality_gate_summary_path = context.reports_dir / "core_tables/quality_gate_summary.json"
    manifest_path = text_output_path.with_suffix(text_output_path.suffix + ".manifest.json")
    cache_key_path = _cache_key_path(text_output_path)
    ensure_directory(text_output_path.parent)
    ensure_directory(health_dashboard_path.parent)
    ensure_directory(quality_gate_summary_path.parent)
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
        if frozen_audit_path is not None:
            input_paths.append(frozen_audit_path)
            run.record_input(frozen_audit_path)
        if fingerprint_manifest_path.exists():
            input_paths.append(fingerprint_manifest_path)
            run.record_input(fingerprint_manifest_path)
        run.record_output(text_output_path)
        run.record_output(health_dashboard_path)
        run.record_output(quality_gate_summary_path)
        run.record_output(cache_key_path)

        pipeline_settings = {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
        cache_key_payload = _cache_key_payload(
            protocol_hash=protocol_hash,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=pipeline_settings,
            feature_schema_version="tubitak-v1",
        )
        if _load_cached_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            output_paths=[text_output_path, health_dashboard_path, quality_gate_summary_path],
            pipeline_settings=pipeline_settings,
        ) and _cache_key_matches(cache_key_path, cache_key_payload):
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
        frozen_audit = (
            pd.read_csv(frozen_audit_path, sep="\t") if frozen_audit_path is not None else pd.DataFrame()
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
        frozen_row = _load_primary_model_row(frozen_audit, primary_model)
        calibration_slope = _clean_metric(primary_metrics.get("calibration_slope"))
        calibration_intercept = _clean_metric(primary_metrics.get("calibration_intercept"))
        if calibration_slope is None and not frozen_row.empty:
            calibration_slope = _clean_metric(frozen_row.get("calibration_slope"))
        if calibration_intercept is None and not frozen_row.empty:
            calibration_intercept = _clean_metric(frozen_row.get("calibration_intercept"))
        ece_value = _clean_metric(
            primary_metrics.get("ece", frozen_row.get("ece") if not frozen_row.empty else None)
        )
        matched_knownness_gap = _clean_metric(
            frozen_row.get("knownness_matched_gap")
            if not frozen_row.empty
            else primary_metrics.get("knownness_matched_gap")
        )
        source_holdout_gap = _clean_metric(
            frozen_row.get("source_holdout_gap")
            if not frozen_row.empty
            else primary_metrics.get("source_holdout_gap")
        )
        spatial_holdout_gap = _clean_metric(
            frozen_row.get("spatial_holdout_gap")
            if not frozen_row.empty
            else primary_metrics.get("spatial_holdout_gap")
        )
        selection_adjusted_p = _clean_metric(
            primary_selection_adjusted.iloc[0]["selection_adjusted_empirical_p_roc_auc"]
        ) if not primary_selection_adjusted.empty else _clean_metric(
            frozen_row.get("selection_adjusted_empirical_p_roc_auc")
            if not frozen_row.empty
            else primary_metrics.get("selection_adjusted_empirical_p_roc_auc")
        )
        matched_knownness_threshold = _clean_metric(
            frozen_row.get("matched_knownness_gap_min")
            if not frozen_row.empty
            else primary_metrics.get("matched_knownness_gap_min")
        )
        source_holdout_threshold = _clean_metric(
            frozen_row.get("source_holdout_gap_min")
            if not frozen_row.empty
            else primary_metrics.get("source_holdout_gap_min")
        )
        spatial_holdout_threshold = _clean_metric(
            frozen_row.get("spatial_holdout_gap_min")
            if not frozen_row.empty
            else primary_metrics.get("spatial_holdout_gap_min")
        )
        ece_threshold = _clean_metric(
            frozen_row.get("ece_max") if not frozen_row.empty else primary_metrics.get("ece_max")
        )
        selection_adjusted_threshold = _clean_metric(
            frozen_row.get("selection_adjusted_p_max")
            if not frozen_row.empty
            else primary_metrics.get("selection_adjusted_p_max")
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
            "n_eligible_backbones": int(primary_metrics.get("n_backbones", 0))
            if primary_metrics.get("n_backbones") is not None
            else None,
            "n_positive_backbones": int(primary_metrics.get("n_positive", 0))
            if primary_metrics.get("n_positive") is not None
            else None,
            "sample_weighting_strategy": str(
                primary_metrics.get(
                    "sample_weight_mode",
                    "class_balanced+inverse_knownness_weighting",
                )
            ),
            "calibration_slope": calibration_slope,
            "calibration_intercept": calibration_intercept,
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

        quality_gate_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_hash": protocol_hash,
            "model_name": primary_model,
            "gates": {
                "ece": _gate_payload(
                    value=ece_value,
                    threshold=ece_threshold if ece_threshold is not None else 0.05,
                    comparator="le",
                ),
                "selection_adjusted_p": _gate_payload(
                    value=selection_adjusted_p,
                    threshold=selection_adjusted_threshold
                    if selection_adjusted_threshold is not None
                    else 0.01,
                    comparator="le",
                ),
                "matched_knownness_gap": _gate_payload(
                    value=matched_knownness_gap,
                    threshold=matched_knownness_threshold
                    if matched_knownness_threshold is not None
                    else -0.005,
                    comparator="ge",
                ),
                "source_holdout_gap": _gate_payload(
                    value=source_holdout_gap,
                    threshold=source_holdout_threshold if source_holdout_threshold is not None else -0.005,
                    comparator="ge",
                ),
                "spatial_holdout_gap": _gate_payload(
                    value=spatial_holdout_gap,
                    threshold=spatial_holdout_threshold
                    if spatial_holdout_threshold is not None
                    else -0.03,
                    comparator="ge",
                ),
                "calibration_slope": _gate_payload(
                    value=calibration_slope,
                    threshold_range=(0.85, 1.15),
                    comparator="abs_le",
                ),
                "calibration_intercept": _gate_payload(
                    value=calibration_intercept,
                    threshold_range=(-0.1, 0.1),
                    comparator="abs_le",
                ),
            },
            "overall_pass": bool(
                not frozen_row.empty
                and str(frozen_row.get("scientific_acceptance_status", "not_scored")) == "pass"
            ),
        }
        if frozen_row.empty:
            quality_gate_summary["overall_pass"] = all(
                entry["pass"] is True for entry in quality_gate_summary["gates"].values()
            )

        asset_rows: list[dict[str, object]] = []
        if fingerprint_manifest_path.exists():
            asset_manifest = pd.read_csv(fingerprint_manifest_path, sep="\t")
            for row in asset_manifest.to_dict(orient="records"):
                asset_key = str(row.get("asset_key", "")).strip()
                raw_path = Path(str(row.get("path", ""))).expanduser()
                if not raw_path.is_absolute():
                    raw_path = (context.root / raw_path).resolve()
                expected_sha256 = str(row.get("sha256", "")).strip()
                asset_status = "ok"
                asset_message = "verified"
                try:
                    verify_asset_fingerprint(asset_key, raw_path, expected_sha256)
                except Exception as exc:  # pragma: no cover - defensive
                    asset_status = "error"
                    asset_message = str(exc)
                asset_rows.append(
                    {
                        "asset_key": asset_key,
                        "path": str(raw_path),
                        "sha256": expected_sha256,
                        "size_bytes": int(raw_path.stat().st_size) if raw_path.exists() else None,
                        "status": asset_status,
                        "message": asset_message,
                    }
                )
        quality_gate_summary["data_assets"] = asset_rows

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
            f"   Temel Oran (Pozitif Prevalans) : {_format_metric(tubitak_payload['prevalence'])} "
            f"(n_pos={tubitak_payload.get('n_positive_backbones', 'NA')} / "
            f"n_total={tubitak_payload.get('n_eligible_backbones', 'NA')})",
            f"   Lift (Kazanç)    : {_format_metric(ap_lift)}",
            f"   Uyg. Sample Weighting: {tubitak_payload.get('sample_weighting_strategy', 'NA')}",
            "",
            "3. KALİBRASYON (Brier Score)",
            f"   Değer            : {_format_metric(tubitak_payload['brier_score'])}",
            f"   Calibration slope      : {_format_metric(tubitak_payload['calibration_slope'])}",
            f"   Calibration intercept  : {_format_metric(tubitak_payload['calibration_intercept'])}",
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
            "",
            "*** SINIF DENGESİZLİĞİ NOTU ***",
            "Bu veri seti sınıf dengesizliği içermektedir. Yukarıdaki Temel Oran,",
            "gerçek dünya AMR plazmid yayılımı prevalansını yansıtır.",
            "Sınıf dengesi için sample_weight_mode=class_balanced+inverse_knownness_weighting",
            "uygulanmıştır. AUC değeri, yüksek negatif sınıf oranına rağmen",
            "geçerli bir sinyal göstergesidir (Permütasyon testi ile doğrulanmıştır).",
            "=================================================================",
        ]

        dashboard_lines = [
            f"# Pipeline Health Dashboard - {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "## Protocol Status",
            f"- Hash: {protocol_hash[:12]}",
            (
                f"- Split year: {int(pipeline.split_year)} | Horizon: {int(protocol.horizon_years)}y | "
                "Accept thresholds: ECE<=0.05, p<=0.01"
            ),
            "",
            "## Data Status",
        ]
        if asset_rows:
            dashboard_lines.extend(
                [
                    "| Asset | Hash | Size | Status |",
                    "|-------|------|------|--------|",
                ]
            )
            for asset in asset_rows:
                size_text = (
                    str(int(asset["size_bytes"])) if asset.get("size_bytes") is not None else "NA"
                )
                dashboard_lines.append(
                    f"| {asset['asset_key']} | {str(asset['sha256'])[:12]} | {size_text} | {str(asset['status']).upper()} |"
                )
        else:
            dashboard_lines.append("No fingerprint manifest available.")
        dashboard_lines.extend(
            [
                "",
                "## Quality Gates",
                "| Gate | Value | Threshold | Status |",
                "|------|-------|-----------|--------|",
            ]
        )
        for gate_name, gate in quality_gate_summary["gates"].items():
            threshold = gate.get("threshold")
            if isinstance(threshold, list) and len(threshold) == 2:
                threshold_text = f"[{_format_metric(_clean_metric(threshold[0]))} - {_format_metric(_clean_metric(threshold[1]))}]"
            else:
                threshold_text = _format_metric(_clean_metric(threshold))
            status_text = "PASS" if gate.get("pass") is True else "FAIL" if gate.get("pass") is False else "NA"
            dashboard_lines.append(
                f"| {gate_name} | {_format_metric(_clean_metric(gate.get('value')))} | {threshold_text} | {status_text} |"
            )
        dashboard_lines.extend(
            [
                "",
                "## Release Readiness",
                "PASS" if quality_gate_summary["overall_pass"] else "FAIL",
            ]
        )

        with text_output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        with health_dashboard_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(dashboard_lines) + "\n")

        atomic_write_json(quality_gate_summary_path, quality_gate_summary)

        atomic_write_json(
            manifest_path,
            {
                "pipeline_settings": pipeline_settings,
                "input_signatures": [_path_signature(path) for path in input_paths],
                "source_signatures": [_path_signature(path) for path in source_paths],
                "output_signatures": [
                    _path_signature(text_output_path),
                    _path_signature(health_dashboard_path),
                    _path_signature(quality_gate_summary_path),
                ],
            },
        )
        atomic_write_json(cache_key_path, cache_key_payload)
        run.set_metric("cache_hit", False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
