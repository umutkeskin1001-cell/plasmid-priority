#!/usr/bin/env python3
"""Normalize T/H/A features and build final backbone score tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    build_discovery_input_contract,
    get_official_model_names,
    validate_discovery_input_contract,
)
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.scoring import build_scored_backbone_table
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)
from plasmid_priority.validation.missingness import audit_backbone_tables, format_missingness_report


def _maybe_write_parquet(frame: pd.DataFrame, path: Path) -> bool:
    try:
        frame.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and score backbone table")
    parser.add_argument(
        "--audit-missingness",
        action="store_true",
        help="Run missingness audit and write artifacts to reports/audits/",
    )
    args = parser.parse_args()

    context = build_context(PROJECT_ROOT)
    backbone_path = context.data_dir / "features/backbone_table.tsv"
    t_path = context.data_dir / "features/feature_T.tsv"
    h_path = context.data_dir / "features/feature_H.tsv"
    a_path = context.data_dir / "features/feature_A.tsv"
    config_path = context.root / "config.yaml"
    manifest_path = context.data_dir / "scores/15_normalize_and_score.manifest.json"
    scored_tsv = context.data_dir / "scores/backbone_scored.tsv"
    scored_parquet = context.data_dir / "scores/backbone_scored.parquet"
    ensure_directory(scored_tsv.parent)
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/15_normalize_and_score.py",
    )
    input_paths = [backbone_path, t_path, h_path, a_path, config_path]
    cache_metadata = {
        "pipeline_settings": {
            "split_year": int(context.pipeline_settings.split_year),
            "min_new_countries_for_spread": int(
                context.pipeline_settings.min_new_countries_for_spread
            ),
        }
    }

    with ManagedScriptRun(context, "15_normalize_and_score") as run:
        for path in (backbone_path, t_path, h_path, a_path, config_path):
            run.record_input(path)
        run.record_output(scored_tsv)
        run.record_output(scored_parquet)
        if load_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached scored backbone tables.")
            run.set_metric("cache_hit", True)

            # Still run missingness audit on cached data if requested
            if args.audit_missingness:
                scored = read_tsv(scored_tsv)
                _run_missingness_audit(context, scored, run)
            return 0

        backbone_table = read_tsv(backbone_path)
        feature_t = read_tsv(t_path)
        feature_h = read_tsv(h_path)
        feature_a = read_tsv(a_path)

        # Memory Optimization: Convert large object columns to categorical
        for df in (backbone_table, feature_t, feature_h, feature_a):
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if df[col].nunique() < len(df) / 2:  # Only if it saves space
                    df[col] = df[col].astype("category")

        scored = build_scored_backbone_table(backbone_table, feature_t, feature_h, feature_a)
        validate_discovery_input_contract(
            scored,
            model_names=get_official_model_names(MODULE_A_FEATURE_SETS.keys()),
            contract=build_discovery_input_contract(int(context.pipeline_settings.split_year)),
            label="Scored backbone table",
        )

        # Runtime Validation: Ensure critical normalized scores remain within [0.0, 1.0] boundaries
        critical_cols = [c for c in scored.columns if c.endswith("_norm") or c.endswith("_index")]
        for col in critical_cols:
            if scored[col].dtype.kind in "fc":
                out_of_bounds = scored[(scored[col] < 0.0) | (scored[col] > 1.0)]
                if not out_of_bounds.empty:
                    raise ValueError(
                        f"Validation failure: Column {col} contains {len(out_of_bounds)} values outside [0.0, 1.0]. "
                        "Fix upstream normalization before scoring."
                    )

        scored.to_csv(scored_tsv, sep="\t", index=False)
        parquet_ok = _maybe_write_parquet(scored, scored_parquet)
        if not parquet_ok:
            run.warn(
                "Parquet output could not be written in the current environment; TSV fallback is available."
            )
        write_signature_manifest(
            manifest_path,
            input_paths=input_paths,
            output_paths=[scored_tsv, scored_parquet],
            source_paths=source_paths,
            metadata=cache_metadata,
        )

        run.set_rows_out("backbone_scored_rows", int(len(scored)))
        run.set_metric("cache_hit", False)

        # Light missingness audit (opt-in, non-invasive)
        if args.audit_missingness:
            _run_missingness_audit(context, scored, run)

    return 0


def _run_missingness_audit(context, scored: pd.DataFrame, run) -> None:
    """Run missingness audit on scored table and write artifacts."""
    audit_dir = context.reports_dir / "audits"
    ensure_directory(audit_dir)

    audit_result = audit_backbone_tables(scored_backbone_table=scored)

    # Write machine-readable JSON
    json_path = audit_dir / "missingness_scored_backbone.json"
    with open(json_path, "w") as f:
        json.dump(audit_result.get("scored_backbone_table", audit_result), f, indent=2)

    # Write human-readable report
    txt_path = audit_dir / "missingness_scored_backbone.txt"
    with open(txt_path, "w") as f:
        if "scored_backbone_table" in audit_result:
            f.write(format_missingness_report(audit_result["scored_backbone_table"]))
        else:
            f.write("No scored_backbone_table audit data available.\n")

    run.note(f"Missingness audit written to {audit_dir}")
    run.set_metric("missingness_audit_status", audit_result.get("overall_status", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
