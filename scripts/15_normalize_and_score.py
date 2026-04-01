#!/usr/bin/env python3
"""Normalize T/H/A features and build final backbone score tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.scoring import build_scored_backbone_table
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory


def _maybe_write_parquet(frame: pd.DataFrame, path: Path) -> bool:
    try:
        frame.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def main() -> int:
    context = build_context(PROJECT_ROOT)
    backbone_path = context.root / "data/features/backbone_table.tsv"
    t_path = context.root / "data/features/feature_T.tsv"
    h_path = context.root / "data/features/feature_H.tsv"
    a_path = context.root / "data/features/feature_A.tsv"
    scored_tsv = context.root / "data/scores/backbone_scored.tsv"
    scored_parquet = context.root / "data/scores/backbone_scored.parquet"
    ensure_directory(scored_tsv.parent)

    with ManagedScriptRun(context, "15_normalize_and_score") as run:
        for path in (backbone_path, t_path, h_path, a_path):
            run.record_input(path)
        run.record_output(scored_tsv)
        run.record_output(scored_parquet)

        backbone_table = read_tsv(backbone_path)
        feature_t = read_tsv(t_path)
        feature_h = read_tsv(h_path)
        feature_a = read_tsv(a_path)
        
        # Memory Optimization: Convert large object columns to categorical
        for df in (backbone_table, feature_t, feature_h, feature_a):
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if df[col].nunique() < len(df) / 2: # Only if it saves space
                    df[col] = df[col].astype('category')

        scored = build_scored_backbone_table(backbone_table, feature_t, feature_h, feature_a)
        
        # Runtime Validation: Ensure critical normalized scores remain within [0.0, 1.0] boundaries
        critical_cols = [c for c in scored.columns if c.endswith("_norm") or c.endswith("_index")]
        for col in critical_cols:
            if scored[col].dtype.kind in 'fc':
                out_of_bounds = scored[(scored[col] < 0.0) | (scored[col] > 1.0)]
                if not out_of_bounds.empty:
                    raise ValueError(
                        f"Validation failure: Column {col} contains {len(out_of_bounds)} values outside [0.0, 1.0]. "
                        "Fix upstream normalization before scoring."
                    )

        scored.to_csv(scored_tsv, sep="\t", index=False)
        parquet_ok = _maybe_write_parquet(scored, scored_parquet)
        if not parquet_ok:
            run.warn("Parquet output could not be written in the current environment; TSV fallback is available.")

        run.set_rows_out("backbone_scored_rows", int(len(scored)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
