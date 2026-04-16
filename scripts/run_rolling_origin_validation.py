#!/usr/bin/env python3
"""Run rolling-origin validation across temporal split years."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plasmid_priority.config import build_context
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory
from plasmid_priority.validation.rolling_origin import run_rolling_origin_validation


def _scored_path(context, scored_path: str | None) -> Path:
    if scored_path:
        return Path(scored_path).expanduser().resolve()
    candidate = context.data_dir / "scores/backbone_scored.tsv"
    if candidate.exists():
        return candidate
    return context.root / "data/scores/backbone_scored.tsv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="governance_linear")
    parser.add_argument("--horizon-years", type=int, default=5)
    parser.add_argument("--assignment-mode", type=str, default="training_only")
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    scored_path = _scored_path(context, args.scored_path)
    if not scored_path.exists():
        print(f"ERROR: scored backbone table not found: {scored_path}", file=sys.stderr)
        return 1

    scored = read_tsv(scored_path)
    report = run_rolling_origin_validation(
        scored,
        model_name=args.model_name,
        horizon_years=int(args.horizon_years),
        assignment_mode=str(args.assignment_mode),
    )

    analysis_dir = context.data_dir / "analysis"
    reports_dir = context.root / "reports" / "core_tables"
    ensure_directory(analysis_dir)
    ensure_directory(reports_dir)

    detail_df = pd.DataFrame([result.__dict__ for result in report.split_results])
    detail_path = analysis_dir / "rolling_origin_validation.tsv"
    report_path = reports_dir / "rolling_origin_validation_summary.json"
    detail_df.to_csv(detail_path, sep="\t", index=False)
    report_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **report.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_name": report.model_name,
                "mean_auc": report.mean_auc,
                "mean_average_precision": report.mean_average_precision,
                "auc_stability_metric": report.auc_stability_metric,
                "detail_path": str(detail_path),
                "summary_path": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
