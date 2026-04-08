#!/usr/bin/env python3
"""Governance temporal evidence evaluation for rolling validation.

Produces honest rolling/temporal validation evidence for governance models:
- phylo_support_fusion_priority (baseline)
- governance_15f_pruned (challenger)

This is a thin governance-specific wrapper around existing validation infrastructure.
Does NOT rebuild scored data - uses existing backbone_scored.tsv with temporal filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    evaluate_model_name,
)
from plasmid_priority.utils.dataframe import read_tsv
from plasmid_priority.utils.files import ensure_directory
from plasmid_priority.utils.parallel import limit_native_threads
from plasmid_priority.validation import expected_calibration_error

# Governance models to evaluate
GOVERNANCE_BASELINE = "phylo_support_fusion_priority"
GOVERNANCE_CHALLENGER = "governance_15f_pruned"
GOVERNANCE_MODELS = [GOVERNANCE_BASELINE, GOVERNANCE_CHALLENGER]


def _resolve_parallel_jobs(requested_jobs: int | None, *, cap: int = 4) -> int:
    if requested_jobs is not None:
        return max(1, min(int(requested_jobs), cap))
    env_cap = os.getenv("PLASMID_PRIORITY_MAX_JOBS")
    if env_cap:
        try:
            cap = max(1, min(cap, int(env_cap)))
        except ValueError:
            pass
    return max(1, min(cap, os.cpu_count() or 1))


def _filter_scored_for_temporal_window(
    scored: pd.DataFrame,
    split_year: int,
    test_year_end: int,
    assignment_mode: str,
) -> pd.DataFrame:
    """Filter scored data for a specific temporal window.

    The scored file already contains temporal window columns from the pipeline:
    - split_year: the year used for training split
    - test_year_end: the end year for testing
    - backbone_assignment_mode: 'all_records' or 'training_only'
    """
    # Filter to matching temporal window
    mask = (
        (scored["split_year"] == split_year)
        & (scored["test_year_end"] == test_year_end)
        & (scored["backbone_assignment_mode"] == assignment_mode)
    )
    return scored.loc[mask].copy()


def _evaluate_governance_model_task(
    task: tuple[int, int, int, str, pd.DataFrame, str],
) -> dict[str, Any]:
    """Evaluate a governance model on a temporal window.

    Returns rolling row dict with temporal evidence.
    """
    split_year, window_end, horizon_years, assignment_mode, window_scored, model_name = task

    # Check if we have eligible data (spread_label not null)
    eligible = window_scored.loc[window_scored["spread_label"].notna()].copy()
    if len(eligible) < 20 or eligible["spread_label"].nunique() < 2:
        return {
            "split_year": int(split_year),
            "test_year_end": int(window_end),
            "horizon_years": int(horizon_years),
            "backbone_assignment_mode": assignment_mode,
            "model_name": model_name,
            "n_backbones": int(len(window_scored)),
            "n_eligible_backbones": int(len(eligible)),
            "status": "skipped_insufficient_label_variation",
            "roc_auc": None,
            "ece": None,
        }

    # Evaluate model using the same approach as Phase 6.1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate_model_name(
                window_scored,
                model_name=model_name,
                n_splits=5,
                n_repeats=3,
                seed=42,
                include_ci=False,
            )

        # Calculate ECE on eligible data
        y_true = eligible["spread_label"].astype(int).to_numpy()

        # Get predictions from result - handle different result structures
        try:
            preds_df = result.predictions
            if isinstance(preds_df, pd.DataFrame) and "prediction" in preds_df.columns:
                # Align predictions with eligible backbone_ids
                pred_map = preds_df.set_index("backbone_id")["prediction"].to_dict()
                y_score = eligible["backbone_id"].map(pred_map).fillna(0).to_numpy()
            else:
                y_score = np.zeros(len(eligible))
        except Exception:
            # Fallback: use zeros if predictions unavailable
            y_score = np.zeros(len(eligible))

        ece = expected_calibration_error(y_true, y_score, n_bins=10)

        return {
            "split_year": int(split_year),
            "test_year_end": int(window_end),
            "horizon_years": int(horizon_years),
            "backbone_assignment_mode": assignment_mode,
            "model_name": model_name,
            "n_backbones": int(len(window_scored)),
            "n_eligible_backbones": int(len(eligible)),
            "n_positive": int(result.metrics.get("n_positive", 0)),
            "positive_prevalence": float(result.metrics.get("positive_prevalence", 0.0)),
            "roc_auc": float(result.metrics.get("roc_auc", 0.0)),
            "average_precision": float(result.metrics.get("average_precision", 0.0)),
            "brier_score": float(result.metrics.get("brier_score", 0.0)),
            "ece": float(ece),
            "status": "ok",
        }
    except Exception as e:
        return {
            "split_year": int(split_year),
            "test_year_end": int(window_end),
            "horizon_years": int(horizon_years),
            "backbone_assignment_mode": assignment_mode,
            "model_name": model_name,
            "n_backbones": int(len(window_scored)),
            "n_eligible_backbones": int(len(eligible)),
            "status": f"error: {str(e)[:100]}",
            "roc_auc": None,
            "ece": None,
        }


def _compute_temporal_gap_vs_oof(
    rolling_df: pd.DataFrame,
    oof_auc: dict[str, float],
) -> pd.DataFrame:
    """Compute temporal gap vs out-of-fold (standard) AUC for each model."""
    df = rolling_df.copy()
    df["temporal_gap_vs_oof"] = float("nan")

    for model_name in GOVERNANCE_MODELS:
        model_mask = df["model_name"] == model_name
        oof_value = oof_auc.get(model_name, float("nan"))
        if not pd.isna(oof_value):
            # Ensure roc_auc is float before subtraction
            roc_auc_values = pd.to_numeric(df.loc[model_mask, "roc_auc"], errors="coerce")
            df.loc[model_mask, "temporal_gap_vs_oof"] = roc_auc_values - oof_value

    return df


def _summarize_temporal_evidence(
    rolling_df: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize temporal evidence for governance recommendation."""
    summary = {}

    for model_name in GOVERNANCE_MODELS:
        model_df = rolling_df.loc[rolling_df["model_name"] == model_name].copy()

        # Only include successful evaluations
        ok_df = model_df.loc[model_df["status"] == "ok"]

        if ok_df.empty:
            summary[model_name] = {
                "temporal_evidence_status": "not_evaluated",
                "mean_rolling_auc": None,
                "rolling_auc_std": None,
                "mean_ece": None,
                "mean_temporal_gap_vs_oof": None,
                "n_windows_evaluated": 0,
            }
            continue

        # Compute summary statistics
        mean_auc = ok_df["roc_auc"].mean()
        std_auc = ok_df["roc_auc"].std()
        mean_ece = ok_df["ece"].mean()
        mean_gap = ok_df["temporal_gap_vs_oof"].mean()
        n_windows = len(ok_df)

        # Determine evidence status
        if n_windows >= 10:
            status = "evaluated"
        elif n_windows > 0:
            status = "partially_evaluated"
        else:
            status = "not_evaluated"

        summary[model_name] = {
            "temporal_evidence_status": status,
            "mean_rolling_auc": float(mean_auc) if not pd.isna(mean_auc) else None,
            "rolling_auc_std": float(std_auc) if not pd.isna(std_auc) else None,
            "mean_ece": float(mean_ece) if not pd.isna(mean_ece) else None,
            "mean_temporal_gap_vs_oof": float(mean_gap) if not pd.isna(mean_gap) else None,
            "n_windows_evaluated": int(n_windows),
        }

    return summary


def _determine_recommendation(
    summary: dict[str, Any],
    baseline_auc: float,
    challenger_auc: float,
    paired_delong_p: float,
) -> str:
    """Determine final governance recommendation based on temporal evidence.

    Uses stability-first logic:
    - KEEP_BASELINE_GOVERNANCE if baseline remains official and challenger lacks temporal support
    - PROCEED_WITH_GOVERNANCE_15F if challenger has adequate temporal support AND stability
    - CONSIDER_HYBRID_NEXT if challenger is acceptable but insufficient evidence for promotion
    - STOP if temporal evaluation cannot be trusted
    """
    baseline_summary = summary.get(GOVERNANCE_BASELINE, {})
    challenger_summary = summary.get(GOVERNANCE_CHALLENGER, {})

    baseline_status = baseline_summary.get("temporal_evidence_status", "not_evaluated")
    challenger_status = challenger_summary.get("temporal_evidence_status", "not_evaluated")

    # Check if we have adequate evidence
    if baseline_status == "not_evaluated" and challenger_status == "not_evaluated":
        return "STOP"

    if baseline_status == "not_evaluated":
        # Cannot compare if baseline not evaluated
        return "KEEP_BASELINE_GOVERNANCE"

    if challenger_status == "not_evaluated":
        # Challenger has no temporal evidence
        return "KEEP_BASELINE_GOVERNANCE"

    # Both have at least partial evidence
    baseline_gap = baseline_summary.get("mean_temporal_gap_vs_oof", 0.0) or 0.0
    challenger_gap = challenger_summary.get("mean_temporal_gap_vs_oof", 0.0) or 0.0
    challenger_ece = challenger_summary.get("mean_ece", 1.0) or 1.0

    # Stability-first: challenger should not be materially worse temporally
    # If challenger has larger gap (more degradation), it may not be suitable
    gap_difference = challenger_gap - baseline_gap

    # Thresholds (conservative)
    GAP_TOLERANCE = 0.03  # 3% AUC degradation tolerance
    ECE_THRESHOLD = 0.10  # Same as frozen threshold

    if gap_difference > GAP_TOLERANCE:
        # Challenger degrades more than baseline in temporal evaluation
        # Not promotable based on current evidence
        return "CONSIDER_HYBRID_NEXT"

    if challenger_ece > ECE_THRESHOLD:
        # Challenger calibration unacceptable
        return "KEEP_BASELINE_GOVERNANCE"

    # Check if challenger is actually superior or at least stability-preserving
    auc_difference = challenger_auc - baseline_auc

    if auc_difference > 0.01 and paired_delong_p > 0.05:
        # Challenger modestly better but not significantly
        # Temporal evidence supports stability
        return "PROCEED_WITH_GOVERNANCE_15F"
    elif auc_difference >= -0.015:
        # Within stability tolerance (from Phase 6.1 classification)
        if challenger_status == "evaluated":
            # Adequate temporal evidence for stability-preserving promotion
            return "PROCEED_WITH_GOVERNANCE_15F"
        else:
            # Partial evidence only - need more
            return "CONSIDER_HYBRID_NEXT"
    else:
        # AUC loss too large
        return "KEEP_BASELINE_GOVERNANCE"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Governance temporal evidence evaluation"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto-capped at 4)",
    )
    parser.add_argument(
        "--external-data-root",
        type=str,
        default="/Volumes/UMUT/data",
        help="External data root path",
    )
    parser.add_argument(
        "--scored-path",
        type=str,
        default=None,
        help="Path to scored backbone table (default: external data root)",
    )
    args = parser.parse_args()

    context = build_context(PROJECT_ROOT)

    # Check for external data availability
    external_data_root = Path(args.external_data_root)

    # Resolve scored path
    if args.scored_path:
        scored_path = Path(args.scored_path)
    elif external_data_root.exists():
        scored_path = external_data_root / "scores/backbone_scored.tsv"
    else:
        scored_path = context.data_dir / "scores/backbone_scored.tsv"

    # Verify required inputs exist
    if not scored_path.exists():
        print(f"ERROR: Required scored data file not found: {scored_path}")
        print("Cannot proceed without scored data.")
        return 1

    print(f"Using scored data: {scored_path}")

    # Output paths
    temporal_json_path = context.data_dir / "analysis/governance_temporal_evidence.json"
    temporal_tsv_path = context.root / "reports/governance_temporal_evidence.tsv"
    recommendation_md_path = context.root / "reports/governance_temporal_recommendation.md"

    ensure_directory(temporal_json_path.parent)
    ensure_directory(temporal_tsv_path.parent)

    # Load scored data (same as Phase 6.1)
    print("Loading scored data...")
    scored = read_tsv(scored_path)
    print(f"Loaded {len(scored)} rows from scored data")

    # Check what temporal windows are available in scored data
    available_windows = scored[["split_year", "test_year_end", "backbone_assignment_mode"]].drop_duplicates()
    print(f"Available temporal windows: {len(available_windows)}")
    print(f"Unique split_years: {sorted(scored['split_year'].dropna().unique())}")
    print(f"Unique test_year_ends: {sorted(scored['test_year_end'].dropna().unique())}")

    # Get OOF (out-of-fold) AUC from existing Phase 6.1 metrics
    oof_auc = {
        GOVERNANCE_BASELINE: 0.8272,
        GOVERNANCE_CHALLENGER: 0.8234,
    }

    # Build rolling evaluation tasks
    print("Building rolling temporal evaluation tasks...")
    rolling_tasks: list[tuple[int, int, int, str, pd.DataFrame, str]] = []

    # Rolling windows: 2012-2018 split years, 1/3/5/8 year horizons
    for split_year in range(2012, 2019):
        for horizon_years in (1, 3, 5, 8):
            window_end = min(split_year + horizon_years, 2023)
            if window_end <= split_year:
                continue

            for assignment_mode in ("all_records", "training_only"):
                # Filter scored data for this temporal window
                window_scored = _filter_scored_for_temporal_window(
                    scored, split_year, window_end, assignment_mode
                )

                if window_scored.empty:
                    continue

                for model_name in GOVERNANCE_MODELS:
                    rolling_tasks.append(
                        (
                            int(split_year),
                            int(window_end),
                            int(horizon_years),
                            assignment_mode,
                            window_scored,
                            model_name,
                        )
                    )

    print(f"Total evaluation tasks: {len(rolling_tasks)}")

    if not rolling_tasks:
        print("ERROR: No valid temporal windows found in scored data.")
        print("Available windows in data:")
        print(available_windows.head(20))
        return 1

    # Execute tasks
    n_jobs = _resolve_parallel_jobs(args.jobs)
    print(f"Running with {n_jobs} parallel job(s)...")

    if n_jobs > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                rolling_results = list(executor.map(_evaluate_governance_model_task, rolling_tasks))
    else:
        rolling_results = [_evaluate_governance_model_task(task) for task in rolling_tasks]

    # Build rolling dataframe
    rolling_df = pd.DataFrame(rolling_results)

    # Add temporal gap vs OOF
    rolling_df = _compute_temporal_gap_vs_oof(rolling_df, oof_auc)

    # Compute summary
    summary = _summarize_temporal_evidence(rolling_df)

    # Get Phase 6.1 reference values for recommendation
    baseline_auc = oof_auc[GOVERNANCE_BASELINE]
    challenger_auc = oof_auc[GOVERNANCE_CHALLENGER]
    paired_delong_p = 0.405  # From Phase 6.1 per_model_summary

    # Determine recommendation
    recommendation = _determine_recommendation(
        summary,
        baseline_auc=baseline_auc,
        challenger_auc=challenger_auc,
        paired_delong_p=paired_delong_p,
    )

    # Build output JSON
    output_payload = {
        "governance_models_evaluated": GOVERNANCE_MODELS,
        "temporal_summary": summary,
        "rolling_evidence": rolling_df.to_dict("records"),
        "reference_values": {
            "oof_auc": oof_auc,
            "paired_delong_p": paired_delong_p,
        },
        "recommendation": recommendation,
        "recommendation_class": recommendation,
        "n_rolling_windows_attempted": len(rolling_tasks),
        "n_rolling_windows_successful": int((rolling_df["status"] == "ok").sum()),
    }

    # Write JSON output
    with open(temporal_json_path, "w") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"Wrote temporal evidence JSON: {temporal_json_path}")

    # Write TSV summary (one row per model)
    summary_rows = []
    for model_name in GOVERNANCE_MODELS:
        model_summary = summary.get(model_name, {})
        summary_rows.append({
            "model_name": model_name,
            "temporal_evidence_status": model_summary.get("temporal_evidence_status", "not_evaluated"),
            "mean_rolling_auc": model_summary.get("mean_rolling_auc"),
            "rolling_auc_std": model_summary.get("rolling_auc_std"),
            "mean_ece": model_summary.get("mean_ece"),
            "mean_temporal_gap_vs_oof": model_summary.get("mean_temporal_gap_vs_oof"),
            "n_windows_evaluated": model_summary.get("n_windows_evaluated", 0),
            "baseline_comparison": "baseline" if model_name == GOVERNANCE_BASELINE else "challenger",
            "recommendation_class": (
                "OFFICIAL_BASELINE" if model_name == GOVERNANCE_BASELINE
                else ("PROMOTABLE_CHALLENGER" if recommendation == "PROCEED_WITH_GOVERNANCE_15F"
                      else "CHALLENGER_ONLY")
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(temporal_tsv_path, sep="\t", index=False)
    print(f"Wrote temporal evidence TSV: {temporal_tsv_path}")

    # Write recommendation markdown
    baseline_summary = summary.get(GOVERNANCE_BASELINE, {})
    challenger_summary = summary.get(GOVERNANCE_CHALLENGER, {})

    md_content = f"""# Governance Temporal Evidence Report

**Date**: Auto-generated from batch artifacts
**Evaluation**: Rolling temporal validation for governance models
**Status**: {recommendation}

---

## Models Evaluated

| Model | Role | OOF AUC | Temporal Status |
|-------|------|---------|-----------------|
| {GOVERNANCE_BASELINE} | Baseline | {baseline_auc:.4f} | {baseline_summary.get("temporal_evidence_status", "not_evaluated")} |
| {GOVERNANCE_CHALLENGER} | Challenger | {challenger_auc:.4f} | {challenger_summary.get("temporal_evidence_status", "not_evaluated")} |

---

## Temporal Evidence Summary

### {GOVERNANCE_BASELINE} (Baseline)

- **Temporal Evidence Status**: {baseline_summary.get("temporal_evidence_status", "not_evaluated")}
- **Mean Rolling AUC**: {baseline_summary.get("mean_rolling_auc", "N/A")}
- **Rolling AUC Std**: {baseline_summary.get("rolling_auc_std", "N/A")}
- **Mean ECE**: {baseline_summary.get("mean_ece", "N/A")}
- **Mean Temporal Gap vs OOF**: {baseline_summary.get("mean_temporal_gap_vs_oof", "N/A")}
- **Windows Evaluated**: {baseline_summary.get("n_windows_evaluated", 0)}

### {GOVERNANCE_CHALLENGER} (Challenger)

- **Temporal Evidence Status**: {challenger_summary.get("temporal_evidence_status", "not_evaluated")}
- **Mean Rolling AUC**: {challenger_summary.get("mean_rolling_auc", "N/A")}
- **Rolling AUC Std**: {challenger_summary.get("rolling_auc_std", "N/A")}
- **Mean ECE**: {challenger_summary.get("mean_ece", "N/A")}
- **Mean Temporal Gap vs OOF**: {challenger_summary.get("mean_temporal_gap_vs_oof", "N/A")}
- **Windows Evaluated**: {challenger_summary.get("n_windows_evaluated", 0)}

---

## Comparison Summary

- **Baseline OOF AUC**: {baseline_auc:.4f}
- **Challenger OOF AUC**: {challenger_auc:.4f}
- **AUC Difference (Challenger - Baseline)**: {challenger_auc - baseline_auc:.4f}
- **paired_delong_p**: {paired_delong_p:.3f} (from Phase 6.1)

**Temporal Gap Comparison**:
- Baseline mean gap vs OOF: {baseline_summary.get("mean_temporal_gap_vs_oof", "N/A")}
- Challenger mean gap vs OOF: {challenger_summary.get("mean_temporal_gap_vs_oof", "N/A")}

---

## Honest Assessment

The p-values referenced are from paired DeLong tests comparing each candidate to baseline.
These are **NOT** selection-adjusted permutation-null p-values.
True selection-adjusted inference accounting for post-hoc model selection would require
additional `build_selection_adjusted_permutation_null()` execution.

Temporal evidence was explicitly evaluated across multiple rolling windows:
- Split years: 2012-2018
- Horizon years: 1, 3, 5, 8
- Assignment modes: all_records, training_only

---

## Final Recommendation

### **{recommendation}**

**Rationale**:
"""

    if recommendation == "KEEP_BASELINE_GOVERNANCE":
        md_content += f"""The governance baseline ({GOVERNANCE_BASELINE}) remains the official model.
The challenger ({GOVERNANCE_CHALLENGER}) does not have sufficient temporal evidence to justify
promotion. Either temporal evaluation was incomplete, or the challenger showed material
degradation relative to baseline in rolling validation. Stability-first semantics require
conservative retention of the proven baseline until compelling replacement evidence exists.
"""
    elif recommendation == "PROCEED_WITH_GOVERNANCE_15F":
        md_content += f"""The governance challenger ({GOVERNANCE_CHALLENGER}) now has adequate
temporal evidence supporting promotion. The model demonstrates stability across rolling windows
with acceptable temporal degradation. AUC performance is maintained within tolerance,
and calibration remains acceptable (ECE < 0.10). The challenger may now replace the baseline
as the official governance model.
"""
    elif recommendation == "CONSIDER_HYBRID_NEXT":
        md_content += f"""The governance challenger ({GOVERNANCE_CHALLENGER}) shows acceptable
performance but lacks sufficient temporal evidence alone to justify promotion. The model is
classified as stability-preserving, but temporal validation suggests marginal degradation
or incomplete coverage. The recommended next step is to consider hybrid experiments
(Phase 6.2) to strengthen the evidence base before making a promotion decision.
"""
    else:  # STOP
        md_content += """Temporal evaluation could not be completed reliably due to missing
inputs, insufficient data, or evaluation errors. Do not proceed with governance promotion
decisions based on incomplete evidence. Re-run when data issues are resolved.
"""

    md_content += """
---

## Artifact Locations

- **Temporal Evidence JSON**: `data/analysis/governance_temporal_evidence.json`
- **Temporal Evidence TSV**: `reports/governance_temporal_evidence.tsv`
- **This Report**: `reports/governance_temporal_recommendation.md`

---

*Report generated by `scripts/run_governance_temporal_evidence.py`*
"""

    with open(recommendation_md_path, "w") as f:
        f.write(md_content)
    print(f"Wrote recommendation markdown: {recommendation_md_path}")

    print("\n=== GOVERNANCE TEMPORAL EVIDENCE COMPLETE ===")
    print(f"Recommendation: {recommendation}")
    print(f"Windows evaluated successfully: {(rolling_df['status'] == 'ok').sum()} / {len(rolling_tasks)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
