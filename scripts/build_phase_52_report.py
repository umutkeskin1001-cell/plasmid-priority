#!/usr/bin/env python3
"""Build Phase 5.2 Discovery Batch human-readable recommendation report.

This script consumes artifacts from run_phase_52_discovery.py and generates:
- reports/phase_52_recommendation.md

The report includes:
- Brief execution summary
- Per-candidate vs baseline table
- Honest batch winner selection result
- Gate pass/fail summary
- Final recommendation: PROCEED / CALIBRATION_WORK / STOP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)


def load_phase_52_artifacts(
    data_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Load all artifacts produced by run_phase_52_discovery.py."""
    metrics_path = data_dir / "analysis/phase_52_metrics.json"
    per_model_path = reports_dir / "phase_52_per_model_summary.tsv"
    batch_winner_path = reports_dir / "phase_52_batch_winner.json"
    gate_eval_path = reports_dir / "phase_52_gate_evaluation.tsv"

    artifacts: dict[str, Any] = {}

    # Load metrics
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
    else:
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    # Load per-model summary
    if per_model_path.exists():
        artifacts["per_model_summary"] = pd.read_csv(per_model_path, sep="\t")
    else:
        raise FileNotFoundError(f"Per-model summary not found: {per_model_path}")

    # Load batch winner
    if batch_winner_path.exists():
        with open(batch_winner_path, "r") as f:
            artifacts["batch_winner"] = json.load(f)
    else:
        raise FileNotFoundError(f"Batch winner file not found: {batch_winner_path}")

    # Load gate evaluation
    if gate_eval_path.exists():
        artifacts["gate_evaluation"] = pd.read_csv(gate_eval_path, sep="\t")
    else:
        raise FileNotFoundError(f"Gate evaluation file not found: {gate_eval_path}")

    return artifacts


def determine_recommendation(
    per_model_summary: pd.DataFrame,
    batch_winner: dict[str, Any],
    gate_evaluation: pd.DataFrame,
) -> tuple[str, str]:
    """Determine final recommendation and rationale.

    NOTE: The p-values used here are from paired DeLong tests vs baseline,
    NOT from full selection-adjusted permutation-null inference. This means
    the comparison does not account for post-hoc model selection across
    the candidate pool. For true selection-adjusted inference, use the
    build_selection_adjusted_permutation_null() function.

    Returns:
        Tuple of (recommendation_code, rationale_text)

    Recommendation rules:
    - PROCEED: ANY candidate satisfies:
        - Meaningful gain (>= 0.025 AUC) with all gates pass, OR
        - Marginal gain (0.015-0.025) with clean gate profile:
            - ECE <= 0.08
            - paired_delong_p <= 0.03 (note: NOT selection-adjusted)
            - discovery contract pass
            - leakage review status acceptable
    - CALIBRATION_WORK: No candidate passes all gates, but at least one has:
        - ECE > 0.10 but other metrics good
        - Can be improved with calibration work
    - STOP: No candidate shows meaningful progress, or serious issues
    """
    # Check passing candidates
    passing_candidates = gate_evaluation.loc[gate_evaluation["overall"]]["model_name"].tolist()

    if not passing_candidates:
        # Check if any candidate is close but has ECE issues
        ece_issues = per_model_summary.loc[per_model_summary["ece"] > 0.10]["model_name"].tolist()

        if ece_issues:
            return (
                "CALIBRATION_WORK",
                f"No candidates pass all gates. Models with ECE > 0.10: {', '.join(ece_issues)}. "
                "Calibration work may improve ECE.",
            )
        return (
            "STOP",
            "No candidates pass acceptance gates. Discovery track shows no viable candidates.",
        )

    # Check for meaningful gain among passing candidates
    meaningful_passing = per_model_summary.loc[
        (per_model_summary["model_name"].isin(passing_candidates))
        & (per_model_summary["gain_class"] == "MEANINGFUL")
    ]

    if not meaningful_passing.empty:
        winner = meaningful_passing.loc[meaningful_passing["raw_auc"].idxmax()]
        return (
            "PROCEED",
            f"Candidate '{winner['model_name']}' shows MEANINGFUL gain ({winner['delta_vs_baseline']:.4f} AUC) "
            f"with clean gate profile. Recommended for promotion.",
        )

    # Check for marginal gain with clean profile
    marginal_passing = per_model_summary.loc[
        (per_model_summary["model_name"].isin(passing_candidates))
        & (per_model_summary["gain_class"] == "MARGINAL")
    ]

    if not marginal_passing.empty:
        # Check for clean profile (ECE <= 0.08, paired_delong_p <= 0.03)
        clean_marginal = marginal_passing.loc[
            (marginal_passing["ece"] <= 0.08) & (marginal_passing["paired_delong_p"] <= 0.03)
        ]

        if not clean_marginal.empty:
            winner = clean_marginal.loc[clean_marginal["raw_auc"].idxmax()]
            return (
                "PROCEED",
                f"Candidate '{winner['model_name']}' shows MARGINAL gain ({winner['delta_vs_baseline']:.4f} AUC) "
                f"with very clean gate profile (ECE {winner['ece']:.4f}, paired_delong_p={winner['paired_delong_p']:.4f}). "
                f"Recommended for promotion.",
            )

    # Default: if passing but no meaningful/marginal gain
    return (
        "CALIBRATION_WORK",
        f"Candidates pass gates but show limited gain: {', '.join(passing_candidates)}. "
        "Consider feature engineering or hyperparameter tuning.",
    )


def build_recommendation_report(
    artifacts: dict[str, Any],
) -> str:
    """Build the human-readable Markdown recommendation report."""
    metrics = artifacts["metrics"]
    per_model = artifacts["per_model_summary"]
    batch_winner = artifacts["batch_winner"]
    gate_eval = artifacts["gate_evaluation"]

    recommendation, rationale = determine_recommendation(per_model, batch_winner, gate_eval)

    lines = [
        "# Phase 5.2 Discovery Batch: Execution Summary and Recommendation",
        "",
        "## Overview",
        "",
        "**Execution Date**: Auto-generated from batch artifacts  ",
        "**Batch**: Phase 5.2 Discovery  ",
        "**Models Evaluated**: 4 (1 baseline + 3 discovery candidates)  ",
        "**Baseline**: bio_clean_priority  ",
        "**Discovery Candidates**: discovery_9f_source, discovery_12f_source, discovery_12f_class_balanced",
        "",
        "---",
        "",
        "## Per-Candidate vs Baseline Results",
        "",
    ]

    # Baseline info
    baseline = metrics.get("baseline", {})
    lines.extend(
        [
            "### Baseline Reference",
            "",
            f"- **Model**: {baseline.get('model_name', 'bio_clean_priority')}",
            f"- **Raw AUC**: {baseline.get('raw_auc', 'N/A'):.4f}",
            f"- **95% CI**: {baseline.get('auc_ci', 'N/A')}",
            f"- **ECE**: {baseline.get('ece', 'N/A'):.4f}",
            "",
            "### Discovery Candidates",
            "",
        ]
    )

    # Candidate table
    lines.append(
        "| Model | Raw AUC | AUC CI | ECE | paired_delong_p | Δ vs Baseline | Gain Class | Gates Pass |"
    )
    lines.append(
        "|-------|---------|--------|-----|-----------------|---------------|------------|------------|"
    )

    for _, row in per_model.iterrows():
        gates_pass = "✓" if row["gate_overall"] else "✗"
        lines.append(
            f"| {row['model_name']} | {row['raw_auc']:.4f} | {row['auc_ci']} | "
            f"{row['ece']:.4f} | {row['paired_delong_p']:.4f} | "
            f"{row['delta_vs_baseline']:+.4f} | {row['gain_class']} | {gates_pass} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Honest Batch Winner Selection",
            "",
            "**Selection Pool**: Discovery candidates only (excluding baseline)",
            "",
            f"- **Selected Config** (highest raw AUC): **{batch_winner['selected_config_name']}**",
            f"  - Raw AUC: {batch_winner['selected_config_raw_auc']:.4f}",
            "",
            f"- **Top-k Configs Considered**: {', '.join(batch_winner['top_k_config_names'])}",
            "",
            f"- **Reported Selection-Adjusted AUC** (top-3 mean): {batch_winner['reported_selection_adjusted_auc']:.4f}",
            f"- **Conservative CI Envelope**: [{batch_winner['reported_ci'][0]:.3f}, {batch_winner['reported_ci'][1]:.3f}]",
            "",
        ]
    )

    # Winner after gates
    winner_after_gates = batch_winner.get("recommended_winner_after_gates")
    if winner_after_gates:
        lines.append(f"- **Recommended Winner After Gates**: **{winner_after_gates}**  ")
    else:
        lines.append("- **Recommended Winner After Gates**: None (no candidate passes all gates)  ")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Gate Evaluation Summary",
            "",
            "| Model | ECE < 0.10 | paired_delong_p < 0.05 | Discovery Contract | Leakage Review | Overall |",
            "|-------|------------|------------------------|-------------------|----------------|---------|",
        ]
    )

    for _, row in gate_eval.iterrows():
        ece_pass = "✓" if row["ece_pass"] else "✗"
        p_pass = "✓" if row["paired_delong_p_pass"] else "✗"
        contract_pass = "✓" if row["discovery_contract_pass"] else "✗"
        leakage = row["leakage_review_status"]
        overall = "✓ PASS" if row["overall"] else "✗ FAIL"

        lines.append(
            f"| {row['model_name']} | {ece_pass} | {p_pass} | {contract_pass} | {leakage} | {overall} |"
        )

    lines.extend(
        [
            "",
            "**Gate Thresholds**:",
            "- ECE < 0.10 (Expected Calibration Error)",
            "- paired_delong_p < 0.05 (paired DeLong vs baseline; NOT selection-adjusted)",
            "- Discovery contract: must pass (temporal safety, assignment mode)",
            "- Leakage review: 'required' = no explicit signal in repo; discovery contract covers key checks",
            "",
            "**Important Notes**:",
            "- The paired_delong_p values are from paired DeLong tests comparing each candidate to baseline.",
            "- This is NOT equivalent to full selection-adjusted permutation-null inference.",
            "- For true selection-adjusted inference accounting for post-hoc model selection, use:",
            "  `build_selection_adjusted_permutation_null()` from `reporting.model_audit`",
            "",
            "---",
            "",
            "## Final Recommendation",
            "",
            f"### **{recommendation}**",
            "",
            f"**Rationale**: {rationale}",
            "",
        ]
    )

    if recommendation == "PROCEED":
        lines.extend(
            [
                "### Next Steps",
                "",
                "1. Document selected model configuration",
                "2. Update production pipeline to use recommended winner",
                "3. Schedule Phase 6 governance experiments if applicable",
                "",
            ]
        )
    elif recommendation == "CALIBRATION_WORK":
        lines.extend(
            [
                "### Next Steps",
                "",
                "1. Investigate calibration issues (ECE, reliability)",
                "2. Consider isotonic regression or Platt scaling",
                "3. Re-run batch after calibration improvements",
                "",
            ]
        )
    else:  # STOP
        lines.extend(
            [
                "### Next Steps",
                "",
                "1. Return to feature engineering",
                "2. Consider alternative model architectures",
                "3. Re-evaluate target variable or labeling strategy",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Artifact Locations",
            "",
            "- **Metrics JSON**: `data/analysis/phase_52_metrics.json`",
            "- **Predictions TSV**: `data/analysis/phase_52_predictions.tsv`",
            "- **Per-Model Summary**: `reports/phase_52_per_model_summary.tsv`",
            "- **Batch Winner**: `reports/phase_52_batch_winner.json`",
            "- **Gate Evaluation**: `reports/phase_52_gate_evaluation.tsv`",
            "- **This Report**: `reports/phase_52_recommendation.md`",
            "",
            "---",
            "",
            "*Report generated by `scripts/build_phase_52_report.py`*",
        ]
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase 5.2 Discovery Batch human-readable recommendation report."
    )
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=None,
        help=(
            "Input data directory containing Phase 5.2 analysis artifacts "
            "(default: data from config). Use to read from external location."
        ),
    )
    parser.add_argument(
        "--input-reports-dir",
        type=Path,
        default=None,
        help=(
            "Input reports directory containing Phase 5.2 TSV artifacts "
            "(default: reports from project root). Use to read from external location."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output report path (default: reports/phase_52_recommendation.md).",
    )
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    data_dir = args.input_data_dir or context.data_dir
    reports_dir = args.input_reports_dir or (PROJECT_ROOT / "reports")
    output_path = args.output_path or (reports_dir / "phase_52_recommendation.md")
    manifest_path = data_dir / "analysis/build_phase_52_report.manifest.json"

    # EXPLICIT WARNING: Check for required input artifacts from run_phase_52_discovery.py
    required_artifacts = [
        data_dir / "analysis/phase_52_metrics.json",
        reports_dir / "phase_52_per_model_summary.tsv",
        reports_dir / "phase_52_batch_winner.json",
        reports_dir / "phase_52_gate_evaluation.tsv",
    ]
    missing_artifacts = [p for p in required_artifacts if not p.exists()]
    if missing_artifacts:
        print(
            "ERROR: Required Phase 5.2 artifacts not found:\n"
            + "\n".join(f"  - {p}" for p in missing_artifacts)
            + "\n\nReport generation CANNOT proceed without batch execution results.\n"
            "To generate these artifacts, first run:\n"
            "  python scripts/run_phase_52_discovery.py --jobs 4\n"
            "\nThen re-run this report script.\n",
            file=sys.stderr,
        )
        return 1

    ensure_directory(output_path.parent)

    # Input artifacts for cache checking
    input_artifacts = [
        data_dir / "analysis/phase_52_metrics.json",
        reports_dir / "phase_52_per_model_summary.tsv",
        reports_dir / "phase_52_batch_winner.json",
        reports_dir / "phase_52_gate_evaluation.tsv",
    ]

    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/build_phase_52_report.py",
    )

    with ManagedScriptRun(context, "build_phase_52_report") as run:
        # Record inputs
        for artifact_path in input_artifacts:
            if artifact_path.exists():
                run.record_input(artifact_path)

        # Check cache
        if load_signature_manifest(
            manifest_path,
            input_paths=[p for p in input_artifacts if p.exists()],
            source_paths=source_paths,
            metadata={},
        ):
            run.note("Inputs unchanged; reusing cached report.")
            run.set_metric("cache_hit", True)
            return 0

        # Load artifacts
        run.note("Loading Phase 5.2 artifacts...")
        artifacts = load_phase_52_artifacts(data_dir, reports_dir)

        # Build report
        run.note("Building recommendation report...")
        report_content = build_recommendation_report(artifacts)

        # Write report
        output_path.write_text(report_content, encoding="utf-8")
        run.record_output(output_path)

        # Write manifest
        write_signature_manifest(
            manifest_path,
            input_paths=[p for p in input_artifacts if p.exists()],
            output_paths=[output_path],
            source_paths=source_paths,
            metadata={},
        )

        # Summary
        recommendation = determine_recommendation(
            artifacts["per_model_summary"],
            artifacts["batch_winner"],
            artifacts["gate_evaluation"],
        )[0]
        run.set_metric("recommendation", recommendation)
        run.set_metric("cache_hit", False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
