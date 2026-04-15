#!/usr/bin/env python3
"""Evaluate sovereign model candidates and select the best balanced option."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plasmid_priority.config import build_context
from plasmid_priority.modeling import (
    MODULE_A_FEATURE_SETS,
    evaluate_model_name,
    fit_full_model_predictions,
)
from plasmid_priority.modeling import module_a_support as module_a_support_impl
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import (
    ensure_directory,
    load_signature_manifest,
    project_python_source_paths,
    write_signature_manifest,
)
from plasmid_priority.utils.parallel import limit_native_threads


def _resolve_training_input_path(data_root: Path) -> Path | None:
    candidates = (
        data_root / "scores/backbone_scored.tsv",
        data_root / "scores/backbone_scored.parquet",
        data_root / "features/module_a_features.tsv",
        data_root / "silver/module_a_scored_train.tsv",
        data_root / "module_a_features.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _best_existing_model_auc(
    analysis_dir: Path, *, exclude_model_name: str | None = None
) -> tuple[str | None, float]:
    metrics_path = analysis_dir / "module_a_metrics.json"
    if not metrics_path.exists():
        return None, float("nan")
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None, float("nan")

    best_model_name: str | None = None
    best_auc = float("nan")
    for model_name, metrics in payload.items():
        if model_name == exclude_model_name or not isinstance(metrics, dict):
            continue
        if str(metrics.get("status", "ok") or "ok") != "ok":
            continue
        auc_value = pd.to_numeric(pd.Series([metrics.get("roc_auc")]), errors="coerce").iloc[0]
        if pd.isna(auc_value):
            continue
        auc = float(auc_value)
        if best_model_name is None or auc > best_auc:
            best_model_name = str(model_name)
            best_auc = auc
    return best_model_name, best_auc


def _build_candidate_scorecard(results: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, result in results.items():
        metrics = result.metrics
        fit_kwargs = module_a_support_impl._model_fit_kwargs(model_name)
        rows.append(
            {
                "model_name": model_name,
                "sample_weight_mode": fit_kwargs.get("sample_weight_mode"),
                "l2": float(fit_kwargs.get("l2", float("nan"))),
                "preprocess_mode": fit_kwargs.get("preprocess_mode"),
                "roc_auc": float(metrics.get("roc_auc", float("nan"))),
                "average_precision": float(metrics.get("average_precision", float("nan"))),
                "brier_score": float(metrics.get("brier_score", float("nan"))),
                "expected_calibration_error": float(
                    metrics.get("expected_calibration_error", float("nan"))
                ),
                "knownness_matched_gap": float(metrics.get("knownness_matched_gap", float("nan"))),
                "feature_count": len(MODULE_A_FEATURE_SETS[model_name]),
            }
        )
    return pd.DataFrame(rows)


def _candidate_model_names() -> list[str]:
    return [
        "sovereign_precision_priority",
        "sovereign_v2_priority",
    ]


def _cache_metadata(
    *, n_splits: int, n_repeats: int, seed: int, jobs: int, skip_full_fit: bool
) -> dict[str, object]:
    return {
        "candidate_models": _candidate_model_names(),
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "seed": int(seed),
        "jobs": int(jobs),
        "skip_full_fit": bool(skip_full_fit),
    }


def _evaluate_candidates(
    scored: pd.DataFrame,
    *,
    candidate_names: list[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    jobs: int,
) -> dict[str, object]:
    def _evaluate(name: str) -> tuple[str, object]:
        return (
            name,
            evaluate_model_name(
                scored,
                model_name=name,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                include_ci=False,
            ),
        )

    if jobs > 1 and len(candidate_names) > 1:
        with limit_native_threads(1):
            with ThreadPoolExecutor(max_workers=min(jobs, len(candidate_names))) as executor:
                return dict(executor.map(_evaluate, candidate_names))
    return {name: _evaluate(name)[1] for name in candidate_names}


def _choose_best_candidate(scorecard: pd.DataFrame) -> dict[str, object]:
    ranked = scorecard.copy()
    ranked = ranked.sort_values(
        [
            "roc_auc",
            "average_precision",
            "brier_score",
            "expected_calibration_error",
            "knownness_matched_gap",
            "feature_count",
        ],
        ascending=[False, False, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    leader = ranked.iloc[0]
    challenger = ranked.iloc[1] if len(ranked) > 1 else None
    if (
        challenger is not None
        and abs(float(leader["roc_auc"]) - float(challenger["roc_auc"])) <= 0.001
    ):
        if (
            float(challenger["brier_score"]) <= float(leader["brier_score"])
            and float(challenger["expected_calibration_error"])
            <= float(leader["expected_calibration_error"])
            and float(challenger["feature_count"]) <= float(leader["feature_count"])
        ):
            leader = challenger
    return leader.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate sovereign candidates with cache-aware native Module A execution."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Cross-validation split count for candidate evaluation.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repeated CV count for candidate evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate evaluation.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Parallel workers for sovereign candidate evaluation.",
    )
    parser.add_argument(
        "--skip-full-fit",
        action="store_true",
        help="Skip full-fit prediction generation after selecting the winner.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached sovereign outputs and recompute everything.",
    )
    args = parser.parse_args(argv)

    print("=" * 80)
    print("SOVEREIGN CANDIDATE EVALUATION")
    print("=" * 80)

    ctx = build_context(PROJECT_ROOT)
    data_root = ctx.data_dir
    analysis_dir = data_root / "analysis"
    manifest_path = analysis_dir / "29_train_sovereign.manifest.json"
    config_path = ctx.root / "config.yaml"
    source_paths = project_python_source_paths(
        PROJECT_ROOT,
        script_path=PROJECT_ROOT / "scripts/29_train_sovereign.py",
    )

    print("\n[1/5] Context loaded")
    print(f"  Data root: {data_root}")

    features_path = _resolve_training_input_path(data_root)
    if features_path is None:
        print("\n❌ Features file not found!")
        return 1

    ensure_directory(analysis_dir)
    cache_metadata = _cache_metadata(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        jobs=max(int(args.jobs), 1),
        skip_full_fit=args.skip_full_fit,
    )

    with ManagedScriptRun(ctx, "29_train_sovereign") as run:
        run.record_input(features_path)
        run.record_input(config_path)
        metrics_path = analysis_dir / "module_a_metrics.json"
        if metrics_path.exists():
            run.record_input(metrics_path)

        if not args.force and load_signature_manifest(
            manifest_path,
            input_paths=[
                path for path in [features_path, config_path, metrics_path] if path.exists()
            ],
            source_paths=source_paths,
            metadata=cache_metadata,
        ):
            run.note("Inputs, code, and config unchanged; reusing cached sovereign outputs.")
            run.set_metric("cache_hit", True)
            print("\n[cache] Reusing cached sovereign outputs")
            return 0

        print(f"\n[2/5] Loading data: {features_path}")
        df = _load_frame(features_path)
        if "spread_label" not in df.columns:
            print("❌ 'spread_label' column not found!")
            return 1

        valid = df.dropna(subset=["spread_label"]).copy()
        print(f"  Rows: {len(df):,}")
        print(f"  Valid rows: {len(valid):,}")
        print(f"  Positive rate: {valid['spread_label'].mean():.3f}")
        run.set_rows_in("scored_rows", int(len(df)))
        run.set_rows_in("eligible_rows", int(len(valid)))

        candidate_names = _candidate_model_names()
        missing = [name for name in candidate_names if name not in MODULE_A_FEATURE_SETS]
        if missing:
            print(f"❌ Missing candidate definitions: {missing}")
            return 1

        print("\n[3/5] Evaluating sovereign candidates")
        jobs = max(int(args.jobs), 1)
        results = _evaluate_candidates(
            valid,
            candidate_names=candidate_names,
            n_splits=int(args.n_splits),
            n_repeats=int(args.n_repeats),
            seed=int(args.seed),
            jobs=jobs,
        )
        scorecard = _build_candidate_scorecard(results)
        print(scorecard.to_string(index=False))

        print("\n[4/5] Selecting best balanced candidate")
        winner = _choose_best_candidate(scorecard)
        winner_name = str(winner["model_name"])
        print(f"  Selected: {winner_name}")

        output_dir = data_root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        scorecard_path = output_dir / "sovereign_candidate_scorecard.tsv"
        scorecard.to_csv(scorecard_path, sep="\t", index=False)
        run.record_output(scorecard_path)

        output_file = output_dir / f"{winner_name}_trained.json"
        pred_file = output_dir / f"{winner_name}_predictions.tsv"

        best_existing_model, best_existing_auc = _best_existing_model_auc(
            analysis_dir,
            exclude_model_name=winner_name,
        )

        payload = {
            "selected_model_name": winner_name,
            "candidate_scorecard_path": str(scorecard_path),
            "sample_weight_mode": winner.get("sample_weight_mode"),
            "l2": float(winner["l2"]),
            "preprocess_mode": winner.get("preprocess_mode"),
            "roc_auc": float(winner["roc_auc"]),
            "average_precision": float(winner["average_precision"]),
            "brier_score": float(winner["brier_score"]),
            "expected_calibration_error": float(winner["expected_calibration_error"]),
            "knownness_matched_gap": float(winner["knownness_matched_gap"]),
            "n_features_used": int(winner["feature_count"]),
            "n_samples": int(len(valid)),
            "positive_rate": float(valid["spread_label"].mean()),
            "training_input_path": str(features_path),
            "best_existing_model_name": best_existing_model,
            "best_existing_roc_auc": float(best_existing_auc)
            if pd.notna(best_existing_auc)
            else None,
            "improvement_over_best": (
                float(winner["roc_auc"]) - float(best_existing_auc)
                if pd.notna(best_existing_auc)
                else None
            ),
            "n_splits": int(args.n_splits),
            "n_repeats": int(args.n_repeats),
            "seed": int(args.seed),
            "jobs": int(jobs),
            "cache_hit": False,
            "full_fit_predictions_generated": not bool(args.skip_full_fit),
        }

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        run.record_output(output_file)

        if args.skip_full_fit:
            run.note("Skipped full-fit prediction generation by request.")
        else:
            winner_predictions = fit_full_model_predictions(
                df,
                model_name=winner_name,
                include_posterior_uncertainty=False,
            )
            winner_predictions.to_csv(pred_file, sep="\t", index=False)
            run.record_output(pred_file)
            run.set_rows_out("winner_prediction_rows", int(len(winner_predictions)))

        write_signature_manifest(
            manifest_path,
            input_paths=[
                path for path in [features_path, config_path, metrics_path] if path.exists()
            ],
            output_paths=[
                path for path in [scorecard_path, output_file, pred_file] if path.exists()
            ],
            source_paths=source_paths,
            metadata=cache_metadata,
        )
        run.record_output(manifest_path)
        run.set_metric("cache_hit", False)
        run.set_metric("candidate_count", len(candidate_names))
        run.set_metric("jobs", jobs)
        run.set_metric("winner_roc_auc", float(winner["roc_auc"]))

        print("\n[5/5] Saved outputs")
        print(f"  Scorecard: {scorecard_path}")
        print(f"  Summary:   {output_file}")
        if args.skip_full_fit:
            print("  Predictions: skipped (--skip-full-fit)")
        else:
            print(f"  Predictions: {pred_file}")
        print("=" * 80)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
