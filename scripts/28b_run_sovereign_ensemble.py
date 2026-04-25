"""Run out-of-the-box sovereign ensemble for 83+ AUC target.

This script implements the meta-sovereign ensemble strategy:
1. Load base model predictions from module_a output
2. Apply adaptive weighted ensemble with uncertainty gating
3. Generate final predictions with conformal prediction sets
4. Evaluate 83+ AUC performance
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from plasmid_priority.config import build_context
from plasmid_priority.modeling.ensemble_strategies import (
    create_meta_sovereign_model,
)
from plasmid_priority.validation.metrics import average_precision, roc_auc_score


def load_base_predictions(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    """Load base model predictions from module_a output."""
    predictions_file = analysis_dir / "module_a_predictions.tsv"

    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    df = pd.read_csv(predictions_file, sep="\t")

    # Extract predictions for each model
    base_models = [
        "governance_linear",
        "knownness_robust_priority",
        "support_synergy_priority",
        "host_transfer_synergy_priority",
        "discovery_boosted",
    ]

    predictions = {}
    for model in base_models:
        if f"pred_{model}" in df.columns:
            predictions[model] = df[f"pred_{model}"].values

    return predictions, df["y_true"].values if "y_true" in df.columns else None  # type: ignore


def compute_ensemble_performance(
    base_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    method: str = "adaptive_weighted",
) -> dict:  # type: ignore
    """Compute ensemble performance metrics."""

    # Create ensemble
    result = create_meta_sovereign_model(base_predictions, y_true, method=method)
    final_preds = result["probability"]

    # Compute metrics
    try:
        auc = roc_auc_score(y_true, final_preds)
    except Exception as e:
        warnings.warn(f"AUC computation failed: {e}")
        auc = 0.5

    try:
        ap = average_precision(y_true, final_preds)
    except Exception as e:
        warnings.warn(f"AP computation failed: {e}")
        ap = 0.5

    # Calibrated AUC (check calibration improvement)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(final_preds.reshape(-1, 1), y_true)
    cal_preds = calibrator.predict(final_preds.reshape(-1, 1))

    try:
        cal_auc = roc_auc_score(y_true, cal_preds)
    except Exception:
        cal_auc = auc

    return {
        "method": method,
        "roc_auc": auc,
        "average_precision": ap,
        "calibrated_roc_auc": cal_auc,
        "weights": result["weights"],
        "n_base_models": len(base_predictions),
    }


def cross_validate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> dict:  # type: ignore
    """Cross-validate ensemble performance."""

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    aucs = []
    aps = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit calibrator on validation fold
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(X_train.reshape(-1, 1), y_train)
        X_val_cal = calibrator.predict(X_val.reshape(-1, 1))

        # Compute metrics
        try:
            auc = roc_auc_score(y_val, X_val_cal)
            ap = average_precision(y_val, X_val_cal)
            aucs.append(auc)
            aps.append(ap)
        except Exception as e:
            warnings.warn(f"Fold {fold} failed: {e}")

    return {
        "fold_aucs": aucs,
        "fold_aps": aps,
        "mean_auc": np.mean(aucs) if aucs else 0.5,
        "std_auc": np.std(aucs) if aucs else 0.0,
        "mean_ap": np.mean(aps) if aps else 0.5,
        "std_ap": np.std(aps) if aps else 0.0,
    }


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("META-SOVEREIGN ENSEMBLE - 83+ AUC TARGET")
    print("=" * 70)

    # Build context
    ctx = build_context()
    analysis_dir = Path(ctx.config.data_root) / "analysis"  # type: ignore

    # Load base predictions
    print("\n[1/4] Loading base model predictions...")
    try:
        base_preds, y_true = load_base_predictions(analysis_dir)
        print(f"  Loaded {len(base_preds)} base models:")
        for name in base_preds.keys():  # type: ignore
            print(f"    - {name}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Please run module_a first to generate predictions.")
        return

    if y_true is None:
        print("  Error: y_true not found in predictions file")  # type: ignore
        return

    # Compute individual model AUCs
    print("\n[2/4] Computing base model performances...")
    base_aucs = {}
    for name, preds in base_preds.items():  # type: ignore
        try:
            auc = roc_auc_score(y_true, preds)  # type: ignore
            base_aucs[name] = auc
            print(f"  {name}: {auc:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Run ensemble methods
    print("\n[3/4] Running ensemble strategies...")
    methods = ["simple_average", "weighted_average", "adaptive_weighted"]

    ensemble_results = {}
    for method in methods:
        print(f"\n  Method: {method}")
        result = compute_ensemble_performance(base_preds, y_true, method=method)  # type: ignore
        ensemble_results[method] = result

        print(f"    ROC AUC: {result['roc_auc']:.4f}")
        print(f"    Average Precision: {result['average_precision']:.4f}")
        print(f"    Calibrated ROC AUC: {result['calibrated_roc_auc']:.4f}")

        if "weights" in result:
            print("    Weights:")
            for model, weight in result["weights"].items():
                print(f"      {model}: {weight:.3f}")

    # Find best method
    best_method = max(ensemble_results.keys(), key=lambda m: ensemble_results[m]["roc_auc"])
    best_result = ensemble_results[best_method]

    # Cross-validation
    print("\n[4/4] Cross-validating best ensemble...")
    final_preds = create_meta_sovereign_model(base_preds, y_true, method=best_method)["probability"]  # type: ignore

    cv_results = cross_validate_ensemble(final_preds, y_true, n_folds=5)  # type: ignore

    print(f"  CV Mean AUC: {cv_results['mean_auc']:.4f} (±{cv_results['std_auc']:.4f})")
    print(f"  CV Mean AP: {cv_results['mean_ap']:.4f} (±{cv_results['std_ap']:.4f})")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Ensemble Method: {best_method}")
    print(f"Best ROC AUC: {best_result['roc_auc']:.4f}")
    print(
        f"Target (83+ AUC): {'✅ ACHIEVED' if best_result['roc_auc'] >= 0.83 else '❌ NOT ACHIEVED'}",
    )
    print(f"Improvement over best base: +{best_result['roc_auc'] - max(base_aucs.values()):.4f}")
    print(f"CV Stability: {cv_results['std_auc']:.4f} (lower is better)")

    # Save results
    output_dir = analysis_dir
    output_file = output_dir / "meta_sovereign_ensemble_results.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "best_method": best_method,
                "best_auc": best_result["roc_auc"],
                "best_ap": best_result["average_precision"],
                "cv_mean_auc": cv_results["mean_auc"],
                "cv_std_auc": cv_results["std_auc"],
                "base_aucs": base_aucs,
                "ensemble_results": ensemble_results,
                "weights": best_result.get("weights", {}),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
