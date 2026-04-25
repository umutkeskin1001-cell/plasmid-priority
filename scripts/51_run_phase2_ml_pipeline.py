"""Phase 2 ML Pipeline Runner: ML Devrimi.

Orchestrates the four Phase 2 components:
1. FT-Transformer deep tabular learning
2. Multi-task learning (3 branches)
3. Deep Ensemble + Conformal Prediction
4. Nested CV + Bayesian model comparison

Usage:
    uv run python scripts/51_run_phase2_ml_pipeline.py \
        --features data/phase1/features.parquet \
        --labels data/phase1/probabilistic_labels.tsv \
        --output data/phase2/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasmid_priority.modeling.conformal import SplitConformalPredictor
from plasmid_priority.modeling.deep_ensemble import DeepEnsemble
from plasmid_priority.modeling.ft_transformer import FTTransformerClassifier
from plasmid_priority.modeling.multi_task import MultiTaskTrainer
from plasmid_priority.modeling.nested_cv_bayesian import (
    NestedCVEvaluator,
    bayesian_model_comparison,
)

_log = logging.getLogger("phase2")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def run_ft_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    _log.info("=== FT-Transformer ===")
    try:
        import torch  # noqa: F401

        model = FTTransformerClassifier(
            d_model=64,
            n_heads=4,
            n_blocks=2,
            num_epochs=100,
            batch_size=128,
            patience=10,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_test, proba)
        _log.info("FT-Transformer AUC: %.4f", auc)
        return {"status": "ok", "auc": float(auc)}
    except ImportError:
        _log.warning("FT-Transformer skipped (torch not installed)")
        return {"status": "skipped", "reason": "torch_missing"}


def run_multi_task(
    X_train: np.ndarray,
    y_dict_train: dict[str, np.ndarray],
    X_test: np.ndarray,
) -> dict[str, object]:
    _log.info("=== Multi-Task Learning ===")
    try:
        import torch  # noqa: F401

        trainer = MultiTaskTrainer(
            input_dim=X_train.shape[1],
            hidden_dim=128,
            num_epochs=50,
            batch_size=64,
            patience=10,
        )
        trainer.fit(X_train, y_dict_train)
        preds = trainer.predict_proba(X_test)
        return {
            "status": "ok",
            "tasks": {t: {"mean_proba": float(p.mean())} for t, p in preds.items()},
        }
    except ImportError:
        _log.warning("Multi-task skipped (torch not installed)")
        return {"status": "skipped", "reason": "torch_missing"}


def run_deep_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    _log.info("=== Deep Ensemble ===")
    ens = DeepEnsemble(
        base_factory=lambda: HistGradientBoostingClassifier(max_iter=200),
        n_members=5,
        bootstrap=True,
        random_state=42,
    )
    ens.fit(X_train, y_train)
    mean_p, unc = ens.predict_proba(X_test)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_test, mean_p)
    _log.info("DeepEnsemble AUC: %.4f, mean_unc: %.4f", auc, unc.mean())
    return {
        "status": "ok",
        "auc": float(auc),
        "mean_uncertainty": float(unc.mean()),
    }


def run_conformal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    _log.info("=== Conformal Prediction ===")
    base = LogisticRegression(max_iter=1000)
    base.fit(X_train, y_train)
    cp = SplitConformalPredictor(base, alpha=0.10, random_state=42)
    cp.fit(X_train, y_train)
    cov = cp.empirical_coverage(X_test, y_test)
    _log.info("Conformal empirical coverage: %.3f (target: %.3f)", cov, 1 - cp.alpha)
    return {
        "status": "ok",
        "empirical_coverage": float(cov),
        "target_coverage": 1 - cp.alpha,
    }


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    _log.info("=== Nested CV + Bayesian Comparison ===")
    ev = NestedCVEvaluator(outer_cv=5, inner_cv=3, random_state=42)
    lr_result = ev.evaluate(X, y, lambda: LogisticRegression(max_iter=1000))
    gb_result = ev.evaluate(X, y, lambda: HistGradientBoostingClassifier(max_iter=200))
    bayes = bayesian_model_comparison(
        {
            "logistic_regression": lr_result["fold_scores"],  # type: ignore
            "gradient_boosting": gb_result["fold_scores"],  # type: ignore
        }
    )
    _log.info(
        "Nested CV complete. LR=%.3f, GB=%.3f", lr_result["mean_score"], gb_result["mean_score"]
    )
    return {
        "status": "ok",
        "logistic_regression": lr_result,
        "gradient_boosting": gb_result,
        "bayesian_comparison": bayes,
    }


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Phase 2 ML Pipeline")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/phase2"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features = (
        pd.read_parquet(args.features)
        if str(args.features).endswith(".parquet")
        else pd.read_csv(args.features, sep="\t")
    )
    labels = pd.read_csv(args.labels, sep="\t")
    _log.info("Loaded %d features x %d dims", len(features), features.shape[1])

    # Prepare matrices
    numeric = features.select_dtypes(include="number").fillna(0)
    X = numeric.to_numpy(dtype=np.float32)
    y = labels["fused_spread_label"].fillna(0).to_numpy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build multi-task labels
    y_dict_train = {
        "geo_spread": y_train,
        "bio_transfer": y_train,
        "clinical_hazard": y_train,
    }

    results: dict[str, object] = {}
    results["ft_transformer"] = run_ft_transformer(X_train_s, y_train, X_test_s, y_test)
    results["multi_task"] = run_multi_task(X_train_s, y_dict_train, X_test_s)
    results["deep_ensemble"] = run_deep_ensemble(X_train, y_train, X_test, y_test)
    results["conformal"] = run_conformal(X_train, y_train, X_test, y_test)
    results["nested_cv"] = run_nested_cv(X, y)

    manifest_path = output_dir / "phase2_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _log.info("Phase 2 complete. Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
