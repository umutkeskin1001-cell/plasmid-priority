"""Rolling-origin validation utilities for temporal robustness checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ModelEvaluator = Callable[[pd.DataFrame, str], dict[str, object]]


def _default_model_evaluator(_scored: pd.DataFrame, _model_name: str) -> dict[str, object]:
    raise RuntimeError(
        "No model evaluator provided. Pass `model_evaluator=` from the caller "
        "(for example via plasmid_priority.modeling.evaluate_model_name wrapper)."
    )


@dataclass(frozen=True)
class RollingOriginSplitResult:
    split_year: int
    test_year_end: int
    horizon_years: int
    assignment_mode: str
    model_name: str
    n_backbones: int
    n_eligible_backbones: int
    status: str
    roc_auc: float | None
    average_precision: float | None
    ece: float | None


@dataclass(frozen=True)
class RollingOriginValidationReport:
    model_name: str
    split_results: tuple[RollingOriginSplitResult, ...]
    mean_auc: float
    mean_average_precision: float
    auc_stability_metric: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split_results": [asdict(result) for result in self.split_results],
            "mean_auc": float(self.mean_auc),
            "mean_average_precision": float(self.mean_average_precision),
            "auc_stability_metric": float(self.auc_stability_metric),
        }


def _window_mask(
    scored: pd.DataFrame,
    *,
    split_year: int,
    test_year_end: int,
    assignment_mode: str,
) -> pd.Series:
    return (
        (pd.to_numeric(scored["split_year"], errors="coerce") == int(split_year))
        & (pd.to_numeric(scored["test_year_end"], errors="coerce") == int(test_year_end))
        & (scored["backbone_assignment_mode"].astype(str) == str(assignment_mode))
    )


def run_rolling_origin_validation(
    scored: pd.DataFrame,
    *,
    model_name: str,
    split_years: range | list[int] | tuple[int, ...] = range(2012, 2019),
    horizon_years: int = 5,
    assignment_mode: str = "training_only",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    model_evaluator: ModelEvaluator | None = None,
) -> RollingOriginValidationReport:
    def _optional_float(metrics: dict[str, object], key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(numeric) if pd.notna(numeric) else None

    evaluator = model_evaluator or _default_model_evaluator
    results: list[RollingOriginSplitResult] = []
    for split_year in split_years:
        test_year_end = int(split_year) + int(horizon_years)
        window = scored.loc[
            _window_mask(
                scored,
                split_year=int(split_year),
                test_year_end=test_year_end,
                assignment_mode=assignment_mode,
            )
        ].copy()
        eligible = window.loc[window["spread_label"].notna()].copy()
        if window.empty or len(eligible) < 20 or eligible["spread_label"].nunique() < 2:
            results.append(
                RollingOriginSplitResult(
                    split_year=int(split_year),
                    test_year_end=test_year_end,
                    horizon_years=int(horizon_years),
                    assignment_mode=assignment_mode,
                    model_name=model_name,
                    n_backbones=int(len(window)),
                    n_eligible_backbones=int(len(eligible)),
                    status="skipped_insufficient_label_variation",
                    roc_auc=None,
                    average_precision=None,
                    ece=None,
                )
            )
            continue

        try:
            metrics = evaluator(window, model_name)
        except RuntimeError:
            metrics = {"status": "skipped_missing_model_evaluator"}
        results.append(
            RollingOriginSplitResult(
                split_year=int(split_year),
                test_year_end=test_year_end,
                horizon_years=int(horizon_years),
                assignment_mode=assignment_mode,
                model_name=model_name,
                n_backbones=int(len(window)),
                n_eligible_backbones=int(len(eligible)),
                status=str(metrics.get("status", "ok")),
                roc_auc=_optional_float(metrics, "roc_auc"),
                average_precision=_optional_float(metrics, "average_precision"),
                ece=_optional_float(metrics, "ece"),
            )
        )

    successful_auc = np.asarray(
        [result.roc_auc for result in results if result.roc_auc is not None], dtype=float
    )
    successful_ap = np.asarray(
        [result.average_precision for result in results if result.average_precision is not None],
        dtype=float,
    )
    mean_auc = float(np.mean(successful_auc)) if successful_auc.size else float("nan")
    mean_ap = float(np.mean(successful_ap)) if successful_ap.size else float("nan")
    stability = (
        float(np.std(successful_auc) / np.mean(successful_auc))
        if successful_auc.size >= 2 and float(np.mean(successful_auc)) != 0.0
        else float("nan")
    )
    return RollingOriginValidationReport(
        model_name=model_name,
        split_results=tuple(results),
        mean_auc=mean_auc,
        mean_average_precision=mean_ap,
        auc_stability_metric=stability,
    )
