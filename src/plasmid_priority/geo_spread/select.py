"""Model blending and selection helpers for the geo spread branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plasmid_priority.geo_spread.dataset import prepare_geo_spread_scored_table
from plasmid_priority.modeling.module_a import _evaluate_prediction_set
from plasmid_priority.modeling.module_a_support import ModelResult, build_failed_model_result
from plasmid_priority.validation.metrics import average_precision, roc_auc_score

GEO_SPREAD_RELIABILITY_BLEND = "geo_reliability_blend"
GEO_SPREAD_ADAPTIVE_PRIORITY = "geo_adaptive_knownness_priority"
GEO_SPREAD_META_PRIORITY = "geo_meta_knownness_priority"
GEO_SPREAD_BLEND_COMPONENTS: tuple[tuple[str, float], ...] = (
    ("geo_context_hybrid_priority", 0.70),
    ("geo_phylo_ecology_priority", 0.30),
)
GEO_SPREAD_RELIABILITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "T_eff_norm",
    "H_obs_specialization_norm",
    "A_eff_norm",
    "coherence_score",
    "backbone_purity_norm",
    "assignment_confidence_norm",
    "mash_neighbor_distance_train_norm",
    "orit_support",
    "H_external_host_range_norm",
    "geo_country_entropy_train",
    "geo_macro_region_entropy_train",
    "geo_dominant_region_share_train",
    "geo_country_record_count_train",
)
GEO_SPREAD_RELIABILITY_ESTIMATOR_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("hist_gradient_boosting", 0.55),
    ("logistic", 0.2025),
    ("random_forest", 0.2475),
)
GEO_SPREAD_ADAPTIVE_SUPPORT_MODEL = "geo_context_hybrid_priority"
GEO_SPREAD_ADAPTIVE_HIGH_KNOWNNESS_MODEL = GEO_SPREAD_RELIABILITY_BLEND
GEO_SPREAD_ADAPTIVE_KNOWNNESS_CENTER = 0.45
GEO_SPREAD_ADAPTIVE_KNOWNNESS_SHARPNESS = 20.0


def _prediction_frame(result: ModelResult) -> pd.DataFrame:
    frame = result.predictions.copy()
    if frame.empty:
        raise ValueError(f"Model `{result.name}` has no prediction rows.")
    score_column = "oof_prediction" if "oof_prediction" in frame.columns else "prediction"
    if score_column not in frame.columns:
        raise ValueError(f"Model `{result.name}` predictions do not contain a usable score column.")
    frame = frame.loc[
        :,
        [
            column
            for column in frame.columns
            if column in {"backbone_id", "spread_label", score_column}
        ],
    ].copy()
    frame = frame.rename(columns={score_column: str(result.name)})
    return frame


def build_geo_spread_blended_result(
    results: Mapping[str, ModelResult],
    *,
    scored: pd.DataFrame | None = None,
    include_ci: bool = True,
    blend_name: str = GEO_SPREAD_RELIABILITY_BLEND,
) -> ModelResult:
    """Blend the strongest reliability-oriented geo models into one derived model."""
    if scored is not None:
        feature_ensemble = _build_geo_spread_feature_ensemble_result(
            scored,
            include_ci=include_ci,
            blend_name=blend_name,
        )
        if feature_ensemble.status == "ok":
            return feature_ensemble

    component_frames: list[pd.DataFrame] = []
    component_weights: dict[str, float] = {}
    for model_name, weight in GEO_SPREAD_BLEND_COMPONENTS:
        result = results.get(model_name)
        if result is None or result.status != "ok":
            return build_failed_model_result(
                blend_name,
                f"Blend requires successful component model `{model_name}`.",
            )
        component_frames.append(_prediction_frame(result))
        component_weights[model_name] = float(weight)

    merged = component_frames[0]
    for frame in component_frames[1:]:
        merge_keys = ["backbone_id"]
        if "spread_label" in merged.columns and "spread_label" in frame.columns:
            merge_keys.append("spread_label")
        merged = merged.merge(frame, on=merge_keys, how="inner")
    if merged.empty:
        return build_failed_model_result(
            blend_name, "Blend components do not share a common evaluation cohort."
        )

    score = np.zeros(len(merged), dtype=float)
    for model_name, weight in component_weights.items():
        score += float(weight) * pd.to_numeric(merged[model_name], errors="coerce").fillna(
            0.0
        ).to_numpy(dtype=float)
    total_weight = max(sum(component_weights.values()), 1e-12)
    score = np.asarray(score / total_weight, dtype=float)
    y = (
        pd.to_numeric(merged["spread_label"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy(dtype=int)
    )
    detail = merged.loc[:, [*component_weights.keys()]].copy()
    detail["blend_prediction"] = score
    return _evaluate_prediction_set(
        blend_name,
        y,
        score,
        merged["backbone_id"].astype(str),
        include_ci=include_ci,
        extra_metrics={
            "blend_component_count": float(len(component_weights)),
            **{f"blend_weight_{name}": float(weight) for name, weight in component_weights.items()},
        },
        prediction_detail=detail,
    )


def _reliability_estimators() -> list[tuple[str, float, Any]]:
    estimators: list[tuple[str, float, Any]] = []
    for name, weight in GEO_SPREAD_RELIABILITY_ESTIMATOR_WEIGHTS:
        if name == "hist_gradient_boosting":
            estimators.append(
                (
                    name,
                    float(weight),
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.035,
                        max_leaf_nodes=10,
                        min_samples_leaf=18,
                        l2_regularization=0.4,
                        random_state=7,
                    ),
                ),
            )
        elif name == "logistic":
            estimators.append(
                (
                    name,
                    float(weight),
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.4, class_weight="balanced"),
                    ),
                ),
            )
        elif name == "random_forest":
            estimators.append(
                (
                    name,
                    float(weight),
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=15,
                        class_weight="balanced_subsample",
                        random_state=7,
                        n_jobs=1,
                    ),
                ),
            )
    return estimators


def _build_geo_spread_feature_ensemble_result(
    scored: pd.DataFrame,
    *,
    include_ci: bool,
    blend_name: str,
) -> ModelResult:
    prepared = prepare_geo_spread_scored_table(scored)
    required_columns = {
        "backbone_id",
        "spread_label",
        "knownness_score",
        *GEO_SPREAD_RELIABILITY_FEATURE_COLUMNS,
    }
    missing_columns = sorted(
        column for column in required_columns if column not in prepared.columns
    )
    if missing_columns:
        return build_failed_model_result(
            blend_name,
            (
                "Feature-ensemble blend requires columns: "
                f"{', '.join(missing_columns)}"
            ),
        )

    frame = prepared.loc[:, list(required_columns)].copy()
    labels = pd.to_numeric(frame["spread_label"], errors="coerce")
    valid = labels.notna()
    if not valid.any():
        return build_failed_model_result(
            blend_name,
            "No labeled rows available for reliability ensemble.",
        )
    frame = frame.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int).reset_index(drop=True)
    class_mask = labels.isin([0, 1]).to_numpy(dtype=bool)
    frame = frame.loc[class_mask].reset_index(drop=True)
    labels = labels.loc[class_mask].reset_index(drop=True)
    y = labels.to_numpy(dtype=int)
    if np.unique(y).size < 2:
        return build_failed_model_result(
            blend_name,
            "Reliability ensemble requires at least two classes.",
        )
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    n_splits = max(2, min(5, positive_count, negative_count))
    if n_splits < 2:
        return build_failed_model_result(
            blend_name,
            "Reliability ensemble requires at least two examples per class.",
        )

    X = (
        frame.loc[:, list(GEO_SPREAD_RELIABILITY_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    knownness = pd.to_numeric(frame["knownness_score"], errors="coerce").fillna(0.45).to_numpy(
        dtype=float
    )
    oof = np.zeros(len(y), dtype=float)
    component_predictions = {
        name: np.zeros(len(y), dtype=float) for name, _ in GEO_SPREAD_RELIABILITY_ESTIMATOR_WEIGHTS
    }
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    for train_idx, test_idx in splitter.split(X, y):
        fold_score = np.zeros(len(test_idx), dtype=float)
        for estimator_name, weight, estimator in _reliability_estimators():
            estimator.fit(X[train_idx], y[train_idx])
            prediction = estimator.predict_proba(X[test_idx])[:, 1]
            component_predictions[estimator_name][test_idx] = prediction
            fold_score += float(weight) * prediction
        oof[test_idx] = np.asarray(fold_score, dtype=float)
    oof = np.clip(oof, 0.0, 1.0)

    detail = pd.DataFrame(
        {
            "knownness_score": knownness,
            **component_predictions,
            "blend_prediction": oof,
        },
    )
    extra_metrics: dict[str, float | int | bool] = {
        "blend_component_count": float(len(GEO_SPREAD_RELIABILITY_ESTIMATOR_WEIGHTS)),
        "blend_feature_count": float(len(GEO_SPREAD_RELIABILITY_FEATURE_COLUMNS)),
        "blend_cv_splits": float(n_splits),
        "blend_is_feature_ensemble": True,
        **{
            f"blend_weight_{name}": float(weight)
            for name, weight in GEO_SPREAD_RELIABILITY_ESTIMATOR_WEIGHTS
        },
    }
    extra_metrics.update(_knownness_metrics(y, oof, knownness))
    return _evaluate_prediction_set(
        blend_name,
        y,
        oof,
        frame["backbone_id"].astype(str),
        include_ci=include_ci,
        knownness_score=knownness,
        extra_metrics=extra_metrics,
        prediction_detail=detail,
    )


def _sigmoid_gate(values: np.ndarray, *, center: float, sharpness: float) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(center)
    logits = np.clip(float(sharpness) * shifted, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(logits)), dtype=float)


def _knownness_metrics(y: np.ndarray, score: np.ndarray, knownness: np.ndarray) -> dict[str, float]:
    knownness = np.asarray(knownness, dtype=float)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    valid_mask = np.isfinite(knownness)
    if not np.any(valid_mask):
        return {}
    valid_knownness = knownness[valid_mask]
    threshold = float(np.nanquantile(valid_knownness, 0.5))
    low_mask = valid_mask & (knownness <= threshold)
    high_mask = valid_mask & (knownness > threshold)
    metrics: dict[str, float] = {
        "low_knownness_share": float(np.mean(low_mask.astype(float))),
        "high_knownness_share": float(np.mean(high_mask.astype(float))),
        "knownness_threshold": threshold,
    }
    if np.unique(y[low_mask]).size >= 2:
        metrics["low_knownness_roc_auc"] = float(roc_auc_score(y[low_mask], score[low_mask]))
        metrics["low_knownness_average_precision"] = float(
            average_precision(y[low_mask], score[low_mask])
        )
    if np.unique(y[high_mask]).size >= 2:
        metrics["high_knownness_roc_auc"] = float(roc_auc_score(y[high_mask], score[high_mask]))
        metrics["high_knownness_average_precision"] = float(
            average_precision(y[high_mask], score[high_mask])
        )
    if valid_knownness.size >= 4:
        quantiles = pd.qcut(
            pd.Series(valid_knownness).rank(method="average"),
            q=min(4, int(pd.Series(valid_knownness).nunique())),
            labels=False,
            duplicates="drop",
        )
        quantile_codes = np.full(len(knownness), -1, dtype=int)
        quantile_codes[np.flatnonzero(valid_mask)] = np.asarray(quantiles, dtype=int)
        quartile_aucs: list[float] = []
        quartile_aps: list[float] = []
        for code in sorted(set(quantile_codes[quantile_codes >= 0].tolist())):
            mask = quantile_codes == int(code)
            if np.unique(y[mask]).size < 2:
                continue
            quartile_aucs.append(float(roc_auc_score(y[mask], score[mask])))
            quartile_aps.append(float(average_precision(y[mask], score[mask])))
        if quartile_aucs:
            metrics["worst_knownness_quartile_roc_auc"] = float(min(quartile_aucs))
            metrics["mean_knownness_quartile_roc_auc"] = float(np.mean(quartile_aucs))
        if quartile_aps:
            metrics["worst_knownness_quartile_average_precision"] = float(min(quartile_aps))
            metrics["mean_knownness_quartile_average_precision"] = float(np.mean(quartile_aps))
    return metrics


def _meta_design_matrix(merged: pd.DataFrame) -> np.ndarray:
    support = (
        pd.to_numeric(merged["geo_context_hybrid_priority"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    legacy_support = (
        pd.to_numeric(merged["geo_support_light_priority"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    phylo = (
        pd.to_numeric(merged["geo_phylo_ecology_priority"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    blend = (
        pd.to_numeric(merged[GEO_SPREAD_RELIABILITY_BLEND], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    adaptive = (
        pd.to_numeric(merged[GEO_SPREAD_ADAPTIVE_PRIORITY], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    knownness = (
        pd.to_numeric(merged["knownness_score"], errors="coerce").fillna(0.45).to_numpy(dtype=float)
    )
    support_gap = support - phylo
    support_delta = support - legacy_support
    agreement = 1.0 - np.abs(support - phylo)
    support_knownness = support * knownness
    blend_knownness = blend * knownness
    adaptive_knownness = adaptive * knownness
    knownness_uncertainty = 1.0 - np.clip(knownness, 0.0, 1.0)
    return np.column_stack(
        [
            support,
            legacy_support,
            phylo,
            blend,
            adaptive,
            knownness,
            support_gap,
            support_delta,
            agreement,
            support_knownness,
            blend_knownness,
            adaptive_knownness,
            knownness_uncertainty,
        ]
    )


def _meta_sample_weight(y: np.ndarray, knownness: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    knownness_values = np.asarray(knownness, dtype=float)
    class_weights = np.ones(len(labels), dtype=float)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) >= 2:
        inverse = {int(value): 1.0 / max(int(count), 1) for value, count in zip(unique, counts)}
        class_weights = np.asarray([inverse.get(int(value), 1.0) for value in labels], dtype=float)
    low_knownness_boost = 1.0 + 1.5 * np.clip(1.0 - knownness_values, 0.0, 1.0)
    return np.asarray(class_weights * low_knownness_boost, dtype=float)


def build_geo_spread_adaptive_result(
    results: Mapping[str, ModelResult],
    *,
    scored: pd.DataFrame,
    include_ci: bool = True,
    adaptive_name: str = GEO_SPREAD_ADAPTIVE_PRIORITY,
    support_model_name: str = GEO_SPREAD_ADAPTIVE_SUPPORT_MODEL,
    high_knownness_model_name: str = GEO_SPREAD_ADAPTIVE_HIGH_KNOWNNESS_MODEL,
    knownness_center: float = GEO_SPREAD_ADAPTIVE_KNOWNNESS_CENTER,
    knownness_sharpness: float = GEO_SPREAD_ADAPTIVE_KNOWNNESS_SHARPNESS,
) -> ModelResult:
    """Build a smooth knownness-aware ensemble specialized for low-support cohorts."""
    support_result = results.get(support_model_name)
    high_knownness_result = results.get(high_knownness_model_name)
    if support_result is None or support_result.status != "ok":
        return build_failed_model_result(
            adaptive_name,
            f"Adaptive ensemble requires successful component model `{support_model_name}`.",
        )
    if high_knownness_result is None or high_knownness_result.status != "ok":
        return build_failed_model_result(
            adaptive_name,
            f"Adaptive ensemble requires successful component model `{high_knownness_model_name}`.",
        )

    support_frame = _prediction_frame(support_result)
    high_frame = _prediction_frame(high_knownness_result)
    prepared = prepare_geo_spread_scored_table(scored)
    knownness_frame = prepared.loc[:, ["backbone_id", "spread_label", "knownness_score"]].copy()
    merged = support_frame.merge(
        high_frame,
        on=["backbone_id", "spread_label"],
        how="inner",
    ).merge(knownness_frame, on=["backbone_id", "spread_label"], how="left")
    if merged.empty:
        return build_failed_model_result(
            adaptive_name,
            "Adaptive ensemble components do not share a common evaluation cohort.",
        )

    support_score = pd.to_numeric(merged[support_model_name], errors="coerce").fillna(0.0)
    high_score = pd.to_numeric(merged[high_knownness_model_name], errors="coerce").fillna(0.0)
    knownness = pd.to_numeric(merged["knownness_score"], errors="coerce").fillna(
        float(knownness_center)
    )
    low_knownness_weight = _sigmoid_gate(
        knownness.to_numpy(dtype=float),
        center=float(knownness_center),
        sharpness=float(knownness_sharpness),
    )
    score = low_knownness_weight * support_score.to_numpy(dtype=float) + (
        1.0 - low_knownness_weight
    ) * high_score.to_numpy(dtype=float)
    y = (
        pd.to_numeric(merged["spread_label"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy(dtype=int)
    )
    detail = pd.DataFrame(
        {
            support_model_name: support_score.to_numpy(dtype=float),
            high_knownness_model_name: high_score.to_numpy(dtype=float),
            "knownness_score": knownness.to_numpy(dtype=float),
            "low_knownness_weight": low_knownness_weight,
            "adaptive_prediction": score,
        }
    )
    extra_metrics: dict[str, float | int | bool] = {
        "adaptive_knownness_center": float(knownness_center),
        "adaptive_knownness_sharpness": float(knownness_sharpness),
        "adaptive_support_model_weight_mean": float(np.mean(low_knownness_weight)),
    }
    extra_metrics.update(_knownness_metrics(y, score, knownness.to_numpy(dtype=float)))
    return _evaluate_prediction_set(
        adaptive_name,
        y,
        np.asarray(score, dtype=float),
        merged["backbone_id"].astype(str),
        include_ci=include_ci,
        knownness_score=knownness.to_numpy(dtype=float),
        extra_metrics=extra_metrics,
        prediction_detail=detail,
    )


def build_geo_spread_meta_result(
    results: Mapping[str, ModelResult],
    *,
    scored: pd.DataFrame,
    include_ci: bool = True,
    meta_name: str = GEO_SPREAD_META_PRIORITY,
) -> ModelResult:
    """Fit a cross-fitted knownness-aware meta-learner on derived/base geo predictions."""
    required_models = [
        "geo_context_hybrid_priority",
        "geo_support_light_priority",
        "geo_phylo_ecology_priority",
        GEO_SPREAD_RELIABILITY_BLEND,
        GEO_SPREAD_ADAPTIVE_PRIORITY,
    ]
    frames: list[pd.DataFrame] = []
    for model_name in required_models:
        result = results.get(model_name)
        if result is None or result.status != "ok":
            return build_failed_model_result(
                meta_name,
                f"Meta learner requires successful component model `{model_name}`.",
            )
        frames.append(_prediction_frame(result))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["backbone_id", "spread_label"], how="inner")
    prepared = prepare_geo_spread_scored_table(scored)
    knownness_frame = prepared.loc[:, ["backbone_id", "spread_label", "knownness_score"]].copy()
    merged = merged.merge(knownness_frame, on=["backbone_id", "spread_label"], how="left")
    if merged.empty:
        return build_failed_model_result(
            meta_name, "Meta learner components do not share a common evaluation cohort."
        )

    y = (
        pd.to_numeric(merged["spread_label"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy(dtype=int)
    )
    knownness = (
        pd.to_numeric(merged["knownness_score"], errors="coerce").fillna(0.45).to_numpy(dtype=float)
    )
    X = _meta_design_matrix(merged)
    if np.unique(y).size < 2 or len(y) < 8:
        return build_failed_model_result(
            meta_name, "Meta learner requires at least two classes and 8 labeled rows."
        )

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    n_splits = max(2, min(5, positives, negatives))
    if n_splits < 2:
        return build_failed_model_result(
            meta_name, "Meta learner requires at least two examples per class."
        )

    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        model = LogisticRegression(
            C=0.7,
            fit_intercept=True,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        model.fit(
            X[train_idx],
            y[train_idx],
            sample_weight=_meta_sample_weight(y[train_idx], knownness[train_idx]),
        )
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    final_model = LogisticRegression(
        C=0.7,
        fit_intercept=True,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    final_model.fit(X, y, sample_weight=_meta_sample_weight(y, knownness))
    detail = merged.loc[:, required_models].copy()
    detail["knownness_score"] = knownness
    detail["meta_prediction"] = oof
    extra_metrics: dict[str, float | int | bool] = {
        "meta_component_count": float(len(required_models)),
        "meta_low_knownness_weight_mean": float(np.mean(_meta_sample_weight(y, knownness))),
        "meta_n_splits": float(n_splits),
    }
    extra_metrics.update(_knownness_metrics(y, oof, knownness))
    for idx, coefficient in enumerate(final_model.coef_[0], start=1):
        extra_metrics[f"meta_coef_{idx}"] = float(coefficient)
    return _evaluate_prediction_set(
        meta_name,
        y,
        np.asarray(oof, dtype=float),
        merged["backbone_id"].astype(str),
        include_ci=include_ci,
        knownness_score=knownness,
        extra_metrics=extra_metrics,
        prediction_detail=detail,
    )


def build_geo_spread_selection_scorecard(
    results: Mapping[str, ModelResult],
    calibration_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a deterministic reliability-aware model selection scorecard."""
    rows: list[dict[str, Any]] = []
    calibration_frame = calibration_summary if calibration_summary is not None else pd.DataFrame()
    for model_name, result in results.items():
        if result.status != "ok":
            continue
        row: dict[str, Any] = {
            "model_name": str(model_name),
            "roc_auc": float(result.metrics.get("roc_auc", float("nan"))),
            "average_precision": float(result.metrics.get("average_precision", float("nan"))),
            "decision_utility_score": float(
                result.metrics.get("decision_utility_score", float("nan"))
            ),
            "novelty_adjusted_average_precision": float(
                result.metrics.get("novelty_adjusted_average_precision", float("nan"))
            ),
            "expected_calibration_error": float(
                result.metrics.get("expected_calibration_error", float("nan"))
            ),
            "brier_score": float(result.metrics.get("brier_score", float("nan"))),
            "abstain_rate": float("nan"),
            "calibrated_expected_calibration_error": float("nan"),
            "calibrated_brier_score": float("nan"),
            "low_knownness_roc_auc": float(
                result.metrics.get("low_knownness_roc_auc", float("nan"))
            ),
            "low_knownness_average_precision": float(
                result.metrics.get("low_knownness_average_precision", float("nan"))
            ),
            "worst_knownness_quartile_roc_auc": float(
                result.metrics.get("worst_knownness_quartile_roc_auc", float("nan"))
            ),
            "event_within_3y_roc_auc": float(
                result.metrics.get("event_within_3y_roc_auc", float("nan"))
            ),
        }
        if not calibration_frame.empty:
            match = calibration_frame.loc[
                calibration_frame.get("model_name", pd.Series(dtype=str))
                .astype(str)
                .eq(str(model_name))
            ]
            if not match.empty:
                row["abstain_rate"] = float(
                    pd.to_numeric(
                        pd.Series([match.iloc[0].get("abstain_rate", np.nan)]), errors="coerce"
                    ).iloc[0]
                )
                row["calibrated_expected_calibration_error"] = float(
                    pd.to_numeric(
                        pd.Series(
                            [match.iloc[0].get("calibrated_expected_calibration_error", np.nan)]
                        ),
                        errors="coerce",
                    ).iloc[0]
                )
                row["calibrated_brier_score"] = float(
                    pd.to_numeric(
                        pd.Series([match.iloc[0].get("calibrated_brier_score", np.nan)]),
                        errors="coerce",
                    ).iloc[0]
                )
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard

    def _normalized(column: str, *, ascending: bool) -> pd.Series:
        values = pd.to_numeric(scorecard[column], errors="coerce")
        valid = values.dropna()
        if valid.empty or float(valid.max()) == float(valid.min()):
            return pd.Series(0.5, index=scorecard.index, dtype=float)
        scaled = (values - float(valid.min())) / (float(valid.max()) - float(valid.min()))
        if ascending:
            scaled = 1.0 - scaled
        return scaled.fillna(0.0).astype(float)

    scorecard["selection_score"] = (
        0.26 * _normalized("roc_auc", ascending=False)
        + 0.18 * _normalized("average_precision", ascending=False)
        + 0.08 * _normalized("decision_utility_score", ascending=False)
        + 0.12 * _normalized("calibrated_expected_calibration_error", ascending=True)
        + 0.10 * _normalized("calibrated_brier_score", ascending=True)
        + 0.04 * _normalized("abstain_rate", ascending=True)
        + 0.05 * _normalized("novelty_adjusted_average_precision", ascending=False)
        + 0.07 * _normalized("low_knownness_roc_auc", ascending=False)
        + 0.06 * _normalized("worst_knownness_quartile_roc_auc", ascending=False)
        + 0.04 * _normalized("event_within_3y_roc_auc", ascending=False)
    )
    abstain_rate = pd.to_numeric(scorecard["abstain_rate"], errors="coerce")
    calibrated_ece = pd.to_numeric(
        scorecard["calibrated_expected_calibration_error"], errors="coerce"
    )
    scorecard["primary_eligible"] = abstain_rate.fillna(1.0).le(0.60) & calibrated_ece.fillna(
        float("inf")
    ).le(0.05)
    scorecard = scorecard.sort_values(
        ["primary_eligible", "selection_score", "roc_auc", "average_precision", "model_name"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scorecard["selection_rank"] = np.arange(1, len(scorecard) + 1, dtype=int)
    return scorecard


def select_geo_spread_primary_model(
    results: Mapping[str, ModelResult],
    calibration_summary: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    """Return the recommended primary model name and the full scorecard."""
    scorecard = build_geo_spread_selection_scorecard(results, calibration_summary)
    if scorecard.empty:
        return "", scorecard
    return str(scorecard.iloc[0]["model_name"]), scorecard
