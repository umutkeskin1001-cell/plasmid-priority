"""Events-per-variable (EPV) audit for model training diagnostics.

EPV = number of positive events (spread_label=1) / number of model features

Low EPV (< 10) is generally considered concerning for logistic regression
and may indicate risk of overfitting or unstable coefficient estimates.
"""

from __future__ import annotations

import pandas as pd

from plasmid_priority.modeling.module_a_support import (
    GOVERNANCE_MODEL_NAME,
    MODULE_A_FEATURE_SETS,
    PRIMARY_MODEL_NAME,
)


def compute_epv_for_model(
    scored: pd.DataFrame,
    model_name: str,
    *,
    feature_set_override: list[str] | None = None,
    split_year: int | None = None,
) -> dict[str, object]:
    """Compute EPV (events-per-variable) for a single model.

    Args:
        scored: Scored backbone table with spread_label column
        model_name: Name of the model to evaluate
        feature_set_override: Optional override for feature set (for custom models)
        split_year: Optional year to restrict training period (if applicable)

    Returns:
        Dictionary with n_features, n_positive_events, epv, and interpretation flags
    """
    # Get feature set
    if feature_set_override is not None:
        features = list(feature_set_override)
    else:
        features = list(MODULE_A_FEATURE_SETS.get(model_name, []))

    n_features = len(features)

    # Determine eligible training rows
    if "spread_label" not in scored.columns:
        return {
            "model_name": model_name,
            "n_features": n_features,
            "n_positive_events": 0,
            "n_eligible_backbones": 0,
            "epv": float("nan"),
            "epv_status": "missing_spread_label",
            "epv_interpretation": "cannot_compute",
        }

    eligible = scored.loc[scored["spread_label"].notna()].copy()

    # Optional year-based restriction
    if split_year is not None and "resolved_year" in eligible.columns:
        years = pd.to_numeric(eligible["resolved_year"], errors="coerce").fillna(0)
        eligible = eligible.loc[years <= split_year].copy()

    n_eligible = int(len(eligible))
    n_positive = int(eligible["spread_label"].fillna(0).astype(int).sum())

    # Compute EPV
    if n_features == 0:
        epv = float("nan")
        status = "no_features"
    elif n_positive == 0:
        epv = 0.0
        status = "no_positive_events"
    else:
        epv = float(n_positive) / float(n_features)
        status = "ok"

    # Interpretation based on EPV rules of thumb:
    # <10 indicates elevated overfitting/instability concern.
    if epv < 5:
        interpretation = "very_high_risk_of_overfitting"
    elif epv < 10:
        interpretation = "high_risk_of_overfitting"
    elif epv < 20:
        interpretation = "moderate_risk_monitor"
    else:
        interpretation = "low_risk"

    return {
        "model_name": model_name,
        "n_features": n_features,
        "n_positive_events": n_positive,
        "n_eligible_backbones": n_eligible,
        "epv": round(epv, 2) if status == "ok" else float("nan"),
        "epv_status": status,
        "epv_interpretation": interpretation,
    }


def build_epv_audit_table(
    scored: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    include_official: bool = True,
    include_challengers: bool = True,
) -> pd.DataFrame:
    """Build EPV audit table for configured models.

    Args:
        scored: Scored backbone table with spread_label column
        model_names: Optional list of specific models to evaluate
        include_official: Whether to include official discovery/governance models
        include_challengers: Whether to include challenger variants

    Returns:
        DataFrame with EPV metrics per model
    """
    models_to_evaluate: set[str] = set()

    if model_names is not None:
        models_to_evaluate.update(model_names)

    if include_official:
        # Primary model (discovery)
        if PRIMARY_MODEL_NAME:
            models_to_evaluate.add(PRIMARY_MODEL_NAME)

        # Governance model
        if GOVERNANCE_MODEL_NAME:
            models_to_evaluate.add(GOVERNANCE_MODEL_NAME)

    if include_challengers:
        # Add all configured models as potential challengers
        models_to_evaluate.update(MODULE_A_FEATURE_SETS.keys())

    # Filter to models with configured feature sets
    available_models = set(MODULE_A_FEATURE_SETS.keys())
    models_to_evaluate = models_to_evaluate & available_models

    if not models_to_evaluate:
        return pd.DataFrame(
            columns=[
                "model_name",
                "n_features",
                "n_positive_events",
                "n_eligible_backbones",
                "epv",
                "epv_status",
                "epv_interpretation",
            ]
        )

    rows = []
    for model_name in sorted(models_to_evaluate):
        row = compute_epv_for_model(scored, model_name)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_epv_concerns(epv_table: pd.DataFrame) -> dict[str, object]:
    """Summarize EPV concerns for reporting.

    Returns a dict with:
    - n_models_evaluated: total models in table
    - n_models_low_epv: models with EPV < 10
    - n_models_very_low_epv: models with EPV < 5
    - models_requiring_review: list of model names with low EPV
    """
    if epv_table.empty:
        return {
            "n_models_evaluated": 0,
            "n_models_low_epv": 0,
            "n_models_very_low_epv": 0,
            "models_requiring_review": [],
        }

    # Filter to rows with valid EPV
    valid = epv_table.loc[epv_table["epv_status"] == "ok"].copy()

    if valid.empty:
        return {
            "n_models_evaluated": int(len(epv_table)),
            "n_models_low_epv": 0,
            "n_models_very_low_epv": 0,
            "models_requiring_review": [],
        }

    low_epv = valid.loc[valid["epv"] < 10.0]
    very_low_epv = valid.loc[valid["epv"] < 5.0]

    return {
        "n_models_evaluated": int(len(valid)),
        "n_models_low_epv": int(len(low_epv)),
        "n_models_very_low_epv": int(len(very_low_epv)),
        "models_requiring_review": sorted(low_epv["model_name"].tolist()),
    }
