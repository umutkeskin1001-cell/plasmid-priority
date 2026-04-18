"""Lead-time bias audit for training visibility vs. outcome associations.

Lead-time bias occurs when models appear to perform well because they rely on
observation intensity (member_count_train, knownness) rather than biological
signal. This module provides lightweight diagnostics to flag such concerns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation safely with validation."""
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].rank(method="average").corr(frame["right"].rank(method="average")))


def compute_lead_time_bias_metrics(
    scored: pd.DataFrame,
    *,
    visibility_column: str = "log1p_member_count_train",
    outcome_column: str = "spread_label",
) -> dict[str, object]:
    """Compute lead-time bias diagnostics between visibility and outcome.

    Args:
        scored: Scored backbone table with visibility and outcome columns
        visibility_column: Column representing observation intensity/knownness
        outcome_column: Binary outcome column (e.g., spread_label)

    Returns:
        Dictionary with correlation metrics and bias indicators
    """
    if outcome_column not in scored.columns:
        return {
            "visibility_column": visibility_column,
            "outcome_column": outcome_column,
            "status": "missing_outcome_column",
            "visibility_outcome_spearman": float("nan"),
            "visibility_outcome_pearson": float("nan"),
            "n_eligible": 0,
            "lead_time_bias_concern": "cannot_assess",
        }

    if visibility_column not in scored.columns:
        return {
            "visibility_column": visibility_column,
            "outcome_column": outcome_column,
            "status": "missing_visibility_column",
            "visibility_outcome_spearman": float("nan"),
            "visibility_outcome_pearson": float("nan"),
            "n_eligible": 0,
            "lead_time_bias_concern": "cannot_assess",
        }

    # Filter to rows with valid outcome
    eligible = scored.loc[scored[outcome_column].notna()].copy()
    n_eligible = int(len(eligible))

    if n_eligible < 10:
        return {
            "visibility_column": visibility_column,
            "outcome_column": outcome_column,
            "status": "insufficient_data",
            "visibility_outcome_spearman": float("nan"),
            "visibility_outcome_pearson": float("nan"),
            "n_eligible": n_eligible,
            "lead_time_bias_concern": "cannot_assess",
        }

    # Compute correlations
    visibility = pd.to_numeric(eligible[visibility_column], errors="coerce")
    outcome = eligible[outcome_column].astype(int)

    spearman_corr = _safe_spearman(visibility, outcome)

    # Pearson correlation
    valid_mask = visibility.notna() & outcome.notna()
    pearson_corr = float("nan")
    if valid_mask.sum() >= 3:
        pearson_corr = float(visibility.loc[valid_mask].corr(outcome.loc[valid_mask]))

    # Assess concern level
    if math.isnan(spearman_corr):
        concern = "cannot_assess"
    elif spearman_corr > 0.3:
        concern = "high"
    elif spearman_corr > 0.15:
        concern = "moderate"
    else:
        concern = "low"

    return {
        "visibility_column": visibility_column,
        "outcome_column": outcome_column,
        "status": "ok",
        "visibility_outcome_spearman": round(spearman_corr, 4) if pd.notna(spearman_corr) else None,
        "visibility_outcome_pearson": round(pearson_corr, 4) if pd.notna(pearson_corr) else None,
        "n_eligible": n_eligible,
        "n_positive": int(outcome.sum()),
        "positive_prevalence": round(float(outcome.mean()), 4),
        "lead_time_bias_concern": concern,
    }


def build_visibility_decile_table(
    scored: pd.DataFrame,
    *,
    visibility_column: str = "log1p_member_count_train",
    outcome_column: str = "spread_label",
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Build decile-based spread rate summary by visibility level.

    Args:
        scored: Scored backbone table
        visibility_column: Column for visibility stratification
        outcome_column: Binary outcome column
        n_quantiles: Number of quantile bins (default 10 = deciles)

    Returns:
        DataFrame with spread rate per visibility decile
    """
    if outcome_column not in scored.columns or visibility_column not in scored.columns:
        return pd.DataFrame(
            columns=[
                "quantile_bin",
                "visibility_min",
                "visibility_max",
                "visibility_median",
                "n_backbones",
                "n_positive",
                "spread_rate",
                "spread_rate_ci_lower",
                "spread_rate_ci_upper",
            ]
        )

    eligible = scored.loc[scored[outcome_column].notna()].copy()
    visibility = pd.to_numeric(eligible[visibility_column], errors="coerce")

    # Filter to rows with valid visibility
    valid_mask = visibility.notna()
    eligible = eligible.loc[valid_mask].copy()
    visibility = visibility.loc[valid_mask]

    if len(eligible) < n_quantiles * 2:
        return pd.DataFrame(
            columns=[
                "quantile_bin",
                "visibility_min",
                "visibility_max",
                "visibility_median",
                "n_backbones",
                "n_positive",
                "spread_rate",
                "spread_rate_ci_lower",
                "spread_rate_ci_upper",
            ]
        )

    # Create quantile bins using rank-based method for stability
    ranks = visibility.rank(method="average", pct=True)
    try:
        bin_labels = [f"Q{i + 1}" for i in range(n_quantiles)]
        bins = pd.qcut(ranks, q=n_quantiles, labels=bin_labels, duplicates="drop")
    except ValueError:
        # Fallback if qcut fails
        bins = pd.cut(ranks, bins=n_quantiles, labels=[f"Q{i + 1}" for i in range(n_quantiles)])

    eligible["_visibility_bin"] = bins
    outcome = eligible[outcome_column].astype(int)

    rows = []
    for bin_name in sorted(eligible["_visibility_bin"].dropna().unique()):
        mask = eligible["_visibility_bin"] == bin_name
        bin_visibility = visibility.loc[mask]
        bin_outcome = outcome.loc[mask]

        n_backbones = int(len(bin_outcome))
        n_positive = int(bin_outcome.sum())
        spread_rate = float(bin_outcome.mean()) if n_backbones > 0 else float("nan")

        # Simple Wilson CI for binomial proportion
        if n_backbones > 0 and n_positive > 0 and n_positive < n_backbones:
            z = 1.96  # 95% CI
            p = spread_rate
            denom = 1 + z**2 / n_backbones
            center = (p + z**2 / (2 * n_backbones)) / denom
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_backbones)) / n_backbones) / denom
            ci_lower = max(0.0, center - margin)
            ci_upper = min(1.0, center + margin)
        elif n_positive == 0:
            ci_lower = 0.0
            ci_upper = 0.0 if n_backbones == 0 else 3.0 / n_backbones  # Rule of three approx
        elif n_positive == n_backbones:
            ci_lower = 1.0 - 3.0 / n_backbones if n_backbones > 0 else 0.0
            ci_upper = 1.0
        else:
            ci_lower = ci_upper = spread_rate

        rows.append(
            {
                "quantile_bin": str(bin_name),
                "visibility_min": round(float(bin_visibility.min()), 4),
                "visibility_max": round(float(bin_visibility.max()), 4),
                "visibility_median": round(float(bin_visibility.median()), 4),
                "n_backbones": n_backbones,
                "n_positive": n_positive,
                "spread_rate": round(spread_rate, 4),
                "spread_rate_ci_lower": round(ci_lower, 4),
                "spread_rate_ci_upper": round(ci_upper, 4),
            }
        )

    return pd.DataFrame(rows)


def build_lead_time_bias_audit(
    scored: pd.DataFrame,
    *,
    visibility_metrics: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build comprehensive lead-time bias audit.

    Args:
        scored: Scored backbone table
        visibility_metrics: List of visibility columns to evaluate
                          (defaults to common knownness proxies)

    Returns:
        Tuple of (summary_metrics_table, decile_detail_table)
    """
    if visibility_metrics is None:
        visibility_metrics = [
            "log1p_member_count_train",
            "log1p_n_countries_train",
            "refseq_share_train",
            "knownness_score",
        ]

    # Filter to metrics that exist in the data
    available_metrics = [col for col in visibility_metrics if col in scored.columns]

    if not available_metrics:
        empty_summary = pd.DataFrame(
            columns=[
                "visibility_column",
                "outcome_column",
                "status",
                "visibility_outcome_spearman",
                "lead_time_bias_concern",
            ]
        )
        return empty_summary, pd.DataFrame()

    # Build summary metrics for each visibility proxy
    summary_rows = []
    for metric in available_metrics:
        result = compute_lead_time_bias_metrics(
            scored,
            visibility_column=metric,
            outcome_column="spread_label",
        )
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)

    # Build decile table for primary visibility metric
    primary_visibility = available_metrics[0]
    decile_df = build_visibility_decile_table(
        scored,
        visibility_column=primary_visibility,
        outcome_column="spread_label",
        n_quantiles=10,
    )

    return summary_df, decile_df


def summarize_lead_time_bias_findings(
    summary_table: pd.DataFrame,
    decile_table: pd.DataFrame,
) -> dict[str, object]:
    """Summarize lead-time bias findings for reporting.

    Returns a dict with:
    - overall_concern_level: max concern across metrics
    - n_metrics_evaluated: number of visibility proxies assessed
    - n_high_concern: count of metrics with high concern
    - n_moderate_concern: count of metrics with moderate concern
    - trend_direction: whether spread rate increases with visibility
    - interpretation: human-readable summary
    """
    if summary_table.empty:
        return {
            "overall_concern_level": "unknown",
            "n_metrics_evaluated": 0,
            "n_high_concern": 0,
            "n_moderate_concern": 0,
            "trend_direction": "unknown",
            "interpretation": "No visibility metrics available for assessment",
        }

    valid = summary_table.loc[summary_table["status"] == "ok"]
    n_metrics = int(len(valid))

    if n_metrics == 0:
        return {
            "overall_concern_level": "unknown",
            "n_metrics_evaluated": 0,
            "n_high_concern": 0,
            "n_moderate_concern": 0,
            "trend_direction": "unknown",
            "interpretation": "No valid visibility metrics could be assessed",
        }

    concern_counts = valid["lead_time_bias_concern"].value_counts()
    n_high = int(concern_counts.get("high", 0))
    n_moderate = int(concern_counts.get("moderate", 0))

    # Overall concern is the max across metrics
    if n_high > 0:
        overall_concern = "high"
    elif n_moderate > 0:
        overall_concern = "moderate"
    else:
        overall_concern = "low"

    # Assess trend direction from decile table if available
    trend_direction = "unknown"
    if not decile_table.empty and len(decile_table) >= 3:
        # Simple linear regression on spread_rate vs quantile index
        x = np.arange(len(decile_table))
        y = decile_table["spread_rate"].to_numpy()
        valid_y = ~np.isnan(y)
        if valid_y.sum() >= 3:
            x_valid = x[valid_y]
            y_valid = y[valid_y]
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            if slope > 0.01:
                trend_direction = "increasing"
            elif slope < -0.01:
                trend_direction = "decreasing"
            else:
                trend_direction = "flat"

    # Build interpretation
    if overall_concern == "high":
        interpretation = (
            f"HIGH concern: {n_high} visibility metrics show strong correlation with outcome. "
            f"Models may be learning from observation intensity rather than biological signal. "
            f"Trend direction: {trend_direction}."
        )
    elif overall_concern == "moderate":
        interpretation = (
            f"MODERATE concern: {n_moderate} visibility metrics show moderate correlation. "
            f"Review model features for potential lead-time bias. "
            f"Trend direction: {trend_direction}."
        )
    else:
        interpretation = (
            f"LOW concern: Visibility metrics show weak correlation with outcome. "
            f"Lead-time bias appears limited. Trend direction: {trend_direction}."
        )

    return {
        "overall_concern_level": overall_concern,
        "n_metrics_evaluated": n_metrics,
        "n_high_concern": n_high,
        "n_moderate_concern": n_moderate,
        "trend_direction": trend_direction,
        "interpretation": interpretation,
    }
