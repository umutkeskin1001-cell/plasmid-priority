"""Variance Inflation Factor (VIF) audit for multicollinearity visibility.

This module provides lightweight VIF computation for model feature sets
to detect multicollinearity without imposing hard rejection gates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif_values(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF = 1 / (1 - R^2) where R^2 is from regressing the feature on all others.
    VIF = 1 indicates no multicollinearity.
    VIF > 5 suggests moderate multicollinearity.
    VIF > 10 indicates strong multicollinearity.

    Args:
        X: Design matrix (n_samples, n_features)
        feature_names: List of feature names corresponding to columns

    Returns:
        DataFrame with columns: feature_name, vif, concern_flag
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    if n_features < 2:
        return pd.DataFrame({
            "feature_name": feature_names[:n_features] if n_features == 1 else [],
            "vif": [1.0] if n_features == 1 else [],
            "concern_flag": ["low"] if n_features == 1 else [],
        })

    # Center the data
    X_centered = X - X.mean(axis=0)

    vif_values = []
    for i in range(n_features):
        # Get feature i and all other features
        y = X_centered[:, i]
        X_others = np.delete(X_centered, i, axis=1)

        # Check for constant features
        if np.var(y) < 1e-10:
            vif_values.append(float("inf"))
            continue

        # Compute R^2 via linear regression using least squares
        try:
            # X_others @ beta ≈ y
            # beta = (X_others^T @ X_others)^+ @ X_others^T @ y
            XtX = X_others.T @ X_others
            # Add small ridge for numerical stability
            ridge = 1e-6 * np.eye(XtX.shape[0])
            beta = np.linalg.solve(XtX + ridge, X_others.T @ y)
            y_pred = X_others @ beta

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum(y ** 2)  # Already centered

            if ss_tot < 1e-10:
                r_squared = 0.0
            else:
                r_squared = 1 - ss_res / ss_tot

            # Clip to avoid numerical issues
            r_squared = np.clip(r_squared, 0.0, 0.999999)
            vif = 1.0 / (1.0 - r_squared) if r_squared < 0.999999 else float("inf")
            vif_values.append(vif)
        except (np.linalg.LinAlgError, ValueError):
            vif_values.append(float("nan"))

    # Define concern thresholds
    concern_flags = []
    for vif in vif_values:
        if np.isnan(vif):
            concern_flags.append("unknown")
        elif np.isinf(vif):
            concern_flags.append("critical")
        elif vif > 10:
            concern_flags.append("high")
        elif vif > 5:
            concern_flags.append("moderate")
        else:
            concern_flags.append("low")

    return pd.DataFrame({
        "feature_name": feature_names,
        "vif": vif_values,
        "concern_flag": concern_flags,
    })


def build_vif_audit_table(
    backbone_table: pd.DataFrame,
    model_feature_sets: dict[str, list[str]],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build VIF audit for specified models.

    Args:
        backbone_table: DataFrame containing backbone features
        model_feature_sets: Dict mapping model_name -> list of feature names
        model_names: Optional list of models to audit (defaults to all keys)

    Returns:
        DataFrame with columns: model_name, feature_name, vif, concern_flag
    """
    if model_names is None:
        model_names = list(model_feature_sets.keys())

    rows = []
    for model_name in model_names:
        features = model_feature_sets.get(model_name, [])
        if not features:
            continue

        # Check which features exist in the table
        available_features = [f for f in features if f in backbone_table.columns]
        if not available_features:
            continue

        # Extract feature matrix
        X = backbone_table[available_features].fillna(0.0).to_numpy(dtype=float)

        # Compute VIF
        vif_df = compute_vif_values(X, available_features)
        vif_df["model_name"] = model_name

        rows.append(vif_df)

    if not rows:
        return pd.DataFrame(columns=["model_name", "feature_name", "vif", "concern_flag"])

    return pd.concat(rows, ignore_index=True)


def summarize_vif_concerns(vif_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize VIF concerns per model.

    Returns:
        DataFrame with model-level VIF summary statistics
    """
    if vif_table.empty or "model_name" not in vif_table.columns:
        return pd.DataFrame()

    summary_rows = []
    for model_name, group in vif_table.groupby("model_name", sort=False):
        vif_values = pd.to_numeric(group["vif"], errors="coerce").dropna()

        n_features = len(group)
        n_high_concern = (group["concern_flag"].isin(["high", "critical"])).sum()
        n_moderate_concern = (group["concern_flag"] == "moderate").sum()

        summary_rows.append({
            "model_name": model_name,
            "n_features": n_features,
            "n_high_concern": n_high_concern,
            "n_moderate_concern": n_moderate_concern,
            "max_vif": vif_values.max() if not vif_values.empty else np.nan,
            "mean_vif": vif_values.mean() if not vif_values.empty else np.nan,
            "median_vif": vif_values.median() if not vif_values.empty else np.nan,
            "pct_vif_gt_5": (vif_values > 5).mean() * 100 if not vif_values.empty else 0.0,
            "pct_vif_gt_10": (vif_values > 10).mean() * 100 if not vif_values.empty else 0.0,
            "overall_status": (
                "review_recommended" if n_high_concern > 0
                else "moderate_concern" if n_moderate_concern > 0
                else "acceptable"
            ),
        })

    return pd.DataFrame(summary_rows)
