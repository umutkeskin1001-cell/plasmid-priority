"""Tests for validation.vif module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.validation.vif import (
    build_vif_audit_table,
    compute_vif_values,
    summarize_vif_concerns,
)


def test_compute_vif_values_single_feature() -> None:
    """Test VIF computation with single feature."""
    X = np.array([[1.0], [2.0], [3.0]])
    feature_names = ["feature1"]
    result = compute_vif_values(X, feature_names)
    assert len(result) == 1
    assert result.iloc[0]["vif"] == 1.0
    assert result.iloc[0]["concern_flag"] == "low"


def test_compute_vif_values_two_features_uncorrelated() -> None:
    """Test VIF computation with uncorrelated features."""
    X = np.array([[1.0, 10.0], [2.0, 5.0], [3.0, 15.0], [4.0, 8.0]])
    feature_names = ["feature1", "feature2"]
    result = compute_vif_values(X, feature_names)
    assert len(result) == 2
    # VIF should be low for uncorrelated features
    assert all(result["vif"] < 10)


def test_compute_vif_values_perfect_correlation() -> None:
    """Test VIF computation with perfectly correlated features."""
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    feature_names = ["feature1", "feature2"]
    result = compute_vif_values(X, feature_names)
    assert len(result) == 2
    # Should have high VIF due to correlation
    assert any(result["concern_flag"] != "low")


def test_compute_vif_values_constant_feature() -> None:
    """Test VIF computation with constant feature."""
    X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    feature_names = ["feature1", "feature2"]
    result = compute_vif_values(X, feature_names)
    assert len(result) == 2
    # Constant feature should have critical concern
    assert "critical" in result["concern_flag"].values


def test_compute_vif_values_empty() -> None:
    """Test VIF computation with no features."""
    X = np.array([[]]).reshape(0, 0)
    feature_names = []
    result = compute_vif_values(X, feature_names)
    assert len(result) == 0


def test_build_vif_audit_table_basic() -> None:
    """Test VIF audit table building."""
    backbone_table = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 3.0, 4.0],
            "feature3": [5.0, 6.0, 7.0],
        }
    )
    model_feature_sets = {
        "model1": ["feature1", "feature2"],
        "model2": ["feature2", "feature3"],
    }
    result = build_vif_audit_table(backbone_table, model_feature_sets)
    assert len(result) == 4  # 2 features per model
    assert "model_name" in result.columns
    assert "feature_name" in result.columns
    assert "vif" in result.columns
    assert "concern_flag" in result.columns


def test_build_vif_audit_table_missing_features() -> None:
    """Test VIF audit with missing features."""
    backbone_table = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
        }
    )
    model_feature_sets = {
        "model1": ["feature1", "feature2"],  # feature2 missing
    }
    result = build_vif_audit_table(backbone_table, model_feature_sets)
    # Should only compute for available features
    assert len(result) == 1
    assert result.iloc[0]["feature_name"] == "feature1"


def test_build_vif_audit_table_specific_models() -> None:
    """Test VIF audit for specific models only."""
    backbone_table = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 3.0, 4.0],
        }
    )
    model_feature_sets = {
        "model1": ["feature1"],
        "model2": ["feature2"],
    }
    result = build_vif_audit_table(backbone_table, model_feature_sets, model_names=["model1"])
    assert len(result) == 1
    assert result.iloc[0]["model_name"] == "model1"


def test_summarize_vif_concerns_empty() -> None:
    """Test VIF summary with empty table."""
    vif_table = pd.DataFrame()
    result = summarize_vif_concerns(vif_table)
    assert len(result) == 0


def test_summarize_vif_concerns_basic() -> None:
    """Test VIF summary with data."""
    vif_table = pd.DataFrame(
        {
            "model_name": ["model1", "model1", "model2"],
            "feature_name": ["f1", "f2", "f3"],
            "vif": [1.5, 2.0, 1.8],
            "concern_flag": ["low", "low", "low"],
        }
    )
    result = summarize_vif_concerns(vif_table)
    assert len(result) == 2
    assert "model_name" in result.columns
    assert "n_features" in result.columns
    assert "max_vif" in result.columns
    assert "overall_status" in result.columns


def test_summarize_vif_concerns_with_high_vif() -> None:
    """Test VIF summary with high VIF values."""
    vif_table = pd.DataFrame(
        {
            "model_name": ["model1", "model1"],
            "feature_name": ["f1", "f2"],
            "vif": [1.5, 15.0],
            "concern_flag": ["low", "high"],
        }
    )
    result = summarize_vif_concerns(vif_table)
    assert len(result) == 1
    assert result.iloc[0]["overall_status"] == "review_recommended"
    assert result.iloc[0]["n_high_concern"] == 1


def test_summarize_vif_concerns_moderate() -> None:
    """Test VIF summary with moderate concerns."""
    vif_table = pd.DataFrame(
        {
            "model_name": ["model1", "model1"],
            "feature_name": ["f1", "f2"],
            "vif": [1.5, 6.0],
            "concern_flag": ["low", "moderate"],
        }
    )
    result = summarize_vif_concerns(vif_table)
    assert len(result) == 1
    assert result.iloc[0]["overall_status"] == "moderate_concern"
