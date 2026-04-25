"""Task dataclass for modeling operations.

This module provides a structured dataclass for modeling tasks,
replacing lambda functions with type-safe task definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from plasmid_priority.modeling.fit_config import FitConfig


@dataclass(slots=True)
class ModelingTask:
    """Structured task definition for modeling operations.

    This replaces lambda functions with a type-safe, serializable
    task definition that can be passed to executors and workflows.
    """

    task_id: str
    model_name: str
    feature_columns: tuple[str, ...]
    fit_config: FitConfig
    label_column: str = "spread_label"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "feature_columns": list(self.feature_columns),
            "fit_config": self.fit_config.to_dict(),
            "label_column": self.label_column,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelingTask:
        """Create from dictionary."""
        fit_config = FitConfig.from_dict(data["fit_config"])
        return cls(
            task_id=data["task_id"],
            model_name=data["model_name"],
            feature_columns=tuple(data["feature_columns"]),
            fit_config=fit_config,
            label_column=data.get("label_column", "spread_label"),
            metadata=data.get("metadata"),
        )


@dataclass(slots=True)
class TaskResult:
    """Result of a modeling task execution."""

    task_id: str
    model_name: str
    status: str  # "ok", "failed", "skipped"
    metrics: dict[str, Any]
    predictions: pd.DataFrame | None = None
    error_message: str | None = None
    execution_time_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "status": self.status,
            "metrics": self.metrics,
            "predictions": self.predictions.to_dict() if self.predictions is not None else None,
            "error_message": self.error_message,
            "execution_time_seconds": self.execution_time_seconds,
        }


def create_task_from_model_name(
    model_name: str,
    feature_sets: dict[str, list[str]],
    fit_configs: dict[str, dict[str, Any]],
    *,
    task_id: str | None = None,
) -> ModelingTask:
    """Create a ModelingTask from a model name.

    Args:
        model_name: Name of the model
        feature_sets: Dictionary mapping model names to feature columns
        fit_configs: Dictionary mapping model names to fit configurations
        task_id: Optional task ID (auto-generated if None)

    Returns:
        ModelingTask instance
    """
    if model_name not in feature_sets:
        raise ValueError(f"Model '{model_name}' not found in feature_sets")

    if model_name not in fit_configs:
        raise ValueError(f"Model '{model_name}' not found in fit_configs")

    fit_config = FitConfig.from_dict(fit_configs[model_name])

    return ModelingTask(
        task_id=task_id or f"task_{model_name}",
        model_name=model_name,
        feature_columns=tuple(feature_sets[model_name]),
        fit_config=fit_config,
    )
