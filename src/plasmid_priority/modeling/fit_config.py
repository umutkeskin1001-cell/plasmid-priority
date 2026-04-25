"""Pydantic models for fit configuration.

This module provides type-safe configuration for model fitting parameters,
replacing dictionary-based fit_kwargs with structured Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SolverType(str, Enum):
    """Supported solver types for logistic regression."""

    LBFGS = "lbfgs"
    LIBLINEAR = "liblinear"
    SAGA = "saga"
    NEWTON_CG = "newton-cg"


class PenaltyType(str, Enum):
    """Supported penalty types for regularization."""

    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"
    NONE = None


class FitConfig(BaseModel):
    """Configuration for model fitting parameters.

    This replaces the dictionary-based fit_kwargs approach with a
    type-safe, validated configuration model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Logistic regression parameters
    solver: SolverType = Field(default=SolverType.LBFGS, description="Solver algorithm")
    penalty: PenaltyType | None = Field(
        default=PenaltyType.L2,
        description="Regularization penalty",
    )
    C: float = Field(default=1.0, ge=0.0, description="Inverse regularization strength")
    max_iter: int = Field(default=1000, ge=1, description="Maximum iterations")
    tol: float = Field(default=1e-4, ge=0.0, description="Tolerance for stopping criteria")

    # Elastic net specific
    l1_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Elastic net mixing parameter (0=L2, 1=L1)",
    )

    # Class weights
    class_weight: str | dict[str, float] | None = Field(
        default=None,
        description="Class weights ('balanced' or dict)",
    )

    # Random state
    random_state: int | None = Field(default=42, description="Random seed")

    @field_validator("l1_ratio")
    @classmethod
    def validate_l1_ratio(cls, v: float | None, info: Any) -> float | None:
        """Validate l1_ratio is only set with elasticnet penalty."""
        if v is not None and info.data.get("penalty") != PenaltyType.ELASTICNET:
            raise ValueError("l1_ratio can only be set with elasticnet penalty")
        return v

    @field_validator("C")
    @classmethod
    def validate_c(cls, v: float) -> float:
        """Validate C is positive."""
        if v <= 0:
            raise ValueError("C must be positive")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for sklearn compatibility."""
        result = {
            "solver": (self.solver.value if isinstance(self.solver, SolverType) else self.solver),
            "penalty": (
                self.penalty.value if isinstance(self.penalty, PenaltyType) else self.penalty
            ),
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

        if self.l1_ratio is not None:
            result["l1_ratio"] = self.l1_ratio

        if self.class_weight is not None:
            result["class_weight"] = self.class_weight

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitConfig:
        """Create FitConfig from dictionary."""
        # Handle enum conversion
        if "solver" in data and isinstance(data["solver"], str):
            data["solver"] = SolverType(data["solver"])

        if "penalty" in data and data["penalty"] is not None:
            data["penalty"] = PenaltyType(data["penalty"])

        return cls(**data)


# Default configurations for common use cases
DEFAULT_FIT_CONFIG = FitConfig()
BALANCED_FIT_CONFIG = FitConfig(class_weight="balanced")
L1_FIT_CONFIG = FitConfig(penalty=PenaltyType.L1, solver=SolverType.LIBLINEAR)
ELASTICNET_FIT_CONFIG = FitConfig(
    penalty=PenaltyType.ELASTICNET,
    solver=SolverType.SAGA,
    l1_ratio=0.5,
)
