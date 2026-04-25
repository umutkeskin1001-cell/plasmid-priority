"""Generic Branch ABC for unified branch interface.

This module provides the abstract base class for all branch implementations,
reducing code duplication across geo_spread, bio_transfer, and clinical_hazard branches.

Expected reduction: ~3,000 lines of duplicated code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from plasmid_priority.config import ProjectContext
from plasmid_priority.shared.specs import BranchConfig


@dataclass
class BranchResult:
    """Result of branch execution."""

    predictions: pd.DataFrame
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    report_card: pd.DataFrame
    calibrated_predictions: pd.DataFrame | None = None
    calibration_summary: pd.DataFrame | None = None


class Branch(ABC):
    """Abstract base class for all branch implementations.

    Concrete branches should implement the abstract methods to provide
    branch-specific logic while inheriting shared functionality.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Branch name identifier."""
        ...

    @property
    @abstractmethod
    def feature_sets(self) -> dict[str, list[str]]:
        """Feature sets for different models."""
        ...

    @abstractmethod
    def load_config(self, context: ProjectContext) -> BranchConfig:
        """Load branch-specific configuration."""
        ...

    @abstractmethod
    def build_labels(
        self,
        scored: pd.DataFrame,
        *,
        split_year: int,
        horizon_years: int,
        records: pd.DataFrame | None = None,
        pd_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build branch-specific labels."""
        ...

    @abstractmethod
    def build_features(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Build branch-specific features."""
        ...

    @abstractmethod
    def evaluate_branch(
        self,
        scored: pd.DataFrame,
        *,
        model_names: Sequence[str] | None = None,
        include_research: bool = False,
        include_ablation: bool = False,
        n_splits: int = 5,
        n_repeats: int = 5,
        seed: int = 42,
        n_jobs: int | None = 1,
    ) -> dict[str, Any]:
        """Evaluate branch models."""
        ...

    @abstractmethod
    def build_predictions(
        self,
        scored: pd.DataFrame,
        *,
        model_name: str,
    ) -> pd.DataFrame:
        """Build predictions for a specific model."""
        ...

    @abstractmethod
    def calibrate(
        self,
        predictions: pd.DataFrame,
        *,
        method: str = "isotonic",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calibrate predictions."""
        ...

    @abstractmethod
    def build_report_card(self, results: dict[str, Any]) -> pd.DataFrame:
        """Build report card."""
        ...

    @abstractmethod
    def format_report_markdown(self, results: dict[str, Any]) -> str:
        """Format report as markdown."""
        ...

    @abstractmethod
    def build_provenance(self, context: ProjectContext) -> dict[str, Any]:
        """Build run provenance."""
        ...

    # SHARED METHODS — same across all branches
    def train(
        self,
        scored: pd.DataFrame,
        *,
        model_names: Sequence[str] | None = None,
        include_research: bool = False,
        include_ablation: bool = False,
        n_splits: int = 5,
        n_repeats: int = 5,
        seed: int = 42,
        n_jobs: int | None = 1,
    ) -> dict[str, Any]:
        """Train branch models (delegates to evaluate_branch)."""
        return self.evaluate_branch(
            scored,
            model_names=model_names,
            include_research=include_research,
            include_ablation=include_ablation,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
        )

    def run_full_branch(
        self,
        scored: pd.DataFrame,
        context: ProjectContext,
        *,
        records: pd.DataFrame | None = None,
        pd_metadata: pd.DataFrame | None = None,
        split_year: int,
        horizon_years: int,
        model_names: Sequence[str] | None = None,
        n_splits: int = 5,
        n_repeats: int = 5,
        seed: int = 42,
        n_jobs: int | None = 1,
    ) -> BranchResult:
        """Run complete branch pipeline: labels → features → train → predict → calibrate."""
        # Build labels
        labeled = self.build_labels(
            scored,
            split_year=split_year,
            horizon_years=horizon_years,
            records=records,
            pd_metadata=pd_metadata,
        )

        # Build features
        featured = self.build_features(labeled)

        # Train/evaluate
        evaluation_results = self.evaluate_branch(
            featured,
            model_names=model_names,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
        )

        # Build predictions for primary model
        primary_model = evaluation_results.get("primary_model_name")
        if primary_model:
            predictions = self.build_predictions(featured, model_name=primary_model)
        else:
            predictions = pd.DataFrame()

        # Calibrate
        calibrated, calibration_summary = self.calibrate(predictions)

        # Build report card
        report_card = self.build_report_card(evaluation_results)

        # Build provenance
        provenance = self.build_provenance(context)

        return BranchResult(
            predictions=predictions,
            metrics=evaluation_results,
            provenance=provenance,
            report_card=report_card,
            calibrated_predictions=calibrated,
            calibration_summary=calibration_summary,
        )


# Branch registry
BRANCH_REGISTRY: dict[str, Branch] = {}


def register_branch(branch: Branch) -> None:
    """Register a branch implementation."""
    BRANCH_REGISTRY[branch.name] = branch


def get_branch(name: str) -> Branch:
    """Get a registered branch by name."""
    if name not in BRANCH_REGISTRY:
        raise ValueError(f"Unknown branch: {name}. Available: {list(BRANCH_REGISTRY.keys())}")
    return BRANCH_REGISTRY[name]
