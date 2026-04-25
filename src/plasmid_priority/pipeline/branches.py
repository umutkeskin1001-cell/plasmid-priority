"""Branch implementations using Generic Branch ABC.

This module provides concrete implementations of the Branch ABC for
geo_spread, bio_transfer, and clinical_hazard branches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from plasmid_priority.config import ProjectContext
from plasmid_priority.geo_spread.calibration import (
    build_geo_spread_calibrated_prediction_table,
    build_geo_spread_calibration_summary,
)
from plasmid_priority.geo_spread.evaluate import (
    build_geo_spread_prediction_table,
    evaluate_geo_spread_branch,
)
from plasmid_priority.geo_spread.features import build_geo_spread_features
from plasmid_priority.geo_spread.inventory import build_geo_spread_inventory
from plasmid_priority.geo_spread.provenance import build_geo_spread_run_provenance
from plasmid_priority.geo_spread.report import (
    build_geo_spread_report_card,
    format_geo_spread_report_markdown,
)
from plasmid_priority.geo_spread.specs import load_geo_spread_config
from plasmid_priority.shared.branch_base import Branch, register_branch
from plasmid_priority.shared.contracts import build_branch_input_contract


class GeoSpreadBranch(Branch):
    """Geo spread branch implementation."""

    @property
    def name(self) -> str:
        return "geo_spread"

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        config = load_geo_spread_config()
        return {k: list(v) for k, v in config.feature_sets.items()}

    def load_config(self, context: ProjectContext) -> Any:
        return load_geo_spread_config()

    def build_labels(
        self,
        scored: pd.DataFrame,
        *,
        split_year: int,
        horizon_years: int,
        records: pd.DataFrame | None = None,
        pd_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build geo spread labels using enrichment."""
        from plasmid_priority.geo_spread.enrichment import enrich_geo_spread_scored_table

        return enrich_geo_spread_scored_table(
            scored,
            split_year=split_year,
            records=records,
        )

    def build_features(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Build geo spread features."""
        return build_geo_spread_features(scored)

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
        """Evaluate geo spread branch models."""
        config = load_geo_spread_config()
        contract = build_branch_input_contract()

        results = evaluate_geo_spread_branch(
            scored,
            config=config,
            contract=contract,
            model_names=model_names,
            include_research=include_research,
            include_ablation=include_ablation,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
        )
        return results

    def build_predictions(
        self,
        scored: pd.DataFrame,
        *,
        model_name: str,
    ) -> pd.DataFrame:
        """Build predictions for geo spread model."""
        config = load_geo_spread_config()
        contract = build_branch_input_contract()

        return build_geo_spread_prediction_table(
            scored,
            model_name=model_name,
            config=config,
            contract=contract,
        )

    def calibrate(
        self,
        predictions: pd.DataFrame,
        *,
        method: str = "isotonic",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calibrate geo spread predictions."""
        calibrated = build_geo_spread_calibrated_prediction_table(predictions)
        summary = build_geo_spread_calibration_summary(predictions)
        return calibrated, summary

    def build_report_card(self, results: dict[str, Any]) -> pd.DataFrame:
        """Build geo spread report card."""
        return build_geo_spread_report_card(results)

    def format_report_markdown(self, results: dict[str, Any]) -> str:
        """Format geo spread report as markdown."""
        return format_geo_spread_report_markdown(results)

    def build_provenance(self, context: ProjectContext) -> dict[str, Any]:
        """Build geo spread run provenance."""
        inventory = build_geo_spread_inventory(context)
        return build_geo_spread_run_provenance(context, inventory)


class BioTransferBranch(Branch):
    """Bio transfer branch implementation."""

    @property
    def name(self) -> str:
        return "bio_transfer"

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        from plasmid_priority.bio_transfer.specs import load_bio_transfer_config

        config = load_bio_transfer_config()
        return {k: list(v) for k, v in config.feature_sets.items()}

    def load_config(self, context: ProjectContext) -> Any:
        from plasmid_priority.bio_transfer.specs import load_bio_transfer_config

        return load_bio_transfer_config()

    def build_labels(
        self,
        scored: pd.DataFrame,
        *,
        split_year: int,
        horizon_years: int,
        records: pd.DataFrame | None = None,
        pd_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build bio transfer labels."""
        from plasmid_priority.bio_transfer.dataset import prepare_bio_transfer_scored_table

        config = self.load_config(None)
        return prepare_bio_transfer_scored_table(
            scored,
            config=config,
            records=records,
            pd_metadata=pd_metadata,
        )

    def build_features(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Build bio transfer features."""
        from plasmid_priority.bio_transfer.features import build_bio_transfer_features

        return build_bio_transfer_features(scored)

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
        """Evaluate bio transfer branch models."""
        from plasmid_priority.bio_transfer.evaluate import evaluate_bio_transfer_branch
        from plasmid_priority.bio_transfer.specs import load_bio_transfer_config
        from plasmid_priority.shared.contracts import build_branch_input_contract

        config = load_bio_transfer_config()
        contract = build_branch_input_contract()

        results = evaluate_bio_transfer_branch(
            scored,
            config=config,
            contract=contract,
            model_names=model_names,
            include_research=include_research,
            include_ablation=include_ablation,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
        )
        return results

    def build_predictions(
        self,
        scored: pd.DataFrame,
        *,
        model_name: str,
    ) -> pd.DataFrame:
        """Build predictions for bio transfer model."""
        from plasmid_priority.bio_transfer.evaluate import build_bio_transfer_prediction_table
        from plasmid_priority.bio_transfer.specs import load_bio_transfer_config
        from plasmid_priority.shared.contracts import build_branch_input_contract

        config = load_bio_transfer_config()
        contract = build_branch_input_contract()

        return build_bio_transfer_prediction_table(
            scored,
            model_name=model_name,
            config=config,
            contract=contract,
        )

    def calibrate(
        self,
        predictions: pd.DataFrame,
        *,
        method: str = "isotonic",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calibrate bio transfer predictions."""
        from plasmid_priority.bio_transfer.calibration import (
            build_bio_transfer_calibrated_prediction_table,
            build_bio_transfer_calibration_summary,
        )

        calibrated = build_bio_transfer_calibrated_prediction_table(predictions)
        summary = build_bio_transfer_calibration_summary(predictions)
        return calibrated, summary

    def build_report_card(self, results: dict[str, Any]) -> pd.DataFrame:
        """Build bio transfer report card."""
        from plasmid_priority.bio_transfer.report import build_bio_transfer_report_card

        return build_bio_transfer_report_card(results)

    def format_report_markdown(self, results: dict[str, Any]) -> str:
        """Format bio transfer report as markdown."""
        from plasmid_priority.bio_transfer.report import format_bio_transfer_report_markdown

        return format_bio_transfer_report_markdown(results)

    def build_provenance(self, context: ProjectContext) -> dict[str, Any]:
        """Build bio transfer run provenance."""
        from plasmid_priority.bio_transfer.provenance import build_bio_transfer_run_provenance
        from plasmid_priority.shared.data_inventory import build_branch_inventory

        inventory = build_branch_inventory(context, branch_name="bio_transfer")
        return build_bio_transfer_run_provenance(context, inventory)


class ClinicalHazardBranch(Branch):
    """Clinical hazard branch implementation."""

    @property
    def name(self) -> str:
        return "clinical_hazard"

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        from plasmid_priority.clinical_hazard.specs import load_clinical_hazard_config

        config = load_clinical_hazard_config()
        return {k: list(v) for k, v in config.feature_sets.items()}

    def load_config(self, context: ProjectContext) -> Any:
        from plasmid_priority.clinical_hazard.specs import load_clinical_hazard_config

        return load_clinical_hazard_config()

    def build_labels(
        self,
        scored: pd.DataFrame,
        *,
        split_year: int,
        horizon_years: int,
        records: pd.DataFrame | None = None,
        pd_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build clinical hazard labels."""
        from plasmid_priority.clinical_hazard.dataset import prepare_clinical_hazard_scored_table

        config = self.load_config(None)
        return prepare_clinical_hazard_scored_table(
            scored,
            config=config,
            records=records,
            pd_metadata=pd_metadata,
        )

    def build_features(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Build clinical hazard features."""
        from plasmid_priority.clinical_hazard.features import build_clinical_hazard_features

        return build_clinical_hazard_features(scored)

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
        """Evaluate clinical hazard branch models."""
        from plasmid_priority.clinical_hazard.evaluate import evaluate_clinical_hazard_branch
        from plasmid_priority.clinical_hazard.specs import load_clinical_hazard_config
        from plasmid_priority.shared.contracts import build_branch_input_contract

        config = load_clinical_hazard_config()
        contract = build_branch_input_contract()

        results = evaluate_clinical_hazard_branch(
            scored,
            config=config,
            contract=contract,
            model_names=model_names,
            include_research=include_research,
            include_ablation=include_ablation,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            n_jobs=n_jobs,
        )
        return results

    def build_predictions(
        self,
        scored: pd.DataFrame,
        *,
        model_name: str,
    ) -> pd.DataFrame:
        """Build predictions for clinical hazard model."""
        from plasmid_priority.clinical_hazard.evaluate import build_clinical_hazard_prediction_table
        from plasmid_priority.clinical_hazard.specs import load_clinical_hazard_config
        from plasmid_priority.shared.contracts import build_branch_input_contract

        config = load_clinical_hazard_config()
        contract = build_branch_input_contract()

        return build_clinical_hazard_prediction_table(
            scored,
            model_name=model_name,
            config=config,
            contract=contract,
        )

    def calibrate(
        self,
        predictions: pd.DataFrame,
        *,
        method: str = "isotonic",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calibrate clinical hazard predictions."""
        from plasmid_priority.clinical_hazard.calibration import (
            build_clinical_hazard_calibrated_prediction_table,
            build_clinical_hazard_calibration_summary,
        )

        calibrated = build_clinical_hazard_calibrated_prediction_table(predictions)
        summary = build_clinical_hazard_calibration_summary(predictions)
        return calibrated, summary

    def build_report_card(self, results: dict[str, Any]) -> pd.DataFrame:
        """Build clinical hazard report card."""
        from plasmid_priority.clinical_hazard.report import build_clinical_hazard_report_card

        return build_clinical_hazard_report_card(results)

    def format_report_markdown(self, results: dict[str, Any]) -> str:
        """Format clinical hazard report as markdown."""
        from plasmid_priority.clinical_hazard.report import format_clinical_hazard_report_markdown

        return format_clinical_hazard_report_markdown(results)

    def build_provenance(self, context: ProjectContext) -> dict[str, Any]:
        """Build clinical hazard run provenance."""
        from plasmid_priority.clinical_hazard.provenance import build_clinical_hazard_run_provenance
        from plasmid_priority.shared.data_inventory import build_branch_inventory

        inventory = build_branch_inventory(context, branch_name="clinical_hazard")
        return build_clinical_hazard_run_provenance(context, inventory)


# Register branches
register_branch(GeoSpreadBranch())
register_branch(BioTransferBranch())
register_branch(ClinicalHazardBranch())
