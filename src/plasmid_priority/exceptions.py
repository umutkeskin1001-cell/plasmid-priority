"""Domain-specific exception hierarchy for Plasmid Priority.

All public modules should raise these instead of bare ``Exception`` so that
callers can catch at the appropriate granularity level.

Hierarchy::

    PlasmidPriorityError
    ├── DataError
    │   ├── SchemaValidationError
    │   ├── MissingnessError
    │   └── DataIntegrityError
    ├── ModelError
    │   ├── ModelFitError
    │   ├── CalibrationError
    │   ├── PredictionError
    │   └── EnsembleError
    ├── ConfigError
    │   ├── BranchConfigError
    │   └── FeatureSetError
    ├── PipelineError
    │   ├── StepFailedError
    │   └── WorkflowError
    └── ProvenanceError
"""

from __future__ import annotations


class PlasmidPriorityError(Exception):
    """Base exception for all plasmid-priority domain errors."""

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        self.detail = detail
        full = f"{message}: {detail}" if detail else message
        super().__init__(full)


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------


class DataError(PlasmidPriorityError):
    """Errors related to data loading, validation, or integrity."""


class SchemaValidationError(DataError):
    """A dataframe failed pandera or contract validation."""

    def __init__(
        self,
        message: str = "Schema validation failed",
        *,
        table_name: str = "",
        detail: str = "",
    ) -> None:
        self.table_name = table_name
        if table_name and not detail:
            detail = f"table={table_name}"
        super().__init__(message, detail=detail)


class MissingnessError(DataError):
    """Unacceptable NaN propagation detected in a pipeline table."""


class DataIntegrityError(DataError):
    """A data integrity invariant was violated (e.g. duplicate IDs, wrong sort)."""


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------


class ModelError(PlasmidPriorityError):
    """Errors arising during model fitting, calibration, or prediction."""


class ModelFitError(ModelError):
    """A single model failed to fit within the branch surface."""

    def __init__(
        self,
        message: str = "Model fit failed",
        *,
        model_name: str = "",
        detail: str = "",
    ) -> None:
        self.model_name = model_name
        if model_name and not detail:
            detail = f"model={model_name}"
        super().__init__(message, detail=detail)


class CalibrationError(ModelError):
    """Calibration (platt / isotonic) could not be performed."""


class PredictionError(ModelError):
    """A model produced invalid or unhandleable predictions."""


class EnsembleError(ModelError):
    """An ensemble strategy failed to combine base predictions."""


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------


class ConfigError(PlasmidPriorityError):
    """Errors related to branch or project configuration."""


class BranchConfigError(ConfigError):
    """A branch configuration block is invalid or incomplete."""

    def __init__(
        self,
        message: str = "Branch config error",
        *,
        branch_key: str = "",
        detail: str = "",
    ) -> None:
        self.branch_key = branch_key
        if branch_key and not detail:
            detail = f"branch={branch_key}"
        super().__init__(message, detail=detail)


class FeatureSetError(ConfigError):
    """A required feature set is missing or contains invalid columns."""

    def __init__(
        self,
        message: str = "Feature set error",
        *,
        model_name: str = "",
        detail: str = "",
    ) -> None:
        self.model_name = model_name
        if model_name and not detail:
            detail = f"model={model_name}"
        super().__init__(message, detail=detail)


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------


class PipelineError(PlasmidPriorityError):
    """Errors occurring during pipeline orchestration."""


class StepFailedError(PipelineError):
    """A single pipeline step returned a non-zero exit code."""

    def __init__(
        self,
        message: str = "Pipeline step failed",
        *,
        step_name: str = "",
        exit_code: int = 1,
        detail: str = "",
    ) -> None:
        self.step_name = step_name
        self.exit_code = exit_code
        if step_name and not detail:
            detail = f"step={step_name}, exit_code={exit_code}"
        super().__init__(message, detail=detail)


class WorkflowError(PipelineError):
    """The workflow orchestrator encountered an unrecoverable error."""


# ---------------------------------------------------------------------------
# Provenance errors
# ---------------------------------------------------------------------------


class ProvenanceError(PlasmidPriorityError):
    """Provenance tracking or reproducibility check failed."""
