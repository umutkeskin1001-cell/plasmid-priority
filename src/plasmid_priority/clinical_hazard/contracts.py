"""Branch-level contract checks for the clinical hazard task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plasmid_priority.clinical_hazard.specs import (
    ClinicalHazardBenchmarkSpec,
    load_clinical_hazard_config,
)
from plasmid_priority.shared.contracts import (
    build_branch_input_contract,
    validate_branch_input_contract,
)


class ClinicalHazardInputContract(BaseModel):
    """Structural contract for clinical hazard backbone tables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: ClinicalHazardBenchmarkSpec = Field(default_factory=ClinicalHazardBenchmarkSpec)
    require_temporal_metadata: bool = True
    require_assignment_mode: bool = True
    require_label_consistency: bool = False

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.benchmark.required_columns


def build_clinical_hazard_input_contract(
    config: Mapping[str, Any] | None = None,
    *,
    split_year: int | None = None,
) -> ClinicalHazardInputContract:
    clinical_config = load_clinical_hazard_config(config)
    benchmark = clinical_config.benchmark
    if split_year is not None and int(split_year) != int(benchmark.split_year):
        benchmark = benchmark.model_copy(update={"split_year": int(split_year)})
    return ClinicalHazardInputContract(benchmark=benchmark)


def validate_clinical_hazard_input_contract(
    scored: Any,
    *,
    contract: ClinicalHazardInputContract | None = None,
    label: str = "clinical hazard input",
) -> None:
    contract = contract or build_clinical_hazard_input_contract()
    branch_contract = build_branch_input_contract(contract.benchmark)
    branch_contract = branch_contract.model_copy(update={"require_label_consistency": False})
    validate_branch_input_contract(scored, contract=branch_contract, label=label)
