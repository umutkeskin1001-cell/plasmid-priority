"""Consensus branch helpers."""

from plasmid_priority.consensus.calibration import (  # type: ignore[attr-defined]
    BranchCalibrationResult,
    build_consensus_calibrated_prediction_table,
    build_consensus_calibration_summary,
    calibrate_consensus_predictions,
)
from plasmid_priority.consensus.dataset import (
    ConsensusDataset,
    prepare_consensus_dataset,
)
from plasmid_priority.consensus.evaluate import (
    build_consensus_model_summary,
    build_consensus_prediction_table,
    evaluate_consensus_branch,
)
from plasmid_priority.consensus.fuse import (
    build_operational_consensus_frame,
    build_research_consensus_frame,
    merge_branch_predictions,
)
from plasmid_priority.consensus.provenance import build_consensus_run_provenance
from plasmid_priority.consensus.report import (
    build_consensus_report_card,
    format_consensus_report_markdown,
)
from plasmid_priority.consensus.specs import (
    CONSENSUS_BENCHMARK_NAME,
    CONSENSUS_CONSERVATIVE_MODEL_NAME,
    CONSENSUS_CORE_MODEL_NAMES,
    CONSENSUS_FEATURE_SETS,
    CONSENSUS_GOVERNANCE_MODEL_NAME,
    CONSENSUS_PRIMARY_MODEL_NAME,
    CONSENSUS_RESEARCH_MODEL_NAME,
    ConsensusBenchmarkSpec,
    ConsensusConfig,
    ConsensusFitConfig,
    load_consensus_config,
    resolve_consensus_fit_config,
    resolve_consensus_model_names,
)

__all__ = [
    "CONSENSUS_BENCHMARK_NAME",
    "CONSENSUS_CONSERVATIVE_MODEL_NAME",
    "CONSENSUS_CORE_MODEL_NAMES",
    "CONSENSUS_FEATURE_SETS",
    "CONSENSUS_GOVERNANCE_MODEL_NAME",
    "CONSENSUS_PRIMARY_MODEL_NAME",
    "CONSENSUS_RESEARCH_MODEL_NAME",
    "ConsensusBenchmarkSpec",
    "BranchCalibrationResult",
    "ConsensusConfig",
    "ConsensusDataset",
    "ConsensusFitConfig",
    "build_consensus_calibrated_prediction_table",
    "build_consensus_calibration_summary",
    "build_consensus_model_summary",
    "build_consensus_prediction_table",
    "build_consensus_report_card",
    "build_consensus_run_provenance",
    "calibrate_consensus_predictions",
    "build_operational_consensus_frame",
    "build_research_consensus_frame",
    "evaluate_consensus_branch",
    "format_consensus_report_markdown",
    "load_consensus_config",
    "merge_branch_predictions",
    "prepare_consensus_dataset",
    "resolve_consensus_fit_config",
    "resolve_consensus_model_names",
]
