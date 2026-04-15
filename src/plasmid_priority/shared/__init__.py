"""Shared branch utilities for split-safe retrospective prediction surfaces."""

from plasmid_priority.shared.calibration import (
    BranchCalibrationResult,
    build_branch_calibrated_prediction_table,
    build_branch_calibration_summary,
    calibrate_branch_predictions,
)
from plasmid_priority.shared.contracts import (
    BranchInputContract,
    build_branch_input_contract,
    ensure_branch_label_alias,
    validate_branch_feature_set,
    validate_branch_input_contract,
)
from plasmid_priority.shared.data_inventory import build_branch_inventory
from plasmid_priority.shared.explanations import (
    build_branch_explanation_table,
    build_branch_review_reason_table,
)
from plasmid_priority.shared.labels import (
    build_bio_transfer_labels,
    build_clinical_hazard_labels,
    build_geo_spread_labels,
)
from plasmid_priority.shared.provenance import (
    BranchRunProvenance,
    build_branch_run_provenance,
    content_hash,
    dataframe_content_hash,
    stable_json_dumps,
)
from plasmid_priority.shared.selection import (
    build_branch_selection_scorecard,
    select_branch_primary_model,
)
from plasmid_priority.shared.specs import (
    BranchBenchmarkSpec,
    BranchConfig,
    BranchFitConfig,
    BranchModelSelectionSpec,
    load_branch_config,
    resolve_branch_fit_config,
    resolve_branch_model_names,
)
from plasmid_priority.shared.temporal import (
    future_window_mask,
    pre_split_mask,
    resolve_window_bounds,
    split_year_window_mask,
)

__all__ = [
    "BranchBenchmarkSpec",
    "BranchCalibrationResult",
    "BranchConfig",
    "BranchFitConfig",
    "BranchInputContract",
    "BranchModelSelectionSpec",
    "BranchRunProvenance",
    "build_branch_calibrated_prediction_table",
    "build_branch_calibration_summary",
    "build_branch_explanation_table",
    "build_branch_input_contract",
    "build_branch_inventory",
    "build_branch_selection_scorecard",
    "build_branch_review_reason_table",
    "build_branch_run_provenance",
    "build_bio_transfer_labels",
    "build_clinical_hazard_labels",
    "build_geo_spread_labels",
    "calibrate_branch_predictions",
    "content_hash",
    "dataframe_content_hash",
    "ensure_branch_label_alias",
    "future_window_mask",
    "load_branch_config",
    "pre_split_mask",
    "resolve_branch_fit_config",
    "resolve_branch_model_names",
    "resolve_window_bounds",
    "select_branch_primary_model",
    "split_year_window_mask",
    "stable_json_dumps",
    "validate_branch_feature_set",
    "validate_branch_input_contract",
]
