"""Clinical hazard branch contracts, specs, and helpers."""

from plasmid_priority.clinical_hazard.calibration import (
    ClinicalHazardCalibrationResult,
    build_clinical_hazard_calibrated_prediction_table,
    build_clinical_hazard_calibration_summary,
    calibrate_clinical_hazard_predictions,
)
from plasmid_priority.clinical_hazard.contracts import (
    ClinicalHazardInputContract,
    build_clinical_hazard_input_contract,
    validate_clinical_hazard_input_contract,
)
from plasmid_priority.clinical_hazard.dataset import (
    ClinicalHazardDataset,
    build_clinical_hazard_dataset_from_prepared,
    prepare_clinical_hazard_dataset,
    prepare_clinical_hazard_scored_table,
    resolve_clinical_hazard_dataset_model_names,
)
from plasmid_priority.clinical_hazard.evaluate import (
    build_clinical_hazard_model_summary,
    build_clinical_hazard_prediction_table,
    evaluate_clinical_hazard_branch,
)
from plasmid_priority.clinical_hazard.features import (
    CLINICAL_HAZARD_ALLOWED_FEATURES,
    CLINICAL_HAZARD_FEATURE_CATEGORIES,
    build_clinical_hazard_features,
    classify_clinical_hazard_feature,
    validate_clinical_hazard_feature_set,
)
from plasmid_priority.clinical_hazard.provenance import build_clinical_hazard_run_provenance
from plasmid_priority.clinical_hazard.report import (
    build_clinical_hazard_report_card,
    format_clinical_hazard_report_markdown,
)
from plasmid_priority.clinical_hazard.specs import (
    CLINICAL_HAZARD_BENCHMARK_NAME,
    CLINICAL_HAZARD_CONSERVATIVE_MODEL_NAME,
    CLINICAL_HAZARD_CORE_MODEL_NAMES,
    CLINICAL_HAZARD_FEATURE_SETS,
    CLINICAL_HAZARD_GOVERNANCE_MODEL_NAME,
    CLINICAL_HAZARD_PRIMARY_MODEL_NAME,
    CLINICAL_HAZARD_RESEARCH_MODEL_NAME,
    ClinicalHazardBenchmarkSpec,
    ClinicalHazardConfig,
    ClinicalHazardFitConfig,
    load_clinical_hazard_config,
    resolve_clinical_hazard_fit_config,
    resolve_clinical_hazard_model_names,
)
from plasmid_priority.clinical_hazard.train import (
    fit_clinical_hazard_branch,
    fit_clinical_hazard_model,
    fit_clinical_hazard_model_predictions,
)

__all__ = [
    "CLINICAL_HAZARD_ALLOWED_FEATURES",
    "CLINICAL_HAZARD_BENCHMARK_NAME",
    "CLINICAL_HAZARD_CONSERVATIVE_MODEL_NAME",
    "CLINICAL_HAZARD_CORE_MODEL_NAMES",
    "CLINICAL_HAZARD_FEATURE_CATEGORIES",
    "CLINICAL_HAZARD_FEATURE_SETS",
    "CLINICAL_HAZARD_GOVERNANCE_MODEL_NAME",
    "CLINICAL_HAZARD_PRIMARY_MODEL_NAME",
    "CLINICAL_HAZARD_RESEARCH_MODEL_NAME",
    "ClinicalHazardBenchmarkSpec",
    "ClinicalHazardCalibrationResult",
    "ClinicalHazardConfig",
    "ClinicalHazardDataset",
    "ClinicalHazardFitConfig",
    "ClinicalHazardInputContract",
    "build_clinical_hazard_calibrated_prediction_table",
    "build_clinical_hazard_calibration_summary",
    "build_clinical_hazard_dataset_from_prepared",
    "build_clinical_hazard_features",
    "build_clinical_hazard_input_contract",
    "build_clinical_hazard_model_summary",
    "build_clinical_hazard_prediction_table",
    "build_clinical_hazard_report_card",
    "build_clinical_hazard_run_provenance",
    "calibrate_clinical_hazard_predictions",
    "classify_clinical_hazard_feature",
    "evaluate_clinical_hazard_branch",
    "fit_clinical_hazard_branch",
    "fit_clinical_hazard_model",
    "fit_clinical_hazard_model_predictions",
    "format_clinical_hazard_report_markdown",
    "load_clinical_hazard_config",
    "prepare_clinical_hazard_dataset",
    "prepare_clinical_hazard_scored_table",
    "resolve_clinical_hazard_dataset_model_names",
    "resolve_clinical_hazard_fit_config",
    "resolve_clinical_hazard_model_names",
    "validate_clinical_hazard_feature_set",
    "validate_clinical_hazard_input_contract",
]
