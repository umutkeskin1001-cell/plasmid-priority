"""Geo spread branch contracts, specs, and helpers."""

from plasmid_priority.geo_spread.calibration import (
    GeoSpreadCalibrationResult,
    build_geo_spread_calibrated_prediction_table,
    build_geo_spread_calibration_summary,
    calibrate_geo_spread_predictions,
)
from plasmid_priority.geo_spread.contracts import (
    GeoSpreadInputContract,
    build_geo_spread_input_contract,
    validate_geo_spread_input_contract,
)
from plasmid_priority.geo_spread.dataset import (
    GeoSpreadDataset,
    build_geo_spread_dataset_from_prepared,
    prepare_geo_spread_dataset,
    prepare_geo_spread_scored_table,
    resolve_geo_spread_dataset_model_names,
)
from plasmid_priority.geo_spread.enrichment import (
    build_geo_spread_context_features,
    enrich_geo_spread_scored_table,
    load_geo_spread_records,
)
from plasmid_priority.geo_spread.evaluate import (
    build_geo_spread_model_summary,
    build_geo_spread_prediction_table,
    evaluate_geo_spread_branch,
)
from plasmid_priority.geo_spread.features import (
    GEO_SPREAD_ALLOWED_FEATURES,
    GEO_SPREAD_FEATURE_CATEGORIES,
    classify_geo_spread_feature,
    validate_geo_spread_feature_set,
)
from plasmid_priority.geo_spread.inventory import build_geo_spread_inventory
from plasmid_priority.geo_spread.provenance import (
    GeoSpreadRunProvenance,
    build_geo_spread_run_provenance,
    content_hash,
    dataframe_content_hash,
    stable_json_dumps,
)
from plasmid_priority.geo_spread.report import (
    build_geo_spread_report_card,
    format_geo_spread_report_markdown,
)
from plasmid_priority.geo_spread.select import (
    GEO_SPREAD_ADAPTIVE_PRIORITY,
    GEO_SPREAD_RELIABILITY_BLEND,
    build_geo_spread_adaptive_result,
    build_geo_spread_blended_result,
    build_geo_spread_meta_result,
    build_geo_spread_selection_scorecard,
    select_geo_spread_primary_model,
)
from plasmid_priority.geo_spread.specs import (
    DEFAULT_GEO_SPREAD_CORE_MODEL_NAMES,
    DEFAULT_GEO_SPREAD_FEATURE_SETS,
    DEFAULT_GEO_SPREAD_RESEARCH_MODEL_NAMES,
    GEO_SPREAD_CONTEXT_HYBRID_PRIORITY,
    GEO_SPREAD_DERIVED_MODEL_NAMES,
    GEO_SPREAD_META_PRIORITY,
    GeoSpreadBenchmarkSpec,
    GeoSpreadConfig,
    GeoSpreadFitConfig,
    GeoSpreadModelSelectionSpec,
    load_geo_spread_config,
    resolve_geo_spread_fit_config,
    resolve_geo_spread_model_names,
)
from plasmid_priority.geo_spread.train import (
    fit_geo_spread_branch,
    fit_geo_spread_model,
    fit_geo_spread_model_predictions,
)

__all__ = [
    "DEFAULT_GEO_SPREAD_CORE_MODEL_NAMES",
    "DEFAULT_GEO_SPREAD_FEATURE_SETS",
    "DEFAULT_GEO_SPREAD_RESEARCH_MODEL_NAMES",
    "GEO_SPREAD_ADAPTIVE_PRIORITY",
    "GEO_SPREAD_DERIVED_MODEL_NAMES",
    "GEO_SPREAD_CONTEXT_HYBRID_PRIORITY",
    "GEO_SPREAD_META_PRIORITY",
    "GEO_SPREAD_RELIABILITY_BLEND",
    "GEO_SPREAD_ALLOWED_FEATURES",
    "GEO_SPREAD_FEATURE_CATEGORIES",
    "GeoSpreadBenchmarkSpec",
    "GeoSpreadConfig",
    "GeoSpreadFitConfig",
    "GeoSpreadInputContract",
    "GeoSpreadDataset",
    "GeoSpreadModelSelectionSpec",
    "build_geo_spread_dataset_from_prepared",
    "build_geo_spread_model_summary",
    "build_geo_spread_prediction_table",
    "build_geo_spread_calibrated_prediction_table",
    "build_geo_spread_calibration_summary",
    "build_geo_spread_adaptive_result",
    "build_geo_spread_blended_result",
    "build_geo_spread_meta_result",
    "build_geo_spread_inventory",
    "build_geo_spread_run_provenance",
    "build_geo_spread_report_card",
    "build_geo_spread_selection_scorecard",
    "build_geo_spread_input_contract",
    "calibrate_geo_spread_predictions",
    "content_hash",
    "dataframe_content_hash",
    "evaluate_geo_spread_branch",
    "build_geo_spread_context_features",
    "classify_geo_spread_feature",
    "GeoSpreadCalibrationResult",
    "GeoSpreadRunProvenance",
    "fit_geo_spread_branch",
    "fit_geo_spread_model",
    "fit_geo_spread_model_predictions",
    "format_geo_spread_report_markdown",
    "enrich_geo_spread_scored_table",
    "load_geo_spread_records",
    "prepare_geo_spread_dataset",
    "prepare_geo_spread_scored_table",
    "load_geo_spread_config",
    "resolve_geo_spread_fit_config",
    "resolve_geo_spread_model_names",
    "resolve_geo_spread_dataset_model_names",
    "select_geo_spread_primary_model",
    "validate_geo_spread_feature_set",
    "validate_geo_spread_input_contract",
    "stable_json_dumps",
]
