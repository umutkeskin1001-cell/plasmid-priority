from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from plasmid_priority.modeling.official.baselines import (
    frozen_biological_prior_score,
    visibility_baseline_score,
)
from plasmid_priority.modeling.official.bounded_tree import fit_bounded_tree_challenger
from plasmid_priority.modeling.official.common import (
    DEFAULT_MIN_CLASS_COUNT,
    DEFAULT_MIN_LABELED_ROWS,
    OfficialSupervisedReadiness,
    assess_supervised_readiness,
    dedupe_feature_columns,
)
from plasmid_priority.modeling.official.consensus import conservative_evidence_consensus
from plasmid_priority.modeling.official.sparse_logistic import fit_sparse_calibrated_logistic

DEFAULT_SUPERVISED_FEATURES: tuple[str, ...] = (
    "log1p_member_count_train",
    "log1p_n_countries_train",
    "T_eff_norm",
    "H_obs_specialization_norm",
    "A_eff_norm",
    "evidence_support_index",
)


def _available_features(
    frame: pd.DataFrame,
    requested_features: Sequence[str] | None,
) -> tuple[str, ...]:
    candidates = dedupe_feature_columns(
        DEFAULT_SUPERVISED_FEATURES if requested_features is None else requested_features,
    )
    return tuple(column for column in dict.fromkeys(candidates) if column in frame.columns)


def _requested_features(requested_features: Sequence[str] | None) -> tuple[str, ...]:
    return dedupe_feature_columns(
        DEFAULT_SUPERVISED_FEATURES if requested_features is None else requested_features,
    )


def _append_supervised_metadata(
    output: pd.DataFrame,
    *,
    readiness: OfficialSupervisedReadiness,
    fitted_model_columns: Sequence[str],
    consensus_score_columns: Sequence[str],
) -> pd.DataFrame:
    metadata = {
        "official_supervised_model_status": readiness.status,
        "official_supervised_feature_count": readiness.available_feature_count,
        "official_supervised_requested_feature_count": readiness.requested_feature_count,
        "official_supervised_missing_feature_count": readiness.missing_feature_count,
        "official_supervised_requested_features": ",".join(readiness.requested_features),
        "official_supervised_available_features": ",".join(readiness.available_features),
        "official_supervised_missing_features": ",".join(readiness.missing_features),
        "official_supervised_labeled_count": readiness.labeled_count,
        "official_supervised_positive_count": readiness.positive_count,
        "official_supervised_negative_count": readiness.negative_count,
        "official_supervised_min_labeled_rows": readiness.min_labeled_rows,
        "official_supervised_min_class_count": readiness.min_class_count,
        "official_supervised_models_used": ",".join(str(column) for column in fitted_model_columns),
        "official_consensus_score_columns": ",".join(
            str(column) for column in consensus_score_columns
        ),
    }
    for column, value in metadata.items():
        output[column] = pd.Series([value] * len(output.index), index=output.index)
    return output


def score_official_model_family(
    frame: pd.DataFrame,
    *,
    label_column: str | None = None,
    supervised_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run the compact official family and return auditable bounded score columns."""
    output = pd.DataFrame(index=frame.index)
    output["visibility_baseline"] = visibility_baseline_score(frame)
    output["frozen_biological_prior"] = frozen_biological_prior_score(frame)
    output["sparse_calibrated_logistic"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    output["bounded_monotonic_tree"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")

    requested_features = _requested_features(supervised_features)
    readiness = assess_supervised_readiness(
        frame,
        label_column=label_column,
        requested_features=requested_features,
        min_labeled_rows=DEFAULT_MIN_LABELED_ROWS,
        min_class_count=DEFAULT_MIN_CLASS_COUNT,
    )
    features = readiness.available_features
    fitted_model_columns: list[str] = []
    if readiness.status == "fit" and label_column is not None:
        sparse_model = fit_sparse_calibrated_logistic(
            frame,
            label_column=label_column,
            feature_columns=features,
        )
        tree_model = fit_bounded_tree_challenger(
            frame,
            label_column=label_column,
            feature_columns=features,
        )
        output["sparse_calibrated_logistic"] = sparse_model.predict_proba(frame)
        output["bounded_monotonic_tree"] = tree_model.predict_proba(frame)
        fitted_model_columns.extend(
            ["sparse_calibrated_logistic", "bounded_monotonic_tree"],
        )

    consensus_score_columns = [
        column
        for column in (
            "visibility_baseline",
            "frozen_biological_prior",
            "sparse_calibrated_logistic",
            "bounded_monotonic_tree",
        )
        if pd.to_numeric(output[column], errors="coerce").notna().any()
    ]
    consensus_input = pd.concat(
        [
            output,
            frame.loc[
                :,
                [
                    column
                    for column in ("evidence_tier", "uncertainty_tier")
                    if column in frame.columns
                ],
            ],
        ],
        axis=1,
    )
    consensus = conservative_evidence_consensus(
        consensus_input,
        score_columns=consensus_score_columns,
    )
    output = pd.concat([output, consensus], axis=1)
    return _append_supervised_metadata(
        output,
        readiness=readiness,
        fitted_model_columns=fitted_model_columns,
        consensus_score_columns=consensus_score_columns,
    )
