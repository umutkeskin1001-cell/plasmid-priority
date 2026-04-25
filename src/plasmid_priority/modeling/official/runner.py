from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from plasmid_priority.modeling.official.baselines import (
    frozen_biological_prior_score,
    visibility_baseline_score,
)
from plasmid_priority.modeling.official.bounded_tree import fit_bounded_tree_challenger
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
    candidates = (
        DEFAULT_SUPERVISED_FEATURES
        if requested_features is None
        else tuple(requested_features)
    )
    return tuple(column for column in dict.fromkeys(candidates) if column in frame.columns)


def _can_fit_supervised(
    frame: pd.DataFrame,
    label_column: str | None,
    features: Sequence[str],
) -> str:
    if label_column is None or label_column not in frame.columns:
        return "not_fit_label_unavailable"
    if not features:
        return "not_fit_no_available_features"
    labels = pd.to_numeric(frame[label_column], errors="coerce").dropna()
    if not set(labels.astype(int).unique()).issubset({0, 1}):
        return "not_fit_label_not_binary"
    if int(labels.nunique()) != 2:
        return "not_fit_label_single_class"
    return "fit"


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
    output["sparse_calibrated_logistic"] = pd.NA
    output["bounded_monotonic_tree"] = pd.NA

    features = _available_features(frame, supervised_features)
    supervised_status = _can_fit_supervised(frame, label_column, features)
    if supervised_status == "fit" and label_column is not None:
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
    output["official_supervised_model_status"] = supervised_status
    output["official_supervised_feature_count"] = len(features)
    return output
