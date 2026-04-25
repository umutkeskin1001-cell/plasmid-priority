from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_BIOLOGICAL_PRIOR_WEIGHTS: dict[str, float] = {
    "mobility": 0.32,
    "host": 0.24,
    "amr": 0.28,
    "evidence": 0.16,
}


def _numeric_column(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def _first_available_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    default: float = 0.0,
) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return _numeric_column(frame, column, default=default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _clip_unit(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.clip(values.to_numpy(dtype=float), 0.0, 1.0),
        index=values.index,
        dtype="float64",
    )


def _minmax_unit(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float64")
    if numeric.empty:
        return pd.Series(dtype="float64")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    scaled = (numeric - minimum) / (maximum - minimum)
    return _clip_unit(scaled)


def visibility_baseline_score(
    frame: pd.DataFrame,
    *,
    member_count_column: str = "log1p_member_count_train",
    country_count_column: str = "log1p_n_countries_train",
) -> pd.Series:
    """Score only training-time visibility proxies, never biological evidence."""
    member_visibility = _numeric_column(frame, member_count_column)
    country_visibility = _numeric_column(frame, country_count_column)
    raw_score = (0.6 * member_visibility) + (0.4 * country_visibility)
    return _minmax_unit(raw_score)


def frozen_biological_prior_score(
    frame: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Conservative predeclared biological prior over mobility, host, AMR, and evidence support."""
    selected_weights = dict(DEFAULT_BIOLOGICAL_PRIOR_WEIGHTS if weights is None else weights)
    total_weight = float(sum(max(0.0, float(value)) for value in selected_weights.values()))
    if total_weight <= 0.0:
        raise ValueError("At least one biological prior weight must be positive")
    normalized_weights = {
        key: max(0.0, float(value)) / total_weight for key, value in selected_weights.items()
    }

    mobility = _clip_unit(
        _first_available_numeric(frame, ["T_eff_norm", "mobility_support", "orit_support_norm"]),
    )
    host = _clip_unit(
        _first_available_numeric(
            frame,
            ["H_eff_norm", "H_obs_specialization_norm", "host_range_support"],
        ),
    )
    amr = _clip_unit(
        _first_available_numeric(
            frame,
            ["A_eff_norm", "amr_support_norm", "amr_class_richness_norm"],
        ),
    )
    evidence = _clip_unit(
        _first_available_numeric(
            frame,
            ["evidence_support_index", "support_depth_norm", "metadata_support_depth_norm"],
        ),
    )

    score = (
        normalized_weights.get("mobility", 0.0) * mobility
        + normalized_weights.get("host", 0.0) * host
        + normalized_weights.get("amr", 0.0) * amr
        + normalized_weights.get("evidence", 0.0) * evidence
    )
    return _clip_unit(score)
