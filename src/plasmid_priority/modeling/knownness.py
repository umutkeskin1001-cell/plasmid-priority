"""Knownness scoring and metadata annotation utilities.

This module provides functions for computing knownness scores and annotating
dataframes with knownness-related metadata. Extracted from module_a.py to
reduce circular import dependencies and improve code organization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _masked_percentile_rank(
    values: pd.Series,
    *,
    cohort_mask: pd.Series | None = None,
) -> pd.Series:
    """Compute percentile rank within a cohort mask."""
    if cohort_mask is None:
        cohort_mask = pd.Series(True, index=values.index, dtype=bool)

    ranks = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    valid_mask = cohort_mask.fillna(False).astype(bool) & numeric.notna()

    if not valid_mask.any():
        return ranks

    valid_values = numeric.loc[valid_mask]
    if valid_values.nunique() < 2:
        ranks.loc[valid_mask] = 0.5
        return ranks

    ranks.loc[valid_mask] = valid_values.rank(pct=True)
    return ranks


def _knownness_score_series(
    frame: pd.DataFrame,
    *,
    cohort_mask: pd.Series | None = None,
) -> pd.Series:
    """Compute knownness score from member count, country count, and refseq share."""
    member_rank = _masked_percentile_rank(
        pd.Series(frame.get("log1p_member_count_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    country_rank = _masked_percentile_rank(
        pd.Series(frame.get("log1p_n_countries_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    source_rank = _masked_percentile_rank(
        pd.Series(frame.get("refseq_share_train", 0.0), index=frame.index),
        cohort_mask=cohort_mask,
    )
    return (member_rank + country_rank + source_rank) / 3.0


def _stable_quantile_labels(
    values: pd.Series,
    *,
    q: int,
    label_prefix: str = "q",
) -> tuple[pd.Series, int]:
    """Create stable quantile labels."""
    labels: pd.Series = pd.Series(np.nan, index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    valid_positions = np.flatnonzero(numeric.notna().to_numpy())
    valid = numeric.loc[numeric.notna()]
    if valid.empty or valid.nunique() < 2:
        return labels, 0

    try:
        labels.iloc[valid_positions] = pd.qcut(valid, q=q, labels=False, duplicates="drop") + 1
        labels = labels.astype("string").str.replace(r"(\d+)", rf"{label_prefix}\1", regex=True)
        n_bins = int(labels.dropna().nunique())
    except ValueError:
        n_bins = 0

    return labels, n_bins


def annotate_knownness_metadata(scored: pd.DataFrame) -> pd.DataFrame:
    """Add training-visibility proxy metadata used in novelty and bias audits.

    Args:
        scored: Input dataframe with scored features

    Returns:
        Dataframe with additional knownness metadata columns:
        - member_rank_norm, country_rank_norm, source_rank_norm
        - knownness_score
        - knownness_half (lower_half/upper_half/out_of_scope)
        - knownness_quartile (q1_lowest/q2/q3/q4_highest)
        - knownness_quartile_supported (boolean)
        - member_count_band, country_count_band, source_band
    """
    working = scored.copy()
    for column in ("log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"):
        if column not in working.columns:
            working[column] = 0.0

    cohort_mask = (
        working["spread_label"].notna()
        if "spread_label" in working.columns
        else pd.Series(True, index=working.index, dtype=bool)
    )

    working["member_rank_norm"] = _masked_percentile_rank(
        working["log1p_member_count_train"],
        cohort_mask=cohort_mask,
    )
    working["country_rank_norm"] = _masked_percentile_rank(
        working["log1p_n_countries_train"],
        cohort_mask=cohort_mask,
    )
    working["source_rank_norm"] = _masked_percentile_rank(
        working["refseq_share_train"],
        cohort_mask=cohort_mask,
    )
    working["knownness_score"] = _knownness_score_series(working, cohort_mask=cohort_mask)

    if not working.empty:
        working["knownness_half"] = pd.Series("out_of_scope", index=working.index, dtype=object)
        knownness_values = pd.to_numeric(working["knownness_score"], errors="coerce")
        valid_mask = cohort_mask.fillna(False).astype(bool) & knownness_values.notna()

        if valid_mask.any():
            median_knownness = float(knownness_values.loc[valid_mask].median())
            working.loc[valid_mask, "knownness_half"] = np.where(
                knownness_values.loc[valid_mask] <= median_knownness,
                "lower_half",
                "upper_half",
            )

        working["knownness_quartile"] = pd.Series(np.nan, index=working.index, dtype=object)
        working["knownness_quartile_supported"] = False
        if valid_mask.any():
            quartile_labels, n_bins = _stable_quantile_labels(knownness_values.loc[valid_mask], q=4)
            if n_bins == 4:
                quartile_labels = quartile_labels.replace(
                    {"q1": "q1_lowest", "q2": "q2", "q3": "q3", "q4": "q4_highest"},
                )
                working.loc[valid_mask, "knownness_quartile"] = quartile_labels.astype(str)
                working.loc[valid_mask, "knownness_quartile_supported"] = True
    else:
        working["knownness_half"] = pd.Series(dtype=object)
        working["knownness_quartile"] = pd.Series(dtype=object)
        working["knownness_quartile_supported"] = pd.Series(dtype=bool)

    # Add count and source bands
    working["member_count_band"] = pd.cut(
        np.expm1(working["log1p_member_count_train"].fillna(0.0)),
        bins=[-np.inf, 1, 2, 4, 9, np.inf],
        labels=["1", "2", "3_4", "5_9", "10_plus"],
    ).astype(str)
    working["country_count_band"] = pd.cut(
        np.expm1(working["log1p_n_countries_train"].fillna(0.0)),
        bins=[-np.inf, 0, 1, 2, 4, np.inf],
        labels=["0", "1", "2", "3_4", "5_plus"],
    ).astype(str)
    working["source_band"] = np.where(
        working["refseq_share_train"].fillna(0.0) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )

    return working
