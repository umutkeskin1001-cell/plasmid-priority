"""Operational backbone assignment and coherence summaries."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from plasmid_priority.utils.temporal import coerce_required_years


def _length_bin(length_val: float) -> str:
    try:
        length = float(length_val)
        if np.isnan(length):
            return "Len_unknown"
    except (ValueError, TypeError):
        return "Len_unknown"

    if length < 50000:
        return "Len_<50k"
    elif length < 100000:
        return "Len_50k-100k"
    elif length < 200000:
        return "Len_100k-200k"
    else:
        return "Len_>200k"


def _fallback_key_from_parts(
    *,
    mobility: str,
    mpf: str,
    replicon: str,
    length_bin: str,
) -> str:
    parts = ["OP", mobility, mpf, replicon]
    if length_bin and length_bin != "Len_unknown":
        parts.append(length_bin)
    return "::".join(parts)


def _fallback_key_series(frame: pd.DataFrame) -> pd.Series:
    mobility = frame["predicted_mobility"].fillna("").astype(str).str.strip().replace("", "unknown")
    mpf = frame["mpf_type"].fillna("").astype(str).str.strip().replace("", "no_mpf")
    replicon = (
        frame["primary_replicon"].fillna("").astype(str).str.strip().replace("", "no_replicon")
    )
    base = "OP::" + mobility + "::" + mpf + "::" + replicon

    lengths = pd.to_numeric(
        frame.get("sequence_length", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    length_bins = pd.Series("Len_unknown", index=frame.index, dtype=object)
    length_bins.loc[lengths < 50000] = "Len_<50k"
    length_bins.loc[(lengths >= 50000) & (lengths < 100000)] = "Len_50k-100k"
    length_bins.loc[(lengths >= 100000) & (lengths < 200000)] = "Len_100k-200k"
    length_bins.loc[lengths >= 200000] = "Len_>200k"

    return base.where(length_bins.eq("Len_unknown"), base + "::" + length_bins)


def fallback_backbone_key(row: pd.Series) -> str:
    mobility = str(row.get("predicted_mobility", "")).strip() or "unknown"
    mpf = str(row.get("mpf_type", "")).strip() or "no_mpf"
    replicon = str(row.get("primary_replicon", "")).strip() or "no_replicon"
    length_str = _length_bin(cast(float, row.get("sequence_length", float("nan"))))
    return _fallback_key_from_parts(
        mobility=mobility,
        mpf=mpf,
        replicon=replicon,
        length_bin=length_str,
    )


def _fallback_backbone(row: pd.Series) -> str:
    return fallback_backbone_key(row)


def _fallback_backbone_key_from_record(record: dict[str, object]) -> str:
    mobility = str(record.get("predicted_mobility", "")).strip() or "unknown"
    mpf = str(record.get("mpf_type", "")).strip() or "no_mpf"
    replicon = str(record.get("primary_replicon", "")).strip() or "no_replicon"
    length_str = _length_bin(cast(float, record.get("sequence_length", float("nan"))))
    return _fallback_key_from_parts(
        mobility=mobility,
        mpf=mpf,
        replicon=replicon,
        length_bin=length_str,
    )


def _training_row_backbone_id(row: pd.Series) -> str:
    primary = str(row.get("primary_cluster_id", "")).strip()
    return primary if primary else fallback_backbone_key(row)


def assign_backbone_ids_training_only(
    records: pd.DataFrame, *, split_year: int = 2015
) -> pd.DataFrame:
    """Assign backbones using only training-period signatures and mark unseen future-only groups."""
    assigned = records.copy()
    years = coerce_required_years(
        assigned,
        "resolved_year",
        context="assign_backbone_ids_training_only",
    )
    training_mask = years <= int(split_year)
    primary = assigned["primary_cluster_id"].fillna("").astype(str).str.strip()
    fallback_key = _fallback_key_series(assigned)

    training = assigned.loc[training_mask].copy()
    if training.empty:
        empty = assigned.copy()
        empty["backbone_id"] = ""
        empty["backbone_assignment_rule"] = "training_only_unavailable"
        empty["backbone_seen_in_training"] = False
        empty["backbone_assignment_mode"] = "training_only"
        empty["backbone_assignment_split_year"] = int(split_year)
        return empty

    training_primary = primary.loc[training_mask]
    training_fallback = fallback_key.loc[training_mask]
    training_backbone_id = training_primary.where(training_primary.ne(""), training_fallback)

    primary_mapping = (
        pd.DataFrame({"primary_cluster_id": training_primary, "backbone_id": training_backbone_id})
        .loc[lambda frame: frame["primary_cluster_id"].ne("")]
        .drop_duplicates("primary_cluster_id")
        .set_index("primary_cluster_id")["backbone_id"]
        .to_dict()
    )
    fallback_training = pd.DataFrame(
        {"fallback_key": training_fallback, "backbone_id": training_backbone_id}
    ).loc[training_primary.eq("")]
    if not fallback_training.empty:
        fallback_mapping = (
            fallback_training[["fallback_key", "backbone_id"]]
            .drop_duplicates("fallback_key")
            .set_index("fallback_key")["backbone_id"]
            .to_dict()
        )
    else:
        fallback_mapping = {}

    backbone_id = pd.Series("", index=assigned.index, dtype=object)
    rule = pd.Series("", index=assigned.index, dtype=object)
    seen = pd.Series(False, index=assigned.index, dtype=bool)

    mapped_primary = primary.map(primary_mapping)
    primary_seen_mask = primary.ne("") & mapped_primary.notna()
    backbone_id.loc[primary_seen_mask] = mapped_primary.loc[primary_seen_mask]
    rule.loc[primary_seen_mask] = "training_primary_cluster_id"
    seen.loc[primary_seen_mask] = True

    mapped_fallback = fallback_key.map(fallback_mapping)
    fallback_seen_mask = primary.eq("") & mapped_fallback.notna()
    backbone_id.loc[fallback_seen_mask] = mapped_fallback.loc[fallback_seen_mask]
    rule.loc[fallback_seen_mask] = "training_mobility_mpf_replicon"
    seen.loc[fallback_seen_mask] = True

    unseen_mask = backbone_id.eq("")
    unseen_key = primary.where(primary.ne(""), fallback_key)
    backbone_id.loc[unseen_mask] = "UNSEEN::" + unseen_key.loc[unseen_mask]
    rule.loc[unseen_mask] = "unseen_after_training"

    assigned["backbone_id"] = backbone_id
    assigned["backbone_assignment_rule"] = rule
    assigned["backbone_seen_in_training"] = seen
    assigned["backbone_assignment_mode"] = "training_only"
    assigned["backbone_assignment_split_year"] = int(split_year)
    return assigned


def assign_backbone_ids(
    records: pd.DataFrame, *, backbone_assignment_mode: str = "all_records"
) -> pd.DataFrame:
    """Assign an operational backbone identifier using plan-defined precedence.

    Args:
        records: Input records to assign backbone IDs to
        backbone_assignment_mode: Explicit backbone assignment mode (required for discovery safety)
            - "training_only": Only training-period backbones are assigned (discovery-safe)
            - "all_records": All records are assigned (not discovery-safe)

    Returns:
        DataFrame with backbone_id, backbone_assignment_rule, and backbone_assignment_mode columns
    """
    if backbone_assignment_mode not in ["training_only", "all_records"]:
        raise ValueError(
            f"backbone_assignment_mode must be 'training_only' or 'all_records', got "
            f"'{backbone_assignment_mode}'. "
            "For discovery safety, use 'training_only' mode explicitly."
        )

    assigned = records.copy()
    primary = assigned["primary_cluster_id"].fillna("").astype(str).str.strip()
    fallback = _fallback_key_series(assigned)

    assigned["backbone_id"] = primary.where(primary.ne(""), fallback)
    assigned["backbone_assignment_rule"] = primary.where(primary.ne(""), "fallback").map(
        lambda value: "primary_cluster_id" if value != "fallback" else "mobility_mpf_replicon"
    )
    assigned["backbone_assignment_mode"] = backbone_assignment_mode
    assigned["backbone_assignment_split_year"] = np.nan
    return assigned


def _dominant_share_by_backbone(
    training: pd.DataFrame, column: str, backbone_order: pd.Series
) -> pd.Series:
    values = training[["backbone_id", column]].copy()
    values[column] = values[column].fillna("").astype(str).str.strip()
    values = values.loc[values[column] != ""]
    if values.empty:
        return pd.Series(0.0, index=backbone_order, dtype=float)
    frequencies = values.groupby("backbone_id", sort=False)[column].value_counts(normalize=True)
    dominant = frequencies.groupby(level=0).max()
    return dominant.reindex(backbone_order, fill_value=0.0).astype(float)


# Alias for backward compatibility with existing tests and consumers
_dominant_share = _dominant_share_by_backbone


def compute_backbone_coherence(records: pd.DataFrame, *, split_year: int = 2015) -> pd.DataFrame:
    """Compute a conservative within-backbone coherence score on training-period rows."""
    years = coerce_required_years(
        records,
        "resolved_year",
        context="compute_backbone_coherence",
    )
    training = records.loc[years <= int(split_year)].copy()
    if training.empty:
        return pd.DataFrame(
            columns=[
                "backbone_id",
                "coherence_score",
                "mobility_dominance",
                "replicon_dominance",
                "topology_dominance",
            ]
        )
    backbone_order = training["backbone_id"].drop_duplicates().astype(str)
    training["backbone_id"] = training["backbone_id"].astype(str)

    mobility_dom = _dominant_share_by_backbone(training, "predicted_mobility", backbone_order)
    replicon_dom = _dominant_share_by_backbone(training, "primary_replicon", backbone_order)
    topology_dom = _dominant_share_by_backbone(training, "topology", backbone_order)
    coherence = (mobility_dom + replicon_dom + topology_dom) / 3.0

    return pd.DataFrame(
        {
            "backbone_id": backbone_order.to_list(),
            "coherence_score": coherence.to_numpy(dtype=float),
            "mobility_dominance": mobility_dom.to_numpy(dtype=float),
            "replicon_dominance": replicon_dom.to_numpy(dtype=float),
            "topology_dominance": topology_dom.to_numpy(dtype=float),
        }
    )
