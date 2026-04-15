"""Split-safe label factories for branch-specific outcome surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from plasmid_priority.shared.temporal import future_window_mask, pre_split_mask

_ENTITY_ID_CANDIDATES = (
    "backbone_id",
    "primary_cluster_id",
    "sequence_accession",
    "accession",
    "plasmid_id",
)
_YEAR_CANDIDATES = (
    "resolved_year",
    "collection_year",
    "year",
    "sample_year",
    "isolation_year",
)
_COUNTRY_CANDIDATES = (
    "country",
    "country_name",
    "host_country",
    "location_country",
    "geo_country",
)
_HOST_GENUS_CANDIDATES = (
    "host_genus",
    "host_genus_name",
    "host_taxon_genus",
    "genus",
    "taxon_genus",
)
_HOST_FAMILY_CANDIDATES = (
    "host_family",
    "host_family_name",
    "host_taxon_family",
    "TAXONOMY_family",
    "family",
    "taxon_family",
)
_CLINICAL_TEXT_CANDIDATES = (
    "clinical_context",
    "context_label",
    "source_type",
    "host_source_context",
    "pd_context",
    "BIOSAMPLE_pathogenicity",
    "ECOSYSTEM_tags",
    "DISEASE_tags",
)
_LAST_RESORT_CANDIDATES = (
    "last_resort_flag",
    "last_resort",
    "is_last_resort",
    "critical_amr_flag",
    "BIOSAMPLE_pathogenicity",
    "DISEASE_tags",
)
_MDR_PROXY_CANDIDATES = (
    "mdr_proxy",
    "mdr_flag",
    "xdr_proxy",
    "xdr_flag",
    "predicted_mobility",
    "is_conjugative",
)
_PD_SUPPORT_CANDIDATES = (
    "pd_clinical_support",
    "pathogen_detection_clinical_support",
    "pathogen_detection_support",
    "source_type",
    "context_label",
)

_LAST_RESORT_TOKENS = (
    "colistin",
    "polymyxin",
    "carbapenem",
    "glycopeptide",
    "oxazolidinone",
    "tigecycline",
)


def _select_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _bool_from_text(series: pd.Series, positive_tokens: Iterable[str]) -> pd.Series:
    text = _normalize_text(series)
    mask = pd.Series(False, index=series.index, dtype=bool)
    for token in positive_tokens:
        mask |= text.str.contains(str(token).lower(), na=False)
    return mask


def _coerce_bool_series(
    frame: pd.DataFrame, candidates: Sequence[str], *, positive_tokens: Iterable[str] = ()
) -> pd.Series:
    column = _select_column(frame, candidates)
    if column is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0) > 0.0
    if positive_tokens:
        return _bool_from_text(series, positive_tokens)
    return _normalize_text(series).isin({"true", "t", "1", "yes", "y"})


def _new_value_counts_by_entity(
    frame: pd.DataFrame, entity_column: str, value_column: str
) -> pd.Series:
    valid = frame.loc[frame[value_column].notna(), [entity_column, value_column]].drop_duplicates()
    if valid.empty:
        return pd.Series(dtype=float)
    return valid.groupby(entity_column, sort=False)[value_column].nunique()


def _new_value_deltas(
    working: pd.DataFrame,
    *,
    entity_column: str,
    value_column: str,
    pre_mask: pd.Series,
    future_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if value_column not in working.columns:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    subset = working.loc[:, [entity_column, value_column]].copy()
    subset[value_column] = _normalize_text(subset[value_column]).replace("", np.nan)
    subset = subset.dropna(subset=[value_column]).drop_duplicates()
    if subset.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    pre_values = subset.loc[pre_mask.loc[subset.index]]
    future_values = subset.loc[future_mask.loc[subset.index]]
    pre_counts = _new_value_counts_by_entity(pre_values, entity_column, value_column)
    future_counts = _new_value_counts_by_entity(future_values, entity_column, value_column)
    if future_values.empty:
        return (
            pd.Series(dtype=float),
            future_counts.astype(float),
            pre_counts.astype(float),
        )
    if pre_values.empty:
        delta = future_counts.astype(float)
    else:
        novel = future_values.merge(
            pre_values,
            on=[entity_column, value_column],
            how="left",
            indicator=True,
        )
        novel = novel.loc[novel["_merge"].eq("left_only"), [entity_column, value_column]]
        delta = _new_value_counts_by_entity(novel, entity_column, value_column).astype(float)
    return delta, future_counts.astype(float), pre_counts.astype(float)


def _aggregate_future_counts(
    records: pd.DataFrame,
    *,
    split_year: int,
    horizon_years: int,
    pd_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["backbone_id"])

    entity_column = _select_column(records, _ENTITY_ID_CANDIDATES)
    year_column = _select_column(records, _YEAR_CANDIDATES)
    if entity_column is None:
        raise KeyError("records must contain a backbone/entity identifier column")
    if year_column is None:
        raise KeyError("records must contain a year-like column")

    working = records.copy()
    working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
    working = working.loc[working[year_column].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["backbone_id"])

    future_mask = future_window_mask(
        working[year_column],
        split_year=split_year,
        horizon_years=horizon_years,
    )
    pre_mask = pre_split_mask(working[year_column], split_year=split_year)

    country_column = _select_column(working, _COUNTRY_CANDIDATES)
    host_genus_column = _select_column(working, _HOST_GENUS_CANDIDATES)
    host_family_column = _select_column(working, _HOST_FAMILY_CANDIDATES)
    clinical_bool = _coerce_bool_series(
        working,
        _CLINICAL_TEXT_CANDIDATES,
        positive_tokens=("clinical", "hospital", "patient", "human", "disease", "pathogen"),
    )
    last_resort_bool = _coerce_bool_series(
        working,
        _LAST_RESORT_CANDIDATES,
        positive_tokens=("pathogen", "disease", "clinical", "hospital", "human"),
    )
    if "amr_class" in working.columns:
        last_resort_bool |= _bool_from_text(working["amr_class"], _LAST_RESORT_TOKENS)
    if "amr_classes" in working.columns:
        last_resort_bool |= _bool_from_text(working["amr_classes"], _LAST_RESORT_TOKENS)
    mdr_bool = _coerce_bool_series(
        working,
        _MDR_PROXY_CANDIDATES,
        positive_tokens=("conjugative", "mdr", "xdr", "multidrug", "extensively drug-resistant"),
    )
    if "drug_class_count" in working.columns:
        mdr_bool |= pd.to_numeric(working["drug_class_count"], errors="coerce").fillna(0.0) >= 3.0
    if "amr_class_count" in working.columns:
        mdr_bool |= pd.to_numeric(working["amr_class_count"], errors="coerce").fillna(0.0) >= 3.0
    if pd_metadata is not None and not pd_metadata.empty:
        clinical_genus_tokens: set[str] = set()
        for column in ("#Organism group", "Scientific name", "scientific_name"):
            if column not in pd_metadata.columns:
                continue
            text = pd_metadata[column].fillna("").astype(str).str.strip().str.lower()
            clinical_genus_tokens.update(token for token in text if token)
            clinical_genus_tokens.update(
                token.split()[0] for token in text if token and token.split()
            )
        genus_column = _select_column(working, _HOST_GENUS_CANDIDATES)
        if genus_column is not None and clinical_genus_tokens:
            pd_support_bool = _normalize_text(working[genus_column]).isin(clinical_genus_tokens)
        else:
            pd_support_bool = _coerce_bool_series(
                working,
                _PD_SUPPORT_CANDIDATES,
                positive_tokens=("clinical", "hospital", "patient"),
            )
    else:
        pd_support_bool = _coerce_bool_series(
            working,
            _PD_SUPPORT_CANDIDATES,
            positive_tokens=("clinical", "hospital", "patient"),
        )

    working[entity_column] = working[entity_column].astype(str)
    entities = pd.Index(pd.unique(working[entity_column]), name="backbone_id")
    future = working.loc[future_mask].copy()
    pre = working.loc[pre_mask].copy()

    result = pd.DataFrame({"backbone_id": entities.astype(str)})
    result["split_year"] = int(split_year)
    result["horizon_years"] = int(horizon_years)

    pre_max_year = pre.groupby(entity_column, sort=False)[year_column].max()
    future_min_year = future.groupby(entity_column, sort=False)[year_column].min()
    result["max_resolved_year_train"] = (
        result["backbone_id"].map(pre_max_year).fillna(int(split_year)).astype(int)
    )
    result["min_resolved_year_test"] = (
        result["backbone_id"].map(future_min_year).fillna(int(split_year + 1)).astype(int)
    )
    result["training_only_future_unseen_backbone_flag"] = ~result["backbone_id"].isin(
        pre[entity_column]
    ) & result["backbone_id"].isin(future[entity_column])

    if country_column is not None:
        country_delta, _, _ = _new_value_deltas(
            working,
            entity_column=entity_column,
            value_column=country_column,
            pre_mask=pre_mask,
            future_mask=future_mask,
        )
        result["n_new_countries_future"] = result["backbone_id"].map(country_delta)
    else:
        result["n_new_countries_future"] = np.nan

    genus_delta = future_genus = pre_genus = pd.Series(dtype=float)
    if host_genus_column is not None:
        genus_delta, future_genus, pre_genus = _new_value_deltas(
            working,
            entity_column=entity_column,
            value_column=host_genus_column,
            pre_mask=pre_mask,
            future_mask=future_mask,
        )
        result["n_new_host_genera_future"] = result["backbone_id"].map(genus_delta)
        result["future_host_genera_count"] = result["backbone_id"].map(future_genus)
        result["pre_host_genera_count"] = result["backbone_id"].map(pre_genus)
    else:
        result["n_new_host_genera_future"] = np.nan
        result["future_host_genera_count"] = np.nan
        result["pre_host_genera_count"] = np.nan

    family_delta = future_family = pre_family = pd.Series(dtype=float)
    if host_family_column is not None:
        family_delta, future_family, pre_family = _new_value_deltas(
            working,
            entity_column=entity_column,
            value_column=host_family_column,
            pre_mask=pre_mask,
            future_mask=future_mask,
        )
        result["n_new_host_families_future"] = result["backbone_id"].map(family_delta)
        result["future_host_families_count"] = result["backbone_id"].map(future_family)
        result["pre_host_families_count"] = result["backbone_id"].map(pre_family)
    else:
        result["n_new_host_families_future"] = np.nan
        result["future_host_families_count"] = np.nan
        result["pre_host_families_count"] = np.nan

    if host_family_column is not None or host_genus_column is not None:
        family_gain = pd.to_numeric(result["future_host_families_count"], errors="coerce").fillna(
            0.0
        ) - pd.to_numeric(result["pre_host_families_count"], errors="coerce").fillna(0.0)
        genus_gain = pd.to_numeric(result["future_host_genera_count"], errors="coerce").fillna(
            0.0
        ) - pd.to_numeric(result["pre_host_genera_count"], errors="coerce").fillna(0.0)
        result["host_phylo_dispersion_gain_future"] = family_gain + 0.5 * genus_gain
    else:
        result["host_phylo_dispersion_gain_future"] = np.nan

    future_entity = future[entity_column]
    pre_entity = pre[entity_column]
    for column_name, series in (
        ("clinical_fraction_future", clinical_bool.loc[future.index]),
        ("last_resort_fraction_future", last_resort_bool.loc[future.index]),
        ("mdr_proxy_fraction_future", mdr_bool.loc[future.index]),
        ("pd_clinical_support_future", pd_support_bool.loc[future.index]),
    ):
        means = pd.to_numeric(series, errors="coerce").groupby(future_entity, sort=False).mean()
        result[column_name] = result["backbone_id"].map(means)
    for column_name, series in (
        ("clinical_fraction_pre", clinical_bool.loc[pre.index]),
        ("last_resort_fraction_pre", last_resort_bool.loc[pre.index]),
        ("mdr_proxy_fraction_pre", mdr_bool.loc[pre.index]),
        ("pd_clinical_support_pre", pd_support_bool.loc[pre.index]),
    ):
        means = pd.to_numeric(series, errors="coerce").groupby(pre_entity, sort=False).mean()
        result[column_name] = result["backbone_id"].map(means)
    return result


def build_geo_spread_labels(
    scored: pd.DataFrame,
    records: pd.DataFrame | None,
    split_year: int,
    horizon_years: int,
) -> pd.DataFrame:
    """Build geo spread labels from future country visibility expansion."""
    base = scored.copy()
    if base.empty and records is not None and not records.empty:
        entity_column = _select_column(records, _ENTITY_ID_CANDIDATES)
        if entity_column is None:
            raise KeyError("records must contain a backbone/entity identifier column")
        base = pd.DataFrame({"backbone_id": pd.unique(records[entity_column].astype(str))})
    if base.empty:
        return pd.DataFrame(
            columns=[
                "backbone_id",
                "split_year",
                "horizon_years",
                "n_new_countries_future",
                "spread_label",
                "label_reason",
            ]
        )

    if records is not None and not records.empty:
        future_stats = _aggregate_future_counts(
            records, split_year=split_year, horizon_years=horizon_years
        )
        base = base.merge(future_stats, on="backbone_id", how="left")

    if "n_new_countries_future" not in base.columns:
        fallback_candidates = [
            "future_n_new_countries",
            "future_country_count",
            "country_count_future",
        ]
        for candidate in fallback_candidates:
            if candidate in base.columns:
                base["n_new_countries_future"] = pd.to_numeric(base[candidate], errors="coerce")
                break
        else:
            base["n_new_countries_future"] = np.nan

    base["spread_label"] = (
        pd.to_numeric(base["n_new_countries_future"], errors="coerce").ge(3).astype(float)
    )
    unseen_mask = (
        base["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool)
        if "training_only_future_unseen_backbone_flag" in base.columns
        else pd.Series(False, index=base.index, dtype=bool)
    )
    base.loc[unseen_mask, "spread_label"] = np.nan
    base["label_reason"] = np.where(
        base["spread_label"].isna(),
        "future_unseen_or_missing_future_window",
        np.where(
            base["spread_label"].astype(float) >= 1.0,
            "future_new_countries>=3",
            "future_new_countries<3",
        ),
    )
    return base


def build_bio_transfer_labels(
    records: pd.DataFrame,
    split_year: int,
    horizon_years: int,
) -> pd.DataFrame:
    """Build host-expansion labels for the bio transfer branch."""
    future_stats = _aggregate_future_counts(
        records, split_year=split_year, horizon_years=horizon_years
    )
    if future_stats.empty:
        return pd.DataFrame(
            columns=[
                "backbone_id",
                "split_year",
                "horizon_years",
                "n_new_host_genera_future",
                "n_new_host_families_future",
                "future_new_host_genera_count",
                "future_new_host_families_count",
                "host_phylo_dispersion_gain_future",
                "bio_transfer_label",
                "bio_transfer_label_reason",
            ]
        )
    future_stats["future_new_host_genera_count"] = future_stats["n_new_host_genera_future"]
    future_stats["future_new_host_families_count"] = future_stats["n_new_host_families_future"]
    future_stats["bio_transfer_label"] = np.where(
        (
            pd.to_numeric(future_stats.get("n_new_host_genera_future"), errors="coerce").fillna(0.0)
            >= 2.0
        )
        | (
            pd.to_numeric(future_stats.get("n_new_host_families_future"), errors="coerce").fillna(
                0.0
            )
            >= 1.0
        ),
        1.0,
        0.0,
    )
    future_stats.loc[
        future_stats["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool),
        "bio_transfer_label",
    ] = np.nan
    future_stats["bio_transfer_label_reason"] = np.where(
        future_stats["bio_transfer_label"].isna(),
        "future_unseen_or_missing_future_window",
        np.where(
            future_stats["bio_transfer_label"].astype(float) >= 1.0,
            "new_host_genera>=2_or_new_host_families>=1",
            "expansion_below_threshold",
        ),
    )
    return future_stats


def build_clinical_hazard_labels(
    records: pd.DataFrame,
    pd_metadata: pd.DataFrame | None,
    split_year: int,
    horizon_years: int,
) -> pd.DataFrame:
    """Build clinical hazard escalation labels from future hazard proxies."""
    future_stats = _aggregate_future_counts(
        records,
        split_year=split_year,
        horizon_years=horizon_years,
        pd_metadata=pd_metadata,
    )
    if future_stats.empty:
        return pd.DataFrame(
            columns=[
                "backbone_id",
                "split_year",
                "horizon_years",
                "clinical_fraction_future",
                "last_resort_fraction_future",
                "mdr_proxy_fraction_future",
                "pd_clinical_support_future",
                "clinical_fraction_future_gain",
                "last_resort_fraction_future_gain",
                "mdr_proxy_fraction_future_gain",
                "pd_clinical_support_future_gain",
                "clinical_hazard_label",
                "clinical_hazard_label_reason",
            ]
        )

    if pd_metadata is not None and not pd_metadata.empty:
        meta = pd_metadata.copy()
        entity_column = _select_column(meta, _ENTITY_ID_CANDIDATES)
        if entity_column is not None:
            meta = meta.rename(columns={entity_column: "backbone_id"})
            meta["backbone_id"] = meta["backbone_id"].astype(str)
            meta_bool = _coerce_bool_series(
                meta,
                _CLINICAL_TEXT_CANDIDATES,
                positive_tokens=("clinical", "hospital", "patient", "human"),
            )
            meta["pd_clinical_support_future"] = meta_bool.astype(float)
            meta["pd_clinical_support_meta"] = meta_bool.astype(float)
            meta = meta.loc[
                :,
                [
                    column
                    for column in meta.columns
                    if column
                    in {"backbone_id", "pd_clinical_support_future", "pd_clinical_support_meta"}
                ],
            ]
            future_stats = future_stats.merge(
                meta.groupby("backbone_id", as_index=False).mean(numeric_only=True),
                on="backbone_id",
                how="left",
                suffixes=("", "_meta"),
            )
            if "pd_clinical_support_future_meta" in future_stats.columns:
                future_stats["pd_clinical_support_future"] = future_stats[
                    "pd_clinical_support_future"
                ].fillna(future_stats["pd_clinical_support_future_meta"])
                future_stats = future_stats.drop(columns=["pd_clinical_support_future_meta"])

    gain_thresholds = {
        "clinical_fraction_future": 0.15,
        "last_resort_fraction_future": 0.10,
        "mdr_proxy_fraction_future": 0.10,
        "pd_clinical_support_future": 0.10,
    }
    for column, threshold in gain_thresholds.items():
        pre_column = column.replace("_future", "_pre")
        future_values = pd.to_numeric(future_stats.get(column), errors="coerce").fillna(0.0)
        pre_values = pd.to_numeric(future_stats.get(pre_column), errors="coerce").fillna(0.0)
        gain = future_values - pre_values
        future_stats[f"{column}_gain"] = gain
    escalation_flags = pd.DataFrame(
        {
            f"{name}_gain": pd.to_numeric(future_stats.get(f"{name}_gain"), errors="coerce").fillna(
                0.0
            )
            >= threshold
            for name, threshold in gain_thresholds.items()
        }
    )
    future_stats["clinical_hazard_label"] = (escalation_flags.sum(axis=1) >= 2).astype(float)
    future_stats.loc[
        future_stats["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool),
        "clinical_hazard_label",
    ] = np.nan
    future_stats["clinical_hazard_label_reason"] = np.where(
        future_stats["clinical_hazard_label"].isna(),
        "future_unseen_or_missing_future_window",
        np.where(
            future_stats["clinical_hazard_label"].astype(float) >= 1.0,
            "at_least_two_future_escalation_gains_met",
            "escalation_conditions_below_threshold",
        ),
    )
    return future_stats
