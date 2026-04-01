"""Rank-based normalization and final backbone score assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.utils.math import geometric_mean as _geometric_mean
from plasmid_priority.utils.math import geometric_mean_frame as _geometric_mean_frame


def _empirical_percentile(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = reference.dropna().astype(float)
    ref = ref.loc[ref > 0.0]
    arr = values.fillna(0.0).astype(float).to_numpy()
    result = np.zeros(len(arr), dtype=float)
    positive_mask = np.isfinite(arr) & (arr > 0.0)
    if ref.empty or not positive_mask.any():
        return pd.Series(result, index=values.index, dtype=float)
    ref_sorted = np.sort(ref.to_numpy())
    ranks = np.searchsorted(ref_sorted, arr[positive_mask], side="right")
    result[positive_mask] = ranks / len(ref_sorted)
    return pd.Series(result, index=values.index, dtype=float)


def _robust_sigmoid(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = reference.dropna().astype(float)
    ref = ref.loc[ref > 0.0]
    arr = values.fillna(0.0).astype(float).to_numpy()
    result = np.zeros(len(arr), dtype=float)
    positive_mask = np.isfinite(arr) & (arr > 0.0)
    if ref.empty or not positive_mask.any():
        return pd.Series(result, index=values.index, dtype=float)
    median = float(ref.median())
    q75 = float(ref.quantile(0.75))
    q25 = float(ref.quantile(0.25))
    scale = q75 - q25
    if scale <= 0:
        scale = float(ref.std())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    z = (arr[positive_mask] - median) / scale
    z = np.clip(z, -12, 12)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z))
    return pd.Series(result, index=values.index, dtype=float)


def _yeo_johnson_sigmoid(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = reference.dropna().astype(float)
    ref = ref.loc[ref > 0.0]
    arr = values.fillna(0.0).astype(float).to_numpy()
    result = np.zeros(len(arr), dtype=float)
    positive_mask = np.isfinite(arr) & (arr > 0.0)
    if ref.empty or not positive_mask.any():
        return pd.Series(result, index=values.index, dtype=float)

    from scipy.stats import yeojohnson, yeojohnson_normmax

    ref_array = ref.to_numpy(dtype=float)
    lmbda = float(yeojohnson_normmax(ref_array))
    transformed_ref = yeojohnson(ref_array, lmbda=lmbda)
    transformed_values = yeojohnson(arr[positive_mask], lmbda=lmbda)
    median = float(np.median(transformed_ref))
    q75 = float(np.quantile(transformed_ref, 0.75))
    q25 = float(np.quantile(transformed_ref, 0.25))
    scale = q75 - q25
    if scale <= 0:
        scale = float(np.std(transformed_ref))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    z = np.clip((transformed_values - median) / scale, -12, 12)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z))
    return pd.Series(result, index=values.index, dtype=float)


def _normalize_component(values: pd.Series, reference: pd.Series, *, method: str) -> pd.Series:
    if method == "rank_percentile":
        return _empirical_percentile(values, reference)
    if method == "robust_sigmoid":
        return _robust_sigmoid(values, reference)
    if method == "yeo_johnson_sigmoid":
        return _yeo_johnson_sigmoid(values, reference)
    raise ValueError(f"Unsupported normalization method: {method}")


def _linear_residual_series(
    values: pd.Series,
    predictors: pd.DataFrame,
    *,
    fit_mask: pd.Series,
) -> pd.Series:
    result = pd.Series(0.0, index=values.index, dtype=float)
    effective_mask = fit_mask.fillna(False).astype(bool) & values.notna()
    if int(effective_mask.sum()) < 3:
        return result
    X = predictors.loc[effective_mask].fillna(0.0).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X), dtype=float), X])
    y = values.loc[effective_mask].fillna(0.0).to_numpy(dtype=float)
    beta = np.linalg.pinv(X) @ y
    fitted = X @ beta
    result.loc[effective_mask] = y - fitted
    return result


def normalize_component(values: pd.Series, reference: pd.Series, *, method: str = "rank_percentile") -> pd.Series:
    """Public wrapper for component normalization against a reference cohort."""
    return _normalize_component(values, reference, method=method)


def _column_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _indicator_from_positive(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0, 1.0, 0.0),
        index=values.index,
        dtype=float,
    )


def recompute_priority_from_reference(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    normalization_method: str = "rank_percentile",
) -> pd.DataFrame:
    """Recompute normalized components and priority index against a supplied reference set."""
    rescored = scored.copy()
    h_raw_values = rescored["H_raw"].fillna(0.0) if "H_raw" in rescored.columns else rescored["H_eff"].fillna(0.0)
    h_raw_reference = reference["H_raw"].fillna(0.0) if "H_raw" in reference.columns else reference["H_eff"].fillna(0.0)
    h_support_values = rescored["host_support_factor"].fillna(0.0) if "host_support_factor" in rescored.columns else pd.Series(0.0, index=rescored.index, dtype=float)
    h_support_reference = reference["host_support_factor"].fillna(0.0) if "host_support_factor" in reference.columns else pd.Series(0.0, index=reference.index, dtype=float)
    t_raw_values = rescored["T_raw"].fillna(0.0) if "T_raw" in rescored.columns else rescored["T_eff"].fillna(0.0)
    t_raw_reference = reference["T_raw"].fillna(0.0) if "T_raw" in reference.columns else reference["T_eff"].fillna(0.0)
    a_raw_values = rescored["A_raw"].fillna(0.0) if "A_raw" in rescored.columns else rescored["A_eff"].fillna(0.0)
    a_raw_reference = reference["A_raw"].fillna(0.0) if "A_raw" in reference.columns else reference["A_eff"].fillna(0.0)
    support_values = rescored["support_shrinkage"].fillna(0.0) if "support_shrinkage" in rescored.columns else pd.Series(0.0, index=rescored.index, dtype=float)
    support_reference = reference["support_shrinkage"].fillna(0.0) if "support_shrinkage" in reference.columns else pd.Series(0.0, index=reference.index, dtype=float)
    amr_support_values = rescored["amr_support_factor"].fillna(0.0) if "amr_support_factor" in rescored.columns else pd.Series(0.0, index=rescored.index, dtype=float)
    amr_support_reference = reference["amr_support_factor"].fillna(0.0) if "amr_support_factor" in reference.columns else pd.Series(0.0, index=reference.index, dtype=float)
    external_host_values = _column_or_zero(rescored, "H_external_host_range_score")
    external_host_reference = _column_or_zero(reference, "H_external_host_range_score")
    h_augmented_values = _column_or_zero(rescored, "H_augmented_raw")
    h_augmented_reference = _column_or_zero(reference, "H_augmented_raw")
    h_phylogenetic_values = _column_or_zero(rescored, "H_phylogenetic_raw")
    h_phylogenetic_reference = _column_or_zero(reference, "H_phylogenetic_raw")
    h_phylogenetic_augmented_values = _column_or_zero(rescored, "H_phylogenetic_augmented_raw")
    h_phylogenetic_augmented_reference = _column_or_zero(reference, "H_phylogenetic_augmented_raw")
    host_phylogenetic_dispersion_values = _column_or_zero(rescored, "phylo_pairwise_dispersion_score")
    host_phylogenetic_dispersion_reference = _column_or_zero(reference, "phylo_pairwise_dispersion_score")
    host_taxon_evenness_values = _column_or_zero(rescored, "host_taxon_evenness_score")
    host_taxon_evenness_reference = _column_or_zero(reference, "host_taxon_evenness_score")
    backbone_purity_values = _column_or_zero(rescored, "backbone_purity_score")
    backbone_purity_reference = _column_or_zero(reference, "backbone_purity_score")
    assignment_confidence_values = _column_or_zero(rescored, "assignment_confidence_score")
    assignment_confidence_reference = _column_or_zero(reference, "assignment_confidence_score")
    pmlst_coherence_values = _column_or_zero(rescored, "pmlst_coherence_score")
    pmlst_coherence_reference = _column_or_zero(reference, "pmlst_coherence_score")
    a_recurrence_values = _column_or_zero(rescored, "A_recurrence")
    a_recurrence_reference = _column_or_zero(reference, "A_recurrence")
    clinical_context_values = _column_or_zero(rescored, "clinical_context_fraction_train")
    clinical_context_reference = _column_or_zero(reference, "clinical_context_fraction_train")
    pathogenic_context_values = _column_or_zero(rescored, "pathogenic_context_fraction_train")
    pathogenic_context_reference = _column_or_zero(reference, "pathogenic_context_fraction_train")
    ecology_diversity_values = _column_or_zero(rescored, "ecology_context_diversity_train")
    ecology_diversity_reference = _column_or_zero(reference, "ecology_context_diversity_train")
    ecology_context_values = _column_or_zero(rescored, "ecology_context_score")
    ecology_context_reference = _column_or_zero(reference, "ecology_context_score")
    pmlst_presence_values = _column_or_zero(rescored, "pmlst_presence_fraction_train")
    pmlst_presence_reference = _column_or_zero(reference, "pmlst_presence_fraction_train")
    mash_distance_values = _column_or_zero(rescored, "mash_neighbor_distance_train_mean")
    mash_distance_reference = _column_or_zero(reference, "mash_neighbor_distance_train_mean")
    mean_n_replicon_values = _column_or_zero(rescored, "mean_n_replicon_types_train")
    mean_n_replicon_reference = _column_or_zero(reference, "mean_n_replicon_types_train")
    multi_replicon_fraction_values = _column_or_zero(rescored, "multi_replicon_fraction_train")
    primary_replicon_diversity_values = _column_or_zero(rescored, "primary_replicon_diversity_train")
    context_observed_values = _indicator_from_positive(
        clinical_context_values
        + _column_or_zero(rescored, "environmental_context_fraction_train")
        + _column_or_zero(rescored, "host_associated_context_fraction_train")
        + _column_or_zero(rescored, "food_context_fraction_train")
        + pathogenic_context_values
    )
    rescored["T_eff_norm"] = _normalize_component(
        rescored["T_eff"].fillna(0.0),
        reference["T_eff"].fillna(0.0),
        method=normalization_method,
    )
    rescored["H_eff_norm"] = _normalize_component(
        rescored["H_eff"].fillna(0.0),
        reference["H_eff"].fillna(0.0),
        method=normalization_method,
    )
    rescored["H_breadth_norm"] = _normalize_component(
        h_raw_values,
        h_raw_reference,
        method=normalization_method,
    )
    rescored["H_specialization_norm"] = (1.0 - rescored["H_breadth_norm"].fillna(0.0)).clip(lower=0.0, upper=1.0)
    rescored["H_support_norm"] = _normalize_component(
        h_support_values,
        h_support_reference,
        method=normalization_method,
    )
    rescored["H_external_host_range_norm"] = _normalize_component(
        external_host_values,
        external_host_reference,
        method=normalization_method,
    )
    rescored["H_augmented_norm"] = _normalize_component(
        h_augmented_values,
        h_augmented_reference,
        method=normalization_method,
    )
    rescored["host_phylogenetic_dispersion_norm"] = _normalize_component(
        host_phylogenetic_dispersion_values,
        host_phylogenetic_dispersion_reference,
        method=normalization_method,
    )
    rescored["host_taxon_evenness_norm"] = _normalize_component(
        host_taxon_evenness_values,
        host_taxon_evenness_reference,
        method=normalization_method,
    )
    rescored["H_phylogenetic_norm"] = _normalize_component(
        h_phylogenetic_values,
        h_phylogenetic_reference,
        method=normalization_method,
    )
    rescored["H_phylogenetic_specialization_norm"] = (
        1.0 - rescored["H_phylogenetic_norm"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    rescored["H_phylogenetic_augmented_norm"] = _normalize_component(
        h_phylogenetic_augmented_values,
        h_phylogenetic_augmented_reference,
        method=normalization_method,
    )
    rescored["H_phylogenetic_augmented_specialization_norm"] = (
        1.0 - rescored["H_phylogenetic_augmented_norm"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    rescored["H_augmented_specialization_norm"] = (
        1.0 - rescored["H_augmented_norm"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    rescored["A_eff_norm"] = _normalize_component(
        rescored["A_eff"].fillna(0.0),
        reference["A_eff"].fillna(0.0),
        method=normalization_method,
    )
    rescored["A_recurrence_norm"] = _normalize_component(
        a_recurrence_values,
        a_recurrence_reference,
        method=normalization_method,
    )
    rescored["T_raw_norm"] = _normalize_component(
        t_raw_values,
        t_raw_reference,
        method=normalization_method,
    )
    rescored["A_raw_norm"] = _normalize_component(
        a_raw_values,
        a_raw_reference,
        method=normalization_method,
    )
    rescored["support_shrinkage_norm"] = _normalize_component(
        support_values,
        support_reference,
        method=normalization_method,
    )
    rescored["amr_support_norm"] = _normalize_component(
        amr_support_values,
        amr_support_reference,
        method=normalization_method,
    )
    rescored["backbone_purity_norm"] = _normalize_component(
        backbone_purity_values,
        backbone_purity_reference,
        method=normalization_method,
    )
    rescored["assignment_confidence_norm"] = _normalize_component(
        assignment_confidence_values,
        assignment_confidence_reference,
        method=normalization_method,
    )
    rescored["pmlst_coherence_norm"] = _normalize_component(
        pmlst_coherence_values,
        pmlst_coherence_reference,
        method=normalization_method,
    )
    rescored["clinical_context_fraction_norm"] = _normalize_component(
        clinical_context_values,
        clinical_context_reference,
        method=normalization_method,
    )
    rescored["pathogenic_context_fraction_norm"] = _normalize_component(
        pathogenic_context_values,
        pathogenic_context_reference,
        method=normalization_method,
    )
    rescored["pmlst_presence_norm"] = _normalize_component(
        pmlst_presence_values,
        pmlst_presence_reference,
        method=normalization_method,
    )
    rescored["ecology_context_diversity_norm"] = _normalize_component(
        ecology_diversity_values,
        ecology_diversity_reference,
        method=normalization_method,
    )
    rescored["ecology_context_norm"] = _normalize_component(
        ecology_context_values,
        ecology_context_reference,
        method=normalization_method,
    )
    rescored["mash_neighbor_distance_train_norm"] = _normalize_component(
        mash_distance_values,
        mash_distance_reference,
        method=normalization_method,
    )
    rescored["clinical_context_sparse_penalty_norm"] = np.clip(
        rescored["clinical_context_fraction_norm"].fillna(0.0)
        * (1.0 - rescored["support_shrinkage_norm"].fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["external_t_synergy_norm"] = np.clip(
        rescored["H_external_host_range_norm"].fillna(0.0)
        * rescored["T_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["T_A_synergy_norm"] = np.clip(
        rescored["T_eff_norm"].fillna(0.0)
        * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["H_A_synergy_norm"] = np.clip(
        rescored["H_specialization_norm"].fillna(0.0)
        * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["clinical_A_synergy_norm"] = np.clip(
        rescored["clinical_context_fraction_norm"].fillna(0.0)
        * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    # Re-adding metadata depth approximation from what was observed in the 0.81 state
    log1p_members = np.log1p(rescored["member_count_train"].fillna(0.0))
    log1p_countries = np.log1p(rescored["n_countries_train"].fillna(0.0))
    member_norm = np.clip(log1p_members / max(float(log1p_members.max()), 1.0), 0.0, 1.0)
    country_norm = np.clip(log1p_countries / max(float(log1p_countries.max()), 1.0), 0.0, 1.0)
    rescored["metadata_support_depth_norm"] = (member_norm + country_norm) / 2.0
    rescored["host_support_observed_flag"] = _indicator_from_positive(h_support_values)
    rescored["external_host_support_observed_flag"] = _indicator_from_positive(
        _column_or_zero(rescored, "H_external_host_range_support")
    )
    rescored["amr_support_observed_flag"] = _indicator_from_positive(amr_support_values)
    rescored["pmlst_support_observed_flag"] = _indicator_from_positive(pmlst_presence_values)
    rescored["context_support_observed_flag"] = context_observed_values
    rescored["metadata_support_depth_norm"] = np.clip(
        (0.26 * rescored["H_support_norm"].fillna(0.0))
        + (0.20 * rescored["amr_support_norm"].fillna(0.0))
        + (0.18 * _column_or_zero(rescored, "H_external_host_range_support").fillna(0.0))
        + (0.16 * rescored["pmlst_presence_norm"].fillna(0.0))
        + (0.10 * rescored["host_support_observed_flag"].fillna(0.0))
        + (0.10 * rescored["context_support_observed_flag"].fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["metadata_missingness_burden"] = (
        1.0 - rescored["metadata_support_depth_norm"].fillna(0.0)
    ).clip(lower=0.0, upper=1.0)
    rescored["context_support_guard_norm"] = np.clip(
        rescored["clinical_context_fraction_norm"].fillna(0.0)
        * rescored["metadata_support_depth_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["mean_n_replicon_types_norm"] = _normalize_component(
        mean_n_replicon_values,
        mean_n_replicon_reference,
        method=normalization_method,
    )
    rescored["replicon_architecture_norm"] = np.clip(
        (0.45 * rescored["mean_n_replicon_types_norm"].fillna(0.0))
        + (0.35 * multi_replicon_fraction_values.fillna(0.0))
        + (0.20 * primary_replicon_diversity_values.fillna(0.0)),
        0.0,
        1.0,
    )
    strict_priority = _geometric_mean_frame(
        rescored[["T_eff_norm", "H_eff_norm", "A_eff_norm"]].fillna(0.0)
    )
    arithmetic_priority = (
        rescored["T_eff_norm"] + rescored["H_eff_norm"] + rescored["A_eff_norm"]
    ) / 3.0
    rescored["strict_priority_index"] = strict_priority
    rescored["strict_operational_priority_index"] = strict_priority
    rescored["legacy_geometric_priority_index"] = strict_priority
    rescored["arithmetic_priority_index"] = arithmetic_priority
    # Main published priority score should not collapse to zero when one axis is
    # absent; keep the harsher geometric form only as a backlog-screening axis.
    rescored["priority_index"] = arithmetic_priority
    rescored["operational_priority_index"] = arithmetic_priority
    rescored["bio_priority_index"] = (
        rescored["T_raw_norm"].fillna(0.0)
        + rescored["H_breadth_norm"].fillna(0.0)
        + rescored["A_raw_norm"].fillna(0.0)
    ) / 3.0
    rescored["evidence_support_index"] = (
        rescored["support_shrinkage_norm"].fillna(0.0)
        + rescored["amr_support_norm"].fillna(0.0)
    ) / 2.0
    return rescored


def build_scored_backbone_table(
    backbone_table: pd.DataFrame,
    feature_t: pd.DataFrame,
    feature_h: pd.DataFrame,
    feature_a: pd.DataFrame,
    *,
    normalization_method: str = "rank_percentile",
) -> pd.DataFrame:
    """Merge feature tables, perform training-cohort normalization, and compute scores."""
    scored = backbone_table.merge(feature_t, on="backbone_id", how="left", validate="1:1", suffixes=("", "_feature_t"))
    scored = scored.merge(feature_h, on="backbone_id", how="left", validate="1:1")
    scored = scored.merge(feature_a, on="backbone_id", how="left", validate="1:1")

    if "member_count_train_feature_t" in scored.columns:
        scored = scored.drop(columns=["member_count_train_feature_t"])

    training_mask = scored["member_count_train"].fillna(0).astype(int) > 0
    ref_pre_a = scored.loc[training_mask].copy()

    scored["amr_class_richness_norm"] = _normalize_component(
        scored["mean_amr_class_count"].fillna(0.0),
        ref_pre_a["mean_amr_class_count"].fillna(0.0),
        method=normalization_method,
    )
    scored["amr_gene_burden_norm"] = _normalize_component(
        scored["mean_amr_gene_count"].fillna(0.0),
        ref_pre_a["mean_amr_gene_count"].fillna(0.0),
        method=normalization_method,
    )
    scored["amr_clinical_threat_norm"] = _normalize_component(
        scored.get("mean_amr_clinical_threat_score", pd.Series(0.0, index=scored.index)).fillna(0.0),
        ref_pre_a.get("mean_amr_clinical_threat_score", pd.Series(0.0, index=ref_pre_a.index)).fillna(0.0),
        method=normalization_method,
    )
    # Keep WHO-derived clinical threat as a descriptive/supportive axis only.
    # The core A score follows the project plan and uses burden + richness,
    # avoiding a hidden WHO-weighted training feature in the main model path.
    scored["A_content_raw"] = 0.5 * scored["amr_class_richness_norm"] + 0.5 * scored["amr_gene_burden_norm"]
    scored["A_raw"] = _geometric_mean_frame(
        scored[["A_content_raw", "A_consistency"]].fillna(0.0)
    )
    scored["A_eff"] = scored["A_raw"] * scored["amr_support_factor"].fillna(0.0)

    ref = scored.loc[training_mask].copy()

    scored = recompute_priority_from_reference(
        scored,
        ref,
        normalization_method=normalization_method,
    )
    scored["log1p_member_count_train"] = np.log1p(scored["member_count_train"].fillna(0.0))
    scored["log1p_n_countries_train"] = np.log1p(scored["n_countries_train"].fillna(0.0))
    knownness_predictors = scored[
        ["log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"]
    ].copy()
    scored["H_support_norm_residual"] = _linear_residual_series(
        scored["H_support_norm"].fillna(0.0),
        knownness_predictors,
        fit_mask=training_mask,
    )
    scored["amr_support_norm_residual"] = _linear_residual_series(
        scored["amr_support_norm"].fillna(0.0),
        knownness_predictors,
        fit_mask=training_mask,
    )

    return scored
