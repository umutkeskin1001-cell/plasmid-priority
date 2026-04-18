"""Normalization and final backbone score assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.utils.math import geometric_mean_frame as _geometric_mean_frame

DEFAULT_NORMALIZATION_METHOD = "rank_percentile"


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


def _saturating_unit_interval(values: pd.Series, *, midpoint: float, steepness: float) -> pd.Series:
    """Map a bounded latent axis into a monotone saturating unit interval."""
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    z = np.clip((arr - float(midpoint)) * float(steepness), -12.0, 12.0)
    return pd.Series(0.5 * (np.tanh(z) + 1.0), index=values.index, dtype=float)


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


def normalize_component(
    values: pd.Series, reference: pd.Series, *, method: str = DEFAULT_NORMALIZATION_METHOD
) -> pd.Series:
    """Public wrapper for component normalization against a reference cohort."""
    return _normalize_component(values, reference, method=method)


def _column_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _indicator_from_positive(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0, 1.0, 0.0
        ),
        index=values.index,
        dtype=float,
    )


def recompute_priority_from_reference(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
) -> pd.DataFrame:
    """Recompute normalized components and priority index against a supplied reference set."""
    rescored = scored.copy()
    h_raw_values = (
        rescored["H_raw"].fillna(0.0)
        if "H_raw" in rescored.columns
        else rescored["H_eff"].fillna(0.0)
    )
    h_raw_reference = (
        reference["H_raw"].fillna(0.0)
        if "H_raw" in reference.columns
        else reference["H_eff"].fillna(0.0)
    )
    h_obs_values = (
        rescored["H_obs"].fillna(0.0)
        if "H_obs" in rescored.columns
        else rescored["H_phylogenetic_raw"].fillna(0.0)
        if "H_phylogenetic_raw" in rescored.columns
        else h_raw_values
    )
    h_obs_reference = (
        reference["H_obs"].fillna(0.0)
        if "H_obs" in reference.columns
        else reference["H_phylogenetic_raw"].fillna(0.0)
        if "H_phylogenetic_raw" in reference.columns
        else h_raw_reference
    )
    h_support_values = (
        rescored["H_support"].fillna(0.0)
        if "H_support" in rescored.columns
        else rescored["host_support_factor"].fillna(0.0)
        if "host_support_factor" in rescored.columns
        else pd.Series(0.0, index=rescored.index, dtype=float)
    )
    h_support_reference = (
        reference["H_support"].fillna(0.0)
        if "H_support" in reference.columns
        else reference["host_support_factor"].fillna(0.0)
        if "host_support_factor" in reference.columns
        else pd.Series(0.0, index=reference.index, dtype=float)
    )
    t_raw_values = (
        rescored["T_raw"].fillna(0.0)
        if "T_raw" in rescored.columns
        else rescored["T_eff"].fillna(0.0)
    )
    t_raw_reference = (
        reference["T_raw"].fillna(0.0)
        if "T_raw" in reference.columns
        else reference["T_eff"].fillna(0.0)
    )
    a_raw_values = (
        rescored["A_raw"].fillna(0.0)
        if "A_raw" in rescored.columns
        else rescored["A_eff"].fillna(0.0)
    )
    a_raw_reference = (
        reference["A_raw"].fillna(0.0)
        if "A_raw" in reference.columns
        else reference["A_eff"].fillna(0.0)
    )
    amr_gene_count_values = _column_or_zero(rescored, "mean_amr_gene_count")
    amr_gene_count_reference = _column_or_zero(reference, "mean_amr_gene_count")
    amr_class_count_values = _column_or_zero(rescored, "mean_amr_class_count")
    amr_class_count_reference = _column_or_zero(reference, "mean_amr_class_count")
    support_values = (
        rescored["support_shrinkage"].fillna(0.0)
        if "support_shrinkage" in rescored.columns
        else pd.Series(0.0, index=rescored.index, dtype=float)
    )
    support_reference = (
        reference["support_shrinkage"].fillna(0.0)
        if "support_shrinkage" in reference.columns
        else pd.Series(0.0, index=reference.index, dtype=float)
    )
    amr_support_values = (
        rescored["amr_support_factor"].fillna(0.0)
        if "amr_support_factor" in rescored.columns
        else pd.Series(0.0, index=rescored.index, dtype=float)
    )
    amr_support_reference = (
        reference["amr_support_factor"].fillna(0.0)
        if "amr_support_factor" in reference.columns
        else pd.Series(0.0, index=reference.index, dtype=float)
    )
    amr_mdr_proxy_values = _column_or_zero(rescored, "mdr_proxy_fraction")
    amr_mdr_proxy_reference = _column_or_zero(reference, "mdr_proxy_fraction")
    amr_xdr_proxy_values = _column_or_zero(rescored, "xdr_proxy_fraction")
    amr_xdr_proxy_reference = _column_or_zero(reference, "xdr_proxy_fraction")
    last_resort_convergence_values = _column_or_zero(rescored, "mean_last_resort_convergence_score")
    last_resort_convergence_reference = _column_or_zero(
        reference, "mean_last_resort_convergence_score"
    )
    amr_mechanism_diversity_values = _column_or_zero(rescored, "mean_amr_mechanism_diversity_proxy")
    amr_mechanism_diversity_reference = _column_or_zero(
        reference, "mean_amr_mechanism_diversity_proxy"
    )
    external_host_values = _column_or_zero(rescored, "H_external_host_range_score")
    external_host_reference = _column_or_zero(reference, "H_external_host_range_score")
    h_augmented_values = _column_or_zero(rescored, "H_augmented_raw")
    h_augmented_reference = _column_or_zero(reference, "H_augmented_raw")
    h_phylogenetic_values = _column_or_zero(rescored, "H_phylogenetic_raw")
    h_phylogenetic_reference = _column_or_zero(reference, "H_phylogenetic_raw")
    h_phylogenetic_augmented_values = _column_or_zero(rescored, "H_phylogenetic_augmented_raw")
    h_phylogenetic_augmented_reference = _column_or_zero(reference, "H_phylogenetic_augmented_raw")
    phylo_breadth_values = _column_or_zero(rescored, "phylo_breadth_score")
    phylo_breadth_reference = _column_or_zero(reference, "phylo_breadth_score")
    host_phylogenetic_dispersion_values = _column_or_zero(
        rescored, "phylo_pairwise_dispersion_score"
    )
    host_phylogenetic_dispersion_reference = _column_or_zero(
        reference, "phylo_pairwise_dispersion_score"
    )
    host_taxon_evenness_values = _column_or_zero(rescored, "host_taxon_evenness_score")
    host_taxon_evenness_reference = _column_or_zero(reference, "host_taxon_evenness_score")
    a_consistency_values = _column_or_zero(rescored, "A_consistency")
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
    plasmidfinder_support_values = _column_or_zero(rescored, "plasmidfinder_support_score")
    plasmidfinder_support_reference = _column_or_zero(reference, "plasmidfinder_support_score")
    plasmidfinder_complexity_values = _column_or_zero(rescored, "plasmidfinder_complexity_score")
    plasmidfinder_complexity_reference = _column_or_zero(
        reference, "plasmidfinder_complexity_score"
    )
    mash_distance_values = _column_or_zero(rescored, "mash_neighbor_distance_train_mean")
    mash_distance_reference = _column_or_zero(reference, "mash_neighbor_distance_train_mean")
    mean_n_replicon_values = _column_or_zero(rescored, "mean_n_replicon_types_train")
    mean_n_replicon_reference = _column_or_zero(reference, "mean_n_replicon_types_train")
    multi_replicon_fraction_values = _column_or_zero(rescored, "multi_replicon_fraction_train")
    primary_replicon_diversity_values = _column_or_zero(
        rescored, "primary_replicon_diversity_train"
    )
    context_observed_values = _indicator_from_positive(
        clinical_context_values
        + _column_or_zero(rescored, "environmental_context_fraction_train")
        + _column_or_zero(rescored, "host_associated_context_fraction_train")
        + _column_or_zero(rescored, "food_context_fraction_train")
        + pathogenic_context_values
    )
    direct_norm_pairs: list[tuple[str, pd.Series, pd.Series]] = [
        ("T_eff_norm", rescored["T_eff"].fillna(0.0), reference["T_eff"].fillna(0.0)),
        ("H_eff_norm", rescored["H_eff"].fillna(0.0), reference["H_eff"].fillna(0.0)),
        ("H_obs_norm", h_obs_values, h_obs_reference),
        ("H_breadth_norm", h_raw_values, h_raw_reference),
        ("H_support_norm", h_support_values, h_support_reference),
        ("H_external_host_range_norm", external_host_values, external_host_reference),
        ("H_augmented_norm", h_augmented_values, h_augmented_reference),
        (
            "host_phylogenetic_dispersion_norm",
            host_phylogenetic_dispersion_values,
            host_phylogenetic_dispersion_reference,
        ),
        ("host_taxon_evenness_norm", host_taxon_evenness_values, host_taxon_evenness_reference),
        ("H_phylogenetic_norm", h_phylogenetic_values, h_phylogenetic_reference),
        (
            "H_phylogenetic_augmented_norm",
            h_phylogenetic_augmented_values,
            h_phylogenetic_augmented_reference,
        ),
        ("A_eff_norm", rescored["A_eff"].fillna(0.0), reference["A_eff"].fillna(0.0)),
        ("A_recurrence_norm", a_recurrence_values, a_recurrence_reference),
        ("T_raw_norm", t_raw_values, t_raw_reference),
        ("A_raw_norm", a_raw_values, a_raw_reference),
        ("support_shrinkage_norm", support_values, support_reference),
        ("amr_support_norm", amr_support_values, amr_support_reference),
        ("backbone_purity_norm", backbone_purity_values, backbone_purity_reference),
        (
            "assignment_confidence_norm",
            assignment_confidence_values,
            assignment_confidence_reference,
        ),
        ("pmlst_coherence_norm", pmlst_coherence_values, pmlst_coherence_reference),
        ("clinical_context_fraction_norm", clinical_context_values, clinical_context_reference),
        (
            "pathogenic_context_fraction_norm",
            pathogenic_context_values,
            pathogenic_context_reference,
        ),
        ("pmlst_presence_norm", pmlst_presence_values, pmlst_presence_reference),
        (
            "plasmidfinder_support_norm",
            plasmidfinder_support_values,
            plasmidfinder_support_reference,
        ),
        (
            "plasmidfinder_complexity_norm",
            plasmidfinder_complexity_values,
            plasmidfinder_complexity_reference,
        ),
        ("ecology_context_diversity_norm", ecology_diversity_values, ecology_diversity_reference),
        ("ecology_context_norm", ecology_context_values, ecology_context_reference),
        ("mash_neighbor_distance_train_norm", mash_distance_values, mash_distance_reference),
    ]
    if normalization_method == "rank_percentile":
        out_cols = [x[0] for x in direct_norm_pairs]
        val_mat = np.column_stack(
            [x[1].fillna(0.0).astype(float).to_numpy() for x in direct_norm_pairs]
        )
        ref_mat = np.column_stack(
            [x[2].dropna().astype(float).to_numpy() for x in direct_norm_pairs]
        )

        results = np.zeros_like(val_mat)
        pos_mask = np.isfinite(val_mat) & (val_mat > 0.0)
        ref_pos_mask = np.isfinite(ref_mat) & (ref_mat > 0.0)

        for i, col in enumerate(out_cols):
            ref_col = ref_mat[:, i][ref_pos_mask[:, i]]
            if len(ref_col) > 0:
                ref_sorted = np.sort(ref_col)
                col_pos = pos_mask[:, i]
                if col_pos.any():
                    ranks = np.searchsorted(ref_sorted, val_mat[col_pos, i], side="right")
                    results[col_pos, i] = ranks / len(ref_sorted)
            rescored[col] = pd.Series(results[:, i], index=rescored.index, dtype=float)
    else:
        for output_column, values, reference_values in direct_norm_pairs:
            rescored[output_column] = _normalize_component(
                values,
                reference_values,
                method=normalization_method,
            )
    for output_column, source_column in (
        ("H_obs_specialization_norm", "H_obs_norm"),
        ("H_specialization_norm", "H_breadth_norm"),
        ("H_phylogenetic_specialization_norm", "H_phylogenetic_norm"),
        ("H_phylogenetic_augmented_specialization_norm", "H_phylogenetic_augmented_norm"),
        ("H_augmented_specialization_norm", "H_augmented_norm"),
    ):
        rescored[output_column] = (1.0 - rescored[source_column].fillna(0.0)).clip(
            lower=0.0, upper=1.0
        )
    amr_load_density_values = pd.Series(
        amr_gene_count_values.to_numpy(dtype=float)
        / np.clip(amr_class_count_values.to_numpy(dtype=float), 1.0, None),
        index=rescored.index,
        dtype=float,
    )
    amr_load_density_reference = pd.Series(
        amr_gene_count_reference.to_numpy(dtype=float)
        / np.clip(amr_class_count_reference.to_numpy(dtype=float), 1.0, None),
        index=reference.index,
        dtype=float,
    )
    rescored["amr_load_density_norm"] = _normalize_component(
        amr_load_density_values,
        amr_load_density_reference,
        method=normalization_method,
    )
    rescored["amr_mdr_proxy_norm"] = _normalize_component(
        amr_mdr_proxy_values,
        amr_mdr_proxy_reference,
        method=normalization_method,
    )
    rescored["amr_xdr_proxy_norm"] = _normalize_component(
        amr_xdr_proxy_values,
        amr_xdr_proxy_reference,
        method=normalization_method,
    )
    rescored["last_resort_convergence_norm"] = _normalize_component(
        last_resort_convergence_values,
        last_resort_convergence_reference,
        method=normalization_method,
    )
    rescored["amr_mechanism_diversity_norm"] = _normalize_component(
        amr_mechanism_diversity_values,
        amr_mechanism_diversity_reference,
        method=normalization_method,
    )
    rescored["amr_clinical_escalation_norm"] = np.clip(
        (0.30 * rescored["amr_mdr_proxy_norm"].fillna(0.0))
        + (0.40 * rescored["amr_xdr_proxy_norm"].fillna(0.0))
        + (0.30 * rescored["last_resort_convergence_norm"].fillna(0.0)),
        0.0,
        1.0,
    )
    evolutionary_jump_values = pd.Series(
        host_phylogenetic_dispersion_values.to_numpy(dtype=float)
        / np.clip(phylo_breadth_values.to_numpy(dtype=float), 0.01, None),
        index=rescored.index,
        dtype=float,
    )
    evolutionary_jump_reference = pd.Series(
        host_phylogenetic_dispersion_reference.to_numpy(dtype=float)
        / np.clip(phylo_breadth_reference.to_numpy(dtype=float), 0.01, None),
        index=reference.index,
        dtype=float,
    )
    rescored["evolutionary_jump_score_norm"] = _normalize_component(
        evolutionary_jump_values,
        evolutionary_jump_reference,
        method=normalization_method,
    )
    rescored["clinical_context_sparse_penalty_norm"] = np.clip(
        rescored["clinical_context_fraction_norm"].fillna(0.0)
        * (1.0 - rescored["support_shrinkage_norm"].fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["external_t_synergy_norm"] = np.clip(
        rescored["H_external_host_range_norm"].fillna(0.0) * rescored["T_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["T_A_synergy_norm"] = np.clip(
        rescored["T_eff_norm"].fillna(0.0) * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["T_H_obs_synergy_norm"] = np.clip(
        rescored["T_eff_norm"].fillna(0.0) * rescored["H_obs_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["A_H_obs_synergy_norm"] = np.clip(
        rescored["A_eff_norm"].fillna(0.0) * rescored["H_obs_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["T_coherence_synergy_norm"] = np.clip(
        rescored["T_eff_norm"].fillna(0.0)
        * _column_or_zero(rescored, "coherence_score").fillna(0.0),
        0.0,
        1.0,
    )
    rescored["H_A_synergy_norm"] = np.clip(
        rescored["H_specialization_norm"].fillna(0.0) * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["clinical_A_synergy_norm"] = np.clip(
        rescored["clinical_context_fraction_norm"].fillna(0.0) * rescored["A_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["clinical_weapon_synergy_norm"] = np.clip(
        rescored["T_eff_norm"].fillna(0.0)
        * rescored["A_eff_norm"].fillna(0.0)
        * rescored["H_specialization_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["endemic_resistance_norm"] = np.clip(
        rescored["A_recurrence_norm"].fillna(0.0) * rescored["H_specialization_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["H_evenness_T_synergy_norm"] = np.clip(
        host_taxon_evenness_values.fillna(0.0) * rescored["T_eff_norm"].fillna(0.0),
        0.0,
        1.0,
    )
    rescored["host_support_observed_flag"] = _indicator_from_positive(h_support_values)
    rescored["external_host_support_observed_flag"] = _indicator_from_positive(
        _column_or_zero(rescored, "H_external_host_range_support")
    )
    rescored["amr_support_observed_flag"] = _indicator_from_positive(amr_support_values)
    rescored["pmlst_support_observed_flag"] = _indicator_from_positive(pmlst_presence_values)
    rescored["context_support_observed_flag"] = context_observed_values
    # Evidence-depth composite: encode corroboration across host, AMR, pMLST,
    # external host-range, and contextual support signals.
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
    rescored["silent_carrier_risk_norm"] = np.clip(
        a_consistency_values.fillna(0.0)
        * (1.0 - rescored["metadata_support_depth_norm"].fillna(0.0)),
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
    rescored["amr_burden_latent_norm"] = np.clip(
        (0.60 * rescored["amr_load_density_norm"].fillna(0.0))
        + (0.40 * _column_or_zero(rescored, "amr_clinical_threat_norm").fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["amr_burden_saturation_norm"] = _saturating_unit_interval(
        rescored["amr_burden_latent_norm"],
        midpoint=0.45,
        steepness=5.5,
    )
    rescored["replicon_multiplicity_latent_norm"] = np.clip(
        (0.55 * rescored["mean_n_replicon_types_norm"].fillna(0.0))
        + (0.45 * rescored["replicon_architecture_norm"].fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["replicon_multiplicity_saturation_norm"] = _saturating_unit_interval(
        rescored["replicon_multiplicity_latent_norm"],
        midpoint=0.40,
        steepness=5.0,
    )
    rescored["host_range_latent_norm"] = np.clip(
        (0.70 * rescored["H_external_host_range_norm"].fillna(0.0))
        + (0.30 * _column_or_zero(rescored, "H_external_host_range_support").fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["host_range_saturation_norm"] = _saturating_unit_interval(
        rescored["host_range_latent_norm"],
        midpoint=0.35,
        steepness=5.0,
    )
    rescored["eco_clinical_context_latent_norm"] = np.clip(
        (0.55 * rescored["clinical_context_fraction_norm"].fillna(0.0))
        + (0.45 * rescored["ecology_context_diversity_norm"].fillna(0.0)),
        0.0,
        1.0,
    )
    rescored["eco_clinical_context_saturation_norm"] = _saturating_unit_interval(
        rescored["eco_clinical_context_latent_norm"],
        midpoint=0.40,
        steepness=4.5,
    )
    rescored["monotonic_latent_priority_index"] = np.clip(
        (0.18 * rescored["T_eff_norm"].fillna(0.0))
        + (0.18 * rescored["H_specialization_norm"].fillna(0.0))
        + (0.15 * rescored["A_eff_norm"].fillna(0.0))
        + (0.16 * rescored["amr_burden_saturation_norm"].fillna(0.0))
        + (0.11 * rescored["replicon_multiplicity_saturation_norm"].fillna(0.0))
        + (0.11 * rescored["host_range_saturation_norm"].fillna(0.0))
        + (0.11 * rescored["eco_clinical_context_saturation_norm"].fillna(0.0)),
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
        rescored["support_shrinkage_norm"].fillna(0.0) + rescored["amr_support_norm"].fillna(0.0)
    ) / 2.0
    return rescored


def build_scored_backbone_table(
    backbone_table: pd.DataFrame,
    feature_t: pd.DataFrame,
    feature_h: pd.DataFrame,
    feature_a: pd.DataFrame,
    *,
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
) -> pd.DataFrame:
    """Merge feature tables, perform training-cohort normalization, and compute scores."""
    scored = backbone_table.merge(
        feature_t, on="backbone_id", how="left", validate="1:1", suffixes=("", "_feature_t")
    )
    scored = scored.merge(feature_h, on="backbone_id", how="left", validate="1:1")
    scored = scored.merge(feature_a, on="backbone_id", how="left", validate="1:1")

    if "member_count_train_feature_t" in scored.columns:
        scored = scored.drop(columns=["member_count_train_feature_t"], errors="ignore")
    if "backbone_assignment_mode" not in scored.columns:
        scored["backbone_assignment_mode"] = "training_only"
    if "max_resolved_year_train" not in scored.columns:
        scored["max_resolved_year_train"] = np.nan
    if "min_resolved_year_test" not in scored.columns:
        scored["min_resolved_year_test"] = np.nan
    if "training_only_future_unseen_backbone_flag" not in scored.columns:
        scored["training_only_future_unseen_backbone_flag"] = False

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
        scored.get("mean_amr_clinical_threat_score", pd.Series(0.0, index=scored.index)).fillna(
            0.0
        ),
        ref_pre_a.get(
            "mean_amr_clinical_threat_score", pd.Series(0.0, index=ref_pre_a.index)
        ).fillna(0.0),
        method=normalization_method,
    )
    # Keep WHO-derived clinical threat out of the core A score itself, while
    # still exposing the normalized column for auxiliary threat-focused models.
    scored["A_content_raw"] = (
        0.5 * scored["amr_class_richness_norm"] + 0.5 * scored["amr_gene_burden_norm"]
    )
    if "A_raw" not in scored.columns:
        scored["A_raw"] = scored["A_content_raw"]
    scored["A_eff"] = scored["A_raw"] * scored.get(
        "amr_support_factor", pd.Series(0.0, index=scored.index, dtype=float)
    ).fillna(0.0)

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
    scored["backbone_assignment_mode"] = (
        scored["backbone_assignment_mode"].fillna("all_records").astype(str)
    )
    scored["training_only_future_unseen_backbone_flag"] = (
        scored["training_only_future_unseen_backbone_flag"].fillna(False).astype(bool)
    )

    return scored
