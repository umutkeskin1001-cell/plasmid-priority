"""Advanced validation and audit helpers for secondary analyses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plasmid_priority.modeling import annotate_knownness_metadata, evaluate_feature_columns
from plasmid_priority.utils.geography import country_to_macro_region
from plasmid_priority.validation import (
    average_precision,
    bootstrap_spearman_ci,
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    positive_prevalence,
    roc_auc_score,
)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _series_or_default(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _split_field_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {token.strip() for token in text.split(",") if token.strip()}


def _safe_qcut(series: pd.Series, *, q: int, default_label: str = "all") -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    if cleaned.notna().sum() < max(q, 4) or cleaned.nunique(dropna=True) < 2:
        return pd.Series(default_label, index=series.index, dtype=object)
    try:
        return pd.qcut(cleaned.rank(method="average"), q=q, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(default_label, index=series.index, dtype=object)


def _parse_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _safe_series_correlation(left: pd.Series, right: pd.Series) -> float:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    valid = left_numeric.notna() & right_numeric.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    left_valid = left_numeric.loc[valid]
    right_valid = right_numeric.loc[valid]
    if left_valid.nunique(dropna=True) < 2 or right_valid.nunique(dropna=True) < 2:
        return float("nan")
    return float(left_valid.corr(right_valid))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return float(2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0))))


def _recommended_cv_splits(labels: pd.Series, default_splits: int = 5) -> int | None:
    y = labels.dropna().astype(int)
    if y.empty or y.nunique() < 2:
        return None
    max_splits = int(min(default_splits, y.sum(), (y == 0).sum()))
    return max_splits if max_splits >= 2 else None


def build_country_upload_propensity(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize archive upload density by country and year."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    working["country_clean"] = _clean_text(
        working.get("country", pd.Series("", index=working.index))
    )
    working["resolved_year"] = (
        pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    )
    working = working.loc[(working["country_clean"] != "") & working["resolved_year"].gt(0)].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "resolved_year",
                "country",
                "macro_region",
                "n_records",
                "n_backbones",
                "year_total_records",
                "year_country_share",
                "inverse_upload_weight",
                "rarity_weight",
            ]
        )
    year_totals = working.groupby("resolved_year", sort=False).size().rename("year_total_records")
    grouped = (
        working.groupby(["resolved_year", "country_clean"], as_index=False)
        .agg(
            n_records=("backbone_id", "size"),
            n_backbones=("backbone_id", "nunique"),
        )
        .rename(columns={"country_clean": "country"})
    )
    grouped = grouped.merge(year_totals, on="resolved_year", how="left")
    grouped["macro_region"] = grouped["country"].map(country_to_macro_region)
    grouped["year_country_share"] = (
        grouped["n_records"].astype(float)
        / grouped["year_total_records"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    grouped["inverse_upload_weight"] = 1.0 / np.sqrt(
        grouped["n_records"].clip(lower=1).astype(float)
    )
    grouped["rarity_weight"] = 1.0 - grouped["year_country_share"].clip(lower=0.0, upper=1.0)
    return grouped.sort_values(
        ["resolved_year", "n_records", "country"], ascending=[True, False, True]
    ).reset_index(drop=True)


def build_macro_region_jump_table(
    records: pd.DataFrame,
    propensity_table: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
) -> pd.DataFrame:
    """Compute new macro-region jumps and weighted new-country burden per backbone."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    working["backbone_id"] = working["backbone_id"].astype(str)
    working["country_clean"] = _clean_text(
        working.get("country", pd.Series("", index=working.index))
    )
    working["macro_region"] = working["country_clean"].map(country_to_macro_region)
    working["resolved_year"] = (
        pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    )
    training = working.loc[working["resolved_year"] <= split_year].copy()
    testing = working.loc[
        (working["resolved_year"] > split_year) & (working["resolved_year"] <= test_year_end)
    ].copy()
    backbone_order = working["backbone_id"].drop_duplicates().tolist()
    result = pd.DataFrame({"backbone_id": backbone_order})

    train_country_pairs = training.loc[
        training["country_clean"].ne(""), ["backbone_id", "country_clean"]
    ].drop_duplicates()
    test_country_first = (
        testing.loc[
            testing["country_clean"].ne(""), ["backbone_id", "country_clean", "resolved_year"]
        ]
        .sort_values(["backbone_id", "resolved_year", "country_clean"], kind="mergesort")
        .drop_duplicates(["backbone_id", "country_clean"], keep="first")
    )
    new_country_first = test_country_first.merge(
        train_country_pairs,
        on=["backbone_id", "country_clean"],
        how="left",
        indicator=True,
    )
    new_country_first = new_country_first.loc[
        new_country_first["_merge"] == "left_only",
        ["backbone_id", "country_clean", "resolved_year"],
    ].copy()

    train_region_pairs = training.loc[
        training["macro_region"].ne(""), ["backbone_id", "macro_region"]
    ].drop_duplicates()
    test_region_pairs = testing.loc[
        testing["macro_region"].ne(""), ["backbone_id", "macro_region"]
    ].drop_duplicates()
    new_region_pairs = test_region_pairs.merge(
        train_region_pairs,
        on=["backbone_id", "macro_region"],
        how="left",
        indicator=True,
    )
    new_region_pairs = new_region_pairs.loc[
        new_region_pairs["_merge"] == "left_only", ["backbone_id", "macro_region"]
    ].copy()

    training["host_family"] = _clean_text(
        training.get("TAXONOMY_family", pd.Series("", index=training.index))
    )
    testing["host_family"] = _clean_text(
        testing.get("TAXONOMY_family", pd.Series("", index=testing.index))
    )
    train_host_family_pairs = training.loc[
        training["host_family"].ne(""), ["backbone_id", "host_family"]
    ].drop_duplicates()
    test_host_family_pairs = testing.loc[
        testing["host_family"].ne(""), ["backbone_id", "host_family"]
    ].drop_duplicates()
    new_host_family_pairs = test_host_family_pairs.merge(
        train_host_family_pairs,
        on=["backbone_id", "host_family"],
        how="left",
        indicator=True,
    )
    new_host_family_pairs = new_host_family_pairs.loc[
        new_host_family_pairs["_merge"] == "left_only", ["backbone_id", "host_family"]
    ].copy()

    training["host_order"] = _clean_text(
        training.get("TAXONOMY_order", pd.Series("", index=training.index))
    )
    testing["host_order"] = _clean_text(
        testing.get("TAXONOMY_order", pd.Series("", index=testing.index))
    )
    train_host_order_pairs = training.loc[
        training["host_order"].ne(""), ["backbone_id", "host_order"]
    ].drop_duplicates()
    test_host_order_pairs = testing.loc[
        testing["host_order"].ne(""), ["backbone_id", "host_order"]
    ].drop_duplicates()
    new_host_order_pairs = test_host_order_pairs.merge(
        train_host_order_pairs,
        on=["backbone_id", "host_order"],
        how="left",
        indicator=True,
    )
    new_host_order_pairs = new_host_order_pairs.loc[
        new_host_order_pairs["_merge"] == "left_only", ["backbone_id", "host_order"]
    ].copy()

    result["n_train_macro_regions"] = (
        result["backbone_id"]
        .map(train_region_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_test_macro_regions"] = (
        result["backbone_id"]
        .map(test_region_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_new_macro_regions"] = (
        result["backbone_id"]
        .map(new_region_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["new_macro_regions"] = (
        result["backbone_id"]
        .map(
            new_region_pairs.groupby("backbone_id", sort=False)["macro_region"].agg(
                lambda values: ",".join(sorted(values.astype(str)))
            )
        )
        .fillna("")
    )
    n_train_countries = (
        result["backbone_id"]
        .map(train_country_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    eligible = n_train_countries.between(1, 3, inclusive="both")
    result["macro_region_jump_label"] = np.where(
        eligible, result["n_new_macro_regions"].ge(1).astype(float), np.nan
    )

    result["n_train_host_families"] = (
        result["backbone_id"]
        .map(train_host_family_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_test_host_families"] = (
        result["backbone_id"]
        .map(test_host_family_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_new_host_families"] = (
        result["backbone_id"]
        .map(new_host_family_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["new_host_families"] = (
        result["backbone_id"]
        .map(
            new_host_family_pairs.groupby("backbone_id", sort=False)["host_family"].agg(
                lambda values: ",".join(sorted(values.astype(str)))
            )
        )
        .fillna("")
    )
    result["host_family_jump_label"] = np.where(
        eligible, result["n_new_host_families"].ge(1).astype(float), np.nan
    )

    result["n_train_host_orders"] = (
        result["backbone_id"]
        .map(train_host_order_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_test_host_orders"] = (
        result["backbone_id"]
        .map(test_host_order_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_new_host_orders"] = (
        result["backbone_id"]
        .map(new_host_order_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["new_host_orders"] = (
        result["backbone_id"]
        .map(
            new_host_order_pairs.groupby("backbone_id", sort=False)["host_order"].agg(
                lambda values: ",".join(sorted(values.astype(str)))
            )
        )
        .fillna("")
    )
    result["host_order_jump_label"] = np.where(
        eligible, result["n_new_host_orders"].ge(1).astype(float), np.nan
    )

    propensity_lookup = (
        propensity_table[
            ["resolved_year", "country", "inverse_upload_weight", "rarity_weight"]
        ].copy()
        if not propensity_table.empty
        else pd.DataFrame()
    )
    if not new_country_first.empty and not propensity_lookup.empty:
        weighted = new_country_first.rename(columns={"country_clean": "country"}).merge(
            propensity_lookup,
            on=["resolved_year", "country"],
            how="left",
        )
        weighted_summary = weighted.groupby("backbone_id", as_index=False).agg(
            weighted_new_country_burden=(
                "inverse_upload_weight",
                lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum()),
            ),
            rarity_weighted_new_country_burden=(
                "rarity_weight",
                lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum()),
            ),
        )
        result = result.merge(weighted_summary, on="backbone_id", how="left")
    else:
        result["weighted_new_country_burden"] = 0.0
        result["rarity_weighted_new_country_burden"] = 0.0
    result["weighted_new_country_burden"] = (
        result["weighted_new_country_burden"].fillna(0.0).astype(float)
    )
    result["rarity_weighted_new_country_burden"] = (
        result["rarity_weighted_new_country_burden"].fillna(0.0).astype(float)
    )
    result["n_new_countries_recomputed"] = (
        result["backbone_id"]
        .map(new_country_first.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    return result


def _prediction_metric_row(
    model_name: str,
    outcome_name: str,
    outcome: pd.Series,
    score: pd.Series,
) -> dict[str, object]:
    valid = outcome.notna() & score.notna()
    y = outcome.loc[valid].astype(int).to_numpy()
    p = score.loc[valid].astype(float).to_numpy()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "model_name": model_name,
            "outcome_name": outcome_name,
            "status": "skipped_insufficient_label_variation",
            "n_backbones": int(valid.sum()),
            "n_positive": int(outcome.loc[valid].fillna(0).astype(int).sum()),
        }
    order = np.argsort(-p, kind="mergesort")
    top_k = min(25, len(order))
    positives = max(int((y == 1).sum()), 1)
    top_precision = float((y[order[:top_k]] == 1).mean()) if top_k else float("nan")
    top_recall = float((y[order[:top_k]] == 1).sum() / positives) if top_k else float("nan")
    return {
        "model_name": model_name,
        "outcome_name": outcome_name,
        "status": "ok",
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision(y, p)),
        "positive_prevalence": float(positive_prevalence(y)),
        "precision_at_top_25": top_precision,
        "recall_at_top_25": top_recall,
        "n_backbones": int(len(y)),
        "n_positive": int((y == 1).sum()),
    }


def build_secondary_outcome_performance(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    outcome_columns: list[str],
    model_names: list[str],
) -> pd.DataFrame:
    """Evaluate OOF model predictions against alternate outcome labels."""
    if predictions.empty or outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = outcomes.copy()
    for model_name in model_names:
        model_rows = predictions.loc[
            predictions["model_name"] == model_name, ["backbone_id", "oof_prediction"]
        ].copy()
        if model_rows.empty:
            continue
        merged = base.merge(model_rows, on="backbone_id", how="inner")
        for outcome_name in outcome_columns:
            if outcome_name not in merged.columns:
                continue
            rows.append(
                _prediction_metric_row(
                    model_name,
                    outcome_name,
                    merged[outcome_name],
                    merged["oof_prediction"],
                )
            )
    return pd.DataFrame(rows)


def build_weighted_outcome_audit(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    burden_column: str = "weighted_new_country_burden",
    model_names: list[str],
) -> pd.DataFrame:
    """Compare model rankings against weighted new-country burden."""
    if predictions.empty or outcomes.empty or burden_column not in outcomes.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = outcomes[["backbone_id", burden_column]].copy()
    for model_name in model_names:
        model_rows = predictions.loc[
            predictions["model_name"] == model_name, ["backbone_id", "oof_prediction"]
        ].copy()
        if model_rows.empty:
            continue
        merged = base.merge(model_rows, on="backbone_id", how="inner")
        valid = merged[burden_column].notna() & merged["oof_prediction"].notna()
        if int(valid.sum()) < 5:
            rows.append({"model_name": model_name, "status": "skipped_insufficient_rows"})
            continue
        burden = merged.loc[valid, burden_column].astype(float)
        score = merged.loc[valid, "oof_prediction"].astype(float)
        spearman_summary = bootstrap_spearman_ci(
            burden.to_numpy(dtype=float),
            score.to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "status": "ok",
                "n_backbones": int(valid.sum()),
                "spearman_corr": float(spearman_summary["statistic"]),
                "spearman_ci_lower": float(spearman_summary["lower"]),
                "spearman_ci_upper": float(spearman_summary["upper"]),
                "pearson_corr": _safe_series_correlation(burden, score),
                "mean_weighted_burden_top_25": float(
                    merged.loc[valid]
                    .sort_values("oof_prediction", ascending=False)
                    .head(25)[burden_column]
                    .mean()
                ),
                "mean_weighted_burden_overall": float(burden.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_count_outcome_alignment(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    count_column: str,
    model_names: list[str],
) -> pd.DataFrame:
    """Compare model rankings against a non-binary count outcome."""
    if predictions.empty or outcomes.empty or count_column not in outcomes.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = outcomes[["backbone_id", count_column]].copy()
    for model_name in model_names:
        model_rows = predictions.loc[
            predictions["model_name"] == model_name, ["backbone_id", "oof_prediction"]
        ].copy()
        if model_rows.empty:
            continue
        merged = base.merge(model_rows, on="backbone_id", how="inner")
        valid = merged[count_column].notna() & merged["oof_prediction"].notna()
        if int(valid.sum()) < 5:
            rows.append(
                {
                    "model_name": model_name,
                    "count_column": count_column,
                    "status": "skipped_insufficient_rows",
                }
            )
            continue
        counts = merged.loc[valid, count_column].astype(float)
        score = merged.loc[valid, "oof_prediction"].astype(float)
        top25 = merged.loc[valid].sort_values("oof_prediction", ascending=False).head(25)
        spearman_summary = bootstrap_spearman_ci(
            counts.to_numpy(dtype=float),
            score.to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_name": model_name,
                "count_column": count_column,
                "status": "ok",
                "n_backbones": int(valid.sum()),
                "spearman_corr": float(spearman_summary["statistic"]),
                "spearman_ci_lower": float(spearman_summary["lower"]),
                "spearman_ci_upper": float(spearman_summary["upper"]),
                "pearson_corr": _safe_series_correlation(counts, score),
                "mean_count_top_25": float(top25[count_column].mean())
                if not top25.empty
                else float("nan"),
                "median_count_top_25": float(top25[count_column].median())
                if not top25.empty
                else float("nan"),
                "mean_count_overall": float(counts.mean()),
                "median_count_overall": float(counts.median()),
            }
        )
    return pd.DataFrame(rows)


def build_exposure_adjusted_event_table(
    records: pd.DataFrame,
    propensity_table: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
) -> pd.DataFrame:
    """Summarize time-to-event and country-expansion burden on an exposure-adjusted scale."""
    if records.empty:
        return pd.DataFrame()
    timing = build_event_timing_outcomes(
        records, split_year=split_year, test_year_end=test_year_end
    )
    macro = build_macro_region_jump_table(
        records, propensity_table, split_year=split_year, test_year_end=test_year_end
    )
    if timing.empty and macro.empty:
        return pd.DataFrame()
    if timing.empty:
        result = macro[["backbone_id"]].drop_duplicates().copy()
    elif macro.empty:
        result = timing[["backbone_id"]].drop_duplicates().copy()
    else:
        result = timing[["backbone_id"]].merge(
            macro[["backbone_id"]], on="backbone_id", how="outer"
        )
    if not timing.empty:
        timing_columns = [
            column
            for column in [
                "backbone_id",
                "n_new_countries_recomputed",
                "time_to_first_new_country_years",
                "time_to_third_new_country_years",
                "spread_severity_bin",
            ]
            if column in timing.columns
        ]
        result = result.merge(timing[timing_columns], on="backbone_id", how="left")
    if not macro.empty:
        macro_columns = [
            column
            for column in [
                "backbone_id",
                "n_train_countries",
                "weighted_new_country_burden",
                "rarity_weighted_new_country_burden",
            ]
            if column in macro.columns
        ]
        result = result.merge(macro[macro_columns], on="backbone_id", how="left")
    exposure_years = max(int(test_year_end - split_year), 1)
    result["exposure_years"] = int(exposure_years)
    new_country_counts = pd.to_numeric(
        result.get("n_new_countries_recomputed", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    weighted_burden = pd.to_numeric(
        result.get("weighted_new_country_burden", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    rarity_weighted_burden = pd.to_numeric(
        result.get("rarity_weighted_new_country_burden", pd.Series(0.0, index=result.index)),
        errors="coerce",
    ).fillna(0.0)
    time_to_first = pd.to_numeric(
        result.get("time_to_first_new_country_years", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    time_to_third = pd.to_numeric(
        result.get("time_to_third_new_country_years", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["new_country_rate_per_year"] = (new_country_counts / float(exposure_years)).astype(float)
    result["weighted_new_country_rate_per_year"] = (weighted_burden / float(exposure_years)).astype(
        float
    )
    result["rarity_weighted_new_country_rate_per_year"] = (
        rarity_weighted_burden / float(exposure_years)
    ).astype(float)
    result["first_event_speed_score"] = np.where(
        time_to_first.notna(),
        1.0 / (1.0 + np.maximum(time_to_first.astype(float), 0.0)),
        0.0,
    )
    result["third_event_speed_score"] = np.where(
        time_to_third.notna(),
        1.0 / (1.0 + np.maximum(time_to_third.astype(float), 0.0)),
        0.0,
    )
    n_train_countries = pd.to_numeric(
        result.get("n_train_countries", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    eligible = n_train_countries.between(1, 3, inclusive="both")
    result["fast_or_broad_expansion_label"] = np.where(
        eligible,
        (
            result["weighted_new_country_rate_per_year"].fillna(0.0).ge(0.30)
            | result["first_event_speed_score"].fillna(0.0).ge(0.25)
        ).astype(float),
        np.nan,
    )
    return result


def build_exposure_adjusted_outcome_audit(
    predictions: pd.DataFrame,
    exposure_outcomes: pd.DataFrame,
    *,
    model_names: list[str],
    outcome_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare model rankings against continuous exposure-adjusted spread outcomes."""
    if predictions.empty or exposure_outcomes.empty:
        return pd.DataFrame()
    outcome_columns = outcome_columns or [
        "new_country_rate_per_year",
        "weighted_new_country_rate_per_year",
        "rarity_weighted_new_country_rate_per_year",
        "first_event_speed_score",
        "third_event_speed_score",
    ]
    frames: list[pd.DataFrame] = []
    for outcome_column in outcome_columns:
        if outcome_column not in exposure_outcomes.columns:
            continue
        aligned = build_count_outcome_alignment(
            predictions,
            exposure_outcomes,
            count_column=outcome_column,
            model_names=model_names,
        )
        if aligned.empty:
            continue
        aligned = aligned.rename(columns={"count_column": "outcome_name"})
        aligned["audit_family"] = "exposure_adjusted"
        frames.append(aligned)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _monotone_binned_probability(
    score: pd.Series,
    target: pd.Series,
    *,
    n_bins: int = 8,
    ordinal_target: bool = False,
) -> pd.Series:
    """Calibrate a score against an outcome with a monotone binned mapping."""
    score_numeric = pd.to_numeric(score, errors="coerce")
    target_numeric = pd.to_numeric(target, errors="coerce")
    result = pd.Series(np.nan, index=score.index, dtype=float)
    valid = score_numeric.notna() & target_numeric.notna()
    if int(valid.sum()) < 5:
        return result

    valid_score = score_numeric.loc[valid].astype(float)
    valid_target = target_numeric.loc[valid].astype(float)
    if ordinal_target:
        target_min = float(valid_target.min())
        target_span = float(valid_target.max() - target_min)
        if not np.isfinite(target_span) or target_span <= 0:
            return result.fillna(valid_score.rank(method="average", pct=True))
        valid_target = (valid_target - target_min) / target_span

    unique_scores = int(valid_score.nunique(dropna=True))
    if unique_scores < 2:
        result.loc[valid] = valid_score.rank(method="average", pct=True).to_numpy(dtype=float)
        return result.clip(0.0, 1.0)

    q = min(int(n_bins), max(2, unique_scores))
    try:
        bins = pd.qcut(valid_score.rank(method="average"), q=q, duplicates="drop")
    except ValueError:
        bins = pd.Series("all", index=valid_score.index)

    calibration_frame = pd.DataFrame({"score": valid_score, "target": valid_target, "bin": bins})
    summary = calibration_frame.groupby("bin", observed=False).agg(
        target_mean=("target", "mean"), n=("target", "size")
    )
    if summary.empty:
        result.loc[valid] = valid_score.rank(method="average", pct=True).to_numpy(dtype=float)
        return result.clip(0.0, 1.0)

    smoothed = (summary["target_mean"] * summary["n"] + 0.5) / (summary["n"] + 1.0)
    smoothed = smoothed.cummax()
    mapped = bins.map(smoothed)
    fallback = valid_score.rank(method="average", pct=True)
    result.loc[valid] = (
        pd.to_numeric(mapped, errors="coerce").fillna(fallback).to_numpy(dtype=float)
    )
    return result.clip(0.0, 1.0)


def build_operational_risk_dictionary(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    scored: pd.DataFrame | None = None,
    model_names: list[str] | None = None,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Project OOF scores into a calibrated, multi-objective operational risk dictionary."""
    if predictions.empty or outcomes.empty:
        return pd.DataFrame()

    requested_models = [
        str(name)
        for name in (
            model_names or predictions["model_name"].dropna().astype(str).drop_duplicates().tolist()
        )
    ]
    if not requested_models:
        return pd.DataFrame()

    base = outcomes.copy()
    if "backbone_id" not in base.columns:
        return pd.DataFrame()
    base["backbone_id"] = base["backbone_id"].astype(str)

    if scored is not None and not scored.empty:
        metadata_source = annotate_knownness_metadata(
            scored.loc[
                :,
                [
                    column
                    for column in (
                        "backbone_id",
                        "log1p_member_count_train",
                        "log1p_n_countries_train",
                        "refseq_share_train",
                        "spread_label",
                    )
                    if column in scored.columns
                ],
            ].drop_duplicates("backbone_id")
        )
        metadata_columns = [
            column
            for column in (
                "backbone_id",
                "knownness_score",
                "knownness_half",
                "knownness_quartile",
                "source_band",
                "member_count_band",
                "country_count_band",
                "member_rank_norm",
                "country_rank_norm",
                "source_rank_norm",
            )
            if column in metadata_source.columns
        ]
        metadata = metadata_source[metadata_columns].copy()
    else:
        metadata = pd.DataFrame({"backbone_id": base["backbone_id"].drop_duplicates()})
        for column in (
            "knownness_score",
            "knownness_half",
            "knownness_quartile",
            "source_band",
            "member_count_band",
            "country_count_band",
            "member_rank_norm",
            "country_rank_norm",
            "source_rank_norm",
        ):
            metadata[column] = np.nan

    rows: list[pd.DataFrame] = []
    weights = pd.Series(
        {
            "risk_spread_probability": 0.30,
            "risk_spread_severity": 0.20,
            "risk_macro_region_jump_3y": 0.20,
            "risk_event_within_3y": 0.15,
            "risk_three_countries_within_5y": 0.15,
        },
        dtype=float,
    )
    for model_name in requested_models:
        model_rows = predictions.loc[
            predictions["model_name"].astype(str) == model_name,
            ["backbone_id", "oof_prediction", "spread_label"]
            if "spread_label" in predictions.columns
            else ["backbone_id", "oof_prediction"],
        ].copy()
        if model_rows.empty:
            continue
        model_rows["backbone_id"] = model_rows["backbone_id"].astype(str)
        merged = base.merge(model_rows, on="backbone_id", how="inner")
        if merged.empty:
            continue
        merged = merged.merge(metadata, on="backbone_id", how="left", validate="m:1")
        merged["model_name"] = model_name
        merged["risk_spread_probability"] = (
            pd.to_numeric(merged["oof_prediction"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )

        risk_columns: dict[str, pd.Series] = {
            "risk_spread_probability": merged["risk_spread_probability"],
        }
        outcome_specs = [
            ("spread_severity_bin", "risk_spread_severity", True),
            ("macro_region_jump_label", "risk_macro_region_jump_3y", False),
            ("event_within_3y_label", "risk_event_within_3y", False),
            ("three_countries_within_5y_label", "risk_three_countries_within_5y", False),
        ]
        for outcome_column, risk_column, ordinal_target in outcome_specs:
            if outcome_column not in merged.columns:
                risk_columns[risk_column] = pd.Series(np.nan, index=merged.index, dtype=float)
                continue
            risk_columns[risk_column] = _monotone_binned_probability(
                merged["risk_spread_probability"],
                merged[outcome_column],
                n_bins=n_bins,
                ordinal_target=ordinal_target,
            )

        component_frame = pd.DataFrame(risk_columns)
        component_weights = weights.reindex(component_frame.columns).fillna(0.0)
        weighted_sum = (component_frame.fillna(0.0) * component_weights).sum(axis=1)
        weight_total = component_frame.notna().mul(component_weights, axis=1).sum(axis=1)
        merged["operational_risk_score"] = np.where(
            weight_total.gt(0.0),
            weighted_sum / weight_total.clip(lower=1e-12),
            merged["risk_spread_probability"],
        ).astype(float)
        merged["operational_risk_score"] = pd.Series(
            merged["operational_risk_score"], index=merged.index, dtype=float
        ).clip(0.0, 1.0)
        filled_components = component_frame.copy()
        for column_name in filled_components.columns:
            filled_components[column_name] = filled_components[column_name].fillna(
                merged["risk_spread_probability"]
            )
        merged["risk_component_std"] = filled_components.std(axis=1, ddof=0).fillna(0.0)
        boundary_proximity = 1.0 - (2.0 * (merged["risk_spread_probability"] - 0.5).abs()).clip(
            0.0, 1.0
        )
        knownness = pd.to_numeric(
            merged.get("knownness_score", pd.Series(np.nan, index=merged.index)), errors="coerce"
        )
        if knownness.notna().any():
            knownness_penalty = 1.0 - knownness.fillna(float(knownness.median())).clip(0.0, 1.0)
            low_knownness_cutoff = float(knownness.quantile(0.25))
        else:
            knownness_penalty = pd.Series(0.5, index=merged.index, dtype=float)
            low_knownness_cutoff = 0.25
        merged["risk_uncertainty"] = np.clip(
            (0.45 * (2.0 * merged["risk_component_std"].fillna(0.0)))
            + (0.35 * boundary_proximity)
            + (0.20 * knownness_penalty),
            0.0,
            1.0,
        )
        merged["risk_uncertainty_quantile"] = (
            merged["risk_uncertainty"].rank(method="average", pct=True).astype(float)
        )
        review_quantile = 0.60
        abstain_quantile = 0.85
        merged["risk_abstain_flag"] = merged["risk_uncertainty_quantile"].ge(abstain_quantile) | (
            merged["risk_spread_probability"].between(0.35, 0.65)
            & knownness.notna()
            & knownness.le(low_knownness_cutoff)
        )
        merged["risk_decision_tier"] = np.select(
            [merged["risk_abstain_flag"], merged["risk_uncertainty_quantile"].ge(review_quantile)],
            ["abstain", "review"],
            default="action",
        )
        merged["risk_route_context"] = (
            merged.get("source_band", pd.Series("", index=merged.index, dtype=object))
            .fillna("")
            .astype(str)
            + "|"
            + merged.get("member_count_band", pd.Series("", index=merged.index, dtype=object))
            .fillna("")
            .astype(str)
            + "|"
            + merged.get("country_count_band", pd.Series("", index=merged.index, dtype=object))
            .fillna("")
            .astype(str)
        )
        for column_name, series in risk_columns.items():
            merged[column_name] = series.astype(float)
        merged["risk_spread_probability"] = merged["risk_spread_probability"].astype(float)
        rows.append(
            merged[
                [
                    "backbone_id",
                    "model_name",
                    "oof_prediction",
                    "risk_spread_probability",
                    "risk_spread_severity",
                    "risk_macro_region_jump_3y",
                    "risk_event_within_3y",
                    "risk_three_countries_within_5y",
                    "operational_risk_score",
                    "risk_component_std",
                    "risk_uncertainty_quantile",
                    "risk_uncertainty",
                    "risk_abstain_flag",
                    "risk_decision_tier",
                    "risk_route_context",
                    "knownness_score",
                    "knownness_half",
                    "knownness_quartile",
                    "source_band",
                    "member_count_band",
                    "country_count_band",
                ]
            ].copy()
        )

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, sort=False)
    return result.sort_values(["model_name", "backbone_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_knownness_matched_validation(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    n_bins: int = 4,
) -> pd.DataFrame:
    """Evaluate models within matched visibility/source strata."""
    if scored.empty or predictions.empty:
        return pd.DataFrame()
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["member_bin"] = pd.qcut(
        np.log1p(eligible["member_count_train"].fillna(0).astype(float)),
        q=min(n_bins, max(2, eligible["member_count_train"].nunique())),
        duplicates="drop",
    ).astype(str)
    eligible["country_bin"] = eligible["n_countries_train"].fillna(0).astype(int).astype(str)
    eligible["source_bin"] = np.where(
        eligible["refseq_share_train"].fillna(0.0).astype(float) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    wide = (
        predictions.loc[
            predictions["model_name"].isin(model_names),
            ["backbone_id", "model_name", "oof_prediction"],
        ]
        .pivot_table(
            index="backbone_id", columns="model_name", values="oof_prediction", aggfunc="first"
        )
        .reset_index()
    )
    eligible = eligible.merge(wide, on="backbone_id", how="left")
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    stratum_configs = [
        ["member_bin", "country_bin", "source_bin"],
        ["member_bin", "source_bin"],
        ["country_bin", "source_bin"],
        ["source_bin"],
    ]
    for stratum_columns in stratum_configs:
        rows = []
        working = eligible.copy()
        working["matched_stratum"] = working[stratum_columns].astype(str).agg("|".join, axis=1)
        for stratum, frame in working.groupby("matched_stratum", sort=False):
            y = frame["spread_label"].astype(int)
            if len(frame) < 6 or y.nunique() < 2:
                continue
            for model_name in model_names:
                if model_name not in frame.columns:
                    continue
                metric_row = _prediction_metric_row(
                    model_name,
                    "matched_primary_outcome",
                    frame["spread_label"],
                    frame[model_name],
                )
                metric_row.update(
                    {
                        "matched_stratum": str(stratum),
                        "member_bin": str(frame["member_bin"].iloc[0])
                        if "member_bin" in frame.columns
                        else "",
                        "country_bin": str(frame["country_bin"].iloc[0])
                        if "country_bin" in frame.columns
                        else "",
                        "source_bin": str(frame["source_bin"].iloc[0])
                        if "source_bin" in frame.columns
                        else "",
                        "stratum_definition": ",".join(stratum_columns),
                    }
                )
                rows.append(metric_row)
        if rows:
            break
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    for model_name, frame in detail.loc[detail["status"] == "ok"].groupby("model_name", sort=False):
        weights = frame["n_backbones"].astype(float)
        weight_total = max(float(weights.sum()), 1.0)
        summary_rows.append(
            {
                "matched_stratum": "__weighted_overall__",
                "member_bin": "",
                "country_bin": "",
                "source_bin": "",
                "model_name": model_name,
                "outcome_name": "matched_primary_outcome",
                "status": "ok",
                "weighted_mean_roc_auc": float(
                    np.average(frame["roc_auc"].astype(float), weights=weights)
                ),
                "weighted_mean_average_precision": float(
                    np.average(frame["average_precision"].astype(float), weights=weights)
                ),
                "n_strata": int(len(frame)),
                "n_backbones": int(frame["n_backbones"].sum()),
                "weighted_positive_prevalence": float(
                    np.average(frame["positive_prevalence"].astype(float), weights=weights)
                ),
                "weight_total": weight_total,
            }
        )
    return pd.concat([detail, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


def _estimate_propensity_scores(covariates: pd.DataFrame, treatment: pd.Series) -> pd.Series:
    treatment_numeric = pd.to_numeric(treatment, errors="coerce").fillna(0).astype(int)
    if treatment_numeric.nunique(dropna=True) < 2:
        return pd.Series(
            float(treatment_numeric.mean()),
            index=treatment.index,
            dtype=float,
        )
    X = covariates.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    means = X.mean(axis=0)
    stds = X.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    X_scaled = (X - means) / stds
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    model.fit(X_scaled, treatment_numeric)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(np.clip(probabilities, 0.05, 0.95), index=treatment.index, dtype=float)


def build_matched_stratum_propensity_audit(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    treatment_quantile: float = 0.75,
    min_stratum_size: int = 10,
) -> pd.DataFrame:
    """Estimate matched-stratum IPW uplift for high-score exposure.

    Treatment is defined per model as membership in the top score quantile among evaluable
    backbones. Propensity is estimated from pre-2016 knownness covariates only, then summarized
    inside matched member/country/source strata.
    """
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["member_bin"] = _safe_qcut(
        np.log1p(eligible["member_count_train"].fillna(0.0)),
        q=4,
    )
    eligible["country_bin"] = _safe_qcut(eligible["n_countries_train"].fillna(0.0), q=4)
    eligible["source_bin"] = np.where(
        eligible["refseq_share_train"].fillna(0.0).astype(float) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    wide = (
        predictions.loc[
            predictions["model_name"].isin(model_names),
            ["backbone_id", "model_name", "oof_prediction"],
        ]
        .pivot_table(
            index="backbone_id", columns="model_name", values="oof_prediction", aggfunc="first"
        )
        .reset_index()
    )
    eligible = eligible.merge(wide, on="backbone_id", how="left")
    stratum_configs = [
        ["member_bin", "country_bin", "source_bin"],
        ["member_bin", "source_bin"],
        ["country_bin", "source_bin"],
        ["source_bin"],
    ]
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        if model_name not in eligible.columns:
            continue
        working = eligible.loc[eligible[model_name].notna()].copy()
        if len(working) < max(min_stratum_size, 12):
            continue
        threshold = float(
            working[model_name].quantile(float(np.clip(treatment_quantile, 0.5, 0.95)))
        )
        treatment = working[model_name].ge(threshold).astype(int)
        if treatment.nunique(dropna=True) < 2:
            threshold = float(working[model_name].median())
            treatment = working[model_name].ge(threshold).astype(int)
        if treatment.nunique(dropna=True) < 2:
            continue
        propensity = _estimate_propensity_scores(
            working[
                [
                    "member_count_train",
                    "n_countries_train",
                    "refseq_share_train",
                ]
            ].assign(
                member_count_train=lambda frame: np.log1p(
                    pd.to_numeric(frame["member_count_train"], errors="coerce").fillna(0.0)
                ),
                n_countries_train=lambda frame: np.log1p(
                    pd.to_numeric(frame["n_countries_train"], errors="coerce").fillna(0.0)
                ),
            ),
            treatment,
        )
        treated_fraction = float(treatment.mean())
        working["treated_flag"] = treatment.to_numpy(dtype=int)
        working["propensity_score"] = propensity.to_numpy(dtype=float)
        working["ipw_weight"] = np.where(
            working["treated_flag"].eq(1),
            treated_fraction / working["propensity_score"].to_numpy(dtype=float),
            (1.0 - treated_fraction) / (1.0 - working["propensity_score"].to_numpy(dtype=float)),
        )
        selected_rows: list[dict[str, object]] = []
        selected_definition = ""
        for stratum_columns in stratum_configs:
            candidate_rows: list[dict[str, object]] = []
            stratum_frame = working.copy()
            stratum_frame["matched_stratum"] = (
                stratum_frame[stratum_columns].astype(str).agg("|".join, axis=1)
            )
            for stratum, frame in stratum_frame.groupby("matched_stratum", sort=False):
                if len(frame) < int(min_stratum_size):
                    continue
                if frame["treated_flag"].nunique(dropna=True) < 2:
                    continue
                if frame["spread_label"].astype(int).nunique(dropna=True) < 2:
                    continue
                outcome = frame["spread_label"].astype(float)
                treated_mask = frame["treated_flag"].eq(1)
                control_mask = ~treated_mask
                treated_weights = frame.loc[treated_mask, "ipw_weight"].astype(float)
                control_weights = frame.loc[control_mask, "ipw_weight"].astype(float)
                if float(treated_weights.sum()) <= 0.0 or float(control_weights.sum()) <= 0.0:
                    continue
                treated_outcome_ipw = float(
                    np.average(outcome.loc[treated_mask], weights=treated_weights)
                )
                control_outcome_ipw = float(
                    np.average(outcome.loc[control_mask], weights=control_weights)
                )
                naive_risk_difference = float(
                    outcome.loc[treated_mask].mean() - outcome.loc[control_mask].mean()
                )
                candidate_rows.append(
                    {
                        "matched_stratum": str(stratum),
                        "member_bin": str(frame["member_bin"].iloc[0]),
                        "country_bin": str(frame["country_bin"].iloc[0]),
                        "source_bin": str(frame["source_bin"].iloc[0]),
                        "stratum_definition": ",".join(stratum_columns),
                        "model_name": model_name,
                        "status": "ok",
                        "n_backbones": int(len(frame)),
                        "n_treated": int(treated_mask.sum()),
                        "n_control": int(control_mask.sum()),
                        "treated_fraction": float(treated_mask.mean()),
                        "treatment_threshold": threshold,
                        "mean_propensity_score": float(frame["propensity_score"].mean()),
                        "propensity_overlap_min": float(
                            min(
                                frame.loc[treated_mask, "propensity_score"].mean(),
                                1.0 - frame.loc[control_mask, "propensity_score"].mean(),
                            )
                        ),
                        "treated_outcome_ipw": treated_outcome_ipw,
                        "control_outcome_ipw": control_outcome_ipw,
                        "ipw_risk_difference": float(treated_outcome_ipw - control_outcome_ipw),
                        "naive_risk_difference": naive_risk_difference,
                    }
                )
            if candidate_rows:
                selected_rows = candidate_rows
                selected_definition = ",".join(stratum_columns)
                break
        if not selected_rows:
            continue
        rows.extend(selected_rows)
        detail = pd.DataFrame(selected_rows)
        weights = detail["n_backbones"].astype(float)
        rows.append(
            {
                "matched_stratum": "__weighted_overall__",
                "member_bin": "",
                "country_bin": "",
                "source_bin": "",
                "stratum_definition": selected_definition,
                "model_name": model_name,
                "status": "ok",
                "n_backbones": int(detail["n_backbones"].sum()),
                "n_treated": int(detail["n_treated"].sum()),
                "n_control": int(detail["n_control"].sum()),
                "treated_fraction": float(np.average(detail["treated_fraction"], weights=weights)),
                "treatment_threshold": threshold,
                "mean_propensity_score": float(
                    np.average(detail["mean_propensity_score"], weights=weights)
                ),
                "propensity_overlap_min": float(detail["propensity_overlap_min"].min()),
                "treated_outcome_ipw": float(
                    np.average(detail["treated_outcome_ipw"], weights=weights)
                ),
                "control_outcome_ipw": float(
                    np.average(detail["control_outcome_ipw"], weights=weights)
                ),
                "ipw_risk_difference": float(
                    np.average(detail["ipw_risk_difference"], weights=weights)
                ),
                "naive_risk_difference": float(
                    np.average(detail["naive_risk_difference"], weights=weights)
                ),
            }
        )
    return pd.DataFrame(rows)


def _residualize(
    values: pd.Series,
    predictors: pd.DataFrame,
    *,
    method: str,
    n_bins: int = 6,
) -> pd.Series:
    y = values.fillna(0.0).astype(float)
    X = predictors.fillna(0.0).astype(float)
    if method == "linear":
        design = np.column_stack([np.ones(len(X), dtype=float), X.to_numpy(dtype=float)])
        beta = np.linalg.pinv(design) @ y.to_numpy(dtype=float)
        fitted = design @ beta
        return pd.Series(y.to_numpy(dtype=float) - fitted, index=values.index, dtype=float)
    if method == "quadratic":
        base = X.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(X), dtype=float), base, np.square(base)])
        beta = np.linalg.pinv(design) @ y.to_numpy(dtype=float)
        fitted = design @ beta
        return pd.Series(y.to_numpy(dtype=float) - fitted, index=values.index, dtype=float)
    if method == "stratified":
        score = X.mean(axis=1)
        try:
            bins = pd.qcut(score, q=min(n_bins, max(2, score.nunique())), duplicates="drop")
        except ValueError:
            bins = pd.Series("all", index=score.index)
        residual = y.copy()
        residual = residual - residual.groupby(bins, observed=False).transform("mean").fillna(0.0)
        return residual.astype(float)
    raise ValueError(f"Unsupported residualization method: {method}")


def build_nonlinear_deconfounding_audit(scored: pd.DataFrame) -> pd.DataFrame:
    """Audit whether proxy-depleted conclusions survive nonlinear deconfounding."""
    if scored.empty:
        return pd.DataFrame()
    eligible = scored.loc[scored["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()
    predictors = eligible[
        ["log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"]
    ].copy()
    knownness_score = predictors.mean(axis=1)
    rows: list[dict[str, object]] = []
    methods = {
        "linear_existing": eligible.get(
            "H_support_norm_residual", pd.Series(0.0, index=eligible.index, dtype=float)
        ).astype(float),
        "quadratic": _residualize(eligible["H_support_norm"], predictors, method="quadratic"),
        "stratified": _residualize(eligible["H_support_norm"], predictors, method="stratified"),
    }
    for method_name, residual in methods.items():
        working = eligible.copy()
        residual_column = f"H_support_residual_{method_name}"
        working[residual_column] = residual.fillna(0.0).astype(float)
        n_splits = _recommended_cv_splits(working["spread_label"])
        if n_splits is None:
            rows.append(
                {
                    "deconfounding_method": method_name,
                    "status": "skipped_insufficient_label_variation",
                }
            )
            continue
        result = evaluate_feature_columns(
            working,
            columns=[
                "T_raw_norm",
                "H_specialization_norm",
                "A_raw_norm",
                "orit_support",
                residual_column,
            ],
            label=f"{method_name}_proxy_depleted_priority",
            n_splits=n_splits,
            n_repeats=3,
            seed=42,
            include_ci=False,
        )
        row = {
            "deconfounding_method": method_name,
            "model_name": f"{method_name}_proxy_depleted_priority",
            "status": "ok",
            "residual_knownness_abs_corr": float(abs(pd.Series(residual).corr(knownness_score))),
        }
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_metadata_quality_table(
    records: pd.DataFrame,
    scored: pd.DataFrame,
    assembly: pd.DataFrame,
    biosample: pd.DataFrame,
    nucc_identical: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Aggregate record-level metadata completeness into backbone-level quality scores."""
    if records.empty:
        return pd.DataFrame()
    training = records.loc[
        pd.to_numeric(records["resolved_year"], errors="coerce").fillna(0).astype(int) <= split_year
    ].copy()
    if training.empty:
        return pd.DataFrame()
    training["sequence_accession"] = training["sequence_accession"].astype(str)
    training["nuccore_uid"] = training["nuccore_uid"].astype(str)
    assembly = assembly.copy()
    biosample = biosample.copy()
    nucc_identical = nucc_identical.copy()
    assembly["NUCCORE_UID"] = assembly["NUCCORE_UID"].astype(str)
    biosample["NUCCORE_UID"] = biosample["NUCCORE_UID"].astype(str)
    nucc_identical["NUCCORE_ACC"] = nucc_identical["NUCCORE_ACC"].astype(str)
    merged = training.merge(
        assembly.rename(columns={"NUCCORE_UID": "nuccore_uid"}),
        on="nuccore_uid",
        how="left",
    )
    merged = merged.merge(
        biosample.rename(columns={"NUCCORE_UID": "nuccore_uid"}),
        on="nuccore_uid",
        how="left",
    )
    merged = merged.merge(
        nucc_identical.rename(columns={"NUCCORE_ACC": "sequence_accession"}),
        on="sequence_accession",
        how="left",
    )
    assembly_status = _clean_text(
        merged.get("ASSEMBLY_Status", pd.Series("", index=merged.index))
    ).str.lower()
    completeness = _clean_text(
        merged.get("NUCCORE_Completeness", pd.Series("", index=merged.index))
    ).str.lower()
    duplicate_flag = (
        _clean_text(merged.get("NUCCORE_DuplicatedEntry", pd.Series("", index=merged.index)))
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    merged["record_metadata_quality"] = pd.DataFrame(
        {
            "has_country": _clean_text(merged.get("country", pd.Series("", index=merged.index))).ne(
                ""
            ),
            "has_species": _clean_text(merged.get("species", pd.Series("", index=merged.index))).ne(
                ""
            ),
            "has_family": _clean_text(
                merged.get("TAXONOMY_family", pd.Series("", index=merged.index))
            ).ne(""),
            "has_assembly_status": assembly_status.ne(""),
            "complete_assembly": assembly_status.str.contains("complete"),
            "complete_nuccore": completeness.str.contains("complete"),
            "low_contig_count": pd.to_numeric(
                merged.get("typing_num_contigs", pd.Series(0.0, index=merged.index)),
                errors="coerce",
            )
            .fillna(0.0)
            .between(1, 10),
            "has_typing_gc": pd.to_numeric(
                merged.get("typing_gc", pd.Series(0.0, index=merged.index)), errors="coerce"
            )
            .fillna(0.0)
            .gt(0),
            "has_typing_size": pd.to_numeric(
                merged.get("typing_size", pd.Series(0.0, index=merged.index)), errors="coerce"
            )
            .fillna(0.0)
            .gt(0),
            "not_duplicate": ~duplicate_flag,
        }
    ).mean(axis=1)
    rows = merged.groupby("backbone_id", as_index=False).agg(
        metadata_quality_mean=("record_metadata_quality", "mean"),
        metadata_quality_min=("record_metadata_quality", "min"),
        n_training_records=("sequence_accession", "nunique"),
        country_coverage_fraction=(
            "country",
            lambda values: _clean_text(pd.Series(values)).ne("").mean(),
        ),
        assembly_status_fraction=(
            "ASSEMBLY_Status",
            lambda values: _clean_text(pd.Series(values)).ne("").mean(),
        ),
        duplicate_fraction=(
            "NUCCORE_DuplicatedEntry",
            lambda values: (
                _clean_text(pd.Series(values)).str.lower().isin(["true", "1", "yes"]).mean()
            ),
        ),
    )
    if not scored.empty and "assignment_confidence_score" in scored.columns:
        rows = rows.merge(
            scored[["backbone_id", "assignment_confidence_score"]].drop_duplicates("backbone_id"),
            on="backbone_id",
            how="left",
        )
        rows["metadata_quality_score"] = (
            0.75 * rows["metadata_quality_mean"].fillna(0.0)
            + 0.25 * rows["assignment_confidence_score"].fillna(0.0)
        ).clip(lower=0.0, upper=1.0)
    else:
        rows["metadata_quality_score"] = (
            rows["metadata_quality_mean"].fillna(0.0).clip(lower=0.0, upper=1.0)
        )
    rows["metadata_quality_tier"] = pd.cut(
        rows["metadata_quality_score"],
        bins=[-0.01, 0.45, 0.70, 1.01],
        labels=["low", "medium", "high"],
    ).astype(str)
    return rows.sort_values(
        ["metadata_quality_score", "n_training_records"], ascending=[False, False]
    ).reset_index(drop=True)


def build_consensus_shortlist(
    consensus_candidates: pd.DataFrame,
    candidate_portfolio: pd.DataFrame,
    candidate_multiverse_stability: pd.DataFrame,
    *,
    top_k: int = 25,
) -> pd.DataFrame:
    """Build a reviewer-facing shortlist from curated portfolio rows plus consensus fill."""
    if candidate_portfolio.empty and consensus_candidates.empty:
        return pd.DataFrame()

    track_order_map = {
        "established_high_risk": 0,
        "novel_signal": 1,
    }
    frames: list[pd.DataFrame] = []
    selected_ids: set[str] = set()

    if not candidate_portfolio.empty:
        portfolio = candidate_portfolio.copy()
        portfolio["portfolio_track"] = (
            portfolio.get("portfolio_track", pd.Series("", index=portfolio.index))
            .fillna("")
            .astype(str)
        )
        portfolio["track_rank"] = pd.to_numeric(
            portfolio.get("track_rank", pd.Series(np.nan, index=portfolio.index)), errors="coerce"
        )
        portfolio["_shortlist_track_order"] = (
            portfolio["portfolio_track"]
            .map(track_order_map)
            .fillna(len(track_order_map))
            .astype(int)
        )
        portfolio["_selection_origin"] = "portfolio"
        sort_columns = [
            column
            for column in [
                "_shortlist_track_order",
                "track_rank",
                "consensus_rank",
                "primary_model_candidate_score",
            ]
            if column in portfolio.columns
        ]
        ascending = []
        for column in sort_columns:
            ascending.append(False if column == "primary_model_candidate_score" else True)
        portfolio = portfolio.sort_values(sort_columns, ascending=ascending, na_position="last")
        portfolio = portfolio.drop_duplicates("backbone_id").head(top_k).copy()
        selected_ids.update(portfolio["backbone_id"].astype(str))
        frames.append(portfolio)

    if len(selected_ids) < top_k and not consensus_candidates.empty:
        remaining = consensus_candidates.loc[
            ~consensus_candidates["backbone_id"].astype(str).isin(selected_ids)
        ].copy()
        if not remaining.empty:
            remaining["_selection_origin"] = "consensus_fill"
            remaining["_shortlist_track_order"] = len(track_order_map) + 1
            remaining = remaining.sort_values(
                ["consensus_rank", "consensus_support_count", "primary_candidate_score"],
                ascending=[True, False, False],
                na_position="last",
            ).head(top_k - len(selected_ids))
            frames.append(remaining)

    shortlist = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if shortlist.empty:
        return shortlist

    if not consensus_candidates.empty:
        shortlist = shortlist.merge(
            consensus_candidates.drop_duplicates("backbone_id"),
            on="backbone_id",
            how="left",
            suffixes=("", "_consensus"),
        )
    if not candidate_multiverse_stability.empty:
        shortlist = shortlist.merge(
            candidate_multiverse_stability.drop_duplicates("backbone_id"),
            on="backbone_id",
            how="left",
            suffixes=("", "_stability"),
        )

    shortlist = shortlist.copy()
    shortlist = shortlist.sort_values(
        [
            column
            for column in [
                "_shortlist_track_order",
                "track_rank",
                "consensus_rank",
                "primary_model_candidate_score",
            ]
            if column in shortlist.columns
        ],
        ascending=[
            False if column == "primary_model_candidate_score" else True
            for column in [
                "_shortlist_track_order",
                "track_rank",
                "consensus_rank",
                "primary_model_candidate_score",
            ]
            if column in shortlist.columns
        ],
        na_position="last",
    ).reset_index(drop=True)
    shortlist["consensus_shortlist_rank"] = np.arange(1, len(shortlist) + 1)
    shortlist["selection_origin"] = (
        shortlist["_selection_origin"].fillna("consensus_fill").astype(str)
    )

    preferred = [
        "consensus_shortlist_rank",
        "selection_origin",
        "portfolio_track",
        "track_rank",
        "backbone_id",
        "candidate_confidence_tier",
        "recommended_monitoring_tier",
        "false_positive_risk_tier",
        "consensus_rank",
        "consensus_candidate_score",
        "consensus_support_count",
        "multiverse_stability_tier",
        "multiverse_stability_score",
        "primary_model_candidate_score",
        "conservative_model_candidate_score",
        "baseline_both_candidate_score",
        "novelty_margin_vs_baseline",
        "bootstrap_top_10_frequency",
        "variant_top_10_frequency",
        "threshold_robustness_score",
        "knownness_half",
        "knownness_score",
        "n_new_countries",
        "spread_label",
    ]
    available = [column for column in preferred if column in shortlist.columns]
    return shortlist[available].head(top_k).reset_index(drop=True)


def build_false_negative_audit(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model_name: str,
    metadata_quality: pd.DataFrame | None = None,
    candidate_threshold_flip: pd.DataFrame | None = None,
    shortlist_cutoffs: tuple[int, ...] = (25, 50),
    top_n: int = 50,
) -> pd.DataFrame:
    """Summarize later positives that were not promoted into the practical shortlist."""
    if scored.empty or predictions.empty:
        return pd.DataFrame()
    primary = predictions.loc[
        predictions["model_name"] == primary_model_name, ["backbone_id", "oof_prediction"]
    ].copy()
    if primary.empty:
        return pd.DataFrame()
    working = scored.loc[scored["spread_label"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    working = working.merge(primary, on="backbone_id", how="inner")
    if working.empty:
        return pd.DataFrame()
    working = working.sort_values("oof_prediction", ascending=False).reset_index(drop=True)
    working["primary_rank"] = np.arange(1, len(working) + 1)
    working["primary_rank_percentile"] = working["primary_rank"] / max(len(working), 1)
    for column in ("log1p_member_count_train", "log1p_n_countries_train", "refseq_share_train"):
        if column in working.columns:
            working[f"{column}_rank"] = working[column].rank(method="average", pct=True)
    if {
        "log1p_member_count_train_rank",
        "log1p_n_countries_train_rank",
        "refseq_share_train_rank",
    } <= set(working.columns):
        working["knownness_score"] = (
            working["log1p_member_count_train_rank"].fillna(0.0)
            + working["log1p_n_countries_train_rank"].fillna(0.0)
            + working["refseq_share_train_rank"].fillna(0.0)
        ) / 3.0
    else:
        working["knownness_score"] = np.nan
    knownness_threshold = (
        float(working["knownness_score"].median())
        if working["knownness_score"].notna().any()
        else math.nan
    )

    if metadata_quality is not None and not metadata_quality.empty:
        meta_columns = [
            column
            for column in [
                "backbone_id",
                "metadata_quality_score",
                "metadata_quality_tier",
                "country_coverage_fraction",
                "duplicate_fraction",
            ]
            if column in metadata_quality.columns
        ]
        if meta_columns:
            working = working.merge(
                metadata_quality[meta_columns].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
    if candidate_threshold_flip is not None and not candidate_threshold_flip.empty:
        flip_columns = [
            column
            for column in [
                "backbone_id",
                "threshold_flip_count",
                "eligible_for_threshold_audit",
                "label_ge_1",
                "label_ge_2",
                "label_ge_3",
                "label_ge_4",
                "label_ge_5",
            ]
            if column in candidate_threshold_flip.columns
        ]
        if flip_columns:
            working = working.merge(
                candidate_threshold_flip[flip_columns].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )
    if "threshold_flip_count" not in working.columns:
        working["threshold_flip_count"] = np.nan
    if "eligible_for_threshold_audit" not in working.columns:
        working["eligible_for_threshold_audit"] = False
    train_countries = pd.to_numeric(
        working.get("n_countries_train", pd.Series(np.nan, index=working.index)), errors="coerce"
    )
    new_countries = pd.to_numeric(
        working.get("n_new_countries", pd.Series(np.nan, index=working.index)), errors="coerce"
    )
    default_status = (new_countries >= 3).astype(float)
    eligible_threshold = train_countries.between(1, 3, inclusive="both").fillna(False)
    inferred_flip = pd.Series(np.nan, index=working.index, dtype=float)
    for idx in working.index[eligible_threshold & working["threshold_flip_count"].isna()]:
        threshold_labels = [
            int(new_countries.loc[idx] >= threshold) for threshold in (1, 2, 3, 4, 5)
        ]
        inferred_flip.loc[idx] = float(
            sum(value != default_status.loc[idx] for value in threshold_labels)
        )
    working.loc[working["threshold_flip_count"].isna(), "threshold_flip_count"] = inferred_flip.loc[
        working["threshold_flip_count"].isna()
    ]
    working.loc[working["eligible_for_threshold_audit"].isna(), "eligible_for_threshold_audit"] = (
        False
    )
    working["eligible_for_threshold_audit"] = (
        working["eligible_for_threshold_audit"].astype(bool) | eligible_threshold
    )

    positives = working.loc[working["spread_label"].fillna(0).astype(int) == 1].copy()
    if positives.empty:
        return pd.DataFrame()
    shortlist_cutoffs = tuple(sorted({int(value) for value in shortlist_cutoffs if int(value) > 0}))
    primary_cutoff = shortlist_cutoffs[0] if shortlist_cutoffs else 25
    positives = positives.loc[positives["primary_rank"] > primary_cutoff].copy()
    if positives.empty:
        return pd.DataFrame()

    positives["low_knownness_flag"] = (
        positives["knownness_score"].fillna(1.0) <= knownness_threshold
        if math.isfinite(knownness_threshold)
        else False
    )
    positives["low_backbone_purity_flag"] = (
        positives.get(
            "backbone_purity_score",
            pd.Series(np.nan, index=positives.index, dtype=float),
        )
        .fillna(1.0)
        .astype(float)
        < 0.55
    )
    positives["low_assignment_confidence_flag"] = (
        positives.get(
            "assignment_confidence_score",
            pd.Series(np.nan, index=positives.index, dtype=float),
        )
        .fillna(1.0)
        .astype(float)
        < 0.55
    )
    positives["low_metadata_quality_flag"] = (
        positives.get(
            "metadata_quality_score",
            pd.Series(np.nan, index=positives.index, dtype=float),
        )
        .fillna(1.0)
        .astype(float)
        < 0.55
    )
    positives["threshold_fragile_flag"] = (
        pd.to_numeric(
            positives.get("threshold_flip_count", pd.Series(np.nan, index=positives.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
        >= 2.0
    )
    positives["low_external_host_support_flag"] = (
        positives.get(
            "H_external_host_range_support",
            pd.Series(np.nan, index=positives.index, dtype=float),
        )
        .fillna(1.0)
        .astype(float)
        < 0.35
    )
    positives["low_training_members_flag"] = (
        positives.get(
            "member_count_train",
            pd.Series(np.nan, index=positives.index, dtype=float),
        )
        .fillna(99.0)
        .astype(float)
        <= 2.0
    )

    flag_columns = [
        "low_knownness_flag",
        "low_backbone_purity_flag",
        "low_assignment_confidence_flag",
        "low_metadata_quality_flag",
        "threshold_fragile_flag",
        "low_external_host_support_flag",
        "low_training_members_flag",
    ]
    positives["miss_driver_count"] = positives[flag_columns].sum(axis=1).astype(int)

    def _driver_text(row: pd.Series) -> str:
        labels = []
        mapping = {
            "low_knownness_flag": "low_knownness",
            "low_backbone_purity_flag": "low_backbone_purity",
            "low_assignment_confidence_flag": "low_assignment_confidence",
            "low_metadata_quality_flag": "low_metadata_quality",
            "threshold_fragile_flag": "threshold_fragile",
            "low_external_host_support_flag": "low_external_host_support",
            "low_training_members_flag": "low_training_members",
        }
        for column, label in mapping.items():
            if bool(row.get(column, False)):
                labels.append(label)
        return ",".join(labels) if labels else "none"

    positives["miss_driver_flags"] = positives.apply(_driver_text, axis=1).fillna("")
    for cutoff in shortlist_cutoffs:
        positives[f"missed_by_top_{cutoff}"] = positives["primary_rank"] > int(cutoff)

    preferred = [
        "backbone_id",
        "primary_rank",
        "primary_rank_percentile",
        "oof_prediction",
        "n_new_countries",
        "spread_label",
        "member_count_train",
        "n_countries_train",
        "knownness_score",
        "metadata_quality_score",
        "metadata_quality_tier",
        "backbone_purity_score",
        "assignment_confidence_score",
        "threshold_flip_count",
        "H_external_host_range_support",
        "miss_driver_count",
        "miss_driver_flags",
    ] + [f"missed_by_top_{cutoff}" for cutoff in shortlist_cutoffs]
    available = [column for column in preferred if column in positives.columns]
    return (
        positives.sort_values(
            ["n_new_countries", "primary_rank", "miss_driver_count"],
            ascending=[False, True, False],
        )[available]
        .head(top_n)
        .reset_index(drop=True)
    )


def build_confirmatory_cohort_summary(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_names: list[str],
    metadata_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate models on stricter internal cohorts with higher metadata integrity."""
    if scored.empty or predictions.empty or not model_names:
        return pd.DataFrame()

    working = scored.copy()
    if metadata_quality is not None and not metadata_quality.empty:
        meta_columns = [
            column
            for column in [
                "backbone_id",
                "metadata_quality_score",
                "metadata_quality_tier",
                "country_coverage_fraction",
                "duplicate_fraction",
            ]
            if column in metadata_quality.columns
        ]
        if meta_columns:
            working = working.merge(
                metadata_quality[meta_columns].drop_duplicates("backbone_id"),
                on="backbone_id",
                how="left",
            )

    eligible = working.loc[working["spread_label"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()

    meta_score = pd.to_numeric(
        eligible.get("metadata_quality_score", pd.Series(np.nan, index=eligible.index)),
        errors="coerce",
    )
    country_coverage = pd.to_numeric(
        eligible.get("country_coverage_fraction", pd.Series(np.nan, index=eligible.index)),
        errors="coerce",
    )
    duplicate_fraction = pd.to_numeric(
        eligible.get("duplicate_fraction", pd.Series(np.nan, index=eligible.index)),
        errors="coerce",
    )

    cohort_masks = {
        "overall_eligible": pd.Series(True, index=eligible.index),
        "confirmatory_internal": (
            meta_score.ge(0.75).fillna(False)
            & country_coverage.ge(0.50).fillna(False)
            & duplicate_fraction.le(0.25).fillna(False)
        ),
        "confirmatory_strict": (
            meta_score.ge(0.85).fillna(False)
            & country_coverage.ge(0.65).fillna(False)
            & duplicate_fraction.le(0.15).fillna(False)
        ),
    }

    rows: list[dict[str, object]] = []
    for cohort_name, mask in cohort_masks.items():
        cohort = eligible.loc[mask].copy()
        if cohort.empty:
            continue
        cohort_size = int(len(cohort))
        share = cohort_size / max(int(len(eligible)), 1)
        for model_name in model_names:
            model_scores = predictions.loc[
                predictions["model_name"].astype(str) == str(model_name),
                ["backbone_id", "oof_prediction"],
            ].copy()
            if model_scores.empty:
                continue
            merged = cohort.merge(model_scores, on="backbone_id", how="inner")
            valid = merged["spread_label"].notna() & merged["oof_prediction"].notna()
            merged = merged.loc[valid].copy()
            if merged.empty:
                rows.append(
                    {
                        "cohort_name": cohort_name,
                        "model_name": model_name,
                        "status": "skipped_no_predictions",
                        "n_backbones": 0,
                        "n_positive": 0,
                        "positive_prevalence": np.nan,
                        "share_of_primary_eligible": share,
                    }
                )
                continue
            y = merged["spread_label"].astype(int).to_numpy()
            p = merged["oof_prediction"].astype(float).to_numpy()
            if len(np.unique(y)) < 2:
                rows.append(
                    {
                        "cohort_name": cohort_name,
                        "model_name": model_name,
                        "status": "skipped_insufficient_label_variation",
                        "n_backbones": int(len(merged)),
                        "n_positive": int(y.sum()),
                        "positive_prevalence": float(positive_prevalence(y)),
                        "share_of_primary_eligible": share,
                    }
                )
                continue
            rows.append(
                {
                    "cohort_name": cohort_name,
                    "model_name": model_name,
                    "status": "ok",
                    "n_backbones": int(len(merged)),
                    "n_positive": int(y.sum()),
                    "positive_prevalence": float(positive_prevalence(y)),
                    "share_of_primary_eligible": float(share),
                    "roc_auc": float(roc_auc_score(y, p)),
                    "average_precision": float(average_precision(y, p)),
                    "brier_score": float(brier_score(y, p)),
                    "ece": float(expected_calibration_error(y, p)),
                    "expected_calibration_error": float(expected_calibration_error(y, p)),
                    "max_calibration_error": float(max_calibration_error(y, p)),
                }
            )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["cohort_name", "status", "roc_auc", "average_precision"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_event_timing_outcomes(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
) -> pd.DataFrame:
    """Derive event-time style outcomes from post-split country expansion."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    working["backbone_id"] = working["backbone_id"].astype(str)
    working["resolved_year"] = (
        pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    )
    working["country_clean"] = _clean_text(_series_or_default(working, "country"))
    training = working.loc[
        (working["resolved_year"] <= split_year) & working["country_clean"].ne("")
    ].copy()
    testing = working.loc[
        (working["resolved_year"] > split_year)
        & (working["resolved_year"] <= test_year_end)
        & working["country_clean"].ne("")
    ].copy()
    backbone_order = working["backbone_id"].drop_duplicates().tolist()
    train_country_pairs = training[["backbone_id", "country_clean"]].drop_duplicates()
    test_country_first = (
        testing[["backbone_id", "country_clean", "resolved_year"]]
        .sort_values(["backbone_id", "resolved_year", "country_clean"], kind="mergesort")
        .drop_duplicates(["backbone_id", "country_clean"], keep="first")
    )
    new_country_events = test_country_first.merge(
        train_country_pairs,
        on=["backbone_id", "country_clean"],
        how="left",
        indicator=True,
    )
    new_country_events = new_country_events.loc[
        new_country_events["_merge"] == "left_only",
        ["backbone_id", "country_clean", "resolved_year"],
    ].copy()
    new_country_events["event_rank"] = new_country_events.groupby(
        "backbone_id", sort=False
    ).cumcount()
    result = pd.DataFrame({"backbone_id": backbone_order})
    n_train_countries = (
        result["backbone_id"]
        .map(train_country_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    result["n_new_countries_recomputed"] = (
        result["backbone_id"]
        .map(new_country_events.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    first_year = new_country_events.groupby("backbone_id", sort=False)["resolved_year"].min()
    third_year = new_country_events.loc[new_country_events["event_rank"] == 2].set_index(
        "backbone_id"
    )["resolved_year"]
    result["time_to_first_new_country_years"] = result["backbone_id"].map(first_year).astype(
        float
    ) - float(split_year)
    result.loc[result["backbone_id"].map(first_year).isna(), "time_to_first_new_country_years"] = (
        math.nan
    )
    result["time_to_third_new_country_years"] = result["backbone_id"].map(third_year).astype(
        float
    ) - float(split_year)
    result.loc[result["backbone_id"].map(third_year).isna(), "time_to_third_new_country_years"] = (
        math.nan
    )
    eligible = n_train_countries.between(1, 3, inclusive="both")
    first_delta = result["time_to_first_new_country_years"]
    third_delta = result["time_to_third_new_country_years"]
    result["event_within_1y_label"] = np.where(
        eligible, ((first_delta <= 1) & first_delta.notna()).astype(float), np.nan
    )
    result["event_within_3y_label"] = np.where(
        eligible, ((first_delta <= 3) & first_delta.notna()).astype(float), np.nan
    )
    result["event_within_5y_label"] = np.where(
        eligible, ((first_delta <= 5) & first_delta.notna()).astype(float), np.nan
    )
    result["three_countries_within_3y_label"] = np.where(
        eligible, ((third_delta <= 3) & third_delta.notna()).astype(float), np.nan
    )
    result["three_countries_within_5y_label"] = np.where(
        eligible, ((third_delta <= 5) & third_delta.notna()).astype(float), np.nan
    )
    severity = pd.Series(np.nan, index=result.index, dtype=float)
    severity.loc[eligible & result["n_new_countries_recomputed"].eq(0)] = 0.0
    severity.loc[
        eligible & result["n_new_countries_recomputed"].between(1, 2, inclusive="both")
    ] = 1.0
    severity.loc[
        eligible & result["n_new_countries_recomputed"].between(3, 4, inclusive="both")
    ] = 2.0
    severity.loc[eligible & result["n_new_countries_recomputed"].ge(5)] = 3.0
    result["spread_severity_bin"] = severity
    return result


def build_ordinal_outcome_alignment(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    ordinal_column: str,
    model_names: list[str],
) -> pd.DataFrame:
    """Evaluate how well rankings align with ordinal severity-style outcomes."""
    if predictions.empty or outcomes.empty or ordinal_column not in outcomes.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base = outcomes[["backbone_id", ordinal_column]].copy()
    for model_name in model_names:
        model_rows = predictions.loc[
            predictions["model_name"] == model_name, ["backbone_id", "oof_prediction"]
        ].copy()
        if model_rows.empty:
            continue
        merged = base.merge(model_rows, on="backbone_id", how="inner")
        valid = merged[ordinal_column].notna() & merged["oof_prediction"].notna()
        if int(valid.sum()) < 5:
            rows.append(
                {
                    "model_name": model_name,
                    "ordinal_column": ordinal_column,
                    "status": "skipped_insufficient_rows",
                }
            )
            continue
        severity = merged.loc[valid, ordinal_column].astype(float)
        score = merged.loc[valid, "oof_prediction"].astype(float)
        top25 = merged.loc[valid].sort_values("oof_prediction", ascending=False).head(25)
        rows.append(
            {
                "model_name": model_name,
                "ordinal_column": ordinal_column,
                "status": "ok",
                "n_backbones": int(valid.sum()),
                "spearman_corr": _safe_series_correlation(
                    severity.rank(method="average"), score.rank(method="average")
                ),
                "pearson_corr": _safe_series_correlation(severity, score),
                "mean_ordinal_top_25": float(top25[ordinal_column].mean())
                if not top25.empty
                else float("nan"),
                "mean_ordinal_overall": float(severity.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_country_missingness_bounds(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
    threshold: int = 3,
) -> pd.DataFrame:
    """Summarize observed vs bounded new-country outcomes under country missingness."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    working["backbone_id"] = working["backbone_id"].astype(str)
    working["resolved_year"] = (
        pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    )
    working["country_clean"] = _clean_text(_series_or_default(working, "country"))
    training = working.loc[working["resolved_year"] <= split_year].copy()
    testing = working.loc[
        (working["resolved_year"] > split_year) & (working["resolved_year"] <= test_year_end)
    ].copy()
    backbone_order = working["backbone_id"].drop_duplicates().tolist()
    result = pd.DataFrame({"backbone_id": backbone_order})
    train_known_pairs = training.loc[
        training["country_clean"].ne(""), ["backbone_id", "country_clean"]
    ].drop_duplicates()
    test_known_pairs = testing.loc[
        testing["country_clean"].ne(""), ["backbone_id", "country_clean", "sequence_accession"]
    ].drop_duplicates()
    observed_new_pairs = test_known_pairs.merge(
        train_known_pairs,
        on=["backbone_id", "country_clean"],
        how="left",
        indicator=True,
    )
    observed_new_pairs = observed_new_pairs.loc[
        observed_new_pairs["_merge"] == "left_only", ["backbone_id", "country_clean"]
    ].drop_duplicates()
    result["n_countries_train"] = (
        result["backbone_id"]
        .map(train_known_pairs.groupby("backbone_id", sort=False).size())
        .fillna(0)
        .astype(int)
    )
    training_missing_fraction = (
        training.assign(country_missing=training["country_clean"].eq(""))
        .groupby("backbone_id", sort=False)["country_missing"]
        .mean()
    )
    testing_missing_fraction = (
        testing.assign(country_missing=testing["country_clean"].eq(""))
        .groupby("backbone_id", sort=False)["country_missing"]
        .mean()
    )
    missing_test_records = (
        testing.loc[testing["country_clean"].eq("")].groupby("backbone_id", sort=False).size()
    )
    known_test_records = (
        testing.loc[testing["country_clean"].ne("")].groupby("backbone_id", sort=False).size()
    )
    observed_new = observed_new_pairs.groupby("backbone_id", sort=False).size()
    result["training_country_missing_fraction"] = (
        result["backbone_id"].map(training_missing_fraction).fillna(0.0).astype(float)
    )
    result["testing_country_missing_fraction"] = (
        result["backbone_id"].map(testing_missing_fraction).fillna(0.0).astype(float)
    )
    result["observed_new_countries"] = result["backbone_id"].map(observed_new).fillna(0).astype(int)
    missing_counts = result["backbone_id"].map(missing_test_records).fillna(0).astype(int)
    known_counts = result["backbone_id"].map(known_test_records).fillna(0).astype(int)
    result["optimistic_new_countries"] = result["observed_new_countries"] + missing_counts
    result["midpoint_new_countries"] = result["observed_new_countries"] + np.ceil(
        missing_counts / 2.0
    ).astype(int)
    result["weighted_new_countries"] = np.where(
        (result["observed_new_countries"] > 0) | (missing_counts > 0),
        result["observed_new_countries"].astype(float)
        * (1.0 + (missing_counts / np.maximum(known_counts, 1))),
        0.0,
    )
    result["eligible_for_country_bounds"] = result["n_countries_train"].between(
        1, 3, inclusive="both"
    )
    result["label_observed"] = np.where(
        result["eligible_for_country_bounds"],
        result["observed_new_countries"].ge(threshold).astype(float),
        np.nan,
    )
    result["label_midpoint"] = np.where(
        result["eligible_for_country_bounds"],
        result["midpoint_new_countries"].ge(threshold).astype(float),
        np.nan,
    )
    result["label_optimistic"] = np.where(
        result["eligible_for_country_bounds"],
        result["optimistic_new_countries"].ge(threshold).astype(float),
        np.nan,
    )
    result["label_weighted"] = np.where(
        result["eligible_for_country_bounds"],
        result["weighted_new_countries"].ge(threshold).astype(float),
        np.nan,
    )
    return result


def build_missingness_sensitivity_performance(
    predictions: pd.DataFrame,
    bounds: pd.DataFrame,
    *,
    model_names: list[str],
) -> pd.DataFrame:
    """Evaluate models under alternate country-missingness label scenarios."""
    if predictions.empty or bounds.empty:
        return pd.DataFrame()
    outcome_columns = ["label_observed", "label_midpoint", "label_optimistic", "label_weighted"]
    return build_secondary_outcome_performance(
        predictions,
        bounds,
        outcome_columns=[column for column in outcome_columns if column in bounds.columns],
        model_names=model_names,
    )


def build_geographic_jump_distance_table(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    test_year_end: int = 2023,
) -> pd.DataFrame:
    """Quantify geographic expansion distance rather than only country counts."""
    if records.empty:
        return pd.DataFrame()
    working = records.copy()
    working["resolved_year"] = (
        pd.to_numeric(working["resolved_year"], errors="coerce").fillna(0).astype(int)
    )
    working["country_clean"] = _clean_text(_series_or_default(working, "country"))
    working["macro_region"] = working["country_clean"].map(country_to_macro_region)
    working["lat"] = _series_or_default(working, "LOCATION_lat").map(_parse_float)
    working["lng"] = _series_or_default(working, "LOCATION_lng").map(_parse_float)
    training = working.loc[working["resolved_year"] <= split_year].copy()
    testing = working.loc[
        (working["resolved_year"] > split_year) & (working["resolved_year"] <= test_year_end)
    ].copy()
    train_groups = {
        str(backbone_id): frame.copy()
        for backbone_id, frame in training.groupby("backbone_id", sort=False)
    }
    test_groups = {
        str(backbone_id): frame.copy()
        for backbone_id, frame in testing.groupby("backbone_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for backbone_id in working["backbone_id"].astype(str).drop_duplicates().tolist():
        train_frame = train_groups.get(
            str(backbone_id), pd.DataFrame(columns=training.columns)
        ).copy()
        test_frame = test_groups.get(str(backbone_id), pd.DataFrame(columns=testing.columns)).copy()
        train_coords = [
            (float(lat), float(lng))
            for lat, lng in zip(train_frame["lat"], train_frame["lng"])
            if lat is not None and lng is not None
        ]
        train_countries = set(_clean_text(train_frame["country_clean"]).loc[lambda s: s != ""])
        test_new = test_frame.loc[~test_frame["country_clean"].isin(train_countries)].copy()
        jump_distances: list[float] = []
        for row in test_new.itertuples(index=False):
            if row.lat is None or row.lng is None or not train_coords:
                continue
            jump_distances.append(
                min(
                    _haversine_km(float(row.lat), float(row.lng), base_lat, base_lng)
                    for base_lat, base_lng in train_coords
                )
            )
        train_regions = set(_clean_text(train_frame["macro_region"]).loc[lambda s: s != ""])
        test_regions = set(_clean_text(test_new["macro_region"]).loc[lambda s: s != ""])
        interregional = int(len(test_regions - train_regions))
        max_jump = float(max(jump_distances)) if jump_distances else 0.0
        mean_jump = float(np.mean(jump_distances)) if jump_distances else 0.0
        dispersion = float(np.std(jump_distances)) if len(jump_distances) >= 2 else 0.0
        eligible = 1 <= len(train_countries) <= 3
        rows.append(
            {
                "backbone_id": str(backbone_id),
                "mean_jump_distance_km": mean_jump,
                "max_jump_distance_km": max_jump,
                "jump_distance_dispersion_km": dispersion,
                "interregional_leap_count": interregional,
                "long_jump_label": float(int(max_jump >= 2000.0 or interregional >= 1))
                if eligible
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_duplicate_completeness_change_audit(
    records: pd.DataFrame,
    nucc_identical: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Aggregate duplicate/completeness/change flags into backbone-level quality audits."""
    if records.empty:
        return pd.DataFrame()
    training = records.loc[
        pd.to_numeric(records["resolved_year"], errors="coerce").fillna(0).astype(int) <= split_year
    ].copy()
    if training.empty:
        return pd.DataFrame()
    training["sequence_accession"] = training["sequence_accession"].astype(str)
    merged = training.merge(
        nucc_identical.rename(columns={"NUCCORE_ACC": "sequence_accession"}),
        on="sequence_accession",
        how="left",
    )
    if not changes.empty:
        change_payload = changes.rename(columns={"NUCCORE_ACC": "sequence_accession"}).copy()
        change_payload["change_flag_present"] = _clean_text(
            _series_or_default(change_payload, "Flag")
        ).ne("") | _clean_text(_series_or_default(change_payload, "Comment")).ne("")
        merged = merged.merge(
            change_payload[["sequence_accession", "change_flag_present"]].drop_duplicates(
                "sequence_accession"
            ),
            on="sequence_accession",
            how="left",
        )
    else:
        merged["change_flag_present"] = False
    completeness = _clean_text(_series_or_default(merged, "NUCCORE_Completeness")).str.lower()
    duplicates = (
        _clean_text(_series_or_default(merged, "NUCCORE_DuplicatedEntry"))
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    merged["is_complete_record"] = completeness.str.contains("complete")
    merged["is_duplicate_record"] = duplicates
    merged["change_flag_present"] = merged["change_flag_present"].fillna(False).astype(bool)
    result = merged.groupby("backbone_id", as_index=False).agg(
        n_training_records=("sequence_accession", "nunique"),
        complete_fraction=("is_complete_record", "mean"),
        duplicate_fraction=("is_duplicate_record", "mean"),
        change_flag_fraction=("change_flag_present", "mean"),
    )
    result["duplicate_completeness_score"] = np.clip(
        0.50 * result["complete_fraction"].fillna(0.0)
        + 0.30 * (1.0 - result["duplicate_fraction"].fillna(0.0))
        + 0.20 * (1.0 - result["change_flag_fraction"].fillna(0.0)),
        0.0,
        1.0,
    )
    return result.sort_values(
        ["duplicate_completeness_score", "n_training_records"], ascending=[False, False]
    ).reset_index(drop=True)


def build_amr_uncertainty_table(
    backbones: pd.DataFrame,
    amr_hits: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Measure annotation agreement across AMR callers at accession and backbone level."""
    if backbones.empty or amr_hits.empty:
        return pd.DataFrame()
    training = backbones.loc[
        pd.to_numeric(backbones["resolved_year"], errors="coerce").fillna(0).astype(int)
        <= split_year
    ].copy()
    if training.empty:
        return pd.DataFrame()
    training["sequence_accession"] = training["sequence_accession"].astype(str)
    hits = amr_hits.copy()
    hits["NUCCORE_ACC"] = hits["NUCCORE_ACC"].astype(str)
    hits["analysis_software_name"] = _clean_text(
        _series_or_default(hits, "analysis_software_name")
    ).str.lower()
    accession_rows: list[dict[str, object]] = []
    software_pairs = {"amrfinderplus", "rgi"}
    training_accessions = set(training["sequence_accession"].astype(str))
    for accession, frame in hits.loc[hits["NUCCORE_ACC"].isin(training_accessions)].groupby(
        "NUCCORE_ACC", sort=False
    ):
        gene_by_tool = {
            tool: set(_clean_text(group["gene_symbol"]).loc[lambda s: s != ""])
            for tool, group in frame.groupby("analysis_software_name", sort=False)
        }
        class_by_tool = {
            tool: set().union(
                *[
                    _split_field_tokens(value)
                    for value in group.get("drug_class", pd.Series(dtype=object))
                ]
            )
            for tool, group in frame.groupby("analysis_software_name", sort=False)
        }
        tools = sorted(set(gene_by_tool) & software_pairs) or sorted(gene_by_tool)
        if len(tools) >= 2:
            left, right = tools[0], tools[1]
            gene_union = gene_by_tool[left] | gene_by_tool[right]
            class_union = class_by_tool[left] | class_by_tool[right]
            gene_jaccard = len(gene_by_tool[left] & gene_by_tool[right]) / max(len(gene_union), 1)
            class_jaccard = len(class_by_tool[left] & class_by_tool[right]) / max(
                len(class_union), 1
            )
        else:
            gene_jaccard = 1.0 if tools else math.nan
            class_jaccard = 1.0 if tools else math.nan
        accession_rows.append(
            {
                "sequence_accession": str(accession),
                "n_software": int(len(gene_by_tool)),
                "gene_agreement_jaccard": float(gene_jaccard)
                if pd.notna(gene_jaccard)
                else math.nan,
                "class_agreement_jaccard": float(class_jaccard)
                if pd.notna(class_jaccard)
                else math.nan,
                "amr_uncertainty_score": float(1.0 - np.nanmean([gene_jaccard, class_jaccard]))
                if pd.notna(gene_jaccard) or pd.notna(class_jaccard)
                else math.nan,
            }
        )
    accession_table = pd.DataFrame(accession_rows)
    if accession_table.empty:
        return accession_table
    merged = (
        training[["backbone_id", "sequence_accession"]]
        .drop_duplicates()
        .merge(accession_table, on="sequence_accession", how="left")
    )
    result = merged.groupby("backbone_id", as_index=False).agg(
        n_training_records=("sequence_accession", "nunique"),
        mean_amr_software_count=("n_software", "mean"),
        mean_gene_agreement_jaccard=("gene_agreement_jaccard", "mean"),
        mean_class_agreement_jaccard=("class_agreement_jaccard", "mean"),
        mean_amr_uncertainty_score=("amr_uncertainty_score", "mean"),
    )
    result["amr_agreement_score"] = np.nanmean(
        [
            result["mean_gene_agreement_jaccard"].fillna(0.0).to_numpy(dtype=float),
            result["mean_class_agreement_jaccard"].fillna(0.0).to_numpy(dtype=float),
        ],
        axis=0,
    )
    return result.sort_values(
        ["mean_amr_uncertainty_score", "n_training_records"], ascending=[True, False]
    ).reset_index(drop=True)


def build_mash_similarity_graph_table(
    records: pd.DataFrame,
    mash_pairs: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Collapse Mash nearest-neighbor edges into training-only backbone graph features."""
    if records.empty or mash_pairs.empty:
        return pd.DataFrame()
    training = records.loc[
        pd.to_numeric(records["resolved_year"], errors="coerce").fillna(0).astype(int) <= split_year
    ].copy()
    training = training.loc[training["sequence_accession"].notna()].copy()
    if training.empty:
        return pd.DataFrame()
    accession_to_backbone = (
        training[["sequence_accession", "backbone_id"]]
        .drop_duplicates("sequence_accession")
        .assign(
            sequence_accession=lambda df: df["sequence_accession"].astype(str),
            backbone_id=lambda df: df["backbone_id"].astype(str),
        )
        .set_index("sequence_accession")["backbone_id"]
        .to_dict()
    )
    if not accession_to_backbone:
        return pd.DataFrame()
    pairs = mash_pairs.copy()
    pairs["left_backbone"] = pairs.iloc[:, 0].astype(str).map(accession_to_backbone)
    pairs["right_backbone"] = pairs.iloc[:, 1].astype(str).map(accession_to_backbone)
    pairs = pairs.loc[pairs["left_backbone"].notna() | pairs["right_backbone"].notna()].copy()
    backbone_order = training["backbone_id"].astype(str).drop_duplicates().tolist()
    result = pd.DataFrame({"backbone_id": backbone_order})
    edge_counts = pd.concat(
        [
            pairs["left_backbone"].dropna().astype(str),
            pairs["right_backbone"].dropna().astype(str),
        ],
        ignore_index=True,
    ).value_counts(sort=False)
    within_counts = (
        pairs.loc[
            pairs["left_backbone"].notna()
            & pairs["right_backbone"].notna()
            & pairs["left_backbone"].astype(str).eq(pairs["right_backbone"].astype(str))
        ]
        .groupby("left_backbone", sort=False)
        .size()
    )
    cross_edges = pairs.loc[
        pairs["left_backbone"].notna()
        & pairs["right_backbone"].notna()
        & pairs["left_backbone"].astype(str).ne(pairs["right_backbone"].astype(str)),
        ["left_backbone", "right_backbone"],
    ].copy()
    if not cross_edges.empty:
        cross_pairs = pd.concat(
            [
                cross_edges.rename(
                    columns={"left_backbone": "backbone_id", "right_backbone": "neighbor_backbone"}
                ),
                cross_edges.rename(
                    columns={"right_backbone": "backbone_id", "left_backbone": "neighbor_backbone"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()
        external_neighbor_count = cross_pairs.groupby("backbone_id", sort=False).size()
    else:
        external_neighbor_count = pd.Series(dtype=int)
    result["mash_graph_edge_count"] = result["backbone_id"].map(edge_counts).fillna(0).astype(int)
    result["mash_graph_within_edge_count"] = (
        result["backbone_id"].map(within_counts).fillna(0).astype(int)
    )
    result["mash_graph_external_neighbor_count"] = (
        result["backbone_id"].map(external_neighbor_count).fillna(0).astype(int)
    )
    result["mash_graph_bridge_fraction"] = (
        (result["mash_graph_edge_count"] - result["mash_graph_within_edge_count"])
        / result["mash_graph_edge_count"].clip(lower=1)
    ).astype(float)
    result["mash_graph_novelty_score"] = (
        1.0 / (1.0 + result["mash_graph_external_neighbor_count"].astype(float))
    ).astype(float)
    return result.sort_values(
        ["mash_graph_external_neighbor_count", "mash_graph_edge_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_counterfactual_shortlist_comparison(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model_name: str,
    baseline_model_name: str,
    top_ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:
    """Compare shortlist yield under matched knownness/source budgets."""
    if scored.empty or predictions.empty:
        return pd.DataFrame()
    eligible = annotate_knownness_metadata(scored.loc[scored["spread_label"].notna()].copy())
    if eligible.empty:
        return pd.DataFrame()
    primary = predictions.loc[
        predictions["model_name"] == primary_model_name, ["backbone_id", "oof_prediction"]
    ].rename(columns={"oof_prediction": "primary_score"})
    baseline = predictions.loc[
        predictions["model_name"] == baseline_model_name, ["backbone_id", "oof_prediction"]
    ].rename(columns={"oof_prediction": "baseline_score"})
    merged = eligible.merge(primary, on="backbone_id", how="inner").merge(
        baseline, on="backbone_id", how="inner"
    )
    if merged.empty:
        return pd.DataFrame()
    merged["matched_source_band"] = np.where(
        merged["refseq_share_train"].fillna(0.0).astype(float) >= 0.5,
        "refseq_leaning",
        "insd_leaning",
    )
    merged["matched_knownness_bin"] = _safe_qcut(merged["knownness_score"], q=4)
    merged["matched_budget_stratum"] = (
        merged["matched_source_band"].astype(str)
        + "|"
        + merged["matched_knownness_bin"].astype(str)
    )

    def _summarize(frame: pd.DataFrame, *, label: str, top_k: int) -> dict[str, object]:
        subset = frame.head(min(top_k, len(frame))).copy()
        positives = subset["spread_label"].fillna(0).astype(int)
        return {
            "selection_mode": label,
            "top_k": int(top_k),
            "n_selected": int(len(subset)),
            "n_positive_selected": int(positives.sum()),
            "precision_at_k": float(positives.mean()) if len(subset) else np.nan,
            "mean_n_new_countries": float(
                pd.to_numeric(
                    subset.get("n_new_countries", pd.Series(dtype=float)), errors="coerce"
                ).mean()
            )
            if len(subset)
            else np.nan,
            "mean_knownness_score": float(
                pd.to_numeric(subset["knownness_score"], errors="coerce").mean()
            )
            if len(subset)
            else np.nan,
        }

    rows: list[dict[str, object]] = []
    primary_ranked = merged.sort_values("primary_score", ascending=False).reset_index(drop=True)
    baseline_ranked = merged.sort_values("baseline_score", ascending=False).reset_index(drop=True)
    for top_k in top_ks:
        rows.append(_summarize(primary_ranked, label="primary_natural", top_k=top_k))
        rows.append(_summarize(baseline_ranked, label="baseline_natural", top_k=top_k))

        baseline_top = baseline_ranked.head(min(top_k, len(baseline_ranked))).copy()
        primary_matched_frames: list[pd.DataFrame] = []
        for stratum, frame in baseline_top.groupby("matched_budget_stratum", sort=False):
            quota = int(len(frame))
            primary_candidates = primary_ranked.loc[
                primary_ranked["matched_budget_stratum"] == stratum
            ].head(quota)
            if not primary_candidates.empty:
                primary_matched_frames.append(primary_candidates)
        primary_matched = (
            pd.concat(primary_matched_frames, ignore_index=True, sort=False)
            if primary_matched_frames
            else pd.DataFrame(columns=merged.columns)
        )
        rows.append(
            _summarize(
                primary_matched.sort_values("primary_score", ascending=False),
                label="primary_matched_to_baseline",
                top_k=top_k,
            )
        )

        primary_top = primary_ranked.head(min(top_k, len(primary_ranked))).copy()
        baseline_matched_frames: list[pd.DataFrame] = []
        for stratum, frame in primary_top.groupby("matched_budget_stratum", sort=False):
            quota = int(len(frame))
            baseline_candidates = baseline_ranked.loc[
                baseline_ranked["matched_budget_stratum"] == stratum
            ].head(quota)
            if not baseline_candidates.empty:
                baseline_matched_frames.append(baseline_candidates)
        baseline_matched = (
            pd.concat(baseline_matched_frames, ignore_index=True, sort=False)
            if baseline_matched_frames
            else pd.DataFrame(columns=merged.columns)
        )
        rows.append(
            _summarize(
                baseline_matched.sort_values("baseline_score", ascending=False),
                label="baseline_matched_to_primary",
                top_k=top_k,
            )
        )
    return pd.DataFrame(rows)
