"""Supportive descriptive analysis against Pathogen Detection metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from plasmid_priority.reporting.external_support import select_priority_groups


class _SupportCount(TypedDict):
    pd_species_record_count: int
    pd_matching_record_count: int
    matched_genes: set[str]
    matched_locations: Counter[str]


def normalize_species_token(value: object) -> str:
    """Normalize species-style labels to a conservative underscore form."""
    if pd.isna(cast(Any, value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return "_".join(text.split())


def extract_pd_gene_symbols(value: object) -> set[str]:
    """Extract AMR gene symbols from a Pathogen Detection genotype string."""
    if pd.isna(cast(Any, value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    genes = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        genes.add(item.split("=", 1)[0].strip())
    return genes


def _top_items(series: pd.Series, *, n: int = 3) -> list[str]:
    counter: Counter[str] = Counter()
    for value in series.fillna(""):
        for item in [part.strip() for part in str(value).split(",") if part.strip()]:
            counter[item] += 1
    return [item for item, _ in counter.most_common(n)]


def build_pathogen_targets(
    scored: pd.DataFrame,
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    *,
    n_per_group: int = 100,
    score_column: str = "priority_index",
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Build descriptive Pathogen Detection lookup targets for high and low priority groups."""
    selected = select_priority_groups(
        scored,
        n_per_group=n_per_group,
        score_column=score_column,
        eligible_only=eligible_only,
    )
    if selected.empty:
        return pd.DataFrame()

    merged = backbones.merge(
        selected[
            [
                "backbone_id",
                "priority_group",
                "priority_index",
                "selection_score",
                "selection_score_column",
            ]
        ],
        on="backbone_id",
        how="inner",
    )
    merged = merged.merge(amr_consensus, on="sequence_accession", how="left")

    rows: list[dict[str, object]] = []
    for backbone_id, frame in merged.groupby("backbone_id", sort=False):
        priority_group = frame["priority_group"].iloc[0]
        priority_index = float(frame["priority_index"].iloc[0])
        selection_score = float(frame["selection_score"].iloc[0])
        species_counts = (
            frame["species"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
            .dropna()
            .value_counts()
        )
        dominant_species = species_counts.index[0] if not species_counts.empty else ""
        dominant_genus = (
            frame["genus"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
            .dropna()
            .value_counts()
            .index[0]
            if frame["genus"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
            .dropna()
            .shape[0]
            > 0
            else ""
        )
        rows.append(
            {
                "backbone_id": backbone_id,
                "priority_group": priority_group,
                "priority_index": priority_index,
                "selection_score": selection_score,
                "selection_score_column": str(frame["selection_score_column"].iloc[0]),
                "dominant_species": dominant_species,
                "dominant_species_norm": normalize_species_token(dominant_species),
                "dominant_genus": dominant_genus,
                "top_amr_genes": ",".join(_top_items(frame["amr_gene_symbols"], n=5)),
                "top_amr_classes": ",".join(_top_items(frame["amr_drug_classes"], n=5)),
            }
        )

    targets = pd.DataFrame(rows)
    targets["target_gene_set"] = targets["top_amr_genes"].map(
        lambda text: {x for x in text.split(",") if x}
    )
    return targets


def build_pathogen_detection_support(
    targets: pd.DataFrame,
    pathogen_metadata_path: Path,
    *,
    chunk_size: int = 50000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute descriptive support counts by matching dominant species and AMR genes."""
    if targets.empty:
        empty = pd.DataFrame()
        return empty, empty

    targets = targets.copy()
    by_species: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in targets.to_dict(orient="records"):
        row_dict = {str(key): value for key, value in row.items()}
        species = str(row_dict["dominant_species_norm"])
        if species:
            by_species[species].append(row_dict)

    support_counts: dict[str, _SupportCount] = {
        row["backbone_id"]: {
            "pd_species_record_count": 0,
            "pd_matching_record_count": 0,
            "matched_genes": set(),
            "matched_locations": Counter(),
        }
        for row in targets.to_dict(orient="records")
    }

    usecols = ["#Organism group", "Scientific name", "AMR genotypes", "Location"]
    for chunk in pd.read_csv(
        pathogen_metadata_path, sep="\t", usecols=usecols, chunksize=chunk_size
    ):
        chunk["species_norm"] = (
            chunk["Scientific name"].fillna(chunk["#Organism group"]).map(normalize_species_token)
        )
        chunk = chunk.loc[chunk["species_norm"].isin(by_species.keys())].copy()
        if chunk.empty:
            continue
        chunk["gene_set"] = chunk["AMR genotypes"].map(extract_pd_gene_symbols)
        species_counts = chunk["species_norm"].value_counts()
        for species_norm, species_count in species_counts.items():
            for target in by_species[str(species_norm)]:
                support_counts[str(target["backbone_id"])]["pd_species_record_count"] += int(
                    species_count
                )
        for row in chunk.itertuples(index=False):
            species_norm = row.species_norm
            location = "" if pd.isna(row.Location) else str(row.Location).strip()
            gene_set = row.gene_set
            for target in by_species[species_norm]:
                record = support_counts[str(target["backbone_id"])]
                matched_genes = gene_set & target["target_gene_set"]
                if matched_genes:
                    record["pd_matching_record_count"] += 1
                    record["matched_genes"].update(matched_genes)
                    if location:
                        record["matched_locations"][location] += 1

    detail_rows: list[dict[str, object]] = []
    for row in targets.to_dict(orient="records"):
        counts = support_counts[str(row["backbone_id"])]
        matched_locations = counts["matched_locations"].most_common(3)
        detail_rows.append(
            {
                "backbone_id": row["backbone_id"],
                "priority_group": row["priority_group"],
                "priority_index": row["priority_index"],
                "selection_score": row["selection_score"],
                "selection_score_column": row["selection_score_column"],
                "dominant_species": row["dominant_species"],
                "dominant_genus": row["dominant_genus"],
                "top_amr_genes": row["top_amr_genes"],
                "top_amr_classes": row["top_amr_classes"],
                "pd_species_record_count": counts["pd_species_record_count"],
                "pd_matching_record_count": counts["pd_matching_record_count"],
                "pd_matching_fraction": (
                    counts["pd_matching_record_count"] / counts["pd_species_record_count"]
                    if counts["pd_species_record_count"] > 0
                    else 0.0
                ),
                "pd_matching_gene_count": len(counts["matched_genes"]),
                "pd_matched_genes": ",".join(sorted(counts["matched_genes"])),
                "pd_top_locations": ",".join(f"{loc}:{count}" for loc, count in matched_locations),
                "pd_any_support": counts["pd_matching_record_count"] > 0,
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["priority_group", "pd_matching_record_count", "selection_score"],
        ascending=[True, False, False],
    )
    summary = (
        detail.groupby("priority_group")
        .agg(
            n_backbones=("backbone_id", "nunique"),
            n_with_any_support=("pd_any_support", "sum"),
            mean_matching_records=("pd_matching_record_count", "mean"),
            median_matching_records=("pd_matching_record_count", "median"),
            mean_matching_fraction=("pd_matching_fraction", "mean"),
            median_matching_fraction=("pd_matching_fraction", "median"),
            mean_matching_genes=("pd_matching_gene_count", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def build_pathogen_strata_group_summary(summary_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset Pathogen Detection summaries into one table."""
    rows: list[pd.DataFrame] = []
    for dataset_name, frame in summary_frames.items():
        if frame.empty:
            continue
        working = frame.copy()
        working.insert(0, "pathogen_dataset", dataset_name)
        rows.append(working)
    if not rows:
        return pd.DataFrame(
            columns=[
                "pathogen_dataset",
                "priority_group",
                "n_backbones",
                "n_with_any_support",
                "mean_matching_records",
                "median_matching_records",
                "mean_matching_fraction",
                "median_matching_fraction",
                "mean_matching_genes",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _permutation_pvalue(
    high_values: np.ndarray,
    low_values: np.ndarray,
    *,
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    if len(high_values) == 0 or len(low_values) == 0:
        return float("nan")
    observed = float(high_values.mean() - low_values.mean())
    pooled = np.concatenate([high_values, low_values])
    n_high = len(high_values)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        delta = float(permuted[:n_high].mean() - permuted[n_high:].mean())
        extreme += int(abs(delta) >= abs(observed))
    return float((extreme + 1) / (n_permutations + 1))


def build_pathogen_group_comparison(
    detail: pd.DataFrame,
    *,
    n_permutations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare high- and low-priority Pathogen Detection support with explicit matching rules."""
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "pathogen_dataset",
                "matching_rule",
                "n_high",
                "n_low",
                "mean_matching_fraction_high",
                "mean_matching_fraction_low",
                "delta_mean_matching_fraction_high_minus_low",
                "support_fraction_high",
                "support_fraction_low",
                "delta_support_fraction_high_minus_low",
                "permutation_p_mean_matching_fraction",
                "permutation_p_support_fraction",
            ]
        )

    working = detail.copy()
    if "pathogen_dataset" not in working.columns:
        working["pathogen_dataset"] = "combined"

    rows: list[dict[str, object]] = []
    for dataset_name, frame in working.groupby("pathogen_dataset", sort=False):
        high = frame.loc[frame["priority_group"] == "high"].copy()
        low = frame.loc[frame["priority_group"] == "low"].copy()
        if high.empty or low.empty:
            continue
        high_fraction = high["pd_matching_fraction"].fillna(0.0).to_numpy(dtype=float)
        low_fraction = low["pd_matching_fraction"].fillna(0.0).to_numpy(dtype=float)
        high_support = high["pd_any_support"].fillna(False).astype(int).to_numpy(dtype=float)
        low_support = low["pd_any_support"].fillna(False).astype(int).to_numpy(dtype=float)
        rows.append(
            {
                "pathogen_dataset": str(dataset_name),
                "matching_rule": (
                    "dominant species exact match plus at least one shared top backbone AMR gene"
                ),
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "mean_matching_fraction_high": float(high_fraction.mean()),
                "mean_matching_fraction_low": float(low_fraction.mean()),
                "delta_mean_matching_fraction_high_minus_low": float(
                    high_fraction.mean() - low_fraction.mean()
                ),
                "support_fraction_high": float(high_support.mean()),
                "support_fraction_low": float(low_support.mean()),
                "delta_support_fraction_high_minus_low": float(
                    high_support.mean() - low_support.mean()
                ),
                "permutation_p_mean_matching_fraction": _permutation_pvalue(
                    high_fraction,
                    low_fraction,
                    n_permutations=n_permutations,
                    seed=seed,
                ),
                "permutation_p_support_fraction": _permutation_pvalue(
                    high_support,
                    low_support,
                    n_permutations=n_permutations,
                    seed=seed + 1,
                ),
            }
        )
    return pd.DataFrame(rows)
