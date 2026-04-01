"""Supportive analyses driven by optional external reference datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd


# Canonical source is schemas.who_mia; re-exported here for backward
# compatibility with scripts that import from this module.
from plasmid_priority.schemas.who_mia import WHO_MIA_CLASS_MAP  # noqa: F401


def split_field_tokens(value: object, *, separators: tuple[str, ...] = (",",)) -> list[str]:
    """Split a delimited text field into cleaned tokens."""
    if pd.isna(value):
        return []
    tokens = [str(value).strip()]
    for separator in separators:
        expanded: list[str] = []
        for token in tokens:
            expanded.extend(token.split(separator))
        tokens = expanded
    return [token.strip() for token in tokens if token and token.strip() and token.strip() != "-"]


def normalize_gene_symbol(value: object) -> str:
    """Normalize gene symbols for conservative exact-or-format-only matching."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalize_drug_class_token(value: object) -> str:
    """Normalize class labels into conservative uppercase tokens."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    if text.endswith(" ANTIBIOTIC"):
        text = text[: -len(" ANTIBIOTIC")]
    return text.strip()


def _dominant_non_empty(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values.loc[values != ""]
    if values.empty:
        return ""
    return str(values.value_counts().index[0])


def _first_tar_member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    candidates = [
        member
        for member in tar.getmembers()
        if member.name.endswith(suffix) and not member.name.split("/")[-1].startswith("._")
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find archive member ending with {suffix}.")
    return candidates[0]


def _top_counter_items(counter: Counter[str], *, n: int = 5) -> str:
    return ",".join(item for item, _ in counter.most_common(n))


def _empty_support_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def select_priority_groups(
    scored: pd.DataFrame,
    *,
    n_per_group: int = 100,
    score_column: str = "priority_index",
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Select high/low backbone groups from a named score column.

    When `eligible_only` is true, only backbones with a defined `spread_label`
    are considered. This keeps external descriptive support aligned with the
    evaluated cohort of the headline model instead of mixing in unevaluable
    backlog-only rows.
    """
    working = scored.copy()
    working = working.loc[working["member_count_train"].fillna(0).astype(int) > 0].copy()
    if eligible_only and "spread_label" in working.columns:
        working = working.loc[working["spread_label"].notna()].copy()
    if score_column not in working.columns:
        raise KeyError(f"Missing score column for group selection: {score_column}")
    working = working.loc[working[score_column].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["backbone_id", "priority_group", "selection_score"])

    high = working.sort_values(score_column, ascending=False).head(n_per_group).copy()
    low = working.sort_values(score_column, ascending=True).head(n_per_group).copy()
    high["priority_group"] = "high"
    low["priority_group"] = "low"
    selected = pd.concat([high, low], ignore_index=True).drop_duplicates("backbone_id", keep="first")
    selected["selection_score"] = selected[score_column].astype(float)
    selected["selection_score_column"] = score_column
    if "priority_index" not in selected.columns:
        selected["priority_index"] = np.nan
    return selected


def build_who_mia_reference_catalog(who_text_path: Path) -> pd.DataFrame:
    """Summarize the curated WHO MIA class map against the local reference text extraction."""
    normalized_text = re.sub(r"\s+", " ", who_text_path.read_text(encoding="utf-8", errors="ignore")).strip().lower()
    class_tokens: dict[tuple[str, str, str], set[str]] = {}
    for amr_token, mapping in WHO_MIA_CLASS_MAP.items():
        key = (
            str(mapping["who_mia_category"]),
            str(mapping["who_mia_class"]),
            str(mapping["who_mapping_scope"]),
        )
        class_tokens.setdefault(key, set()).add(amr_token)

    rows: list[dict[str, object]] = []
    for (category, who_class, scope), tokens in sorted(class_tokens.items()):
        reference_present = re.sub(r"\s+", " ", who_class).strip().lower() in normalized_text
        rows.append(
            {
                "who_mia_category": category,
                "who_mia_class": who_class,
                "who_mapping_scope": scope,
                "mapped_amr_tokens": ",".join(sorted(tokens)),
                "reference_class_present_in_text": reference_present,
            }
        )
    return pd.DataFrame(rows)


def build_priority_backbone_support_frame(
    scored: pd.DataFrame,
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    *,
    n_per_group: int = 100,
    score_column: str = "priority_index",
    eligible_only: bool = False,
) -> pd.DataFrame:
    """Aggregate per-backbone support inputs for high- and low-priority groups."""
    selected = select_priority_groups(
        scored,
        n_per_group=n_per_group,
        score_column=score_column,
        eligible_only=eligible_only,
    )
    if selected.empty:
        return pd.DataFrame()

    merged = backbones.merge(
        selected[["backbone_id", "priority_group", "priority_index", "selection_score", "selection_score_column"]],
        on="backbone_id",
        how="inner",
    )
    merged = merged.merge(amr_consensus, on="sequence_accession", how="left")

    rows: list[dict[str, object]] = []
    for backbone_id, frame in merged.groupby("backbone_id", sort=False):
        gene_set: set[str] = set()
        class_set: set[str] = set()
        replicon_tokens: set[str] = set()
        for value in frame["amr_gene_symbols"].fillna(""):
            gene_set.update(split_field_tokens(value, separators=(",",)))
        for value in frame["amr_drug_classes"].fillna(""):
            class_set.update(split_field_tokens(value, separators=(",",)))
        for value in frame["replicon_types"].fillna(""):
            replicon_tokens.update(split_field_tokens(value, separators=(",",)))

        primary_replicon = _dominant_non_empty(frame["primary_replicon"])
        if not primary_replicon and replicon_tokens:
            primary_replicon = sorted(replicon_tokens)[0]

        rows.append(
            {
                "backbone_id": backbone_id,
                "priority_group": str(frame["priority_group"].iloc[0]),
                "priority_index": float(frame["priority_index"].iloc[0]),
                "selection_score": float(frame["selection_score"].iloc[0]),
                "selection_score_column": str(frame["selection_score_column"].iloc[0]),
                "dominant_species": _dominant_non_empty(frame["species"]),
                "dominant_genus": _dominant_non_empty(frame["genus"]),
                "sequence_count": int(frame["sequence_accession"].nunique()),
                "primary_replicon": primary_replicon,
                "replicon_types": ",".join(sorted(replicon_tokens)),
                "amr_gene_symbols": ",".join(sorted(gene_set)),
                "amr_gene_count": len(gene_set),
                "amr_drug_classes": ",".join(sorted(class_set)),
                "amr_class_count": len(class_set),
            }
        )

    return pd.DataFrame(rows)


def read_card_ontology(card_archive_path: Path) -> pd.DataFrame:
    """Read CARD ARO index from the provided archive as a normalized lookup table."""
    with tarfile.open(card_archive_path, "r:bz2") as archive:
        member = _first_tar_member(archive, "aro_index.tsv")
        frame = pd.read_csv(
            archive.extractfile(member),
            sep="\t",
            usecols=[
                "CARD Short Name",
                "ARO Name",
                "AMR Gene Family",
                "Drug Class",
                "Resistance Mechanism",
            ],
        )

    rows: list[dict[str, object]] = []
    for row in frame.fillna("").itertuples(index=False):
        name_variants = {str(row[0]).strip(), str(row[1]).strip()} - {""}
        drug_classes = split_field_tokens(row[3], separators=(";",))
        if not drug_classes:
            drug_classes = [""]
        for name_variant in name_variants:
            for drug_class in drug_classes:
                rows.append(
                    {
                        "card_name_variant": name_variant,
                        "card_name_norm": normalize_gene_symbol(name_variant),
                        "card_amr_gene_family": str(row[2]).strip(),
                        "card_drug_class": drug_class,
                        "card_resistance_mechanism": str(row[4]).strip(),
                    }
                )

    lookup = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return lookup


def _build_prevalence_comparison(
    membership: pd.DataFrame,
    *,
    value_column: str,
    group_totals: dict[str, int] | None = None,
) -> pd.DataFrame:
    if membership.empty:
        return _empty_support_frame([value_column, "high", "low", "prevalence_delta_high_minus_low"])

    membership = membership.drop_duplicates(["backbone_id", value_column])
    totals = group_totals or membership.groupby("priority_group")["backbone_id"].nunique().to_dict()
    summary = (
        membership.groupby(["priority_group", value_column])["backbone_id"]
        .nunique()
        .reset_index(name="n_backbones")
    )
    summary["group_total_backbones"] = summary["priority_group"].map(totals)
    summary["prevalence"] = summary["n_backbones"] / summary["group_total_backbones"]
    pivot = summary.pivot(index=value_column, columns="priority_group", values="prevalence").fillna(0.0)
    pivot["prevalence_delta_high_minus_low"] = pivot.get("high", 0.0) - pivot.get("low", 0.0)
    return pivot.reset_index().sort_values("prevalence_delta_high_minus_low", ascending=False)


def build_card_support(
    priority_backbones: pd.DataFrame,
    card_archive_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map observed backbone AMR genes onto CARD families and mechanisms."""
    detail_columns = [
        "backbone_id",
        "priority_group",
        "priority_index",
        "selection_score",
        "selection_score_column",
        "dominant_species",
        "dominant_genus",
        "primary_replicon",
        "amr_gene_count",
        "card_matched_gene_count",
        "card_exact_match_gene_count",
        "card_normalized_match_gene_count",
        "card_match_fraction",
        "card_gene_family_count",
        "card_mechanism_count",
        "card_drug_class_count",
        "card_top_gene_families",
        "card_top_mechanisms",
        "card_top_drug_classes",
        "card_any_support",
    ]
    summary_columns = [
        "priority_group",
        "n_backbones",
        "n_with_any_card_support",
        "mean_card_match_fraction",
        "median_card_match_fraction",
        "mean_card_gene_family_count",
        "mean_card_mechanism_count",
        "mean_card_drug_class_count",
    ]
    family_columns = ["card_amr_gene_family", "high", "low", "prevalence_delta_high_minus_low"]
    mechanism_columns = ["card_resistance_mechanism", "high", "low", "prevalence_delta_high_minus_low"]

    if priority_backbones.empty:
        return (
            _empty_support_frame(detail_columns),
            _empty_support_frame(summary_columns),
            _empty_support_frame(family_columns),
            _empty_support_frame(mechanism_columns),
        )

    lookup = read_card_ontology(card_archive_path)
    detail_rows: list[dict[str, object]] = []
    family_membership: list[dict[str, object]] = []
    mechanism_membership: list[dict[str, object]] = []

    for row in priority_backbones.to_dict(orient="records"):
        genes = split_field_tokens(row["amr_gene_symbols"], separators=(",",))
        exact_count = 0
        normalized_count = 0
        family_counter: Counter[str] = Counter()
        mechanism_counter: Counter[str] = Counter()
        class_counter: Counter[str] = Counter()
        matched_gene_set: set[str] = set()

        for gene in genes:
            gene_norm = normalize_gene_symbol(gene)
            if not gene_norm:
                continue
            matches = lookup.loc[lookup["card_name_norm"] == gene_norm].copy()
            if matches.empty:
                continue

            matched_gene_set.add(gene)
            if (matches["card_name_variant"] == gene).any():
                exact_count += 1
            else:
                normalized_count += 1

            for family in sorted({value for value in matches["card_amr_gene_family"] if value}):
                family_counter[family] += 1
                family_membership.append(
                    {
                        "backbone_id": row["backbone_id"],
                        "priority_group": row["priority_group"],
                        "card_amr_gene_family": family,
                    }
                )
            for mechanism in sorted({value for value in matches["card_resistance_mechanism"] if value}):
                mechanism_counter[mechanism] += 1
                mechanism_membership.append(
                    {
                        "backbone_id": row["backbone_id"],
                        "priority_group": row["priority_group"],
                        "card_resistance_mechanism": mechanism,
                    }
                )
            for drug_class in sorted({value for value in matches["card_drug_class"] if value}):
                class_counter[drug_class] += 1

        matched_gene_count = len(matched_gene_set)
        gene_count = int(row["amr_gene_count"])
        detail_rows.append(
            {
                "backbone_id": row["backbone_id"],
                "priority_group": row["priority_group"],
                "priority_index": row["priority_index"],
                "selection_score": row.get("selection_score", np.nan),
                "selection_score_column": row.get("selection_score_column", "priority_index"),
                "dominant_species": row["dominant_species"],
                "dominant_genus": row["dominant_genus"],
                "primary_replicon": row["primary_replicon"],
                "amr_gene_count": gene_count,
                "card_matched_gene_count": matched_gene_count,
                "card_exact_match_gene_count": exact_count,
                "card_normalized_match_gene_count": normalized_count,
                "card_match_fraction": matched_gene_count / gene_count if gene_count > 0 else 0.0,
                "card_gene_family_count": len(family_counter),
                "card_mechanism_count": len(mechanism_counter),
                "card_drug_class_count": len(class_counter),
                "card_top_gene_families": _top_counter_items(family_counter),
                "card_top_mechanisms": _top_counter_items(mechanism_counter),
                "card_top_drug_classes": _top_counter_items(class_counter),
                "card_any_support": matched_gene_count > 0,
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["priority_group", "card_match_fraction", "selection_score"],
        ascending=[True, False, False],
    )
    summary = (
        detail.groupby("priority_group")
        .agg(
            n_backbones=("backbone_id", "nunique"),
            n_with_any_card_support=("card_any_support", "sum"),
            mean_card_match_fraction=("card_match_fraction", "mean"),
            median_card_match_fraction=("card_match_fraction", "median"),
            mean_card_gene_family_count=("card_gene_family_count", "mean"),
            mean_card_mechanism_count=("card_mechanism_count", "mean"),
            mean_card_drug_class_count=("card_drug_class_count", "mean"),
        )
        .reset_index()
    )
    family_comparison = _build_prevalence_comparison(
        pd.DataFrame(family_membership),
        value_column="card_amr_gene_family",
        group_totals=detail.groupby("priority_group")["backbone_id"].nunique().astype(int).to_dict(),
    )
    mechanism_comparison = _build_prevalence_comparison(
        pd.DataFrame(mechanism_membership),
        value_column="card_resistance_mechanism",
        group_totals=detail.groupby("priority_group")["backbone_id"].nunique().astype(int).to_dict(),
    )
    return detail, summary, family_comparison, mechanism_comparison


def build_who_mia_support(
    priority_backbones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map AMR class tokens to curated WHO MIA categories where the class match is unambiguous."""
    detail_columns = [
        "backbone_id",
        "priority_group",
        "priority_index",
        "selection_score",
        "selection_score_column",
        "dominant_species",
        "dominant_genus",
        "amr_class_count",
        "who_mia_mapped_class_count",
        "who_mia_unmapped_class_count",
        "who_mia_mapped_fraction",
        "who_mia_top_categories",
        "who_mia_top_classes",
        "who_mia_scope_labels",
        "who_mia_any_hpecia",
        "who_mia_any_cia",
        "who_mia_any_hia",
        "who_mia_any_ia",
        "who_mia_any_support",
    ]
    summary_columns = [
        "priority_group",
        "n_backbones",
        "n_with_any_who_support",
        "n_with_hpecia_support",
        "n_with_cia_support",
        "n_with_hia_support",
        "n_with_ia_support",
        "mean_who_mia_mapped_fraction",
        "median_who_mia_mapped_fraction",
        "mean_who_mia_mapped_class_count",
    ]
    comparison_columns = ["who_mia_category", "high", "low", "prevalence_delta_high_minus_low"]

    if priority_backbones.empty:
        return (
            _empty_support_frame(detail_columns),
            _empty_support_frame(summary_columns),
            _empty_support_frame(comparison_columns),
        )

    category_membership: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for row in priority_backbones.to_dict(orient="records"):
        class_tokens = {
            normalize_drug_class_token(token)
            for token in split_field_tokens(row["amr_drug_classes"], separators=(",", ";"))
        } - {""}
        matched_tokens = []
        category_counter: Counter[str] = Counter()
        class_counter: Counter[str] = Counter()
        scope_counter: Counter[str] = Counter()

        for token in sorted(class_tokens):
            mapping = WHO_MIA_CLASS_MAP.get(token)
            if mapping is None:
                continue
            matched_tokens.append(token)
            category = str(mapping["who_mia_category"])
            who_class = str(mapping["who_mia_class"])
            scope = str(mapping["who_mapping_scope"])
            category_counter[category] += 1
            class_counter[who_class] += 1
            scope_counter[scope] += 1
            category_membership.append(
                {
                    "backbone_id": row["backbone_id"],
                    "priority_group": row["priority_group"],
                    "who_mia_category": category,
                }
            )

        total_class_count = len(class_tokens)
        mapped_class_count = len(matched_tokens)
        detail_rows.append(
            {
                "backbone_id": row["backbone_id"],
                "priority_group": row["priority_group"],
                "priority_index": row["priority_index"],
                "selection_score": row.get("selection_score", np.nan),
                "selection_score_column": row.get("selection_score_column", "priority_index"),
                "dominant_species": row["dominant_species"],
                "dominant_genus": row["dominant_genus"],
                "amr_class_count": total_class_count,
                "who_mia_mapped_class_count": mapped_class_count,
                "who_mia_unmapped_class_count": max(total_class_count - mapped_class_count, 0),
                "who_mia_mapped_fraction": mapped_class_count / total_class_count if total_class_count > 0 else 0.0,
                "who_mia_top_categories": _top_counter_items(category_counter),
                "who_mia_top_classes": _top_counter_items(class_counter),
                "who_mia_scope_labels": _top_counter_items(scope_counter),
                "who_mia_any_hpecia": category_counter["HPCIA"] > 0,
                "who_mia_any_cia": category_counter["CIA"] > 0,
                "who_mia_any_hia": category_counter["HIA"] > 0,
                "who_mia_any_ia": category_counter["IA"] > 0,
                "who_mia_any_support": mapped_class_count > 0,
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["priority_group", "who_mia_mapped_fraction", "selection_score"],
        ascending=[True, False, False],
    )
    summary = (
        detail.groupby("priority_group")
        .agg(
            n_backbones=("backbone_id", "nunique"),
            n_with_any_who_support=("who_mia_any_support", "sum"),
            n_with_hpecia_support=("who_mia_any_hpecia", "sum"),
            n_with_cia_support=("who_mia_any_cia", "sum"),
            n_with_hia_support=("who_mia_any_hia", "sum"),
            n_with_ia_support=("who_mia_any_ia", "sum"),
            mean_who_mia_mapped_fraction=("who_mia_mapped_fraction", "mean"),
            median_who_mia_mapped_fraction=("who_mia_mapped_fraction", "median"),
            mean_who_mia_mapped_class_count=("who_mia_mapped_class_count", "mean"),
        )
        .reset_index()
    )
    category_comparison = _build_prevalence_comparison(
        pd.DataFrame(category_membership),
        value_column="who_mia_category",
        group_totals=detail.groupby("priority_group")["backbone_id"].nunique().astype(int).to_dict(),
    )
    return detail, summary, category_comparison


def read_mobsuite_host_range_table(mobsuite_tar_path: Path) -> pd.DataFrame:
    """Read the MOB-suite literature host-range table from the bundled archive."""
    with tarfile.open(mobsuite_tar_path, "r:") as archive:
        member = _first_tar_member(archive, "host_range_literature_plasmidDB.txt")
        frame = pd.read_csv(
            archive.extractfile(member),
            sep="\t",
            usecols=[
                "sample_id",
                "rep_type(s)",
                "host_species",
                "reported_host_range_taxid",
                "pmid",
                "year",
                "notes",
            ],
        )
    return frame.drop_duplicates().reset_index(drop=True)


def read_mobsuite_clusters_table(mobsuite_tar_path: Path) -> pd.DataFrame:
    """Read the MOB-suite cluster catalog from the bundled archive."""
    with tarfile.open(mobsuite_tar_path, "r:") as archive:
        member = _first_tar_member(archive, "clusters.txt")
        frame = pd.read_csv(
            archive.extractfile(member),
            sep="\t",
            usecols=[
                "sample_id",
                "taxid",
                "rep_type(s)",
                "relaxase_type(s)",
                "mpf_type",
                "orit_type(s)",
                "predicted_mobility",
            ],
        )
    return frame.drop_duplicates().reset_index(drop=True)


def build_mobsuite_support(
    priority_backbones: pd.DataFrame,
    mobsuite_tar_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize external MOB-suite host-range and cluster support by backbone."""
    detail_columns = [
        "backbone_id",
        "priority_group",
        "priority_index",
        "selection_score",
        "selection_score_column",
        "dominant_species",
        "primary_replicon",
        "replicon_types",
        "mobsuite_literature_record_count",
        "mobsuite_literature_sample_count",
        "mobsuite_literature_host_species_count",
        "mobsuite_reported_host_range_taxid_count",
        "mobsuite_literature_pmid_count",
        "mobsuite_literature_year_min",
        "mobsuite_literature_year_max",
        "mobsuite_literature_conjugative_note_fraction",
        "mobsuite_cluster_record_count",
        "mobsuite_cluster_taxid_count",
        "mobsuite_cluster_mobilizable_fraction",
        "mobsuite_cluster_conjugative_fraction",
        "mobsuite_cluster_relaxase_presence_fraction",
        "mobsuite_cluster_mpf_presence_fraction",
        "mobsuite_cluster_orit_presence_fraction",
        "mobsuite_any_literature_support",
        "mobsuite_any_cluster_support",
    ]
    summary_columns = [
        "priority_group",
        "n_backbones",
        "n_with_literature_support",
        "n_with_cluster_support",
        "mean_reported_host_range_taxid_count",
        "median_reported_host_range_taxid_count",
        "mean_cluster_taxid_count",
        "mean_cluster_conjugative_fraction",
        "mean_cluster_relaxase_presence_fraction",
    ]

    if priority_backbones.empty:
        return _empty_support_frame(detail_columns), _empty_support_frame(summary_columns)

    host_range = read_mobsuite_host_range_table(mobsuite_tar_path)
    host_range = host_range.loc[host_range["rep_type(s)"].fillna("").astype(str).str.strip().ne("")]
    host_range = host_range.loc[host_range["rep_type(s)"].astype(str).str.strip().ne("-")]
    host_summary = (
        host_range.groupby("rep_type(s)")
        .agg(
            mobsuite_literature_record_count=("sample_id", "size"),
            mobsuite_literature_sample_count=("sample_id", "nunique"),
            mobsuite_literature_host_species_count=("host_species", lambda series: series.fillna("").astype(str).str.strip().replace({"": pd.NA}).dropna().nunique()),
            mobsuite_reported_host_range_taxid_count=("reported_host_range_taxid", lambda series: pd.Series(series).dropna().nunique()),
            mobsuite_literature_pmid_count=("pmid", lambda series: pd.Series(series).dropna().nunique()),
            mobsuite_literature_year_min=("year", "min"),
            mobsuite_literature_year_max=("year", "max"),
            mobsuite_literature_conjugative_note_fraction=(
                "notes",
                lambda series: series.fillna("").astype(str).str.contains("conjugative", case=False).mean(),
            ),
        )
        .reset_index()
        .rename(columns={"rep_type(s)": "primary_replicon"})
    )

    clusters = read_mobsuite_clusters_table(mobsuite_tar_path)
    clusters = clusters.loc[clusters["rep_type(s)"].fillna("").astype(str).str.strip().ne("")]
    clusters = clusters.loc[clusters["rep_type(s)"].astype(str).str.strip().ne("-")]
    cluster_summary = (
        clusters.groupby("rep_type(s)")
        .agg(
            mobsuite_cluster_record_count=("sample_id", "size"),
            mobsuite_cluster_taxid_count=("taxid", lambda series: pd.Series(series).dropna().nunique()),
            mobsuite_cluster_mobilizable_fraction=(
                "predicted_mobility",
                lambda series: series.fillna("").astype(str).str.strip().str.lower().eq("mobilizable").mean(),
            ),
            mobsuite_cluster_conjugative_fraction=(
                "predicted_mobility",
                lambda series: series.fillna("").astype(str).str.contains("conjugative", case=False).mean(),
            ),
            mobsuite_cluster_relaxase_presence_fraction=(
                "relaxase_type(s)",
                lambda series: (~series.fillna("").astype(str).str.strip().isin(["", "-"])).mean(),
            ),
            mobsuite_cluster_mpf_presence_fraction=(
                "mpf_type",
                lambda series: (~series.fillna("").astype(str).str.strip().isin(["", "-"])).mean(),
            ),
            mobsuite_cluster_orit_presence_fraction=(
                "orit_type(s)",
                lambda series: (~series.fillna("").astype(str).str.strip().isin(["", "-"])).mean(),
            ),
        )
        .reset_index()
        .rename(columns={"rep_type(s)": "primary_replicon"})
    )

    detail = priority_backbones.copy()
    if "selection_score" not in detail.columns:
        detail["selection_score"] = pd.to_numeric(detail.get("priority_index", 0.0), errors="coerce").fillna(0.0)
    if "selection_score_column" not in detail.columns:
        detail["selection_score_column"] = "priority_index"
    keep_columns = [
        "backbone_id",
        "priority_group",
        "priority_index",
        "selection_score",
        "selection_score_column",
        "dominant_species",
        "primary_replicon",
        "replicon_types",
    ]
    detail = detail[[column for column in keep_columns if column in detail.columns]].copy()
    detail = detail.merge(host_summary, on="primary_replicon", how="left")
    detail = detail.merge(cluster_summary, on="primary_replicon", how="left")

    numeric_fill_columns = [
        "mobsuite_literature_record_count",
        "mobsuite_literature_sample_count",
        "mobsuite_literature_host_species_count",
        "mobsuite_reported_host_range_taxid_count",
        "mobsuite_literature_pmid_count",
        "mobsuite_literature_conjugative_note_fraction",
        "mobsuite_cluster_record_count",
        "mobsuite_cluster_taxid_count",
        "mobsuite_cluster_mobilizable_fraction",
        "mobsuite_cluster_conjugative_fraction",
        "mobsuite_cluster_relaxase_presence_fraction",
        "mobsuite_cluster_mpf_presence_fraction",
        "mobsuite_cluster_orit_presence_fraction",
    ]
    for column in numeric_fill_columns:
        if column in detail.columns:
            detail[column] = detail[column].fillna(0.0)

    detail["mobsuite_any_literature_support"] = detail["mobsuite_literature_record_count"].gt(0)
    detail["mobsuite_any_cluster_support"] = detail["mobsuite_cluster_record_count"].gt(0)
    detail = detail.sort_values(
        ["priority_group", "mobsuite_reported_host_range_taxid_count", "selection_score"],
        ascending=[True, False, False],
    )
    summary = (
        detail.groupby("priority_group")
        .agg(
            n_backbones=("backbone_id", "nunique"),
            n_with_literature_support=("mobsuite_any_literature_support", "sum"),
            n_with_cluster_support=("mobsuite_any_cluster_support", "sum"),
            mean_reported_host_range_taxid_count=("mobsuite_reported_host_range_taxid_count", "mean"),
            median_reported_host_range_taxid_count=("mobsuite_reported_host_range_taxid_count", "median"),
            mean_cluster_taxid_count=("mobsuite_cluster_taxid_count", "mean"),
            mean_cluster_conjugative_fraction=("mobsuite_cluster_conjugative_fraction", "mean"),
            mean_cluster_relaxase_presence_fraction=("mobsuite_cluster_relaxase_presence_fraction", "mean"),
        )
        .reset_index()
    )
    return detail, summary
