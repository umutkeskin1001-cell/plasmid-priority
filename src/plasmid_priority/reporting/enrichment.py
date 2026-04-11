"""Independent categorical enrichment analyses for candidate interpretation."""

from __future__ import annotations

import re
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from plasmid_priority.reporting.external_support import (
    normalize_drug_class_token,
    split_field_tokens,
)


def _format_qvalue(value: float) -> str:
    if not np.isfinite(value):
        return "q=NA"
    if value < 0.001:
        return "q<0.001"
    return f"q={value:.3f}"


_NON_PUBLIC_HEALTH_AMR_TERMS = (
    "MERCURY",
    "TELLURIUM",
    "CADMIUM",
    "ARSENIC",
    "COPPER",
    "SILVER",
    "NICKEL",
    "COBALT",
    "ZINC",
    "LEAD",
    "BISMUTH",
    "CHROMATE",
    "QUATERNARY AMMONIUM",
    "DISINFECTING AGENTS",
    "ANTISEPTICS",
)


def _is_public_health_amr_class(token: str) -> bool:
    upper = token.upper()
    return not any(term in upper for term in _NON_PUBLIC_HEALTH_AMR_TERMS)


def _dominant_non_empty(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values.loc[(values != "") & (values != "-")]
    if values.empty:
        return ""
    value_counts = values.value_counts()
    if value_counts.empty:
        return ""
    return str(value_counts.index[0])


def _gene_family(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", token).lower()
    if not cleaned:
        return ""
    match = re.match(r"([a-z]+)", cleaned)
    if not match:
        return cleaned
    prefix = match.group(1)
    if prefix == "bla":
        rest = re.sub(r"[^A-Za-z]+", "", cleaned[3:])
        return f"bla{rest[:6]}".upper() if rest else "BLA"
    return prefix[:8].upper()


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    working = pvalues.astype(float).copy()
    ranked = working.sort_values(kind="mergesort")
    n = max(len(ranked), 1)
    adjusted = pd.Series(index=ranked.index, dtype=float)
    running = 1.0
    for rank, (index, value) in enumerate(reversed(list(ranked.items())), start=1):
        order_rank = n - rank + 1
        corrected = min(running, float(value) * n / max(order_rank, 1))
        running = corrected
        adjusted.loc[index] = corrected
    return adjusted.reindex(working.index)


def build_backbone_identity_table(
    scored: pd.DataFrame,
    backbones: pd.DataFrame,
    amr_consensus: pd.DataFrame,
    *,
    split_year: int = 2015,
) -> pd.DataFrame:
    """Aggregate training-period categorical identities per evaluable backbone."""
    label_column = (
        "spread_label" if "spread_label" in scored.columns else "visibility_expansion_label"
    )
    if label_column not in scored.columns:
        return pd.DataFrame()
    eligible = scored.loc[scored[label_column].notna(), ["backbone_id", label_column]].copy()
    eligible = eligible.rename(columns={label_column: "spread_label"})
    if eligible.empty:
        return pd.DataFrame()
    training_records = backbones.copy()
    years = pd.to_numeric(training_records["resolved_year"], errors="coerce").fillna(0).astype(int)
    training_records = training_records.loc[years <= split_year].copy()
    training_records = training_records.loc[
        training_records["backbone_id"].isin(eligible["backbone_id"])
    ].copy()
    training_records = training_records.merge(eligible, on="backbone_id", how="left")
    training_records = training_records.merge(
        amr_consensus[["sequence_accession", "amr_drug_classes", "amr_gene_symbols"]],
        on="sequence_accession",
        how="left",
    )

    rows: list[dict[str, object]] = []
    for backbone_id, frame in training_records.groupby("backbone_id", sort=False):
        amr_classes: set[str] = set()
        gene_families: Counter[str] = Counter()
        for value in frame["amr_drug_classes"].fillna(""):
            for token in split_field_tokens(str(value).replace(";", ","), separators=(",",)):
                normalized = normalize_drug_class_token(token)
                if normalized and _is_public_health_amr_class(normalized):
                    amr_classes.add(normalized)
        for value in frame["amr_gene_symbols"].fillna(""):
            for token in split_field_tokens(value, separators=(",",)):
                family = _gene_family(token)
                if family:
                    gene_families[family] += 1

        dominant_replicon = _dominant_non_empty(frame["primary_replicon"])
        if not dominant_replicon:
            replicon_tokens: Counter[str] = Counter()
            for value in frame["replicon_types"].fillna(""):
                for token in split_field_tokens(value, separators=(",",)):
                    replicon_tokens[token] += 1
            dominant_replicon = replicon_tokens.most_common(1)[0][0] if replicon_tokens else ""

        rows.append(
            {
                "backbone_id": str(backbone_id),
                "spread_label": int(frame["spread_label"].iloc[0]),
                "visibility_expansion_label": int(frame["spread_label"].iloc[0]),
                "dominant_genus": _dominant_non_empty(frame["genus"]),
                "primary_replicon": dominant_replicon,
                "dominant_mpf_type": _dominant_non_empty(frame["mpf_type"]),
                "amr_class_tokens": ",".join(sorted(amr_classes)),
                "dominant_amr_gene_family": gene_families.most_common(1)[0][0]
                if gene_families
                else "",
            }
        )
    return pd.DataFrame(rows)


def build_module_f_enrichment_table(
    identity: pd.DataFrame,
    *,
    label_column: str = "spread_label",
    min_backbones: int = 10,
) -> pd.DataFrame:
    """Test independent categorical identities for enrichment in visibility-positive backbones."""
    if identity.empty or label_column not in identity.columns:
        return pd.DataFrame()
    working = identity.copy()
    y = working[label_column].astype(int)
    n_positive = int(y.sum())
    n_negative = int((1 - y).sum())
    rows: list[dict[str, object]] = []

    def add_rows(feature_group: str, memberships: dict[str, pd.Series]) -> None:
        for feature_value, mask in memberships.items():
            mask = mask.fillna(False).astype(bool)
            n_with = int(mask.sum())
            if n_with < min_backbones:
                continue
            a = int((y[mask] == 1).sum())
            b = int((y[mask] == 0).sum())
            c = int((y[~mask] == 1).sum())
            d = int((y[~mask] == 0).sum())
            if min(a + b, c + d) == 0:
                continue
            odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            prevalence_with = a / max(a + b, 1)
            prevalence_without = c / max(c + d, 1)
            rows.append(
                {
                    "feature_group": feature_group,
                    "feature_value": feature_value,
                    "n_backbones_with_feature": n_with,
                    "positive_with_feature": a,
                    "negative_with_feature": b,
                    "positive_without_feature": c,
                    "negative_without_feature": d,
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(pvalue),
                    "prevalence_with_feature": float(prevalence_with),
                    "prevalence_without_feature": float(prevalence_without),
                    "prevalence_delta": float(prevalence_with - prevalence_without),
                    "positive_prevalence": float(n_positive / max(n_positive + n_negative, 1)),
                }
            )

    for column, group_name in (
        ("dominant_genus", "dominant_genus"),
        ("primary_replicon", "primary_replicon"),
        ("dominant_mpf_type", "dominant_mpf_type"),
        ("dominant_amr_gene_family", "dominant_amr_gene_family"),
    ):
        memberships = {
            str(value): working[column].fillna("").astype(str).eq(str(value))
            for value in sorted(set(working[column].fillna("").astype(str)) - {""})
        }
        add_rows(group_name, memberships)

    amr_classes = {
        token
        for value in working["amr_class_tokens"].fillna("").astype(str)
        for token in split_field_tokens(value, separators=(",",))
    }
    add_rows(
        "amr_class",
        {
            token: working["amr_class_tokens"]
            .fillna("")
            .astype(str)
            .str.contains(rf"(?:^|,){re.escape(token)}(?:,|$)")
            for token in sorted(amr_classes)
        },
    )

    combo_memberships: dict[str, pd.Series] = {}
    for row in working.itertuples(index=False):
        classes = split_field_tokens(getattr(row, "amr_class_tokens", ""), separators=(",",))
        for token in classes:
            if getattr(row, "primary_replicon", ""):
                combo = f"{row.primary_replicon} + {token}"
                combo_memberships.setdefault(combo, pd.Series(False, index=working.index))
                combo_memberships[combo].loc[working["backbone_id"] == row.backbone_id] = True
            if getattr(row, "dominant_genus", ""):
                combo = f"{row.dominant_genus} + {token}"
                combo_memberships.setdefault(combo, pd.Series(False, index=working.index))
                combo_memberships[combo].loc[working["backbone_id"] == row.backbone_id] = True
    add_rows(
        "replicon_amr_class",
        {
            k: v
            for k, v in combo_memberships.items()
            if " + " in k and not k.split(" + ")[0].istitle()
        },
    )
    add_rows(
        "genus_amr_class",
        {k: v for k, v in combo_memberships.items() if " + " in k and k.split(" + ")[0].istitle()},
    )

    enrichment = pd.DataFrame(rows)
    if enrichment.empty:
        return enrichment
    enrichment["q_value"] = _benjamini_hochberg(enrichment["p_value"])
    enrichment["enriched_in_positive"] = enrichment["odds_ratio"].replace(np.inf, 1e9) > 1.0
    enrichment = enrichment.sort_values(
        ["q_value", "odds_ratio", "prevalence_delta", "n_backbones_with_feature"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return enrichment


def build_module_f_top_hits(
    enrichment: pd.DataFrame,
    *,
    q_threshold: float = 0.05,
    max_per_group: int = 3,
    max_total: int = 20,
) -> pd.DataFrame:
    """Return the most decision-relevant independent enrichment hits."""
    if enrichment.empty:
        return pd.DataFrame()
    working = enrichment.copy()
    working = working.loc[
        working["enriched_in_positive"].fillna(False).astype(bool)
        & working["q_value"].fillna(1.0).le(q_threshold)
        & working["odds_ratio"].replace(np.inf, 1e9).gt(1.0)
    ].copy()
    if working.empty:
        return working
    working["log2_odds_ratio"] = np.log2(
        working["odds_ratio"].replace(0.0, np.nan).replace(np.inf, 1e9)
    )
    selected_frames = []
    for _, frame in working.groupby("feature_group", sort=False):
        selected_frames.append(frame.head(max_per_group))
    top_hits = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if top_hits.empty:
        return top_hits
    top_hits = top_hits.sort_values(
        ["q_value", "odds_ratio", "prevalence_delta", "n_backbones_with_feature"],
        ascending=[True, False, False, False],
    ).head(max_total)
    return top_hits.reset_index(drop=True)


def build_candidate_signature_context(
    candidates: pd.DataFrame,
    identity: pd.DataFrame,
    enrichment: pd.DataFrame,
    *,
    q_threshold: float = 0.05,
    max_signatures_per_candidate: int = 5,
) -> pd.DataFrame:
    """Annotate candidate backbones with independently enriched categorical signatures."""
    if candidates.empty or identity.empty or enrichment.empty:
        return pd.DataFrame()
    candidate_ids = candidates["backbone_id"].astype(str).tolist()
    working = identity.loc[identity["backbone_id"].astype(str).isin(candidate_ids)].copy()
    if working.empty:
        return pd.DataFrame()

    positive_hits = enrichment.loc[
        enrichment["enriched_in_positive"].fillna(False).astype(bool)
        & enrichment["q_value"].fillna(1.0).le(q_threshold)
        & enrichment["odds_ratio"].replace(np.inf, 1e9).gt(1.0)
    ].copy()
    if positive_hits.empty:
        return pd.DataFrame()

    feature_maps: dict[str, dict[str, dict[str, object]]] = {}
    for row in positive_hits.to_dict(orient="records"):
        feature_maps.setdefault(str(row["feature_group"]), {})[str(row["feature_value"])] = row

    rows: list[dict[str, object]] = []
    for row in working.to_dict(orient="records"):
        amr_classes = set(
            split_field_tokens(str(row.get("amr_class_tokens", "")), separators=(",",))
        )
        matched: list[tuple[float, str]] = []

        def maybe_add(group: str, value: str) -> None:
            if not value:
                return
            hit = feature_maps.get(group, {}).get(value)
            if hit is None:
                return
            q_value = float(cast(float, hit.get("q_value", 1.0)))
            odds_ratio = float(cast(float, hit.get("odds_ratio", np.nan)))
            matched.append(
                (
                    q_value,
                    f"global {group}:{value} (OR={odds_ratio:.2f}, {_format_qvalue(q_value)})",
                )
            )

        maybe_add("dominant_genus", str(row.get("dominant_genus", "")))
        maybe_add("primary_replicon", str(row.get("primary_replicon", "")))
        maybe_add("dominant_mpf_type", str(row.get("dominant_mpf_type", "")))
        maybe_add("dominant_amr_gene_family", str(row.get("dominant_amr_gene_family", "")))
        for token in sorted(amr_classes):
            maybe_add("amr_class", token)
            maybe_add("replicon_amr_class", f"{row.get('primary_replicon', '')} + {token}")
            maybe_add("genus_amr_class", f"{row.get('dominant_genus', '')} + {token}")

        matched = sorted(matched, key=lambda item: (item[0], item[1]))
        unique_signatures = []
        seen = set()
        for _, label in matched:
            if label in seen:
                continue
            seen.add(label)
            unique_signatures.append(label)
        rows.append(
            {
                "backbone_id": str(row["backbone_id"]),
                "dominant_genus": str(row.get("dominant_genus", "")),
                "primary_replicon": str(row.get("primary_replicon", "")),
                "dominant_mpf_type": str(row.get("dominant_mpf_type", "")),
                "dominant_amr_gene_family": str(row.get("dominant_amr_gene_family", "")),
                "module_f_enriched_signature_count": int(len(unique_signatures)),
                "module_f_enriched_signatures": " | ".join(
                    unique_signatures[:max_signatures_per_candidate]
                ),
            }
        )
    return pd.DataFrame(rows)
