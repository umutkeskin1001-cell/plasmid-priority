#!/usr/bin/env python3
"""Build official model release artifacts from a tabular candidate input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from plasmid_priority.config import build_context
from plasmid_priority.modeling.official import (
    build_official_release_artifacts,
    write_official_release_artifacts,
)


def _copy_first_available(
    frame: pd.DataFrame,
    target_column: str,
    source_columns: tuple[str, ...],
) -> None:
    if target_column in frame.columns:
        return
    for source_column in source_columns:
        if source_column in frame.columns:
            frame[target_column] = pd.to_numeric(frame[source_column], errors="coerce")
            return


def _uncertainty_tier_from_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(1.0)
    return pd.Series(
        np.select(
            [numeric <= 0.20, numeric <= 0.45],
            ["low", "moderate"],
            default="high",
        ),
        index=values.index,
        dtype="object",
    )


def _evidence_tier_from_support(frame: pd.DataFrame) -> pd.Series:
    if "evidence_support_index" in frame.columns:
        support = pd.to_numeric(frame["evidence_support_index"], errors="coerce").fillna(0.0)
        return pd.Series(
            np.select(
                [support >= 0.80, support >= 0.45],
                ["high", "moderate"],
                default="low",
            ),
            index=frame.index,
            dtype="object",
        )
    if "source_support_tier" in frame.columns:
        support_tier = frame["source_support_tier"].fillna("").astype(str).str.lower()
        return pd.Series(
            np.select(
                [
                    support_tier.str.contains("cross_source|multi_modal", regex=True),
                    support_tier.str.contains("supported|partial", regex=True),
                ],
                ["high", "moderate"],
                default="low",
            ),
            index=frame.index,
            dtype="object",
        )
    return pd.Series("insufficient", index=frame.index, dtype="object")


def _normalize_evidence_tier(frame: pd.DataFrame) -> None:
    supported_tiers = {"high", "moderate", "low", "insufficient"}
    derived = _evidence_tier_from_support(frame)
    if "evidence_tier" not in frame.columns:
        frame["evidence_tier"] = derived
        return
    normalized = frame["evidence_tier"].fillna("").astype(str).str.lower().str.strip()
    frame["evidence_tier"] = normalized.where(normalized.isin(supported_tiers), derived)


def prepare_official_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt existing release candidate portfolio columns to official-family inputs."""
    prepared = frame.copy()
    if "log1p_member_count_train" not in prepared.columns and "member_count_train" in prepared.columns:
        prepared["log1p_member_count_train"] = np.log1p(
            pd.to_numeric(prepared["member_count_train"], errors="coerce").fillna(0.0),
        )
    if "log1p_n_countries_train" not in prepared.columns and "n_countries_train" in prepared.columns:
        prepared["log1p_n_countries_train"] = np.log1p(
            pd.to_numeric(prepared["n_countries_train"], errors="coerce").fillna(0.0),
        )

    _copy_first_available(prepared, "T_eff_norm", ("bio_priority_index", "priority_index"))
    _copy_first_available(
        prepared,
        "H_obs_specialization_norm",
        ("bio_priority_index", "coherence_score", "knownness_score"),
    )
    _copy_first_available(
        prepared,
        "A_eff_norm",
        ("priority_index", "primary_model_candidate_score", "bio_priority_index"),
    )
    _copy_first_available(
        prepared,
        "evidence_support_index",
        ("evidence_support_index", "candidate_confidence_score"),
    )
    _normalize_evidence_tier(prepared)
    if "uncertainty_tier" not in prepared.columns and "model_prediction_uncertainty" in prepared.columns:
        prepared["uncertainty_tier"] = _uncertainty_tier_from_numeric(
            prepared["model_prediction_uncertainty"],
        )
    return prepared


def build_official_artifacts_from_tsv(
    input_path: Path,
    output_dir: Path,
    id_column: str,
    label_column: str | None = None,
) -> dict[str, Path]:
    frame = prepare_official_candidate_frame(pd.read_csv(input_path, sep="\t"))
    artifacts = build_official_release_artifacts(
        frame,
        id_column=id_column,
        label_column=label_column,
    )
    return write_official_release_artifacts(artifacts, output_dir)


def main(argv: list[str] | None = None) -> int:
    context = build_context()
    default_input = context.root / "reports/core_tables/candidate_portfolio.tsv"
    default_output_dir = context.root / "reports/core_tables"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=default_input, type=Path)
    parser.add_argument("--output-dir", default=default_output_dir, type=Path)
    parser.add_argument("--id-column", default="backbone_id")
    parser.add_argument("--label-column", default=None)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(
            f"official release artifact input is missing: {args.input}",
            file=sys.stderr,
        )
        return 2

    build_official_artifacts_from_tsv(
        input_path=args.input,
        output_dir=args.output_dir,
        id_column=args.id_column,
        label_column=args.label_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
