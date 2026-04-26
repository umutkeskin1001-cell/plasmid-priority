from __future__ import annotations

import numpy as np
import pandas as pd

_DERIVABLE_SOURCES: dict[str, tuple[tuple[str, ...], ...]] = {
    "H_obs_norm": (("H_phylogenetic_norm",), ("H_breadth_norm",)),
    "H_obs_specialization_norm": (
        ("H_obs_norm",),
        ("H_phylogenetic_norm",),
        ("H_breadth_norm",),
    ),
    "H_specialization_norm": (("H_breadth_norm",),),
    "H_augmented_specialization_norm": (("H_augmented_norm",),),
    "H_phylogenetic_specialization_norm": (("H_phylogenetic_norm",),),
    "T_H_obs_synergy_norm": (
        ("T_eff_norm", "H_obs_norm"),
        ("T_eff_norm", "H_phylogenetic_norm"),
        ("T_eff_norm", "H_breadth_norm"),
    ),
    "A_H_obs_synergy_norm": (
        ("A_eff_norm", "H_obs_norm"),
        ("A_eff_norm", "H_phylogenetic_norm"),
        ("A_eff_norm", "H_breadth_norm"),
    ),
    "T_coherence_synergy_norm": (("T_eff_norm", "coherence_score"),),
}


def _numeric_series(values: pd.Series | float, index: pd.Index) -> pd.Series:
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce").fillna(0.0)


def ensure_feature_columns(
    scored: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    """Fill derivable feature columns so downstream model surfaces stay deterministic."""
    requested = [str(column) for column in columns]
    working = scored.copy()

    needs_h_obs_norm = any(
        column in requested
        for column in (
            "H_obs_norm",
            "H_obs_specialization_norm",
            "T_H_obs_synergy_norm",
            "A_H_obs_synergy_norm",
        )
    )
    if needs_h_obs_norm and "H_obs_norm" not in working.columns:
        if "H_phylogenetic_norm" in working.columns:
            working["H_obs_norm"] = _numeric_series(
                working["H_phylogenetic_norm"],
                working.index,
            ).clip(lower=0.0, upper=1.0)
        elif "H_breadth_norm" in working.columns:
            working["H_obs_norm"] = _numeric_series(
                working["H_breadth_norm"],
                working.index,
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_obs_norm"] = 0.0

    if (
        "H_obs_specialization_norm" in requested
        and "H_obs_specialization_norm" not in working.columns
    ):
        if "H_obs_norm" in working.columns:
            working["H_obs_specialization_norm"] = (
                1.0 - _numeric_series(working["H_obs_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        elif "H_phylogenetic_norm" in working.columns:
            working["H_obs_specialization_norm"] = (
                1.0 - _numeric_series(working["H_phylogenetic_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        elif "H_breadth_norm" in working.columns:
            working["H_obs_specialization_norm"] = (
                1.0 - _numeric_series(working["H_breadth_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_obs_specialization_norm"] = 0.0

    if "H_specialization_norm" in requested and "H_specialization_norm" not in working.columns:
        if "H_breadth_norm" in working.columns:
            working["H_specialization_norm"] = (
                1.0 - _numeric_series(working["H_breadth_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_specialization_norm"] = 0.0

    if (
        "H_augmented_specialization_norm" in requested
        and "H_augmented_specialization_norm" not in working.columns
    ):
        if "H_augmented_norm" in working.columns:
            working["H_augmented_specialization_norm"] = (
                1.0 - _numeric_series(working["H_augmented_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_augmented_specialization_norm"] = 0.0

    if (
        "H_phylogenetic_specialization_norm" in requested
        and "H_phylogenetic_specialization_norm" not in working.columns
    ):
        if "H_phylogenetic_norm" in working.columns:
            working["H_phylogenetic_specialization_norm"] = (
                1.0 - _numeric_series(working["H_phylogenetic_norm"], working.index)
            ).clip(lower=0.0, upper=1.0)
        else:
            working["H_phylogenetic_specialization_norm"] = 0.0

    if "T_H_obs_synergy_norm" in requested and "T_H_obs_synergy_norm" not in working.columns:
        working["T_H_obs_synergy_norm"] = np.clip(
            _numeric_series(working.get("T_eff_norm", 0.0), working.index)
            * _numeric_series(working.get("H_obs_norm", 0.0), working.index),
            0.0,
            1.0,
        )

    if "A_H_obs_synergy_norm" in requested and "A_H_obs_synergy_norm" not in working.columns:
        working["A_H_obs_synergy_norm"] = np.clip(
            _numeric_series(working.get("A_eff_norm", 0.0), working.index)
            * _numeric_series(working.get("H_obs_norm", 0.0), working.index),
            0.0,
            1.0,
        )

    if (
        "T_coherence_synergy_norm" in requested
        and "T_coherence_synergy_norm" not in working.columns
    ):
        working["T_coherence_synergy_norm"] = np.clip(
            _numeric_series(working.get("T_eff_norm", 0.0), working.index)
            * _numeric_series(working.get("coherence_score", 0.0), working.index),
            0.0,
            1.0,
        )

    for column in requested:
        if column not in working.columns:
            working[column] = 0.0
    return working


def assert_feature_columns_present(
    scored: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | set[str],
    *,
    label: str,
) -> None:
    """Fail loudly when required columns are absent and cannot be derived."""
    available = set(scored.columns.astype(str))
    missing: list[str] = []
    for column in dict.fromkeys(str(column) for column in columns):
        if column in available:
            continue
        source_sets = _DERIVABLE_SOURCES.get(column)
        if source_sets is not None and any(
            all(candidate in available for candidate in source_set) for source_set in source_sets
        ):
            continue
        missing.append(column)
    if missing:
        formatted = ", ".join(f"`{column}`" for column in missing)
        raise ValueError(
            f"{label} is missing required scored feature columns: {formatted}. "
            "Rerun `python3 scripts/15_normalize_and_score.py` before downstream modeling.",
        )
