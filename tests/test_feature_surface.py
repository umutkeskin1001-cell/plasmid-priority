from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.modeling.feature_surface import (
    assert_feature_columns_present,
    ensure_feature_columns,
)


def test_ensure_feature_columns_derives_observed_host_surfaces_and_synergies() -> None:
    frame = pd.DataFrame(
        {
            "T_eff_norm": [0.2, 0.8],
            "A_eff_norm": [0.1, 0.6],
            "H_phylogenetic_norm": [0.25, 0.75],
            "coherence_score": [0.3, 0.9],
        },
    )

    derived = ensure_feature_columns(
        frame,
        [
            "H_obs_norm",
            "H_obs_specialization_norm",
            "T_H_obs_synergy_norm",
            "A_H_obs_synergy_norm",
            "T_coherence_synergy_norm",
        ],
    )

    assert derived["H_obs_norm"].tolist() == pytest.approx([0.25, 0.75])
    assert derived["H_obs_specialization_norm"].tolist() == pytest.approx([0.75, 0.25])
    assert derived["T_H_obs_synergy_norm"].tolist() == pytest.approx([0.05, 0.6])
    assert derived["A_H_obs_synergy_norm"].tolist() == pytest.approx([0.025, 0.45])
    assert derived["T_coherence_synergy_norm"].tolist() == pytest.approx([0.06, 0.72])


def test_ensure_feature_columns_backfills_unknown_requested_columns_with_zero() -> None:
    frame = pd.DataFrame({"T_eff_norm": [0.2, 0.8]})

    derived = ensure_feature_columns(frame, ["T_eff_norm", "unknown_feature"])

    assert derived["T_eff_norm"].tolist() == pytest.approx([0.2, 0.8])
    assert derived["unknown_feature"].tolist() == pytest.approx([0.0, 0.0])


def test_assert_feature_columns_present_accepts_derivable_columns() -> None:
    frame = pd.DataFrame(
        {
            "T_eff_norm": [0.2, 0.8],
            "H_breadth_norm": [0.4, 0.9],
            "coherence_score": [0.3, 0.9],
        },
    )

    assert_feature_columns_present(
        frame,
        [
            "H_obs_norm",
            "H_obs_specialization_norm",
            "T_H_obs_synergy_norm",
            "T_coherence_synergy_norm",
        ],
        label="test feature surface",
    )


def test_assert_feature_columns_present_fails_for_non_derivable_columns() -> None:
    frame = pd.DataFrame({"T_eff_norm": [0.2, 0.8]})

    with pytest.raises(ValueError, match="clinical_context_sparse_penalty_norm"):
        assert_feature_columns_present(
            frame,
            ["T_eff_norm", "clinical_context_sparse_penalty_norm"],
            label="test feature surface",
        )
