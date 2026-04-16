from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sample DataFrames for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_scored_backbone_df() -> pd.DataFrame:
    """Minimal scored backbone table with required columns."""
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "backbone_id": [f"BB_{i:03d}" for i in range(n)],
            "spread_label": rng.choice([0.0, 1.0], size=n),
            "priority_index": rng.uniform(0, 1, size=n),
            "bio_priority_index": rng.uniform(0, 1, size=n),
            "T_eff_norm": rng.uniform(0, 1, size=n),
            "H_eff_norm": rng.uniform(0, 1, size=n),
            "A_eff_norm": rng.uniform(0, 1, size=n),
            "log1p_member_count_train": rng.uniform(0, 5, size=n),
            "log1p_n_countries_train": rng.uniform(0, 3, size=n),
            "refseq_share_train": rng.uniform(0, 1, size=n),
            "coherence_score": rng.uniform(0, 1, size=n),
            "knownness_score": rng.uniform(0, 1, size=n),
        }
    )


@pytest.fixture()
def sample_records_df() -> pd.DataFrame:
    """Minimal records table for branch dataset assembly."""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sequence_accession": [f"ACC_{i:05d}" for i in range(n)],
            "backbone_id": [f"BB_{i % 50:03d}" for i in range(n)],
            "resolved_year": rng.integers(2000, 2025, size=n).astype(float),
            "country": rng.choice(["USA", "DEU", "GBR", "CHN", "IND"], size=n),
        }
    )


@pytest.fixture()
def sample_model_result() -> dict[str, object]:
    """Minimal model result dict for evaluation tests."""
    from plasmid_priority.modeling.module_a_support import build_failed_model_result

    return build_failed_model_result("test_model", "test error")
