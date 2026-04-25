from __future__ import annotations

import pandas as pd

from plasmid_priority.reporting.literature_validation import (
    build_literature_inventory,
    build_literature_support_matrix,
)


def test_build_literature_support_matrix_marks_matches() -> None:
    scored = pd.DataFrame(
        {"backbone_id": ["IncF_A", "Unknown_X"], "priority_index": [0.9, 0.1]},
    )
    literature = pd.DataFrame(
        {
            "pmid": ["1", "2"],
            "title": ["IncF plasmid spread in hospitals", "random title"],
            "pub_year": ["2024", "2025"],
        },
    )
    matrix = build_literature_support_matrix(scored, literature, top_k=2)
    assert len(matrix) == 2
    row = matrix.loc[matrix["backbone_id"] == "IncF_A"].iloc[0]
    assert int(row["literature_match_count"]) >= 1
    assert row["claim_level"] == "literature_supported"


def test_build_literature_inventory_aggregates_year_counts() -> None:
    literature = pd.DataFrame(
        {"pmid": ["1", "2", "3"], "title": ["a", "b", "c"], "pub_year": ["2024", "2024", "2025"]},
    )
    inventory = build_literature_inventory(literature)
    assert set(inventory["pub_year"].astype(str)) == {"2024", "2025"}
    assert int(inventory.loc[inventory["pub_year"] == "2024", "n_records"].iloc[0]) == 2
