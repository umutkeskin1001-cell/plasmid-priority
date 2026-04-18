from __future__ import annotations

import pytest

from plasmid_priority.pipeline.branch_runner import run_branch, supported_branches


def test_supported_branches_are_stable() -> None:
    assert supported_branches() == (
        "geo_spread",
        "bio_transfer",
        "clinical_hazard",
        "consensus",
    )


def test_run_branch_rejects_unknown_branch() -> None:
    with pytest.raises(ValueError, match="Unsupported branch"):
        run_branch("unknown_branch")
