"""Unified branch runner dispatch used by CLI entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

BranchMain = Callable[[list[str] | None], int]


def _geo_spread_main(argv: list[str] | None = None) -> int:
    from plasmid_priority.geo_spread.cli import main

    return int(main(argv))


def _bio_transfer_main(argv: list[str] | None = None) -> int:
    from plasmid_priority.bio_transfer.cli import main

    return int(main(argv))


def _clinical_hazard_main(argv: list[str] | None = None) -> int:
    from plasmid_priority.clinical_hazard.cli import main

    return int(main(argv))


def _consensus_main(argv: list[str] | None = None) -> int:
    from plasmid_priority.consensus.cli import main

    return int(main(argv))


BRANCH_MAIN_MAP: Final[dict[str, BranchMain]] = {
    "geo_spread": _geo_spread_main,
    "bio_transfer": _bio_transfer_main,
    "clinical_hazard": _clinical_hazard_main,
    "consensus": _consensus_main,
}


def supported_branches() -> tuple[str, ...]:
    return tuple(BRANCH_MAIN_MAP.keys())


def run_branch(branch_name: str, *, branch_args: Sequence[str] | None = None) -> int:
    normalized = str(branch_name).strip().lower()
    if normalized not in BRANCH_MAIN_MAP:
        supported = ", ".join(supported_branches())
        raise ValueError(f"Unsupported branch `{branch_name}`. Supported: {supported}")
    dispatch = BRANCH_MAIN_MAP[normalized]
    return int(dispatch(list(branch_args or [])))
