"""Canonical evidence claim levels."""


from enum import Enum


class EvidenceLevel(str, Enum):
    observed = "observed"
    proxy = "proxy"
    literature_supported = "literature_supported"
    externally_validated = "externally_validated"


CLAIM_LEVEL_ORDER: tuple[str, ...] = tuple(level.value for level in EvidenceLevel)


def is_valid_claim_level(value: str) -> bool:
    return value in CLAIM_LEVEL_ORDER


def normalize_claim_levels(levels: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for level in CLAIM_LEVEL_ORDER:
        if level in levels and level not in seen:
            normalized.append(level)
            seen.add(level)
    return normalized
