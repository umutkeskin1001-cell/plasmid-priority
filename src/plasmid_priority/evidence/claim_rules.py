"""Deterministic claim-level rules for official outputs."""


from plasmid_priority.evidence.levels import CLAIM_LEVEL_ORDER


def default_claim_levels() -> list[str]:
    return list(CLAIM_LEVEL_ORDER)


def validate_claim_levels_present(claim_levels: list[str] | tuple[str, ...]) -> list[str]:
    present = set(claim_levels)
    return [level for level in CLAIM_LEVEL_ORDER if level not in present]


def derive_claim_level(
    *,
    observed_signal: bool,
    proxy_only: bool,
    literature_support: bool,
    external_validation: bool,
) -> str:
    if external_validation:
        return "externally_validated"
    if literature_support:
        return "literature_supported"
    if observed_signal and not proxy_only:
        return "observed"
    return "proxy"
