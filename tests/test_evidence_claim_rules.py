from plasmid_priority.evidence import (
    default_claim_levels,
    derive_claim_level,
    normalize_claim_levels,
    validate_claim_levels_present,
)


def test_default_claim_levels_are_canonical() -> None:
    assert default_claim_levels() == [
        "observed",
        "proxy",
        "literature_supported",
        "externally_validated",
    ]


def test_normalize_claim_levels_orders_and_deduplicates() -> None:
    assert normalize_claim_levels(["proxy", "observed", "proxy"]) == ["observed", "proxy"]


def test_validate_claim_levels_present_reports_missing_levels() -> None:
    missing = validate_claim_levels_present(["observed", "proxy"])
    assert missing == ["literature_supported", "externally_validated"]


def test_derive_claim_level_prioritizes_external_over_literature() -> None:
    assert (
        derive_claim_level(
            observed_signal=True,
            proxy_only=False,
            literature_support=True,
            external_validation=True,
        )
        == "externally_validated"
    )
