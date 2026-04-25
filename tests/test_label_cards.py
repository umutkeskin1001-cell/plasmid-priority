from plasmid_priority.validation.label_cards import index_label_cards, validate_label_cards


def test_validate_label_cards_accepts_minimal_valid_cards() -> None:
    cards = [
        {"label_name": "spread_label", "description": "desc", "caveats": "caveat"},
    ]
    assert validate_label_cards(cards) == []


def test_validate_label_cards_rejects_missing_fields() -> None:
    cards = [{"label_name": "spread_label", "description": ""}]
    errors = validate_label_cards(cards)
    assert any("missing 'description'" in err for err in errors)
    assert any("missing 'caveats'" in err for err in errors)


def test_index_label_cards_builds_lookup_map() -> None:
    cards = [
        {"label_name": "spread_label", "description": "desc", "caveats": "caveat"},
        {
            "label_name": "visibility_expansion_label",
            "description": "desc2",
            "caveats": "caveat2",
        },
    ]
    indexed = index_label_cards(cards)
    assert "spread_label" in indexed
    assert "visibility_expansion_label" in indexed
