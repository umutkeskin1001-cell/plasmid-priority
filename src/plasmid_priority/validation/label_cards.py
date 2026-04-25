"""Validation helpers for machine-readable label cards."""

from __future__ import annotations

from typing import Any

REQUIRED_LABEL_CARD_FIELDS: tuple[str, ...] = (
    "label_name",
    "description",
    "caveats",
)


def validate_label_cards(cards: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(cards, list) or not cards:
        return ["label_cards must be a non-empty list"]
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            errors.append(f"label_cards[{index}] must be an object")
            continue
        for field in REQUIRED_LABEL_CARD_FIELDS:
            value = card.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"label_cards[{index}] missing '{field}'")
    return errors


def index_label_cards(cards: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for card in cards:
        key = str(card.get("label_name", "")).strip()
        if key:
            indexed[key] = card
    return indexed
