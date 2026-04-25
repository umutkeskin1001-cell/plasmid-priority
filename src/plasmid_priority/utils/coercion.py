"""Type coercion utilities for consistent data type handling."""

from __future__ import annotations

from collections.abc import Mapping


def coerce_int(value: object, *, default: int = 0) -> int:
    """Safely coerce a value to int with a default fallback.

    Args:
        value: The value to coerce.
        default: Default value if coercion fails.

    Returns:
        Integer value or default.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def coerce_float(value: object, *, default: float = 0.0) -> float:
    """Safely coerce a value to float with a default fallback.

    Args:
        value: The value to coerce.
        default: Default value if coercion fails.

    Returns:
        Float value or default.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def coerce_float_mapping(value: object) -> dict[str, float]:
    """Coerce a mapping to dict[str, float].

    Args:
        value: The value to coerce.

    Returns:
        Dictionary with string keys and float values.
    """
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result
