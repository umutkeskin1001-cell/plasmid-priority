"""Backward-compat shim for sensitivity cache manager."""

from plasmid_priority.sensitivity.variant_cache import (
    VariantCacheManager as SensitivityCacheManager,
)

__all__ = ["SensitivityCacheManager"]
