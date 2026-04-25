"""Tests for SensitivityCacheManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from plasmid_priority.utils.sensitivity_cache import SensitivityCacheManager


def test_sensitivity_cache_manager_init() -> None:
    """Test SensitivityCacheManager initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures = {"records": "abc123", "amr_hits": "def456"}
        cache = SensitivityCacheManager(cache_dir, source_signatures)
        assert cache.cache_dir == cache_dir / "sensitivity_cache"
        assert cache.source_signatures == source_signatures
        assert cache._memory == {}
        assert cache.cache_dir.exists()


def test_sensitivity_cache_key_generation() -> None:
    """Test cache key generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures = {"records": "abc123"}
        cache = SensitivityCacheManager(cache_dir, source_signatures)

        key1 = cache._key(2015, mode="all")
        key2 = cache._key(2015, mode="all")
        key3 = cache._key(2016, mode="all")

        assert key1 == key2
        assert key1 != key3


def test_sensitivity_cache_components() -> None:
    """Test components cache (disk tier)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures = {"records": "abc123"}
        cache = SensitivityCacheManager(cache_dir, source_signatures)

        # Test cache miss
        result = cache.get_components(2015, mode="all")
        assert result is None

        # Test cache put
        components = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        cache.put_components(2015, components, mode="all")

        # Test cache hit
        result = cache.get_components(2015, mode="all")
        assert result is not None
        assert len(result) == 3


def test_sensitivity_cache_scored() -> None:
    """Test scored cache (memory tier)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures = {"records": "abc123"}
        cache = SensitivityCacheManager(cache_dir, source_signatures)

        # Test cache miss
        result = cache.get_scored(2015, 2023, mode="all")
        assert result is None

        # Test cache put
        scored = pd.DataFrame({"backbone_id": ["A", "B"], "score": [0.5, 0.8]})
        cache.put_scored(2015, 2023, scored, mode="all")

        # Test cache hit
        result = cache.get_scored(2015, 2023, mode="all")
        assert result is not None
        assert len(result) == 2


def test_sensitivity_cache_clear_memory() -> None:
    """Test memory cache clearing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures = {"records": "abc123"}
        cache = SensitivityCacheManager(cache_dir, source_signatures)

        scored = pd.DataFrame({"backbone_id": ["A"], "score": [0.5]})
        cache.put_scored(2015, 2023, scored, mode="all")

        assert cache.get_scored(2015, 2023, mode="all") is not None

        cache.clear_memory()

        assert cache.get_scored(2015, 2023, mode="all") is None


def test_sensitivity_cache_invalidation() -> None:
    """Test cache invalidation with different source signatures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        source_signatures1 = {"records": "abc123"}
        cache1 = SensitivityCacheManager(cache_dir, source_signatures1)

        components = pd.DataFrame({"col1": [1, 2, 3]})
        cache1.put_components(2015, components)

        # Different signatures should invalidate cache
        source_signatures2 = {"records": "xyz789"}
        cache2 = SensitivityCacheManager(cache_dir, source_signatures2)
        result = cache2.get_components(2015)
        assert result is None
