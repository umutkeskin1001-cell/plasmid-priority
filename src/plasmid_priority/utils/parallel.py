"""Parallel execution helpers with safe native-thread limiting."""

from __future__ import annotations

from contextlib import contextmanager

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - optional dependency
    _threadpool_limits = None


@contextmanager
def limit_native_threads(limits: int = 1):
    """Temporarily cap BLAS/OpenMP thread pools to avoid oversubscription."""
    if _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=limits):
        yield
