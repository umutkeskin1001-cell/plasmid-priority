"""Parallel execution helpers with safe native-thread limiting.

Phase 0 compute efficiency optimization:
- BLAS/OpenMP thread limiting prevents nested parallelism chaos
- Cross-platform: works on Mac, Windows, Linux
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - optional dependency
    _threadpool_limits = None


def configure_blas_threads(n: int = 1) -> None:
    """Set BLAS/LAPACK threads globally to prevent nested parallelism.

    Call this at the start of every script's main() function.
    Prevents the "64 threads fighting for 8 cores" problem where
    each process opens its own BLAS thread pool.

    Works on all platforms: Mac (Accelerate/VECLIB), Windows (MKL/OpenBLAS),
    Linux (OpenBLAS/MKL).

    Args:
        n: Number of threads per BLAS library. Default 1 ( safest for
           multi-process workloads). Set to -1 to let BLAS auto-detect.
    """
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_DOMAIN_NUM_THREADS'] = str(n)


@contextmanager
def limit_native_threads(limits: int = 1) -> Iterator[None]:
    """Temporarily cap BLAS/OpenMP thread pools to avoid oversubscription."""
    if _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=limits):
        yield


@contextmanager
def limit_all_threads(blas_limit: int = 1, openmp_limit: int = 1) -> Iterator[None]:
    """Context manager to limit both BLAS and OpenMP threads.

    Use this when spawning ProcessPoolExecutor workers to prevent
    each worker from opening 8 BLAS threads.
    """
    if _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=blas_limit, user_api='blas'):
        with _threadpool_limits(limits=openmp_limit, user_api='openmp'):
            yield
