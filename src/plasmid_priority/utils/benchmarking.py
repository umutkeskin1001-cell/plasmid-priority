"""Runtime benchmarking utilities for performance tracking."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable


@contextmanager
def benchmark_runtime(operation_name: str):
    """Context manager to benchmark runtime of an operation.
    
    Args:
        operation_name: Name of the operation being benchmarked
        
    Yields:
        None
        
    Example:
        with benchmark_runtime("data_loading"):
            data = load_data()
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        print(f"[BENCHMARK] {operation_name}: {elapsed_seconds:.3f}s")


def measure_runtime(func: Callable, *args, **kwargs) -> tuple[object, float]:
    """Measure runtime of a function call.
    
    Args:
        func: Function to measure
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Tuple of (result, elapsed_seconds)
        
    Example:
        result, runtime = measure_runtime(load_data, path="data.tsv")
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    return result, elapsed_seconds
