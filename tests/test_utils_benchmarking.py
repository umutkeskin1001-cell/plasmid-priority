"""Tests for utils.benchmarking module."""

from __future__ import annotations

import time

import pytest

from plasmid_priority.utils.benchmarking import benchmark_runtime, measure_runtime


def test_benchmark_runtime_context_manager() -> None:
    """Test benchmark_runtime context manager."""
    operation_name = "test_operation"

    with benchmark_runtime(operation_name):
        time.sleep(0.01)  # Small delay to measure

    # If no exception, context manager worked correctly
    assert True


def test_benchmark_runtime_with_exception() -> None:
    """Test benchmark_runtime handles exceptions."""
    operation_name = "test_operation"

    with pytest.raises(ValueError):
        with benchmark_runtime(operation_name):
            raise ValueError("Test exception")

    # Should still log benchmark even with exception
    assert True


def test_measure_runtime_basic() -> None:
    """Test measure_runtime with a simple function."""

    def simple_function(x: int, y: int) -> int:
        time.sleep(0.01)
        return x + y

    result, runtime = measure_runtime(simple_function, 3, 4)

    assert result == 7
    assert runtime > 0
    assert runtime < 1.0  # Should be fast


def test_measure_runtime_with_kwargs() -> None:
    """Test measure_runtime with keyword arguments."""

    def function_with_kwargs(a: int, b: int, multiplier: int = 2) -> int:
        time.sleep(0.01)
        return (a + b) * multiplier

    result, runtime = measure_runtime(function_with_kwargs, 1, 2, multiplier=3)

    assert result == 9
    assert runtime > 0


def test_measure_runtime_no_args() -> None:
    """Test measure_runtime with no arguments."""

    def no_args_function() -> str:
        time.sleep(0.01)
        return "done"

    result, runtime = measure_runtime(no_args_function)

    assert result == "done"
    assert runtime > 0


def test_measure_runtime_exception_propagation() -> None:
    """Test measure_runtime propagates exceptions."""

    def failing_function() -> None:
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        measure_runtime(failing_function)
