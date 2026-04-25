"""Universal batch parallel executor for optimizing external tool throughput."""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def batch_parallel_execute(
    items: Iterable[T],
    process_fn: Callable[[list[T], Path], list[R]],
    n_workers: int | None = None,
) -> list[R]:
    """Execute a function over batches of items in parallel.

    This is highly effective for external tools (like MOB-suite or AMRfinder)
    that are single-threaded. By splitting the input into N batches and
    running N instances in parallel, we achieve near-linear speedup.

    Works on Mac (spawn), Windows (spawn), and Linux (fork).

    Args:
        items: List of items to process.
        process_fn: Function that takes (batch_items, tmp_dir) and returns list of results.
        n_workers: Number of parallel processes. Defaults to CPU count - 1.

    Returns:
        Flattened list of all results.
    """
    item_list = list(items)
    if not item_list:
        return []

    if n_workers is None:
        # Leave one core for the OS/runner
        n_workers = max(1, (multiprocessing.cpu_count() or 4) - 1)

    # Platform-safe context (spawn is safest across Mac/Win/Linux)
    ctx = multiprocessing.get_context('spawn')

    # Split into batches
    batch_size = max(1, len(item_list) // n_workers)
    batches = [item_list[i:i + batch_size] for i in range(0, len(item_list), batch_size)]

    # Adjust n_workers if we have fewer batches than workers
    actual_workers = min(n_workers, len(batches))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with ctx.Pool(processes=actual_workers) as pool:
            # We use starmap to pass the batch and a unique tmp subdirectory for each worker
            results = pool.starmap(
                process_fn,
                [(batch, tmp_path / f"batch_{i}") for i, batch in enumerate(batches)]
            )

    # Flatten results
    return [item for sublist in results for item in sublist]
