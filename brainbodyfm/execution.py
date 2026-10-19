"""Bounded, process-isolated execution for independent frozen analysis fits."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os


def _initialize_fit_worker():
    # Separate processes isolate sklearn warning capture and convergence checks.
    from threadpoolctl import threadpool_limits
    global _native_limits
    _native_limits = threadpool_limits(limits=1)
    os.environ.pop("BBFM_PROGRESS", None)


def run_bounded_fit_tasks(function, tasks, workers=1):
    """Consume tasks in order; completion order cannot influence task creation.

    At most `workers` tasks are in flight, bounding serialization and memory.
    Callers generate all random choices from task indices or ordered parent RNG
    draws, and sort returned scientific records into their canonical order.
    """
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    if workers == 1:
        yield from map(function, tasks)
        return
    executor = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_initialize_fit_worker)
    pending = set()
    iterator = iter(tasks)
    try:
        for _ in range(workers):
            task = next(iterator, None)
            if task is not None:
                pending.add(executor.submit(function, task))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                task = next(iterator, None)
                if task is not None:
                    pending.add(executor.submit(function, task))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
