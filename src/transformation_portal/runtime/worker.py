"""Spawn-safe worker utilities for isolated process execution.

This module provides utilities for running functions in isolated
processes using multiprocessing.spawn, which is required for
CUDA safety across process boundaries.

Key features:
- Spawn-safe execution (no fork issues with CUDA)
- Exception propagation from child processes
- Timeout support

Example:
    >>> def gpu_inference(device_id, inputs):
    ...     model = load_model(device_id)
    ...     return model(inputs)
    >>>
    >>> result = run_spawned(gpu_inference, device_id=0, inputs=data)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SpawnError(RuntimeError):
    """Raised when spawned process fails."""


def _worker_entry(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    result_queue: mp.Queue,
) -> None:
    """Worker entry point for spawned process.

    Args:
        fn: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        result_queue: Queue for returning results
    """
    try:
        result = fn(*args, **kwargs)
        result_queue.put(("ok", result))
    except Exception as e:
        # Serialize exception as string (exception objects may not pickle)
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def run_spawned(
    fn: Callable,
    *args: Any,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Execute function in a spawned child process.

    Uses multiprocessing.spawn context which is safe for CUDA
    (unlike fork, which can corrupt GPU state).

    Args:
        fn: Function to execute
        *args: Positional arguments to pass to fn
        timeout: Maximum time to wait in seconds (None = no timeout)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Return value of fn

    Raises:
        SpawnError: If child process fails or times out

    Example:
        >>> def load_and_run(device_id, data):
        ...     import torch
        ...     torch.cuda.set_device(device_id)
        ...     model = torch.load("model.pt")
        ...     return model(data)
        >>>
        >>> result = run_spawned(load_and_run, device_id=0, data=inputs)
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_worker_entry,
        args=(fn, args, kwargs, result_queue),
    )

    logger.debug("Spawning worker process for %s", fn.__name__)
    proc.start()

    # Wait for process to complete
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Timeout - terminate process
        logger.warning("Worker process timed out, terminating")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        raise SpawnError(f"Worker process timed out after {timeout}s")

    # Check exit code
    if proc.exitcode != 0:
        raise SpawnError(f"Worker process exited with code {proc.exitcode}")

    # Get result from queue
    if result_queue.empty():
        raise SpawnError("Worker process did not return a result")

    status, payload = result_queue.get_nowait()

    if status == "error":
        raise SpawnError(f"Worker process failed: {payload}")

    logger.debug("Worker process completed successfully")
    return payload


def run_with_gpu(
    semaphore: "GPUSemaphore",  # type: ignore
    fn: Callable,
    *args: Any,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Execute function with exclusive GPU access in a spawned process.

    Combines GPU semaphore acquisition with spawn-safe execution.

    Args:
        semaphore: GPUSemaphore instance
        fn: Function to execute (receives device_id as first arg)
        *args: Additional positional arguments
        timeout: Maximum time to wait
        **kwargs: Keyword arguments

    Returns:
        Return value of fn

    Example:
        >>> def inference(device_id, model_path, inputs):
        ...     model = load_on_device(model_path, device_id)
        ...     return model(inputs)
        >>>
        >>> sem = GPUSemaphore(num_devices=2)
        >>> result = run_with_gpu(sem, inference, "model.pt", inputs)
    """
    with semaphore.acquire() as slot:
        # Prepend device_id to args
        return run_spawned(fn, slot.device_id, *args, timeout=timeout, **kwargs)
