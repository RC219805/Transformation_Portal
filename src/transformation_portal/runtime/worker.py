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
import os
import pickle
import time
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

logger = logging.getLogger(__name__)


class SpawnError(RuntimeError):
    """Raised when spawned process fails."""


class _ResultPublisher(Protocol):
    """The worker supports both test queues and its private result publisher."""

    def put(self, value: Any) -> None: ...


class _ResultFilePublisher:
    """Publish a complete result privately, without inherited IPC writers."""

    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path

    def put(self, value: Any) -> None:
        """Leave serialization/disk failures unpublished, as unsendable results."""
        partial_path = self.result_path.with_suffix(".partial")
        descriptor = None
        try:
            descriptor = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                # Match multiprocessing's registered result reducers/protocol.
                ForkingPickler(stream).dump(value)
            os.replace(partial_path, self.result_path)
        except Exception:
            logger.exception("Worker result serialization or transport failed")
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _worker_entry(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    result_queue: _ResultPublisher,
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
        import traceback

        # Serialize exception with full traceback for debugging
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result_queue.put(("error", error_msg))


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
    # The child opens its result only after fn returns. A descendant forked by
    # fn cannot inherit a writer that blocks result collection after child exit.
    # At most one serialized result (partial or complete) occupies this private
    # directory. Context cleanup also removes interrupted/error publications.
    with TemporaryDirectory(prefix="tp-worker-result-") as result_directory:
        result_path = Path(result_directory) / "result.pickle"
        proc = ctx.Process(target=_worker_entry, args=(fn, args, kwargs, _ResultFilePublisher(result_path)))
        process_started = False
        try:
            logger.debug("Spawning worker process for %s", fn.__name__)
            proc.start()
            process_started = True
            deadline = None if timeout is None else time.monotonic() + timeout
            # Result serialization/publication happens before the child exits,
            # so it shares this one execution deadline with function/shutdown.
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                logger.warning("Worker process timed out, terminating")
                raise SpawnError(f"Worker process timed out after {timeout}s")
            if proc.exitcode != 0:
                raise SpawnError(f"Worker process exited with code {proc.exitcode}")
            if not result_path.is_file():
                raise SpawnError("Worker process did not return a result")
        finally:
            try:
                if process_started and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=5)
                    if proc.is_alive():
                        raise SpawnError("Worker process could not be terminated")
            finally:
                if not process_started or not proc.is_alive():
                    proc.close()

        with result_path.open("rb") as stream:
            status, payload = pickle.load(stream)
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
