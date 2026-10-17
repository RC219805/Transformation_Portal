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
import pickle
import time
from multiprocessing.connection import Connection
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from transformation_portal.runtime.gpu_semaphore import GPUSemaphore

logger = logging.getLogger(__name__)


class SpawnError(RuntimeError):
    """Raised when spawned process fails."""


class _ResultPublisher(Protocol):
    """The worker supports both test queues and the pipe-backed publisher."""

    def put(self, value: Any) -> None: ...


class _ResultSender:
    """Keep the worker's queue-like publisher over a synchronous pipe."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def put(self, value: Any) -> None:
        """Leave an unsendable result absent, as the queue feeder did."""
        try:
            self.connection.send(value)
        except Exception:
            logger.exception("Worker result serialization or transport failed")


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


def _pipe_worker_entry(fn: Callable, args: tuple, kwargs: dict, sender: Connection) -> None:
    """Close the child's only result writer on every exit path."""
    try:
        _worker_entry(fn, args, kwargs, _ResultSender(sender))
    finally:
        sender.close()


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
    receiver, sender = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_pipe_worker_entry,
        args=(fn, args, kwargs, sender),
    )

    received = []

    def receive_result() -> None:
        try:
            received.append(receiver.recv_bytes())
        except (EOFError, OSError):
            # A crashed child may leave no frame or only a partial frame.
            pass
        finally:
            receiver.close()

    reader = Thread(target=receive_result, name="tp-worker-result", daemon=True)
    process_started = False
    reader_started = False
    try:
        logger.debug("Spawning worker process for %s", fn.__name__)
        proc.start()
        process_started = True
        deadline = None if timeout is None else time.monotonic() + timeout
        # Only the child may retain a writer: its death must wake the reader.
        sender.close()
        reader.start()
        reader_started = True

        # The reader drains concurrently, including a partially written frame.
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        proc.join(timeout=remaining)
        if proc.is_alive():
            logger.warning("Worker process timed out, terminating")
            raise SpawnError(f"Worker process timed out after {timeout}s")
        if proc.exitcode != 0:
            raise SpawnError(f"Worker process exited with code {proc.exitcode}")

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        reader.join(timeout=remaining)
        if reader.is_alive():
            raise SpawnError(f"Worker process timed out after {timeout}s")
        if not received:
            raise SpawnError("Worker process did not return a result")
    finally:
        try:
            sender.close()
            if process_started and proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                if proc.is_alive():
                    raise SpawnError("Worker process could not be terminated")
        finally:
            try:
                if reader_started:
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise SpawnError("Worker result receiver did not stop")
            finally:
                receiver.close()
                if not process_started or not proc.is_alive():
                    proc.close()

    status, payload = pickle.loads(received[0])
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
