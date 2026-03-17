"""GPU session pool with persistent workers and warm models.

This module provides a pool of long-lived GPU worker processes that:
- Own exclusive GPU devices
- Load models once (warm start)
- Execute tasks via IPC queues
- Avoid repeated model loads and VRAM fragmentation

Design:
    Workers are spawned once and stay alive, accepting tasks from a queue.
    Each worker initializes its GPU context and loads models during startup.
    Subsequent tasks reuse the warm models for low-latency inference.

Example:
    >>> def init_worker(device_id):
    ...     import torch
    ...     torch.cuda.set_device(device_id)
    ...     return {"device": device_id, "model": load_model()}
    >>>
    >>> pool = GPUSessionPool(num_workers=2, init_fn=init_worker)
    >>> pool.submit(inference_task, inputs)
    >>> result = pool.get()
    >>> pool.shutdown()
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionTask:
    """Task to execute in a GPU session worker.

    Attributes:
        fn: Function to execute (receives worker state as first arg)
        args: Positional arguments
        kwargs: Keyword arguments
        task_id: Unique task identifier
    """

    fn: Callable
    args: tuple
    kwargs: dict
    task_id: str = ""


class SessionPoolError(RuntimeError):
    """Raised for session pool errors."""


def _worker_loop(
    device_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    init_fn: Callable,
    worker_id: str,
) -> None:
    """Worker process main loop.

    Args:
        device_id: CUDA device ID for this worker
        task_queue: Queue for receiving tasks
        result_queue: Queue for sending results
        init_fn: Initialization function (receives device_id, returns state)
        worker_id: Worker identifier for logging
    """
    import os

    # Set CUDA device via environment (before any torch imports)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    logger.info("Worker %s starting on device %d", worker_id, device_id)

    # Initialize worker state (load models, etc.)
    try:
        state = init_fn(device_id)
        logger.info("Worker %s initialized successfully", worker_id)
    except Exception as exc:
        import traceback

        error_detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Worker %s initialization failed: %s", worker_id, error_detail)
        result_queue.put(("init_error", error_detail))
        return

    result_queue.put(("init_ok", worker_id))

    # Main task loop
    while True:
        try:
            task: Optional[SessionTask] = task_queue.get()

            if task is None:
                # Shutdown signal
                logger.info("Worker %s shutting down", worker_id)
                break

            start_time = time.time()

            try:
                result = task.fn(state, *task.args, **task.kwargs)
                elapsed = time.time() - start_time

                result_queue.put(
                    (
                        "ok",
                        {
                            "result": result,
                            "task_id": task.task_id,
                            "worker_id": worker_id,
                            "elapsed_ms": elapsed * 1000,
                        },
                    )
                )

            except Exception as exc:
                logger.error(
                    "Worker %s task %s failed: %s",
                    worker_id,
                    task.task_id,
                    exc,
                )
                result_queue.put(
                    (
                        "error",
                        {
                            "error": str(exc),
                            "task_id": task.task_id,
                            "worker_id": worker_id,
                        },
                    )
                )

        except Exception as exc:
            logger.error("Worker %s loop error: %s", worker_id, exc)


class GPUSessionPool:
    """Pool of persistent GPU worker processes with warm models.

    Workers are spawned once and stay alive, accepting tasks from a queue.
    Each worker initializes its GPU context and loads models during startup.

    Example:
        >>> def init_worker(device_id):
        ...     import torch
        ...     torch.cuda.set_device(device_id)
        ...     model = load_model().to(f"cuda:{device_id}")
        ...     return {"device": device_id, "model": model}
        >>>
        >>> def inference_task(state, inputs):
        ...     return state["model"](inputs)
        >>>
        >>> pool = GPUSessionPool(num_workers=2, init_fn=init_worker)
        >>> pool.submit(inference_task, batch1)
        >>> pool.submit(inference_task, batch2)
        >>> result1 = pool.get()
        >>> result2 = pool.get()
        >>> pool.shutdown()
    """

    def __init__(
        self,
        num_workers: int,
        init_fn: Callable[[int], Any],
        *,
        init_timeout: float = 300.0,
    ) -> None:
        """Initialize GPU session pool.

        Args:
            num_workers: Number of worker processes (typically = num GPUs)
            init_fn: Worker initialization function.
                    Receives device_id (int), returns worker state dict.
            init_timeout: Timeout for worker initialization (seconds)

        Raises:
            SessionPoolError: If worker initialization fails
        """
        self.num_workers = num_workers
        self.init_fn = init_fn

        ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue = ctx.Queue()
        self.result_queue: mp.Queue = ctx.Queue()
        self.workers: list[mp.Process] = []
        self._task_counter = 0
        self._shutdown = False

        logger.info("Starting GPU session pool with %d workers", num_workers)

        # Start workers
        for i in range(num_workers):
            worker_id = f"gpu_worker_{i}"
            p = ctx.Process(
                target=_worker_loop,
                args=(i, self.task_queue, self.result_queue, init_fn, worker_id),
                name=worker_id,
            )
            p.start()
            self.workers.append(p)

        # Wait for all workers to initialize
        init_results = []
        for _ in range(num_workers):
            try:
                status, payload = self.result_queue.get(timeout=init_timeout)
                init_results.append((status, payload))
            except Exception as exc:
                self.shutdown()
                raise SessionPoolError(f"Worker initialization timed out after {init_timeout}s") from exc

        # Check for initialization errors
        errors = [p for s, p in init_results if s == "init_error"]
        if errors:
            self.shutdown()
            raise SessionPoolError(f"Worker initialization failed: {errors}")

        logger.info("GPU session pool ready with %d workers", num_workers)

    def submit(
        self,
        fn: Callable,
        *args: Any,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a task to the pool.

        Args:
            fn: Function to execute (receives worker state as first arg)
            *args: Positional arguments for fn
            task_id: Optional task identifier (auto-generated if None)
            **kwargs: Keyword arguments for fn

        Returns:
            Task ID

        Raises:
            SessionPoolError: If pool is shutdown
        """
        if self._shutdown:
            raise SessionPoolError("Cannot submit to shutdown pool")

        if task_id is None:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

        task = SessionTask(fn=fn, args=args, kwargs=kwargs, task_id=task_id)
        self.task_queue.put(task)

        logger.debug("Submitted task %s", task_id)
        return task_id

    def get(
        self,
        timeout: Optional[float] = None,
    ) -> Any:
        """Get result from the pool.

        Blocks until a result is available.

        Args:
            timeout: Maximum time to wait (None = forever)

        Returns:
            Task result

        Raises:
            SessionPoolError: If task failed or timeout
        """
        try:
            status, payload = self.result_queue.get(timeout=timeout)
        except Exception as exc:
            raise SessionPoolError(f"Result timeout: {exc}") from exc

        if status == "error":
            raise SessionPoolError(f"Task {payload.get('task_id')} failed: {payload.get('error')}")

        return payload.get("result")

    def get_with_metadata(
        self,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Get result with execution metadata.

        Args:
            timeout: Maximum time to wait

        Returns:
            Dict with result, task_id, worker_id, elapsed_ms
        """
        try:
            status, payload = self.result_queue.get(timeout=timeout)
        except Exception as exc:
            raise SessionPoolError(f"Result timeout: {exc}") from exc

        if status == "error":
            raise SessionPoolError(f"Task {payload.get('task_id')} failed: {payload.get('error')}")

        return payload

    def map(
        self,
        fn: Callable,
        items: list[Any],
        *,
        timeout: Optional[float] = None,
    ) -> list[Any]:
        """Map a function over items using the pool.

        Args:
            fn: Function to apply (receives worker state and item)
            items: List of items to process
            timeout: Timeout per result

        Returns:
            List of results in order
        """
        # Submit all tasks
        task_ids = []
        for i, item in enumerate(items):
            task_id = self.submit(fn, item, task_id=f"map_{i}")
            task_ids.append(task_id)

        # Collect results (may be out of order)
        results_map = {}
        for _ in range(len(items)):
            payload = self.get_with_metadata(timeout=timeout)
            results_map[payload["task_id"]] = payload["result"]

        # Return in order
        return [results_map[tid] for tid in task_ids]

    def shutdown(
        self,
        wait: bool = True,
        timeout: float = 10.0,
    ) -> None:
        """Shutdown the pool.

        Args:
            wait: If True, wait for workers to finish
            timeout: Timeout for worker termination
        """
        if self._shutdown:
            return

        self._shutdown = True
        logger.info("Shutting down GPU session pool")

        # Send shutdown signals
        for _ in self.workers:
            try:
                self.task_queue.put(None)
            except Exception:
                pass

        if wait:
            for p in self.workers:
                p.join(timeout=timeout)
                if p.is_alive():
                    logger.warning("Force terminating worker %s", p.name)
                    p.terminate()
                    p.join(timeout=5)

        logger.info("GPU session pool shutdown complete")

    def __enter__(self) -> "GPUSessionPool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
