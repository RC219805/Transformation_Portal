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

    Each worker is pinned to one physical GPU through ``CUDA_VISIBLE_DEVICES``,
    so inside the worker that GPU is always ordinal 0; ``init_fn`` receives
    that visible ordinal.

    Tasks and results are pickled eagerly (tasks in ``GPUSessionPool.submit``,
    results in the worker) so an unpicklable object fails loudly for its own
    task instead of being dropped by a queue feeder thread.

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
import queue
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Poll interval used while waiting for a result, so a pool whose workers
# have all exited raises instead of blocking forever.
_LIVENESS_POLL_SECONDS = 1.0


def _dumps(obj: Any) -> bytes:
    """Pickle ``obj`` exactly as a multiprocessing queue would."""
    return bytes(ForkingPickler.dumps(obj))


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
        device_id: Physical CUDA device ID for this worker (exposed to the
            worker as ``CUDA_VISIBLE_DEVICES``; ``init_fn`` gets ordinal 0)
        task_queue: Queue for receiving ``(task_id, pickled SessionTask)``
        result_queue: Queue for sending results
        init_fn: Initialization function (receives device_id, returns state)
        worker_id: Worker identifier for logging
    """
    import os

    # Set CUDA device via environment (before any torch imports)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    logger.info("Worker %s starting on device %d", worker_id, device_id)

    # Initialize worker state (load models, etc.). Only the pinned device is
    # visible to this process, so its in-process ordinal is always 0.
    try:
        state = init_fn(0)
        logger.info("Worker %s initialized successfully", worker_id)
    except Exception as exc:
        import traceback

        error_detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Worker %s initialization failed: %s", worker_id, error_detail)
        result_queue.put(("init_error", error_detail))
        return

    result_queue.put(("init_ok", worker_id))

    def _put_error(task_id: str, error: str) -> None:
        logger.error("Worker %s task %s failed: %s", worker_id, task_id, error)
        result_queue.put(("error", {"error": error, "task_id": task_id, "worker_id": worker_id}))

    # Main task loop
    while True:
        try:
            item: Optional[Tuple[str, bytes]] = task_queue.get()

            if item is None:
                # Shutdown signal
                logger.info("Worker %s shutting down", worker_id)
                break

            task_id, task_blob = item
            try:
                task: SessionTask = ForkingPickler.loads(task_blob)
            except Exception as exc:
                _put_error(task_id, f"task could not be unpickled in worker: {type(exc).__name__}: {exc}")
                continue

            start_time = time.time()

            try:
                result = task.fn(state, *task.args, **task.kwargs)
            except Exception as exc:
                _put_error(task_id, str(exc))
                continue

            elapsed = time.time() - start_time

            # Pickle here, not in the queue feeder thread, so an unpicklable
            # result is reported for its task instead of silently lost.
            try:
                result_blob = _dumps(result)
            except Exception as exc:
                _put_error(task_id, f"result could not be pickled: {type(exc).__name__}: {exc}")
                continue

            result_queue.put(
                (
                    "ok",
                    {
                        "result_blob": result_blob,
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "elapsed_ms": elapsed * 1000,
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
                    Receives the in-process CUDA ordinal of the worker's
                    pinned GPU (always 0, see ``CUDA_VISIBLE_DEVICES``),
                    returns the worker state dict.
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
        self._map_counter = 0
        self._shutdown = False
        # Results received while map() collected its own batch; get() serves them first.
        self._buffered: Deque[Tuple[str, Dict[str, Any]]] = deque()
        # Task ids whose results nobody will claim (a map() that timed out or
        # could not submit its whole batch); their results are discarded.
        self._abandoned: Set[str] = set()

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
            SessionPoolError: If pool is shutdown or the task cannot be pickled
        """
        if self._shutdown:
            raise SessionPoolError("Cannot submit to shutdown pool")

        if task_id is None:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

        task = SessionTask(fn=fn, args=args, kwargs=kwargs, task_id=task_id)
        try:
            task_blob = _dumps(task)
        except Exception as exc:
            raise SessionPoolError(f"Task {task_id} cannot be sent to a worker process: {type(exc).__name__}: {exc}") from exc
        self.task_queue.put((task_id, task_blob))

        logger.debug("Submitted task %s", task_id)
        return task_id

    def _receive(
        self,
        timeout: Optional[float],
        *,
        use_buffer: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return the next ``(status, payload)`` task message.

        Results of abandoned tasks are discarded. While waiting, the pool
        raises if every worker process has exited.
        """
        if use_buffer and self._buffered:
            return self._buffered.popleft()

        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        while True:
            if deadline is None:
                wait = _LIVENESS_POLL_SECONDS
            else:
                wait = max(0.0, min(deadline - time.monotonic(), _LIVENESS_POLL_SECONDS))
            try:
                status, payload = self.result_queue.get(timeout=wait)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise SessionPoolError(f"Result timeout after {timeout}s") from None
                if not any(p.is_alive() for p in self.workers):
                    raise SessionPoolError("All pool workers have exited; no result can arrive") from None
                continue
            except Exception as exc:
                raise SessionPoolError(f"Result queue error: {exc}") from exc

            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if task_id in self._abandoned:
                self._abandoned.discard(task_id)
                logger.debug("Discarding result of abandoned task %s", task_id)
                continue

            if status == "ok" and "result_blob" in payload:
                try:
                    payload["result"] = ForkingPickler.loads(payload.pop("result_blob"))
                except Exception as exc:
                    return "error", {
                        "error": f"result could not be unpickled: {type(exc).__name__}: {exc}",
                        "task_id": task_id,
                        "worker_id": payload.get("worker_id"),
                    }
            return status, payload

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
        status, payload = self._receive(timeout)

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
        status, payload = self._receive(timeout)

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

        Raises:
            SessionPoolError: If any task failed (raised only after the whole
                batch has been collected) or a result timed out
        """
        # Task ids are unique per call so results of an earlier batch can
        # never be mistaken for this batch's results.
        self._map_counter += 1
        prefix = f"map_{self._map_counter}"

        # Submit all tasks
        task_ids: List[str] = []
        try:
            for i, item in enumerate(items):
                task_ids.append(self.submit(fn, item, task_id=f"{prefix}_{i}"))
        except SessionPoolError:
            self._abandoned.update(task_ids)
            raise

        # Collect results (may be out of order). Drain the whole batch before
        # reporting a failure so no result of this batch is left queued.
        pending = set(task_ids)
        results_map: Dict[str, Any] = {}
        first_error: Optional[Dict[str, Any]] = None
        foreign: List[Tuple[str, Dict[str, Any]]] = []
        try:
            while pending:
                status, payload = self._receive(timeout, use_buffer=False)
                tid = payload.get("task_id")
                if tid not in pending:
                    foreign.append((status, payload))
                    continue
                pending.discard(tid)
                if status == "error":
                    if first_error is None:
                        first_error = payload
                else:
                    results_map[tid] = payload.get("result")
        except SessionPoolError:
            self._abandoned.update(pending)
            raise
        finally:
            self._buffered.extend(foreign)

        if first_error is not None:
            raise SessionPoolError(f"Task {first_error.get('task_id')} failed: {first_error.get('error')}")

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
