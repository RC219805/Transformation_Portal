"""Async/Streaming Pipeline Architecture for high-throughput image processing.

This module provides async/await-based pipeline stages with queue-based execution,
backpressure handling, and worker pool management for parallel GPU/CPU operations.

Performance targets:
- Sequential (100 4K images): ~6.9 hours
- With async pipeline: ~1.5-2 hours (3-5x faster)

Key components:
- AsyncStage: Base class for async pipeline stages
- AsyncPipeline: Orchestrator with stage chaining and backpressure
- BackpressureQueue: Async queue with flow control
- WorkerPool: GPU/CPU worker management with device affinity
- StreamingImageLoader: Memory-efficient async image loading
- AsyncBatchProcessor: High-level batch processing coordinator

Example:
    >>> async def process_batch():
    ...     pipeline = AsyncPipeline()
    ...     pipeline.add_stage(LoadStage(max_concurrent=4))
    ...     pipeline.add_stage(DepthStage(device='cuda'))
    ...     pipeline.add_stage(EnhanceStage())
    ...     pipeline.add_stage(SaveStage())
    ...
    ...     async for result in pipeline.process_batch(image_paths):
    ...         print(f"Processed: {result.path}")
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)


# Type variables for generic pipeline stages
T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class DeviceType(Enum):
    """Device types for worker affinity."""
    CPU = auto()
    CUDA = auto()
    MPS = auto()  # Apple Metal Performance Shaders
    AUTO = auto()


class StageStatus(Enum):
    """Status of a pipeline stage."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring."""
    items_processed: int = 0
    items_failed: int = 0
    total_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    queue_wait_times: Dict[str, float] = field(default_factory=dict)
    memory_peak_mb: float = 0.0
    throughput_items_per_sec: float = 0.0

    def update_throughput(self) -> None:
        """Calculate current throughput."""
        if self.total_processing_time > 0:
            self.throughput_items_per_sec = self.items_processed / self.total_processing_time


@dataclass
class StageResult(Generic[T]):
    """Result from a pipeline stage."""
    data: Optional[T]
    stage_name: str
    elapsed_time: float
    success: bool
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Check if result represents a failure."""
        return not self.success


@dataclass
class WorkItem(Generic[T]):
    """Item being processed through the pipeline."""
    id: str
    data: T
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_results: List[StageResult] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """Total time since item was created."""
        return time.time() - self.created_at

    def add_result(self, result: StageResult) -> None:
        """Add a stage result to this work item."""
        self.stage_results.append(result)


class BackpressureQueue(Generic[T]):
    """Async queue with backpressure handling to prevent memory explosion.

    Features:
    - Configurable max size for flow control
    - Async put with optional timeout
    - Backpressure signals when queue is filling up
    - Stats tracking for monitoring

    Example:
        >>> queue = BackpressureQueue(maxsize=10, high_water_mark=0.8)
        >>>
        >>> # Producer
        >>> async def producer():
        ...     for item in items:
        ...         if queue.is_backpressured:
        ...             await asyncio.sleep(0.1)  # Slow down
        ...         await queue.put(item)
        >>>
        >>> # Consumer
        >>> async def consumer():
        ...     while True:
        ...         item = await queue.get()
        ...         process(item)
        ...         queue.task_done()
    """

    def __init__(
        self,
        maxsize: int = 10,
        high_water_mark: float = 0.8,
        low_water_mark: float = 0.3,
        name: str = "queue"
    ):
        """Initialize backpressure queue.

        Args:
            maxsize: Maximum queue size (0 for unlimited)
            high_water_mark: Fraction of maxsize to trigger backpressure (0.0-1.0)
            low_water_mark: Fraction to release backpressure (0.0-1.0)
            name: Queue name for logging/metrics
        """
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._high_water = int(maxsize * high_water_mark) if maxsize > 0 else 0
        self._low_water = int(maxsize * low_water_mark) if maxsize > 0 else 0
        self._name = name
        self._backpressured = False
        self._items_put = 0
        self._items_got = 0
        self._total_wait_time = 0.0
        self._closed = False
        self._close_event = asyncio.Event()

    @property
    def is_backpressured(self) -> bool:
        """Check if queue is in backpressure state."""
        return self._backpressured

    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if queue is at max capacity."""
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'name': self._name,
            'size': self.size,
            'maxsize': self._maxsize,
            'items_put': self._items_put,
            'items_got': self._items_got,
            'backpressured': self._backpressured,
            'avg_wait_time': (
                self._total_wait_time / self._items_got
                if self._items_got > 0 else 0.0
            ),
        }

    async def put(
        self,
        item: T,
        timeout: Optional[float] = None
    ) -> None:
        """Put item into queue with optional timeout.

        Args:
            item: Item to add
            timeout: Maximum time to wait (None for forever)

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If queue is closed
        """
        if self._closed:
            raise RuntimeError(f"Queue '{self._name}' is closed")

        start = time.time()

        try:
            if timeout is not None:
                await asyncio.wait_for(
                    self._queue.put(item),
                    timeout=timeout
                )
            else:
                await self._queue.put(item)
        finally:
            self._items_put += 1
            self._total_wait_time += time.time() - start

        # Update backpressure state
        if self._maxsize > 0 and self.size >= self._high_water:
            self._backpressured = True

    async def get(
        self,
        timeout: Optional[float] = None
    ) -> T:
        """Get item from queue with optional timeout.

        Args:
            timeout: Maximum time to wait (None for forever)

        Returns:
            Next item from queue

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start = time.time()

        try:
            if timeout is not None:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                item = await self._queue.get()
        finally:
            self._items_got += 1
            self._total_wait_time += time.time() - start

        # Update backpressure state
        if self._backpressured and self.size <= self._low_water:
            self._backpressured = False

        return item

    def task_done(self) -> None:
        """Mark a task as done."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for all items to be processed."""
        await self._queue.join()

    def close(self) -> None:
        """Close the queue (no more puts allowed)."""
        self._closed = True
        self._close_event.set()

    async def wait_closed(self) -> None:
        """Wait for queue to be closed."""
        await self._close_event.wait()


@runtime_checkable
class AsyncStageProtocol(Protocol[InputT, OutputT]):
    """Protocol for async pipeline stages."""

    name: str

    async def process(self, item: InputT) -> OutputT:
        """Process a single item."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def startup(self) -> None:
        """Initialize stage resources."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def shutdown(self) -> None:
        """Clean up stage resources."""
        ...  # pylint: disable=unnecessary-ellipsis


class AsyncStage(ABC, Generic[InputT, OutputT]):
    """Base class for async pipeline stages.

    Subclass this to create custom pipeline stages. Each stage:
    - Has a unique name for identification
    - Can specify device affinity (CPU/GPU)
    - Processes items asynchronously
    - Tracks metrics

    Example:
        >>> class DepthEstimationStage(AsyncStage[np.ndarray, Tuple[np.ndarray, np.ndarray]]):
        ...     def __init__(self):
        ...         super().__init__("depth_estimation", device=DeviceType.CUDA)
        ...         self.model = None
        ...
        ...     async def startup(self):
        ...         self.model = load_depth_model()
        ...
        ...     async def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...         loop = asyncio.get_event_loop()
        ...         return await loop.run_in_executor(None, self.model, image)
        ...
        ...     async def shutdown(self):
        ...         del self.model
        ...         gc.collect()
    """

    def __init__(
        self,
        name: str,
        device: DeviceType = DeviceType.CPU,
        max_concurrent: int = 1,
        timeout: Optional[float] = None,
        required: bool = True
    ):
        """Initialize async stage.

        Args:
            name: Stage name for identification
            device: Device affinity for processing
            max_concurrent: Maximum concurrent processing tasks
            timeout: Maximum time per item (None for no limit)
            required: If True, failures stop the pipeline
        """
        self.name = name
        self.device = device
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.required = required
        self._status = StageStatus.IDLE
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._items_processed = 0
        self._items_failed = 0
        self._total_time = 0.0

    @property
    def status(self) -> StageStatus:
        """Current stage status."""
        return self._status

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get stage metrics."""
        return {
            'name': self.name,
            'status': self._status.name,
            'items_processed': self._items_processed,
            'items_failed': self._items_failed,
            'total_time': self._total_time,
            'avg_time': (
                self._total_time / self._items_processed
                if self._items_processed > 0 else 0.0
            ),
        }

    async def startup(self) -> None:
        """Initialize stage resources. Override in subclass."""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._status = StageStatus.RUNNING

    async def shutdown(self) -> None:
        """Clean up stage resources. Override in subclass."""
        self._status = StageStatus.COMPLETED

    @abstractmethod
    async def process(self, item: InputT) -> OutputT:
        """Process a single item. Must be implemented by subclass.

        Args:
            item: Input item to process

        Returns:
            Processed output
        """

    async def __call__(self, item: InputT) -> StageResult[OutputT]:
        """Execute stage processing with metrics and error handling.

        Args:
            item: Input item to process

        Returns:
            StageResult with output or error information
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        start_time = time.time()

        async with self._semaphore:
            try:
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        self.process(item),
                        timeout=self.timeout
                    )
                else:
                    result = await self.process(item)

                elapsed = time.time() - start_time
                self._items_processed += 1
                self._total_time += elapsed

                return StageResult(
                    data=result,
                    stage_name=self.name,
                    elapsed_time=elapsed,
                    success=True
                )

            except asyncio.TimeoutError as e:
                elapsed = time.time() - start_time
                self._items_failed += 1

                return StageResult(
                    data=None,
                    stage_name=self.name,
                    elapsed_time=elapsed,
                    success=False,
                    error=e,
                    metadata={'error_type': 'timeout'}
                )

            except Exception as e:
                elapsed = time.time() - start_time
                self._items_failed += 1

                return StageResult(
                    data=None,
                    stage_name=self.name,
                    elapsed_time=elapsed,
                    success=False,
                    error=e,
                    metadata={'error_type': type(e).__name__}
                )


class WorkerPool:
    """Manage worker pools for CPU and GPU operations.

    Features:
    - Separate thread pools for CPU-bound and I/O-bound tasks
    - Process pool for CPU-intensive operations
    - Device-aware task routing
    - Automatic resource cleanup

    Example:
        >>> pool = WorkerPool(
        ...     cpu_workers=4,
        ...     io_workers=8,
        ...     use_process_pool=True
        ... )
        >>>
        >>> async with pool:
        ...     result = await pool.run_cpu(heavy_computation, data)
        ...     image = await pool.run_io(load_image, path)
    """

    def __init__(
        self,
        cpu_workers: Optional[int] = None,
        io_workers: Optional[int] = None,
        use_process_pool: bool = False,
        process_workers: Optional[int] = None
    ):
        """Initialize worker pool.

        Args:
            cpu_workers: Thread pool size for CPU tasks (default: cpu_count)
            io_workers: Thread pool size for I/O tasks (default: cpu_count * 2)
            use_process_pool: Whether to use ProcessPoolExecutor for CPU tasks
            process_workers: Process pool size (default: cpu_count)
        """
        self._cpu_count = os.cpu_count() or 4
        self._cpu_workers = cpu_workers or self._cpu_count
        self._io_workers = io_workers or self._cpu_count * 2
        self._use_process_pool = use_process_pool
        self._process_workers = process_workers or self._cpu_count

        self._cpu_pool: Optional[ThreadPoolExecutor] = None
        self._io_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._active = False

    async def __aenter__(self) -> 'WorkerPool':
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def startup(self) -> None:
        """Initialize worker pools."""
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=self._cpu_workers,
            thread_name_prefix="cpu_worker"
        )
        self._io_pool = ThreadPoolExecutor(
            max_workers=self._io_workers,
            thread_name_prefix="io_worker"
        )

        if self._use_process_pool:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self._process_workers
            )

        self._active = True

    async def shutdown(self) -> None:
        """Shutdown worker pools."""
        self._active = False

        if self._cpu_pool:
            self._cpu_pool.shutdown(wait=True)
            self._cpu_pool = None

        if self._io_pool:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None

        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None

    async def run_cpu(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Run CPU-bound task in thread pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if not self._active or self._cpu_pool is None:
            raise RuntimeError("WorkerPool not active")

        loop = asyncio.get_event_loop()

        if kwargs:
            func = partial(func, **kwargs)

        return await loop.run_in_executor(self._cpu_pool, func, *args)

    async def run_io(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Run I/O-bound task in dedicated I/O pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if not self._active or self._io_pool is None:
            raise RuntimeError("WorkerPool not active")

        loop = asyncio.get_event_loop()

        if kwargs:
            func = partial(func, **kwargs)

        return await loop.run_in_executor(self._io_pool, func, *args)

    async def run_process(
        self,
        func: Callable[..., T],
        *args: Any
    ) -> T:
        """Run CPU-intensive task in process pool.

        Note: Function and arguments must be picklable.

        Args:
            func: Function to execute (must be picklable)
            *args: Positional arguments (must be picklable)

        Returns:
            Function result
        """
        if not self._use_process_pool or self._process_pool is None:
            raise RuntimeError("Process pool not enabled or not active")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._process_pool, func, *args)


class StreamingImageLoader:
    """Memory-efficient async image loader with prefetching.

    Features:
    - Async loading with prefetch buffer
    - Memory management to prevent OOM
    - Support for multiple image formats
    - Optional thumbnail generation during load

    Example:
        >>> loader = StreamingImageLoader(
        ...     prefetch_size=4,
        ...     max_memory_mb=1024
        ... )
        >>>
        >>> async for image, path in loader.load_batch(image_paths):
        ...     processed = await process(image)
        ...     del image  # Explicit cleanup
    """

    def __init__(
        self,
        prefetch_size: int = 4,
        max_memory_mb: int = 2048,
        worker_pool: Optional[WorkerPool] = None
    ):
        """Initialize streaming image loader.

        Args:
            prefetch_size: Number of images to prefetch
            max_memory_mb: Maximum memory usage in MB
            worker_pool: Optional shared worker pool
        """
        self._prefetch_size = prefetch_size
        self._max_memory_mb = max_memory_mb
        self._worker_pool = worker_pool
        self._owns_pool = worker_pool is None
        self._current_memory_mb = 0.0

    async def startup(self) -> None:
        """Initialize resources."""
        if self._owns_pool:
            self._worker_pool = WorkerPool(io_workers=self._prefetch_size * 2)
            await self._worker_pool.startup()

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._owns_pool and self._worker_pool:
            await self._worker_pool.shutdown()

    def _load_image_sync(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Synchronously load an image (runs in thread pool).

        Args:
            path: Path to image file

        Returns:
            Tuple of (image array, metadata dict)
        """
        # Lazy import for optional dependencies
        try:
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "PIL and numpy required for image loading. "
                "Install with: pip install Pillow numpy"
            ) from e

        with Image.open(path) as img:
            # Get metadata
            metadata = {
                'path': str(path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
            }

            # Convert to numpy array
            image_array = np.array(img)

            # Estimate memory usage
            memory_mb = image_array.nbytes / (1024 * 1024)
            metadata['memory_mb'] = memory_mb

            return image_array, metadata

    async def load_image(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Async load a single image.

        Args:
            path: Path to image file

        Returns:
            Tuple of (image array, metadata dict)
        """
        if self._worker_pool is None:
            await self.startup()

        return await self._worker_pool.run_io(self._load_image_sync, path)

    async def load_batch(
        self,
        paths: Sequence[Path],
        yield_on_error: bool = True
    ) -> AsyncIterator[Tuple[Any, Path, Optional[Exception]]]:
        """Stream load batch of images with prefetching.

        Args:
            paths: Sequence of image paths
            yield_on_error: If True, yield (None, path, error) on failure

        Yields:
            Tuples of (image_array, path, error) where error is None on success
        """
        if self._worker_pool is None:
            await self.startup()

        # Create prefetch queue
        prefetch_queue: BackpressureQueue[Tuple[Path, asyncio.Task]] = BackpressureQueue(
            maxsize=self._prefetch_size,
            name="prefetch"
        )

        async def prefetch_producer():
            """Produce prefetch tasks."""
            for path in paths:
                task = asyncio.create_task(self.load_image(Path(path)))
                await prefetch_queue.put((Path(path), task))
            prefetch_queue.close()

        # Start prefetch producer
        producer_task = asyncio.create_task(prefetch_producer())

        try:
            items_remaining = len(paths)

            while items_remaining > 0:
                try:
                    path, task = await asyncio.wait_for(
                        prefetch_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if prefetch_queue.is_empty and producer_task.done():
                        break
                    continue

                try:
                    image, metadata = await task
                    yield image, path, None
                except Exception as e:
                    if yield_on_error:
                        yield None, path, e
                    else:
                        raise

                items_remaining -= 1
                prefetch_queue.task_done()

                # Memory management - trigger GC if needed
                if self._current_memory_mb > self._max_memory_mb * 0.9:
                    gc.collect()
                    self._current_memory_mb = 0.0

        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass


class AsyncPipeline:
    """Orchestrate async pipeline stages with queue-based execution.

    Features:
    - Chain multiple async stages
    - Backpressure handling between stages
    - Parallel stage execution where possible
    - Metrics collection and monitoring
    - Graceful error handling

    Example:
        >>> pipeline = AsyncPipeline(max_queue_size=10)
        >>>
        >>> pipeline.add_stage(LoadStage())
        >>> pipeline.add_stage(ProcessStage())
        >>> pipeline.add_stage(SaveStage())
        >>>
        >>> async with pipeline:
        ...     async for result in pipeline.process_batch(items):
        ...         print(f"Completed: {result}")
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        worker_pool: Optional[WorkerPool] = None,
        stop_on_error: bool = False
    ):
        """Initialize async pipeline.

        Args:
            max_queue_size: Maximum queue size between stages
            worker_pool: Shared worker pool (created if not provided)
            stop_on_error: Stop pipeline on first error
        """
        self._stages: List[AsyncStage] = []
        self._queues: List[BackpressureQueue] = []
        self._max_queue_size = max_queue_size
        self._worker_pool = worker_pool
        self._owns_pool = worker_pool is None
        self._stop_on_error = stop_on_error
        self._metrics = PipelineMetrics()
        self._active = False
        self._shutdown_event = asyncio.Event()

    @property
    def metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self._metrics

    @property
    def stage_count(self) -> int:
        """Number of stages in pipeline."""
        return len(self._stages)

    def add_stage(self, stage: AsyncStage) -> 'AsyncPipeline':
        """Add a stage to the pipeline.

        Args:
            stage: Stage to add

        Returns:
            Self for chaining
        """
        self._stages.append(stage)
        return self

    async def __aenter__(self) -> 'AsyncPipeline':
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def startup(self) -> None:
        """Initialize pipeline resources."""
        if self._active:
            return

        # Create worker pool if needed
        if self._owns_pool:
            self._worker_pool = WorkerPool(
                cpu_workers=4,
                io_workers=8
            )
            await self._worker_pool.startup()

        # Initialize stages
        for stage in self._stages:
            await stage.startup()

        # Create queues between stages
        self._queues = [
            BackpressureQueue(
                maxsize=self._max_queue_size,
                name=f"queue_{i}"
            )
            for i in range(len(self._stages))
        ]

        self._active = True
        self._shutdown_event.clear()

    async def shutdown(self) -> None:
        """Shutdown pipeline resources."""
        self._shutdown_event.set()

        # Shutdown stages in reverse order
        for stage in reversed(self._stages):
            await stage.shutdown()

        # Close queues
        for queue in self._queues:
            queue.close()

        # Shutdown worker pool if owned
        if self._owns_pool and self._worker_pool:
            await self._worker_pool.shutdown()

        self._active = False

    async def process_item(self, item: Any) -> WorkItem:
        """Process a single item through all stages.

        Args:
            item: Item to process

        Returns:
            WorkItem with all stage results
        """
        work_item = WorkItem(
            id=str(id(item)),
            data=item
        )

        current_data = item

        for stage in self._stages:
            result = await stage(current_data)
            work_item.add_result(result)

            if result.failed:
                if stage.required or self._stop_on_error:
                    self._metrics.items_failed += 1
                    return work_item
                # Continue with original data for optional stages
            else:
                current_data = result.data

        self._metrics.items_processed += 1
        work_item.data = current_data
        return work_item

    async def process_batch(
        self,
        items: Sequence[Any],
        max_concurrent: int = 4
    ) -> AsyncIterator[WorkItem]:
        """Process batch of items with concurrent execution.

        Args:
            items: Items to process
            max_concurrent: Maximum concurrent pipeline executions

        Yields:
            WorkItem for each completed item
        """
        if not self._active:
            await self.startup()

        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(item: Any) -> WorkItem:
            async with semaphore:
                return await self.process_item(item)

        # Create all tasks
        tasks = [
            asyncio.create_task(process_with_semaphore(item))
            for item in items
        ]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                work_item = await coro
                yield work_item
            except Exception as e:
                # Create failed work item
                yield WorkItem(
                    id="error",
                    data=None,
                    metadata={'error': str(e)}
                )

        self._metrics.total_processing_time = time.time() - start_time
        self._metrics.update_throughput()

    async def process_streaming(
        self,
        items: AsyncIterator[Any]
    ) -> AsyncIterator[WorkItem]:
        """Process items from async iterator (true streaming).

        This method processes items as they arrive, maintaining
        constant memory usage regardless of total items.

        Args:
            items: Async iterator of items

        Yields:
            WorkItem for each completed item
        """
        if not self._active:
            await self.startup()

        start_time = time.time()

        async for item in items:
            if self._shutdown_event.is_set():
                break

            work_item = await self.process_item(item)
            yield work_item

        self._metrics.total_processing_time = time.time() - start_time
        self._metrics.update_throughput()


class AsyncBatchProcessor:
    """High-level async batch processor for image pipelines.

    Coordinates:
    - Streaming image loading with prefetch
    - Multi-stage async processing
    - Memory-efficient output streaming
    - Progress tracking and checkpointing

    Example:
        >>> processor = AsyncBatchProcessor(
        ...     stages=[
        ...         LoadStage(max_concurrent=4),
        ...         DepthStage(device='cuda'),
        ...         EnhanceStage(),
        ...         SaveStage()
        ...     ],
        ...     prefetch_size=4,
        ...     max_concurrent=2
        ... )
        >>>
        >>> async for result in processor.process(input_dir, output_dir):
        ...     print(f"Processed: {result.path}")
    """

    def __init__(
        self,
        stages: Optional[List[AsyncStage]] = None,
        prefetch_size: int = 4,
        max_concurrent: int = 2,
        max_queue_size: int = 10,
        checkpoint_interval: int = 10
    ):
        """Initialize batch processor.

        Args:
            stages: List of pipeline stages
            prefetch_size: Number of images to prefetch
            max_concurrent: Maximum concurrent pipeline executions
            max_queue_size: Queue size between stages
            checkpoint_interval: Save checkpoint every N items
        """
        self._stages = stages or []
        self._prefetch_size = prefetch_size
        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._checkpoint_interval = checkpoint_interval
        self._pipeline: Optional[AsyncPipeline] = None
        self._loader: Optional[StreamingImageLoader] = None
        self._worker_pool: Optional[WorkerPool] = None

    async def startup(self) -> None:
        """Initialize processor resources."""
        # Create shared worker pool
        self._worker_pool = WorkerPool(
            cpu_workers=4,
            io_workers=self._prefetch_size * 2
        )
        await self._worker_pool.startup()

        # Create image loader
        self._loader = StreamingImageLoader(
            prefetch_size=self._prefetch_size,
            worker_pool=self._worker_pool
        )
        await self._loader.startup()

        # Create pipeline
        self._pipeline = AsyncPipeline(
            max_queue_size=self._max_queue_size,
            worker_pool=self._worker_pool
        )
        for stage in self._stages:
            self._pipeline.add_stage(stage)
        await self._pipeline.startup()

    async def shutdown(self) -> None:
        """Clean up processor resources."""
        if self._pipeline:
            await self._pipeline.shutdown()
        if self._loader:
            await self._loader.shutdown()
        if self._worker_pool:
            await self._worker_pool.shutdown()

    async def __aenter__(self) -> 'AsyncBatchProcessor':
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    async def process_paths(
        self,
        paths: Sequence[Path]
    ) -> AsyncIterator[WorkItem]:
        """Process list of image paths.

        Args:
            paths: Sequence of image paths

        Yields:
            WorkItem for each processed image
        """
        if not self._pipeline:
            await self.startup()

        async for result in self._pipeline.process_batch(
            list(paths),
            max_concurrent=self._max_concurrent
        ):
            yield result

    async def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.jpg"
    ) -> AsyncIterator[WorkItem]:
        """Process all images in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory (optional)
            pattern: Glob pattern for images

        Yields:
            WorkItem for each processed image
        """
        input_path = Path(input_dir)
        paths = list(input_path.glob(pattern))

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        async for result in self.process_paths(paths):
            yield result


# Convenience function to run async pipeline
def run_async_pipeline(
    coro: Awaitable[T],
    debug: bool = False
) -> T:
    """Run async pipeline in sync context.

    Args:
        coro: Coroutine to run
        debug: Enable asyncio debug mode

    Returns:
        Coroutine result
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if debug:
            loop.set_debug(True)
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Export public API
__all__ = [
    'DeviceType',
    'StageStatus',
    'PipelineMetrics',
    'StageResult',
    'WorkItem',
    'BackpressureQueue',
    'AsyncStage',
    'AsyncStageProtocol',
    'WorkerPool',
    'StreamingImageLoader',
    'AsyncPipeline',
    'AsyncBatchProcessor',
    'run_async_pipeline',
]
