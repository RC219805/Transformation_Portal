#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for async/streaming pipeline architecture.

Tests cover:
- BackpressureQueue flow control
- AsyncStage base class and execution
- WorkerPool management
- AsyncPipeline orchestration
- Integration with concrete stages
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transformation_portal.streaming.async_pipeline import (
    AsyncBatchProcessor,
    AsyncPipeline,
    AsyncStage,
    BackpressureQueue,
    DeviceType,
    PipelineMetrics,
    StageResult,
    StageStatus,
    StreamingImageLoader,
    WorkerPool,
    WorkItem,
    run_async_pipeline,
)


# ============================================================================
# BackpressureQueue Tests
# ============================================================================

class TestBackpressureQueue:
    """Tests for BackpressureQueue class."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test queue initialization with default values."""
        queue = BackpressureQueue(maxsize=10, name="test_queue")

        assert queue.size == 0
        assert not queue.is_full
        assert queue.is_empty
        assert not queue.is_backpressured

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test basic put and get operations."""
        queue = BackpressureQueue(maxsize=10)

        await queue.put("item1")
        await queue.put("item2")

        assert queue.size == 2

        item1 = await queue.get()
        item2 = await queue.get()

        assert item1 == "item1"
        assert item2 == "item2"
        assert queue.is_empty

    @pytest.mark.asyncio
    async def test_backpressure_triggers(self):
        """Test that backpressure triggers at high water mark."""
        queue = BackpressureQueue(
            maxsize=10,
            high_water_mark=0.5,  # 50% = 5 items
            low_water_mark=0.2   # 20% = 2 items
        )

        # Fill to high water mark
        for i in range(5):
            await queue.put(f"item_{i}")

        assert queue.is_backpressured

    @pytest.mark.asyncio
    async def test_backpressure_releases(self):
        """Test that backpressure releases at low water mark."""
        queue = BackpressureQueue(
            maxsize=10,
            high_water_mark=0.5,
            low_water_mark=0.2
        )

        # Fill to trigger backpressure
        for i in range(6):
            await queue.put(f"item_{i}")

        assert queue.is_backpressured

        # Drain to below low water mark
        for _ in range(5):
            await queue.get()

        assert not queue.is_backpressured

    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics."""
        queue = BackpressureQueue(maxsize=10, name="stats_test")

        await queue.put("item1")
        await queue.put("item2")
        await queue.get()

        stats = queue.stats

        assert stats['name'] == "stats_test"
        assert stats['size'] == 1
        assert stats['items_put'] == 2
        assert stats['items_got'] == 1

    @pytest.mark.asyncio
    async def test_put_timeout(self):
        """Test put with timeout on full queue."""
        queue = BackpressureQueue(maxsize=2)

        await queue.put("item1")
        await queue.put("item2")

        with pytest.raises(asyncio.TimeoutError):
            await queue.put("item3", timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_timeout(self):
        """Test get with timeout on empty queue."""
        queue = BackpressureQueue(maxsize=10)

        with pytest.raises(asyncio.TimeoutError):
            await queue.get(timeout=0.1)

    @pytest.mark.asyncio
    async def test_closed_queue_raises(self):
        """Test that put raises on closed queue."""
        queue = BackpressureQueue(maxsize=10)
        queue.close()

        with pytest.raises(RuntimeError, match="closed"):
            await queue.put("item")

    @pytest.mark.asyncio
    async def test_task_done_and_join(self):
        """Test task_done and join functionality."""
        queue = BackpressureQueue(maxsize=10)

        await queue.put("item1")

        # Process item
        item = await queue.get()
        queue.task_done()

        # Join should complete immediately
        await asyncio.wait_for(queue.join(), timeout=1.0)


# ============================================================================
# AsyncStage Tests
# ============================================================================

class SimpleStage(AsyncStage[int, int]):
    """Simple test stage that doubles input."""

    def __init__(self, **kwargs):
        super().__init__(name="simple", **kwargs)
        self.process_count = 0

    async def process(self, item: int) -> int:
        self.process_count += 1
        await asyncio.sleep(0.01)  # Simulate work
        return item * 2


class FailingStage(AsyncStage[int, int]):
    """Stage that fails on specific inputs."""

    def __init__(self, fail_on: int = 0, **kwargs):
        super().__init__(name="failing", **kwargs)
        self.fail_on = fail_on

    async def process(self, item: int) -> int:
        if item == self.fail_on:
            raise ValueError(f"Cannot process {item}")
        return item


class TestAsyncStage:
    """Tests for AsyncStage base class."""

    @pytest.mark.asyncio
    async def test_stage_initialization(self):
        """Test stage initialization."""
        stage = SimpleStage(
            device=DeviceType.CPU,
            max_concurrent=4,
            timeout=10.0,
            required=True
        )

        assert stage.name == "simple"
        assert stage.device == DeviceType.CPU
        assert stage.max_concurrent == 4
        assert stage.timeout == 10.0
        assert stage.required is True
        assert stage.status == StageStatus.IDLE

    @pytest.mark.asyncio
    async def test_stage_startup_shutdown(self):
        """Test stage startup and shutdown."""
        stage = SimpleStage()

        await stage.startup()
        assert stage.status == StageStatus.RUNNING

        await stage.shutdown()
        assert stage.status == StageStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stage_process_success(self):
        """Test successful stage processing."""
        stage = SimpleStage()
        await stage.startup()

        result = await stage(5)

        assert result.success
        assert result.data == 10
        assert result.stage_name == "simple"
        assert result.elapsed_time > 0
        assert result.error is None

        await stage.shutdown()

    @pytest.mark.asyncio
    async def test_stage_process_failure(self):
        """Test stage processing failure."""
        stage = FailingStage(fail_on=42, required=False)
        await stage.startup()

        result = await stage(42)

        assert not result.success
        assert result.data is None
        assert result.error is not None
        assert "42" in str(result.error)

        await stage.shutdown()

    @pytest.mark.asyncio
    async def test_stage_timeout(self):
        """Test stage timeout handling."""

        class SlowStage(AsyncStage[int, int]):
            async def process(self, item: int) -> int:
                await asyncio.sleep(10)  # Very slow
                return item

        stage = SlowStage(name="slow", timeout=0.1)
        await stage.startup()

        result = await stage(5)

        assert not result.success
        assert result.metadata.get('error_type') == 'timeout'

        await stage.shutdown()

    @pytest.mark.asyncio
    async def test_stage_concurrency_limit(self):
        """Test that max_concurrent limits parallel processing."""

        class TrackedStage(AsyncStage[int, int]):
            def __init__(self):
                super().__init__(name="tracked", max_concurrent=2)
                self.current_active = 0
                self.max_active = 0

            async def process(self, item: int) -> int:
                self.current_active += 1
                self.max_active = max(self.max_active, self.current_active)
                await asyncio.sleep(0.1)
                self.current_active -= 1
                return item

        stage = TrackedStage()
        await stage.startup()

        # Process many items concurrently
        tasks = [stage(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Max active should not exceed 2
        assert stage.max_active <= 2

        await stage.shutdown()

    @pytest.mark.asyncio
    async def test_stage_metrics(self):
        """Test stage metrics collection."""
        stage = SimpleStage()
        await stage.startup()

        # Process several items
        for i in range(5):
            await stage(i)

        metrics = stage.metrics

        assert metrics['name'] == "simple"
        assert metrics['items_processed'] == 5
        assert metrics['items_failed'] == 0
        assert metrics['total_time'] > 0
        assert metrics['avg_time'] > 0

        await stage.shutdown()


# ============================================================================
# WorkerPool Tests
# ============================================================================

class TestWorkerPool:
    """Tests for WorkerPool class."""

    @pytest.mark.asyncio
    async def test_pool_context_manager(self):
        """Test worker pool as async context manager."""
        async with WorkerPool(cpu_workers=2, io_workers=4) as pool:
            assert pool._active
            assert pool._cpu_pool is not None
            assert pool._io_pool is not None

        # Should be cleaned up after exit
        assert not pool._active

    @pytest.mark.asyncio
    async def test_run_cpu_task(self):
        """Test running CPU-bound task."""

        def cpu_work(x: int) -> int:
            return x * x

        async with WorkerPool(cpu_workers=2) as pool:
            result = await pool.run_cpu(cpu_work, 5)
            assert result == 25

    @pytest.mark.asyncio
    async def test_run_io_task(self):
        """Test running I/O-bound task."""
        import time

        def io_work(delay: float) -> float:
            time.sleep(delay)
            return delay

        async with WorkerPool(io_workers=4) as pool:
            start = time.time()
            result = await pool.run_io(io_work, 0.1)
            elapsed = time.time() - start

            assert result == 0.1
            assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_run_cpu_with_kwargs(self):
        """Test running CPU task with keyword arguments."""

        def add(a: int, b: int = 0) -> int:
            return a + b

        async with WorkerPool() as pool:
            result = await pool.run_cpu(add, 5, b=3)
            assert result == 8

    @pytest.mark.asyncio
    async def test_parallel_io_tasks(self):
        """Test running multiple I/O tasks in parallel."""
        import time

        def io_work(x: int) -> int:
            time.sleep(0.1)
            return x

        async with WorkerPool(io_workers=4) as pool:
            start = time.time()

            # Run 4 tasks in parallel
            tasks = [pool.run_io(io_work, i) for i in range(4)]
            results = await asyncio.gather(*tasks)

            elapsed = time.time() - start

            assert sorted(results) == [0, 1, 2, 3]
            # Should complete in ~0.1s (parallel) not ~0.4s (sequential)
            assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_pool_not_active_raises(self):
        """Test that running on inactive pool raises."""
        pool = WorkerPool()

        with pytest.raises(RuntimeError, match="not active"):
            await pool.run_cpu(lambda x: x, 1)


# ============================================================================
# AsyncPipeline Tests
# ============================================================================

class TestAsyncPipeline:
    """Tests for AsyncPipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AsyncPipeline(max_queue_size=10)

        assert pipeline.stage_count == 0
        assert not pipeline._active

    @pytest.mark.asyncio
    async def test_add_stages(self):
        """Test adding stages to pipeline."""
        pipeline = AsyncPipeline()

        stage1 = SimpleStage()
        stage2 = SimpleStage()

        pipeline.add_stage(stage1).add_stage(stage2)

        assert pipeline.stage_count == 2

    @pytest.mark.asyncio
    async def test_pipeline_context_manager(self):
        """Test pipeline as async context manager."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(SimpleStage())

        async with pipeline:
            assert pipeline._active

        assert not pipeline._active

    @pytest.mark.asyncio
    async def test_process_single_item(self):
        """Test processing single item through pipeline."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(SimpleStage())

        async with pipeline:
            result = await pipeline.process_item(5)

            assert result.data == 10
            assert len(result.stage_results) == 1
            assert result.stage_results[0].success

    @pytest.mark.asyncio
    async def test_process_multi_stage(self):
        """Test processing through multiple stages."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(SimpleStage())  # doubles
        pipeline.add_stage(SimpleStage())  # doubles again

        async with pipeline:
            result = await pipeline.process_item(5)

            assert result.data == 20  # 5 * 2 * 2

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(SimpleStage())

        items = [1, 2, 3, 4, 5]

        async with pipeline:
            results = []
            async for result in pipeline.process_batch(items, max_concurrent=2):
                results.append(result.data)

            # Results may be out of order due to async
            assert sorted(results) == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_pipeline_metrics(self):
        """Test pipeline metrics collection."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(SimpleStage())

        async with pipeline:
            async for _ in pipeline.process_batch([1, 2, 3]):
                pass

            metrics = pipeline.metrics

            assert metrics.items_processed == 3
            assert metrics.total_processing_time > 0

    @pytest.mark.asyncio
    async def test_optional_stage_failure(self):
        """Test that optional stage failure doesn't stop pipeline."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(FailingStage(fail_on=2, required=False))

        async with pipeline:
            results = []
            async for result in pipeline.process_batch([1, 2, 3]):
                results.append(result)

            # All items should be processed
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_required_stage_failure(self):
        """Test that required stage failure marks item as failed."""
        pipeline = AsyncPipeline()
        pipeline.add_stage(FailingStage(fail_on=2, required=True))

        async with pipeline:
            results = []
            async for result in pipeline.process_batch([1, 2, 3]):
                results.append(result)

            # Item 2 should have failed
            failed_items = [r for r in results if any(sr.failed for sr in r.stage_results)]
            assert len(failed_items) == 1


# ============================================================================
# WorkItem and StageResult Tests
# ============================================================================

class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_work_item_creation(self):
        """Test work item creation."""
        item = WorkItem(id="test-1", data={"key": "value"})

        assert item.id == "test-1"
        assert item.data == {"key": "value"}
        assert item.created_at > 0
        assert len(item.stage_results) == 0

    def test_work_item_elapsed_time(self):
        """Test elapsed time calculation."""
        item = WorkItem(id="test", data=None)
        time.sleep(0.1)

        elapsed = item.elapsed_time
        assert elapsed >= 0.1

    def test_add_result(self):
        """Test adding stage results."""
        item = WorkItem(id="test", data=None)

        result = StageResult(
            data="processed",
            stage_name="test_stage",
            elapsed_time=0.1,
            success=True
        )

        item.add_result(result)

        assert len(item.stage_results) == 1
        assert item.stage_results[0].stage_name == "test_stage"


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_successful_result(self):
        """Test successful result creation."""
        result = StageResult(
            data="output",
            stage_name="test",
            elapsed_time=0.5,
            success=True
        )

        assert result.success
        assert not result.failed
        assert result.data == "output"
        assert result.error is None

    def test_failed_result(self):
        """Test failed result creation."""
        error = ValueError("test error")
        result = StageResult(
            data=None,
            stage_name="test",
            elapsed_time=0.1,
            success=False,
            error=error
        )

        assert not result.success
        assert result.failed
        assert result.error is error


# ============================================================================
# PipelineMetrics Tests
# ============================================================================

class TestPipelineMetrics:
    """Tests for PipelineMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PipelineMetrics()

        assert metrics.items_processed == 0
        assert metrics.items_failed == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.throughput_items_per_sec == 0.0

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = PipelineMetrics(
            items_processed=100,
            total_processing_time=10.0
        )

        metrics.update_throughput()

        assert metrics.throughput_items_per_sec == 10.0

    def test_throughput_zero_time(self):
        """Test throughput with zero time."""
        metrics = PipelineMetrics(
            items_processed=100,
            total_processing_time=0.0
        )

        metrics.update_throughput()

        assert metrics.throughput_items_per_sec == 0.0


# ============================================================================
# StreamingImageLoader Tests
# ============================================================================

class TestStreamingImageLoader:
    """Tests for StreamingImageLoader class."""

    @pytest.mark.asyncio
    async def test_loader_initialization(self):
        """Test loader initialization."""
        loader = StreamingImageLoader(
            prefetch_size=4,
            max_memory_mb=1024
        )

        assert loader._prefetch_size == 4
        assert loader._max_memory_mb == 1024

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        loader = StreamingImageLoader()
        await loader.startup()

        try:
            with pytest.raises(Exception):
                await loader.load_image(Path("/nonexistent/image.jpg"))
        finally:
            await loader.shutdown()


# ============================================================================
# Integration Tests
# ============================================================================

class TestAsyncPipelineIntegration:
    """Integration tests for async pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete pipeline flow with multiple stages."""

        class AddStage(AsyncStage[int, int]):
            def __init__(self, value: int):
                super().__init__(name=f"add_{value}")
                self.value = value

            async def process(self, item: int) -> int:
                return item + self.value

        class MultiplyStage(AsyncStage[int, int]):
            def __init__(self, factor: int):
                super().__init__(name=f"multiply_{factor}")
                self.factor = factor

            async def process(self, item: int) -> int:
                return item * self.factor

        pipeline = AsyncPipeline()
        pipeline.add_stage(AddStage(10))      # x + 10
        pipeline.add_stage(MultiplyStage(2))  # (x + 10) * 2

        async with pipeline:
            result = await pipeline.process_item(5)

            # (5 + 10) * 2 = 30
            assert result.data == 30

    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        """Test AsyncBatchProcessor initialization."""
        processor = AsyncBatchProcessor(
            stages=[SimpleStage()],
            prefetch_size=2,
            max_concurrent=1
        )

        assert processor._prefetch_size == 2
        assert processor._max_concurrent == 1

    def test_run_async_pipeline_helper(self):
        """Test run_async_pipeline convenience function."""
        # Note: This test is not marked as asyncio since run_async_pipeline
        # creates its own event loop (for use in sync contexts)

        async def simple_async():
            return 42

        result = run_async_pipeline(simple_async())
        assert result == 42


# ============================================================================
# Device Type Tests
# ============================================================================

class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types(self):
        """Test all device types are available."""
        assert DeviceType.CPU
        assert DeviceType.CUDA
        assert DeviceType.MPS
        assert DeviceType.AUTO


# ============================================================================
# Stage Status Tests
# ============================================================================

class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_status_values(self):
        """Test all status values are available."""
        assert StageStatus.IDLE
        assert StageStatus.RUNNING
        assert StageStatus.PAUSED
        assert StageStatus.COMPLETED
        assert StageStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
