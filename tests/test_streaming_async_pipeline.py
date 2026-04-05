"""Regression coverage for streaming async pipeline primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

import transformation_portal.streaming.async_pipeline as async_pipeline_module
from transformation_portal.streaming.async_pipeline import AsyncPipeline, AsyncStage, BackpressureQueue


def _patch_time(monkeypatch: pytest.MonkeyPatch, *values: float) -> None:
    iterator = iter(values)
    monkeypatch.setattr(async_pipeline_module.time, "time", lambda: next(iterator))


def _make_clock(*values: float) -> Callable[[], float]:
    """Create a fake clock that returns values in sequence."""
    iterator = iter(values)
    return lambda: next(iterator)


def test_backpressure_queue_get_timeout_does_not_increment_items_got() -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="timeout_get")

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(queue.get(timeout=0.01))

    stats = queue.stats
    assert stats["items_got"] == 0
    assert stats["items_put"] == 0
    assert stats["avg_wait_time"] == 0.0


def test_backpressure_queue_put_timeout_does_not_increment_items_put() -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="timeout_put")
    queue._queue.put_nowait("existing")  # Prime a full queue without affecting success counters.

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(queue.put("blocked", timeout=0.01))

    stats = queue.stats
    assert stats["items_put"] == 0
    assert stats["items_got"] == 0
    assert stats["avg_wait_time"] == 0.0


def test_backpressure_queue_successful_get_updates_stats() -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="success_get", clock=_make_clock(10.0, 10.25))
    queue._queue.put_nowait("payload")

    result = asyncio.run(queue.get())

    stats = queue.stats
    assert result == "payload"
    assert stats["items_got"] == 1
    assert stats["items_put"] == 0
    assert stats["size"] == 0
    assert stats["avg_wait_time"] == pytest.approx(0.25)


def test_backpressure_queue_successful_put_updates_stats() -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="success_put", clock=_make_clock(1.0, 1.5))

    asyncio.run(queue.put("payload"))

    stats = queue.stats
    assert stats["items_put"] == 1
    assert stats["items_got"] == 0
    assert stats["size"] == 1
    assert stats["avg_wait_time"] == pytest.approx(0.5)


def test_backpressure_queue_timeout_does_not_distort_avg_wait_time() -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="avg_wait_time", clock=_make_clock(5.0, 5.4, 6.0))
    asyncio.run(queue.put("payload"))

    baseline = queue.stats["avg_wait_time"]

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(queue.put("blocked", timeout=0.01))

    stats = queue.stats
    assert stats["items_put"] == 1
    assert stats["items_got"] == 0
    assert stats["avg_wait_time"] == pytest.approx(baseline)


class _AddOneStage(AsyncStage[int, int]):
    def __init__(self) -> None:
        super().__init__(name="add_one", max_concurrent=8)

    async def process(self, item: int) -> int:
        await asyncio.sleep(0)
        return item + 1


class _DoubleStage(AsyncStage[int, int]):
    def __init__(self) -> None:
        super().__init__(name="double", max_concurrent=8)

    async def process(self, item: int) -> int:
        await asyncio.sleep(0)
        return item * 2


@dataclass
class _TaskCounter:
    created: int = 0
    live: int = 0
    peak_live: int = 0


class _BarrierStage(AsyncStage[int, int]):
    def __init__(
        self,
        *,
        release_event: asyncio.Event,
        ready_count: int,
        transform: int = 10,
        fail_items: Optional[set[int]] = None,
    ) -> None:
        super().__init__(name="barrier", max_concurrent=64)
        self.release_event = release_event
        self.ready_count = ready_count
        self.transform = transform
        self.fail_items = fail_items or set()
        self.started = 0
        self.active = 0
        self.peak_active = 0
        self.cancelled = 0
        self.ready_event = asyncio.Event()

    async def process(self, item: int) -> int:
        self.started += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if self.started >= self.ready_count:
            self.ready_event.set()

        try:
            await self.release_event.wait()
            if item in self.fail_items:
                raise RuntimeError(f"boom:{item}")
            return item + self.transform
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


class _SelectiveShutdownStage(AsyncStage[int, int]):
    def __init__(self) -> None:
        super().__init__(name="selective_shutdown", max_concurrent=64)
        self.started = 0
        self.started_event = asyncio.Event()
        self.cancelled = 0

    async def process(self, item: int) -> int:
        self.started += 1
        if self.started >= 2:
            self.started_event.set()

        try:
            if item == 0:
                await asyncio.sleep(0)
                return 100
            await asyncio.Event().wait()
            return item
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


async def _collect_process_batch(
    pipeline: AsyncPipeline,
    items: list[int],
    *,
    max_concurrent: int,
) -> list[Any]:
    results = []
    async for result in pipeline.process_batch(items, max_concurrent=max_concurrent):
        results.append(result)
    return results


def _install_task_counter(monkeypatch: pytest.MonkeyPatch) -> _TaskCounter:
    counter = _TaskCounter()
    original = async_pipeline_module._create_process_batch_task

    def instrumented(coro: Any) -> asyncio.Task[Any]:
        task = original(coro)
        counter.created += 1
        counter.live += 1
        counter.peak_live = max(counter.peak_live, counter.live)

        def _on_done(_: asyncio.Task[Any]) -> None:
            counter.live -= 1

        task.add_done_callback(_on_done)
        return task

    monkeypatch.setattr(async_pipeline_module, "_create_process_batch_task", instrumented)
    return counter


def test_async_pipeline_with_fake_stages_processes_end_to_end() -> None:
    async def runner() -> None:
        pipeline = AsyncPipeline().add_stage(_AddOneStage()).add_stage(_DoubleStage())

        async with pipeline:
            results = await _collect_process_batch(pipeline, [1, 2, 3], max_concurrent=2)

        assert sorted(result.data for result in results) == [4, 6, 8]
        assert all(len(result.stage_results) == 2 for result in results)
        assert all(stage_result.success for result in results for stage_result in result.stage_results)
        assert {result.stage_results[0].stage_name for result in results} == {"add_one"}
        assert {result.stage_results[1].stage_name for result in results} == {"double"}
        assert pipeline.metrics.items_processed == 3

    asyncio.run(runner())


def test_process_batch_caps_in_flight_work_at_max_concurrent() -> None:
    async def runner() -> None:
        release_event = asyncio.Event()
        stage = _BarrierStage(release_event=release_event, ready_count=3)
        pipeline = AsyncPipeline().add_stage(stage)

        collector = asyncio.create_task(_collect_process_batch(pipeline, list(range(8)), max_concurrent=3))
        await asyncio.wait_for(stage.ready_event.wait(), timeout=1.0)
        assert stage.peak_active == 3

        release_event.set()
        results = await collector
        assert len(results) == 8

    asyncio.run(runner())


def test_process_batch_handles_large_input_without_one_task_per_item(monkeypatch: pytest.MonkeyPatch) -> None:
    async def runner() -> None:
        release_event = asyncio.Event()
        counter = _install_task_counter(monkeypatch)
        stage = _BarrierStage(release_event=release_event, ready_count=4)
        pipeline = AsyncPipeline().add_stage(stage)

        collector = asyncio.create_task(_collect_process_batch(pipeline, list(range(50)), max_concurrent=4))
        await asyncio.wait_for(stage.ready_event.wait(), timeout=1.0)
        await asyncio.sleep(0.05)

        assert counter.created == 4
        assert counter.peak_live == 4

        release_event.set()
        results = await collector
        assert len(results) == 50

    asyncio.run(runner())


def test_process_batch_yields_all_results_under_bounded_scheduler() -> None:
    async def runner() -> None:
        release_event = asyncio.Event()
        release_event.set()
        stage = _BarrierStage(release_event=release_event, ready_count=1)
        pipeline = AsyncPipeline().add_stage(stage)

        results = await _collect_process_batch(pipeline, list(range(6)), max_concurrent=2)

        assert len(results) == 6
        assert sorted(result.data for result in results) == [10, 11, 12, 13, 14, 15]
        assert all(len(result.stage_results) == 1 for result in results)
        assert all(result.stage_results[0].success for result in results)

    asyncio.run(runner())


def test_process_batch_preserves_failure_results_under_bounded_scheduler() -> None:
    async def runner() -> None:
        release_event = asyncio.Event()
        release_event.set()
        fail_items = {2, 5}
        stage = _BarrierStage(release_event=release_event, ready_count=1, fail_items=fail_items)
        pipeline = AsyncPipeline(stop_on_error=False).add_stage(stage)

        results = await _collect_process_batch(pipeline, list(range(6)), max_concurrent=2)

        assert len(results) == 6
        failures = [result for result in results if result.stage_results[0].failed]
        successes = [result for result in results if not result.stage_results[0].failed]

        assert {result.data for result in failures} == fail_items
        assert all(isinstance(result.stage_results[0].error, RuntimeError) for result in failures)
        assert sorted(result.data for result in successes) == [10, 11, 13, 14]

    asyncio.run(runner())


def test_process_batch_shutdown_cleans_up_pending_work(monkeypatch: pytest.MonkeyPatch) -> None:
    async def runner() -> None:
        release_event = asyncio.Event()
        counter = _install_task_counter(monkeypatch)
        stage = _BarrierStage(release_event=release_event, ready_count=3)
        pipeline = AsyncPipeline().add_stage(stage)

        collector = asyncio.create_task(_collect_process_batch(pipeline, list(range(10)), max_concurrent=3))
        await asyncio.wait_for(stage.ready_event.wait(), timeout=1.0)
        await pipeline.shutdown()

        results = await asyncio.wait_for(collector, timeout=1.0)

        assert results == []
        assert stage.cancelled == 3
        assert counter.live == 0

    asyncio.run(runner())


def test_process_batch_rejects_non_positive_max_concurrent() -> None:
    async def runner() -> None:
        pipeline = AsyncPipeline()

        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            await _collect_process_batch(pipeline, [1], max_concurrent=0)

    asyncio.run(runner())


def test_process_batch_shutdown_preserves_completed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def runner() -> None:
        stage = _SelectiveShutdownStage()
        pipeline = AsyncPipeline().add_stage(stage)
        original = async_pipeline_module._create_process_batch_task

        def instrumented(coro: Any) -> asyncio.Task[Any]:
            task = original(coro)

            def trigger_shutdown(completed: asyncio.Task[Any]) -> None:
                if completed.cancelled():
                    return
                if completed.exception() is not None:
                    return
                work_item = completed.result()
                if work_item.data == 100:
                    pipeline._shutdown_event.set()

            task.add_done_callback(trigger_shutdown)
            return task

        monkeypatch.setattr(async_pipeline_module, "_create_process_batch_task", instrumented)

        results = await _collect_process_batch(pipeline, [0, 1], max_concurrent=2)

        assert [result.data for result in results] == [100]
        assert stage.cancelled == 1

    asyncio.run(runner())


def test_process_batch_wraps_unexpected_exceptions_in_failure_work_item(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_process_item(item: int) -> Any:
        raise RuntimeError(f"broken:{item}")

    async def runner() -> None:
        pipeline = AsyncPipeline()
        monkeypatch.setattr(pipeline, "process_item", broken_process_item)

        results = await _collect_process_batch(pipeline, [7], max_concurrent=1)

        assert len(results) == 1
        failure = results[0]
        assert failure.data == 7
        assert failure.metadata["error"] == "broken:7"
        assert len(failure.stage_results) == 1
        assert failure.stage_results[0].stage_name == "process_batch"
        assert failure.stage_results[0].failed
        assert isinstance(failure.stage_results[0].error, RuntimeError)
        assert pipeline.metrics.items_failed == 1

    asyncio.run(runner())
