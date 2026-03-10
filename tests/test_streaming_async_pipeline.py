"""Regression coverage for streaming async pipeline primitives."""

from __future__ import annotations

import asyncio

import pytest

import transformation_portal.streaming.async_pipeline as async_pipeline_module
from transformation_portal.streaming.async_pipeline import BackpressureQueue


def _patch_time(monkeypatch: pytest.MonkeyPatch, *values: float) -> None:
    iterator = iter(values)
    monkeypatch.setattr(async_pipeline_module.time, "time", lambda: next(iterator))


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


def test_backpressure_queue_successful_get_updates_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="success_get")
    queue._queue.put_nowait("payload")
    _patch_time(monkeypatch, 10.0, 10.25)

    result = asyncio.run(queue.get())

    stats = queue.stats
    assert result == "payload"
    assert stats["items_got"] == 1
    assert stats["items_put"] == 0
    assert stats["size"] == 0
    assert stats["avg_wait_time"] == pytest.approx(0.25)


def test_backpressure_queue_successful_put_updates_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="success_put")
    _patch_time(monkeypatch, 1.0, 1.5)

    asyncio.run(queue.put("payload"))

    stats = queue.stats
    assert stats["items_put"] == 1
    assert stats["items_got"] == 0
    assert stats["size"] == 1
    assert stats["avg_wait_time"] == pytest.approx(0.5)


def test_backpressure_queue_timeout_does_not_distort_avg_wait_time(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: BackpressureQueue[str] = BackpressureQueue(maxsize=1, name="avg_wait_time")
    _patch_time(monkeypatch, 5.0, 5.4, 6.0)
    asyncio.run(queue.put("payload"))

    baseline = queue.stats["avg_wait_time"]

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(queue.put("blocked", timeout=0.01))

    stats = queue.stats
    assert stats["items_put"] == 1
    assert stats["items_got"] == 0
    assert stats["avg_wait_time"] == pytest.approx(baseline)
