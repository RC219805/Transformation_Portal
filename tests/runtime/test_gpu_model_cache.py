"""Tests for ModelCache single-flight loading and eviction accounting."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List

import pytest

from transformation_portal.runtime.gpu_model_cache import ModelCache, ModelCacheError

pytestmark = [pytest.mark.unit, pytest.mark.regression]

_WATCHDOG_SEC = 5.0


def _wait_until(predicate: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + _WATCHDOG_SEC
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.001)


def _requests_seen(cache: ModelCache) -> int:
    stats = cache.get_stats()
    return int(stats["hit_count"]) + int(stats["miss_count"])


def _start_callers(cache: ModelCache, key: str, load_fn: Callable[[], Any], count: int) -> tuple:
    results: List[Any] = []
    errors: List[BaseException] = []
    guard = threading.Lock()

    def call() -> None:
        try:
            value = cache.get_or_load(key, load_fn)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            with guard:
                errors.append(exc)
        else:
            with guard:
                results.append(value)

    threads = [threading.Thread(target=call, daemon=True) for _ in range(count)]
    for thread in threads:
        thread.start()
    return threads, results, errors


def _join(threads: List[threading.Thread]) -> None:
    for thread in threads:
        thread.join(_WATCHDOG_SEC)
        assert not thread.is_alive(), "get_or_load caller did not finish"


def test_concurrent_get_or_load_runs_load_fn_once() -> None:
    cache = ModelCache()
    release = threading.Event()
    loads: List[int] = []
    model = object()

    def load() -> object:
        loads.append(1)
        assert release.wait(_WATCHDOG_SEC)
        return model

    threads, results, errors = _start_callers(cache, "llava-large", load, 4)
    # Every caller has registered its request (hit or miss) before the load finishes.
    _wait_until(lambda: _requests_seen(cache) == 4, "all callers to reach the cache")
    release.set()
    _join(threads)

    assert errors == []
    assert len(loads) == 1
    assert len(results) == 4 and all(r is model for r in results)
    assert cache.get_stats()["load_count"] == 1


def test_waiters_share_the_load_failure_and_a_later_call_retries() -> None:
    cache = ModelCache()
    release = threading.Event()
    calls: List[int] = []

    def failing_load() -> object:
        calls.append(1)
        assert release.wait(_WATCHDOG_SEC)
        raise OSError("checkpoint missing")

    threads, results, errors = _start_callers(cache, "m", failing_load, 3)
    _wait_until(lambda: _requests_seen(cache) == 3, "all callers to reach the cache")
    release.set()
    _join(threads)

    assert len(calls) == 1
    assert results == []
    assert len(errors) == 3 and all(isinstance(e, ModelCacheError) for e in errors)

    recovered = object()
    assert cache.get_or_load("m", lambda: recovered) is recovered
    assert cache.get("m") is recovered


def test_max_memory_accounts_for_the_entry_being_inserted() -> None:
    cache = ModelCache(max_memory_bytes=10)
    cache.get_or_load("a", object, size_bytes=8)
    cache.get_or_load("b", object, size_bytes=8)

    stats = cache.get_stats()
    assert stats["total_size_bytes"] <= 10
    assert cache.keys() == ["b"]


def test_entry_larger_than_memory_budget_is_returned_uncached() -> None:
    cache = ModelCache(max_memory_bytes=10)
    cache.get_or_load("small", object, size_bytes=4)
    big = object()

    assert cache.get_or_load("big", lambda: big, size_bytes=11) is big
    assert "big" not in cache
    # The over-budget model must not have evicted the entries that do fit.
    assert cache.keys() == ["small"]


def test_max_entries_zero_does_not_crash_and_caches_nothing() -> None:
    cache = ModelCache(max_entries=0)
    model = object()

    assert cache.get_or_load("a", lambda: model) is model
    assert len(cache) == 0
