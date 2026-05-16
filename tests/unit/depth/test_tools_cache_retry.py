"""Unit coverage for depth tools cache and retry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


def test_bounded_cache_evicts_lru_and_returns_copies() -> None:
    cache = tools.BoundedCache(maxsize=2)
    first = np.array([[1.0, 2.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0]], dtype=np.float32)
    third = np.array([[5.0, 6.0]], dtype=np.float32)

    cache.put("first", first)
    cache.put("second", second)
    cached = cache.get("first")
    assert cached is not None
    cached[0, 0] = 99.0

    cache.put("third", third)

    assert cache.get("second") is None
    assert np.array_equal(cache.get("first"), first)
    assert cache.stats()["size"] == 2
    assert cache.stats()["hits"] == 2
    assert cache.stats()["misses"] == 1


def test_bounded_cache_clear_resets_entries_and_counters() -> None:
    cache = tools.BoundedCache(maxsize=1)
    cache.put("item", np.ones((1, 1), dtype=np.float32))
    assert cache.get("missing") is None
    assert cache.stats()["size"] == 1

    cache.clear()

    assert cache.stats() == {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0}


def test_clear_all_caches_resets_depth_and_mask_caches() -> None:
    tools._depth_cache.put("depth", np.ones((1, 1), dtype=np.float32))
    tools._mask_cache.put("mask", np.ones((1, 1), dtype=np.float32))

    tools.clear_all_caches()

    assert tools._depth_cache.stats()["size"] == 0
    assert tools._mask_cache.stats()["size"] == 0


def test_retry_on_io_error_retries_os_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    attempts = 0

    monkeypatch.setattr(tools.time, "sleep", delays.append)

    @tools.retry_on_io_error(max_attempts=3, initial_delay=0.25, backoff_factor=3.0)
    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary I/O failure")
        return "ok"

    assert flaky() == "ok"
    assert attempts == 3
    assert delays == [0.25, 0.75]


def test_retry_on_io_error_does_not_retry_non_io_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    monkeypatch.setattr(tools.time, "sleep", lambda delay: pytest.fail(f"unexpected sleep {delay}"))

    @tools.retry_on_io_error(max_attempts=3)
    def invalid() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("contract failure")

    with pytest.raises(ValueError, match="contract failure"):
        invalid()
    assert attempts == 1


def test_retry_on_io_error_reraises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.time, "sleep", lambda delay: None)

    @tools.retry_on_io_error(max_attempts=2)
    def always_fails() -> None:
        raise OSError("disk still unavailable")

    with pytest.raises(OSError, match="disk still unavailable"):
        always_fails()


def test_retry_on_io_error_raises_runtime_error_when_loop_never_executes() -> None:
    # max_attempts=0 makes range(1, 1) empty; the defensive RuntimeError
    # fallback path (lines 244-246) must fire instead of returning None.
    @tools.retry_on_io_error(max_attempts=0)
    def never_runs() -> str:
        return "unreachable"

    with pytest.raises(RuntimeError, match="Retry wrapper failed without exception"):
        never_runs()


def test_format_cache_stats_emits_debug_log(caplog: pytest.LogCaptureFixture) -> None:
    cache = tools.BoundedCache(maxsize=4)
    cache.put("k", np.ones((1, 1), dtype=np.float32))
    cache.get("k")
    cache.get("missing")

    caplog.set_level("DEBUG", logger="depth_tools")
    tools._format_cache_stats("UnitProbe", cache.stats())

    assert any("UnitProbe cache:" in record.message for record in caplog.records)
