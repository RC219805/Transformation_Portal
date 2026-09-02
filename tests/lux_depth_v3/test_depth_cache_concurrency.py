"""Concurrency contracts for the identity-v3 depth cache."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tests.lux_depth_v3.test_depth_cache_identity_v3 import _identity as _materialized_identity
from transformation_portal.core.execution_identity_v3 import MaterializedExecutionIdentityV3
from transformation_portal.lux_depth_v3.depth_cache import DepthCache

pytestmark = pytest.mark.unit


def _identity(index: int = 0) -> MaterializedExecutionIdentityV3:
    return _materialized_identity(input_label=f"input-{index}")


def _multiprocess_writer(cache_dir: str, start_index: int, start_event) -> None:
    if not start_event.wait(timeout=10):
        raise TimeoutError("multiprocess cache test did not start")
    cache = DepthCache(Path(cache_dir), max_size_gb=1.0)
    for index in range(start_index, start_index + 16):
        depth = np.full((12, 12), index, dtype=np.float32)
        if not cache.store(_identity(index), depth):
            raise AssertionError(f"multiprocess cache store failed for identity {index}")
        time.sleep(0.001)


def _multiprocess_housekeeper(cache_dir: str, start_event) -> None:
    if not start_event.wait(timeout=10):
        raise TimeoutError("multiprocess cache test did not start")
    cache = DepthCache(Path(cache_dir), max_size_gb=1.0)
    for _ in range(8):
        cache.stats()
        cache.clear()
        time.sleep(0.002)


def _hard_killed_temp_publisher(cache_dir: str, destination_suffix: str, ready_connection) -> None:
    """Pause a child after fsyncing a temp so the parent can kill it."""

    import transformation_portal.lux_depth_v3.depth_cache as depth_cache_module

    cache = DepthCache(Path(cache_dir), max_size_gb=1.0)
    real_replace = os.replace

    def pause_before_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        destination_name = os.fspath(destination)
        if destination_name != depth_cache_module._QUOTA_STATE_NAME and destination_name.endswith(destination_suffix):
            ready_connection.send(destination_suffix)
            while True:
                time.sleep(1)
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    depth_cache_module.os.replace = pause_before_replace
    cache.store(_identity(), np.ones((64, 64), dtype=np.float32))


def _directory_lock_waiter(cache_dir: str, ready_event, start_event, acquired_event) -> None:
    cache = DepthCache(Path(cache_dir), max_size_gb=1.0)
    ready_event.set()
    if not start_event.wait(timeout=10):
        raise TimeoutError("directory-lock waiter did not start")
    with cache._fixed_shard_lock(7):
        acquired_event.set()


def _replacement_namespace_constructor(cache_dir: str, start_event, attempting_event, acquired_event) -> None:
    import transformation_portal.lux_depth_v3.depth_cache as depth_cache_module

    if not start_event.wait(timeout=10):
        raise TimeoutError("replacement namespace constructor did not start")
    real_acquire = depth_cache_module._acquire_platform_file_lock

    def reporting_acquire(descriptor):
        attempting_event.set()
        return real_acquire(descriptor)

    depth_cache_module._acquire_platform_file_lock = reporting_acquire
    DepthCache(Path(cache_dir), max_size_gb=1.0)
    acquired_event.set()


def _preinitialized_quota_writer(
    cache_dir: str, index: int, max_size_gb: float, ready_queue, start_event, result_queue
) -> None:
    cache = DepthCache(Path(cache_dir), max_size_gb=max_size_gb)
    ready_queue.put(index)
    if not start_event.wait(timeout=10):
        raise TimeoutError("quota writer did not start")
    outcome = cache.store(_identity(index), np.full((16, 16), index, dtype=np.float32))
    result_queue.put((index, outcome))


def test_concurrent_same_identity_writers_publish_one_verified_result(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    identity = _identity()
    outcomes: list[bool] = []
    outcome_lock = threading.Lock()

    def write(value: int) -> None:
        stored = cache.store(identity, np.full((32, 32), value, dtype=np.float32))
        with outcome_lock:
            outcomes.append(stored)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The first complete publication wins. Different bytes for the same
    # authoritative identity are rejected as nondeterministic output.
    assert sum(outcomes) == 1
    cached = cache.get(identity)
    assert cached is not None
    assert cached.shape == (32, 32)
    assert np.unique(cached).size == 1


def test_concurrent_same_bytes_are_idempotent_across_cache_instances(tmp_path) -> None:
    identity = _identity()
    depth = np.arange(1024, dtype=np.float32).reshape(32, 32)
    caches = [DepthCache(tmp_path, max_size_gb=1.0) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda cache: cache.store(identity, depth), caches))

    assert all(outcomes)
    cached = caches[0].get(identity)
    assert cached is not None
    np.testing.assert_array_equal(cached, depth)
    assert len(list((tmp_path / ".depth_cache" / "v1" / "objects").glob("*/*.npy"))) == 1


def test_concurrent_different_identity_writes_are_isolated(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)

    def write(index: int) -> bool:
        return cache.store(_identity(index), np.full((20, 20), index, dtype=np.float32))

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(write, range(40)))

    assert all(outcomes)
    for index in range(40):
        cached = cache.get(_identity(index))
        assert cached is not None
        assert np.all(cached == index)
    assert cache.stats()["entry_count"] == 40


def test_concurrent_reads_never_observe_partial_publication(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    identity = _identity()
    depth = np.arange(4096, dtype=np.float32).reshape(64, 64)
    observations: list[np.ndarray | None] = []
    observation_lock = threading.Lock()
    start = threading.Barrier(9)

    def reader() -> None:
        start.wait()
        result = cache.get(identity)
        with observation_lock:
            observations.append(result)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait()
    assert cache.store(identity, depth)
    for thread in threads:
        thread.join()

    for observation in observations:
        if observation is not None:
            np.testing.assert_array_equal(observation, depth)
    final = cache.get(identity)
    assert final is not None
    np.testing.assert_array_equal(final, depth)


def test_stats_and_clear_are_pair_safe_during_writes(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    errors: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            for index in range(offset, offset + 8):
                cache.store(_identity(index), np.full((12, 12), index, dtype=np.float32))
        except BaseException as exc:  # pragma: no cover - assertion reports worker failures
            errors.append(exc)

    def housekeeper() -> None:
        try:
            cache.stats()
            cache.clear()
            cache.stats()
        except BaseException as exc:  # pragma: no cover - assertion reports worker failures
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(index * 10,)) for index in range(4)]
    threads.append(threading.Thread(target=housekeeper))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    final_identity = _identity(999)
    final_depth = np.full((4, 4), 999, dtype=np.float32)
    assert cache.store(final_identity, final_depth)
    np.testing.assert_array_equal(cache.get(final_identity), final_depth)


def test_multiprocess_writers_and_housekeeper_preserve_verified_pairs(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    cache_dir = str(tmp_path / "cache")
    processes = [
        context.Process(target=_multiprocess_writer, args=(cache_dir, 0, start_event)),
        context.Process(target=_multiprocess_writer, args=(cache_dir, 100, start_event)),
        context.Process(target=_multiprocess_housekeeper, args=(cache_dir, start_event)),
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)

    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert alive == []
    assert [process.exitcode for process in processes] == [0, 0, 0]

    cache = DepthCache(Path(cache_dir), max_size_gb=1.0)
    stats = cache.stats()
    pointer_paths = list((cache.cache_dir / "v1" / "entries").glob("*/*.json"))
    object_paths = list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))
    referenced_digests = {json.loads(path.read_bytes())["npy_sha256"] for path in pointer_paths}
    assert stats["entry_count"] == len(pointer_paths)
    assert referenced_digests == {path.stem for path in object_paths}
    for index in (*range(16), *range(100, 116)):
        cached = cache.get(_identity(index))
        if cached is not None:
            np.testing.assert_array_equal(cached, np.full((12, 12), index, dtype=np.float32))

    identity = _identity(999)
    depth = np.full((4, 4), 999, dtype=np.float32)
    assert cache.store(identity, depth)
    np.testing.assert_array_equal(cache.get(identity), depth)
    assert cache.stats()["entry_count"] >= 1


def test_lock_authority_uses_directory_inode_without_lock_files(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    for index in range(4):
        assert cache.store(_identity(index), np.array([[index]], dtype=np.float32))

    cache.stats()
    lock_files = list((tmp_path / ".depth_cache" / "v1" / "locks").glob("*.publication.lock"))
    assert lock_files == []


def test_replaced_fixed_lock_name_cannot_split_directory_lease(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "cache"
    cache = DepthCache(cache_dir, max_size_gb=1.0)
    ready_event = context.Event()
    start_event = context.Event()
    acquired_event = context.Event()
    process = context.Process(
        target=_directory_lock_waiter,
        args=(str(cache_dir), ready_event, start_event, acquired_event),
    )
    process.start()
    assert ready_event.wait(timeout=15)

    lock_path = cache._locks_dir / ".shard-07.publication.lock"
    backup_path = cache._locks_dir / ".shard-07.original.lock"
    with cache._fixed_shard_lock(7):
        if lock_path.exists():
            os.replace(lock_path, backup_path)
        lock_path.write_bytes(b"replacement-inode")
        start_event.set()
        assert not acquired_event.wait(timeout=0.5)

    assert acquired_event.wait(timeout=10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert not process.is_alive()
    assert process.exitcode == 0


def test_v1_replacement_cannot_split_base_directory_lease(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "cache"
    cache = DepthCache(cache_dir, max_size_gb=1.0)
    start_event = context.Event()
    attempting_event = context.Event()
    acquired_event = context.Event()
    process = context.Process(
        target=_replacement_namespace_constructor,
        args=(str(cache_dir), start_event, attempting_event, acquired_event),
    )
    process.start()
    moved_v1 = tmp_path / "v1-moved-outside"

    with pytest.raises(OSError, match="namespace root"):
        with cache._fixed_shard_lock(7):
            cache._v1_dir.rename(moved_v1)
            cache._v1_dir.mkdir()
            for child in ("entries", "objects", "locks"):
                (cache._v1_dir / child).mkdir()
            start_event.set()
            assert attempting_event.wait(timeout=15)
            assert not acquired_event.wait(timeout=0.5)

    assert acquired_event.wait(timeout=10)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert not process.is_alive()
    assert process.exitcode == 0


def test_preinitialized_cache_instances_enforce_limit_on_every_publication(tmp_path) -> None:
    max_bytes = 3_000
    caches = [DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3)) for _ in range(20)]
    for index, cache in enumerate(caches):
        assert cache.store(_identity(index), np.full((16, 16), index, dtype=np.float32))
        pointer_paths = list(cache._entries_dir.glob("*/*.json"))
        object_paths = list(cache._objects_dir.glob("*/*.npy"))
        physical_bytes = sum(path.stat().st_size for path in pointer_paths + object_paths)
        referenced = {json.loads(path.read_bytes())["npy_sha256"] for path in pointer_paths}
        assert physical_bytes <= max_bytes
        assert referenced == {path.stem for path in object_paths}


def test_preinitialized_processes_enforce_shared_limit(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    max_bytes = 3_000
    max_size_gb = max_bytes / (1024**3)
    cache_dir = tmp_path / "cache"
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_preinitialized_quota_writer,
            args=(str(cache_dir), index, max_size_gb, ready_queue, start_event, result_queue),
        )
        for index in range(3)
    ]
    for process in processes:
        process.start()
    assert sorted(ready_queue.get(timeout=15) for _ in processes) == [0, 1, 2]
    start_event.set()
    results = sorted(result_queue.get(timeout=15) for _ in processes)
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert results == [(0, True), (1, True), (2, True)]
    assert [process.exitcode for process in processes] == [0, 0, 0]
    cache = DepthCache(cache_dir, max_size_gb=max_size_gb)
    pointer_paths = list(cache._entries_dir.glob("*/*.json"))
    object_paths = list(cache._objects_dir.glob("*/*.npy"))
    physical_bytes = sum(path.stat().st_size for path in pointer_paths + object_paths)
    referenced = {json.loads(path.read_bytes())["npy_sha256"] for path in pointer_paths}
    assert physical_bytes <= max_bytes
    assert referenced == {path.stem for path in object_paths}


@pytest.mark.parametrize("destination_suffix", [".npy", ".json"])
def test_restart_cleans_temp_left_by_hard_killed_publisher(tmp_path, destination_suffix) -> None:
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "cache"
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_hard_killed_temp_publisher,
        args=(str(cache_dir), destination_suffix, sending_connection),
    )
    process.start()
    sending_connection.close()
    try:
        assert receiving_connection.poll(15), "publisher did not reach the pre-replace boundary"
        assert receiving_connection.recv() == destination_suffix
    finally:
        receiving_connection.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=5)

    assert not process.is_alive()
    assert process.exitcode is not None and process.exitcode != 0
    stale_temps = list((cache_dir / ".depth_cache" / "v1").glob("*/*/*.tmp-*"))
    assert len(stale_temps) == 1

    restarted = DepthCache(cache_dir, max_size_gb=1.0)
    assert list((cache_dir / ".depth_cache" / "v1").glob("*/*/*.tmp-*")) == []
    assert restarted.stats()["entry_count"] == 0
    assert list((cache_dir / ".depth_cache" / "v1" / "objects").glob("*/*.npy")) == []
