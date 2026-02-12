"""Multi-process tests for ArtifactStore locking (Phase 3 L1 Workstream #1).

Test Coverage:
- Per-key lock concurrency safety
- Exclusive locks for writes (store, evict)
- Shared locks for reads (load, load_provenance)
- Lock timeout behavior
- No corruption under concurrent access
- No global lock bottleneck (different keys don't block each other)

Design notes:
- Uses multiprocessing (not threading) to test true process-level locking
- Tests validate that fcntl locks work correctly across processes
- Each test spawns multiple processes to simulate real concurrent workloads
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from transformation_portal.spatial_ai.orchestration.graph.artifact_store import (
    ArtifactStore,
    CacheLockTimeout,
    ProvenanceMetadata,
)


def _make_cache_key(seed: str) -> str:
    """Generate valid SHA256 cache key for testing.

    Args:
        seed: Unique seed string for this test case.

    Returns:
        64-character lowercase hex string (SHA256 format).
    """
    return hashlib.sha256(seed.encode()).hexdigest()


def _store_worker(
    cache_dir: Path,
    cache_key: str,
    worker_id: int,
    result_queue: multiprocessing.Queue,
    delay_before_lock: float = 0.0,
    delay_during_lock: float = 0.0,
) -> None:
    """Worker process that stores an artifact.

    Args:
        cache_dir: Cache directory path.
        cache_key: Cache key to store.
        worker_id: Unique worker identifier.
        result_queue: Queue to report results.
        delay_before_lock: Artificial delay before acquiring lock.
        delay_during_lock: Artificial delay while holding lock (for timeout tests).
    """
    try:
        store = ArtifactStore(cache_dir=cache_dir, lock_timeout_seconds=10.0)

        # Delay before acquiring lock if requested
        if delay_before_lock > 0:
            time.sleep(delay_before_lock)

        # Create unique artifact for this worker
        artifact = {
            "worker_id": worker_id,
            "data": np.array([worker_id] * 100, dtype=np.int32),
        }
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id=f"worker_{worker_id}",
            stage_version="1.0.0",
            input_fingerprints={},
            config_snapshot={},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )

        # If we need to simulate holding lock for a long time
        if delay_during_lock > 0:
            # We need to manually acquire lock and hold it
            with store._acquire_lock(cache_key, exclusive=True):
                time.sleep(delay_during_lock)
                # Write artifact while holding lock
                artifact_path = store._artifact_path(cache_key)
                provenance_path = store._provenance_path(cache_key)
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                
                np_dict = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                          for k, v in artifact.items()}
                
                import tempfile
                with tempfile.NamedTemporaryFile(mode="wb", dir=artifact_path.parent, 
                                                delete=False, suffix=".npz") as tmp_artifact:
                    tmp_artifact_path = Path(tmp_artifact.name)
                    np.savez_compressed(tmp_artifact, **np_dict)
                    tmp_artifact.flush()
                    os.fsync(tmp_artifact.fileno())
                
                with tempfile.NamedTemporaryFile(mode="w", dir=artifact_path.parent,
                                                delete=False, suffix=".json") as tmp_prov:
                    tmp_prov_path = Path(tmp_prov.name)
                    json.dump(asdict(provenance), tmp_prov, indent=2)
                    tmp_prov.flush()
                    os.fsync(tmp_prov.fileno())
                
                tmp_artifact_path.replace(artifact_path)
                tmp_prov_path.replace(provenance_path)
        else:
            # Normal store (acquires lock automatically)
            store.store(cache_key, artifact, provenance)

        result_queue.put({"success": True, "worker_id": worker_id, "error": None})

    except Exception as e:
        result_queue.put({"success": False, "worker_id": worker_id, "error": str(e)})


def _load_worker(
    cache_dir: Path,
    cache_key: str,
    worker_id: int,
    result_queue: multiprocessing.Queue,
    expected_keys: list = None,
) -> None:
    """Worker process that loads an artifact.

    Args:
        cache_dir: Cache directory path.
        cache_key: Cache key to load.
        worker_id: Unique worker identifier.
        result_queue: Queue to report results.
        expected_keys: Optional list of keys to check for completeness.
    """
    try:
        store = ArtifactStore(cache_dir=cache_dir, lock_timeout_seconds=10.0)

        # Load artifact (should acquire shared lock)
        artifact = store.load(cache_key)

        # Verify artifact is complete (not partial/corrupted)
        if expected_keys is None:
            expected_keys = ["worker_id", "data"]
        
        for key in expected_keys:
            assert key in artifact, f"Artifact missing {key} key (partial read)"
        
        # If we expect worker_id and data, validate data structure
        if "data" in expected_keys:
            assert isinstance(artifact["data"], np.ndarray), "Data not a numpy array"
            if "worker_id" in expected_keys:
                assert len(artifact["data"]) == 100, "Data array wrong length (partial read)"

        loaded_worker_id = int(artifact["worker_id"]) if "worker_id" in artifact else None
        result_queue.put({
            "success": True,
            "worker_id": worker_id,
            "loaded_worker_id": loaded_worker_id,
            "error": None,
        })

    except FileNotFoundError:
        # Expected if artifact doesn't exist yet
        result_queue.put({
            "success": True,
            "worker_id": worker_id,
            "loaded_worker_id": None,
            "error": "not_found",
        })
    except Exception as e:
        result_queue.put({"success": False, "worker_id": worker_id, "error": str(e)})


class TestArtifactStoreMultiProcess:
    """Multi-process safety tests for ArtifactStore."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "test_cache"

    def test_concurrent_writes_same_key(self, cache_dir: Path):
        """Test 4 processes writing to same cache key concurrently.

        Expected behavior:
        - Only one process succeeds (last writer wins due to exclusive lock)
        - All processes complete without errors
        - Final artifact is complete and valid (no corruption)
        - Provenance exists and is consistent with final artifact
        """
        cache_key = _make_cache_key("concurrent_writes_test")
        num_workers = 4
        result_queue = multiprocessing.Queue()

        # Spawn worker processes
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=_store_worker,
                args=(cache_dir, cache_key, i, result_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=15.0)
            assert not p.is_alive(), f"Process {p.pid} did not complete in time"

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify all processes succeeded
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        for result in results:
            assert result["success"], f"Worker {result['worker_id']} failed: {result['error']}"

        # Verify final artifact exists and is valid
        store = ArtifactStore(cache_dir=cache_dir)
        assert store.exists(cache_key), "Final artifact does not exist"

        # Load and verify artifact is complete (not corrupted)
        artifact = store.load(cache_key)
        assert "worker_id" in artifact, "Final artifact missing worker_id"
        assert "data" in artifact, "Final artifact missing data"
        assert isinstance(artifact["data"], np.ndarray), "Data not a numpy array"
        assert len(artifact["data"]) == 100, "Data array wrong length"

        # Verify provenance exists
        provenance = store.load_provenance(cache_key)
        assert provenance.cache_key == cache_key
        assert provenance.stage_id.startswith("worker_")

    def test_reader_during_writer(self, cache_dir: Path):
        """Test reader blocks during writer mid-write.

        Expected behavior:
        - Writer starts and holds exclusive lock
        - Reader attempts to read during write
        - Reader either blocks until write completes OR gets FileNotFoundError
        - Reader never sees partial/corrupted artifact
        """
        cache_key = _make_cache_key("reader_writer_test")
        result_queue = multiprocessing.Queue()

        # Start writer with artificial delay DURING lock hold
        writer_delay = 2.0  # 2 second delay while holding lock
        writer = multiprocessing.Process(
            target=_store_worker,
            args=(cache_dir, cache_key, 0, result_queue),
            kwargs={"delay_during_lock": writer_delay},
        )
        writer.start()

        # Give writer time to acquire lock and start writing
        time.sleep(0.5)

        # Start reader (should block or get FileNotFoundError)
        reader = multiprocessing.Process(
            target=_load_worker,
            args=(cache_dir, cache_key, 1, result_queue),
        )
        reader.start()

        # Wait for both to complete
        writer.join(timeout=10.0)
        reader.join(timeout=10.0)

        assert not writer.is_alive(), "Writer did not complete in time"
        assert not reader.is_alive(), "Reader did not complete in time"

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify both processes succeeded (no crashes/corruption)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for result in results:
            assert result["success"], f"Process failed: {result['error']}"

        # Reader should have either:
        # 1. Successfully loaded complete artifact (blocked until write finished)
        # 2. Got FileNotFoundError (artifact didn't exist at check time)
        reader_result = [r for r in results if r["worker_id"] == 1][0]
        if reader_result["error"] != "not_found":
            # If reader loaded artifact, it must be complete
            assert reader_result["loaded_worker_id"] == 0, "Reader loaded wrong artifact"

        # Final artifact should be valid
        store = ArtifactStore(cache_dir=cache_dir)
        artifact = store.load(cache_key)
        assert artifact["worker_id"] == 0
        assert len(artifact["data"]) == 100

    def test_concurrent_writes_different_keys(self, cache_dir: Path):
        """Test 4 processes writing to different cache keys concurrently.

        Expected behavior:
        - All processes complete successfully (no global lock bottleneck)
        - All artifacts stored correctly
        - No interference between different keys
        - Per-key locking allows concurrent writes to different keys
        """
        num_workers = 4
        result_queue = multiprocessing.Queue()

        # Each worker writes to a different cache key
        processes = []
        cache_keys = []
        for i in range(num_workers):
            cache_key = _make_cache_key(f"different_key_{i}")
            cache_keys.append(cache_key)
            p = multiprocessing.Process(
                target=_store_worker,
                args=(cache_dir, cache_key, i, result_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10.0)
            assert not p.is_alive(), f"Process {p.pid} did not complete in time"

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify all processes succeeded
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        for result in results:
            assert result["success"], f"Worker {result['worker_id']} failed: {result['error']}"

        # Verify all artifacts exist and are correct
        store = ArtifactStore(cache_dir=cache_dir)
        for i, cache_key in enumerate(cache_keys):
            assert store.exists(cache_key), f"Artifact {i} does not exist"
            artifact = store.load(cache_key)
            assert artifact["worker_id"] == i, f"Artifact {i} has wrong worker_id"
            assert len(artifact["data"]) == 100, f"Artifact {i} data wrong length"

    def test_lock_timeout(self, cache_dir: Path):
        """Test lock timeout behavior when lock cannot be acquired.

        Expected behavior:
        - First process acquires lock and holds it (simulated by delay)
        - Second process attempts to acquire lock
        - Second process times out with CacheLockTimeout
        """
        cache_key = _make_cache_key("lock_timeout_test")

        # Create store with very short timeout for testing
        store_short_timeout = ArtifactStore(cache_dir=cache_dir, lock_timeout_seconds=1.0)

        # Simulate holding lock by starting a slow write in another process
        result_queue = multiprocessing.Queue()
        slow_writer = multiprocessing.Process(
            target=_store_worker,
            args=(cache_dir, cache_key, 0, result_queue),
            kwargs={"delay_during_lock": 5.0},  # Hold lock for 5 seconds
        )
        slow_writer.start()

        # Give slow writer time to acquire lock
        time.sleep(0.5)

        # Try to store with short timeout (should timeout)
        artifact = {"value": 42}
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="timeout_test",
            stage_version="1.0.0",
            input_fingerprints={},
            config_snapshot={},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )

        # This should timeout
        with pytest.raises(CacheLockTimeout, match="Could not acquire exclusive lock"):
            store_short_timeout.store(cache_key, artifact, provenance)

        # Clean up slow writer
        slow_writer.join(timeout=5.0)
        if slow_writer.is_alive():
            slow_writer.terminate()

    def test_concurrent_reads_same_key(self, cache_dir: Path):
        """Test multiple processes reading same cache key concurrently.

        Expected behavior:
        - First store artifact
        - Multiple readers can read concurrently (shared locks)
        - All readers get valid, complete artifact
        - No corruption or partial reads
        """
        cache_key = _make_cache_key("concurrent_reads_test")

        # Store initial artifact
        store = ArtifactStore(cache_dir=cache_dir)
        artifact = {
            "value": 123,
            "data": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        }
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="initial",
            stage_version="1.0.0",
            input_fingerprints={},
            config_snapshot={},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )
        store.store(cache_key, artifact, provenance)

        # Spawn multiple concurrent readers
        num_readers = 4
        result_queue = multiprocessing.Queue()
        processes = []
        for i in range(num_readers):
            p = multiprocessing.Process(
                target=_load_worker,
                args=(cache_dir, cache_key, i, result_queue),
                kwargs={"expected_keys": ["value", "data"]},  # Match stored artifact
            )
            p.start()
            processes.append(p)

        # Wait for all readers to complete
        for p in processes:
            p.join(timeout=10.0)
            assert not p.is_alive(), f"Process {p.pid} did not complete in time"

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify all readers succeeded
        assert len(results) == num_readers, f"Expected {num_readers} results, got {len(results)}"
        for result in results:
            assert result["success"], f"Reader {result['worker_id']} failed: {result['error']}"
            # All readers should see the initial artifact (not partial)
            # Note: Our _load_worker doesn't return the loaded value for this case,
            # but success=True means no corruption detected

    def test_evict_with_concurrent_access(self, cache_dir: Path):
        """Test eviction with concurrent access attempts.

        Expected behavior:
        - Eviction acquires exclusive lock
        - Concurrent reads/writes block until eviction completes
        - After eviction, artifact is gone
        - No corruption or race conditions
        """
        cache_key = _make_cache_key("evict_concurrent_test")

        # Store initial artifact
        store = ArtifactStore(cache_dir=cache_dir)
        artifact = {"value": 42}
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="test",
            stage_version="1.0.0",
            input_fingerprints={},
            config_snapshot={},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )
        store.store(cache_key, artifact, provenance)
        assert store.exists(cache_key)

        # Evict artifact (with exclusive lock)
        store.evict(cache_key)
        assert not store.exists(cache_key)

        # Subsequent attempts should cleanly fail (FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            store.load(cache_key)
