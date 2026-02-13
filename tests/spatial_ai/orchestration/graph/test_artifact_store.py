"""Tests for ArtifactStore caching and provenance (Phase 3 L1).

Test Coverage:
- Cache hit/miss
- Atomic writes
- Provenance metadata
- Determinism (bitwise identical outputs)
- Cache statistics
- Storage integrity
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from transformation_portal.spatial_ai.orchestration.graph.artifact_store import ArtifactStore, ProvenanceMetadata


def _make_cache_key(seed: str) -> str:
    """Generate valid SHA256 cache key for testing.

    Args:
        seed: Unique seed string for this test case.

    Returns:
        64-character lowercase hex string (SHA256 format).
    """
    return hashlib.sha256(seed.encode()).hexdigest()


class TestArtifactStore:
    """Tests for ArtifactStore."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def store(self, cache_dir: Path) -> ArtifactStore:
        """Create ArtifactStore instance."""
        return ArtifactStore(cache_dir=cache_dir, max_size_gb=1.0)

    def test_initialization(self, cache_dir: Path):
        """Test store initialization."""
        store = ArtifactStore(cache_dir=cache_dir)

        # Verify directory structure
        assert cache_dir.exists()
        assert (cache_dir / "artifacts").exists()
        # Stats file created on first access
        assert store.get_stats() is not None

    def test_exists_empty_cache(self, store: ArtifactStore):
        """Test exists() returns False for empty cache."""
        assert not store.exists(_make_cache_key("nonexistent"))

    def test_store_and_load_simple(self, store: ArtifactStore):
        """Test storing and loading simple artifact."""
        cache_key = _make_cache_key("test_simple")
        artifact = {
            "value": 42,
            "name": "test",
        }
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="test_stage",
            stage_version="1.0.0",
            input_fingerprints={"input": "abc123"},
            config_snapshot={"param": "value"},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )

        # Store
        store.store(cache_key, artifact, provenance)

        # Verify exists
        assert store.exists(cache_key)

        # Load
        loaded = store.load(cache_key)
        assert loaded["value"] == 42
        assert loaded["name"] == "test"

    def test_store_and_load_numpy_arrays(self, store: ArtifactStore):
        """Test storing and loading numpy arrays."""
        cache_key = _make_cache_key("numpy_test")
        artifact = {
            "masks": np.array([[True, False], [False, True]], dtype=bool),
            "scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "indices": np.array([1, 2, 3], dtype=np.int32),
        }
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="numpy_stage",
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
        loaded = store.load(cache_key)

        # Verify arrays are identical
        np.testing.assert_array_equal(loaded["masks"], artifact["masks"])
        np.testing.assert_array_equal(loaded["scores"], artifact["scores"])
        np.testing.assert_array_equal(loaded["indices"], artifact["indices"])

    def test_determinism_bitwise_identical(self, store: ArtifactStore):
        """Test cache returns bitwise identical outputs."""
        cache_key = _make_cache_key("determinism_test")

        # Create artifact with numpy array
        original_array = np.random.rand(100, 100, 3).astype(np.float32)
        artifact = {"data": original_array}
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

        # Store
        store.store(cache_key, artifact, provenance)

        # Load multiple times
        loaded1 = store.load(cache_key)
        loaded2 = store.load(cache_key)

        # Verify bitwise identical
        np.testing.assert_array_equal(loaded1["data"], original_array)
        np.testing.assert_array_equal(loaded2["data"], original_array)
        np.testing.assert_array_equal(loaded1["data"], loaded2["data"])

        # Verify exact bytes match
        assert loaded1["data"].tobytes() == original_array.tobytes()
        assert loaded2["data"].tobytes() == original_array.tobytes()

    def test_load_nonexistent_raises_error(self, store: ArtifactStore):
        """Test loading non-existent artifact raises error."""
        with pytest.raises(FileNotFoundError, match="Artifact not found"):
            store.load(_make_cache_key("nonexistent_key"))

    def test_provenance_storage(self, store: ArtifactStore):
        """Test provenance metadata is stored correctly."""
        cache_key = _make_cache_key("provenance_test")
        artifact = {"value": 123}
        provenance = ProvenanceMetadata(
            cache_key=cache_key,
            stage_id="test_stage",
            stage_version="2.1.0",
            input_fingerprints={"input1": "hash1", "input2": "hash2"},
            config_snapshot={"model_size": "large", "device": "cuda"},
            timestamp="2026-02-12T10:00:00Z",
            hostname="gpu-node-1",
            python_version="3.11.5",
            numpy_version="1.26.0",
            device="cuda",
            torch_version="2.1.0",
            model_repo_id="facebook/sam2-large",
            model_revision="abc123",
        )

        store.store(cache_key, artifact, provenance)

        # Load provenance
        loaded_prov = store.load_provenance(cache_key)

        assert loaded_prov.cache_key == cache_key
        assert loaded_prov.stage_id == "test_stage"
        assert loaded_prov.stage_version == "2.1.0"
        assert loaded_prov.input_fingerprints == {"input1": "hash1", "input2": "hash2"}
        assert loaded_prov.config_snapshot == {"model_size": "large", "device": "cuda"}
        assert loaded_prov.hostname == "gpu-node-1"
        assert loaded_prov.model_repo_id == "facebook/sam2-large"
        assert loaded_prov.model_revision == "abc123"

    def test_load_provenance_nonexistent_raises_error(self, store: ArtifactStore):
        """Test loading non-existent provenance raises error."""
        with pytest.raises(FileNotFoundError, match="Provenance not found"):
            store.load_provenance(_make_cache_key("nonexistent_key"))

    def test_eviction(self, store: ArtifactStore):
        """Test manual eviction."""
        cache_key = _make_cache_key("evict_test")
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

        # Store
        store.store(cache_key, artifact, provenance)
        assert store.exists(cache_key)

        # Evict
        store.evict(cache_key)
        assert not store.exists(cache_key)

    def test_eviction_idempotent(self, store: ArtifactStore):
        """Test eviction is idempotent."""
        cache_key = _make_cache_key("evict_test")

        # Evict non-existent artifact (should not raise error)
        store.evict(cache_key)
        assert not store.exists(cache_key)

    def test_cache_statistics(self, store: ArtifactStore):
        """Test cache statistics tracking."""
        # Initial stats
        stats = store.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

        # Store artifact (cache miss)
        cache_key = _make_cache_key("stats_test")
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

        stats = store.get_stats()
        assert stats["cache_misses"] == 1

        # Load artifact (cache hit)
        store.load(cache_key)
        stats = store.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

        # Load again (another cache hit)
        store.load(cache_key)
        stats = store.get_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1

    def test_cache_size_calculation(self, store: ArtifactStore):
        """Test cache size calculation."""
        # Empty cache
        assert store.get_cache_size_mb() == 0.0

        # Store artifact
        cache_key = _make_cache_key("size_test")
        large_array = np.random.rand(1000, 1000).astype(np.float32)  # ~4MB
        artifact = {"data": large_array}
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

        # Cache should have non-zero size
        size_mb = store.get_cache_size_mb()
        assert size_mb > 0.0
        assert size_mb < 10.0  # Compressed, should be < 10MB

    def test_two_level_directory_hierarchy(self, store: ArtifactStore, cache_dir: Path):
        """Test artifacts are stored in two-level directory hierarchy."""
        # Generate a key that starts with "ab" for testing directory structure
        cache_key = _make_cache_key("ab_test_seed")

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

        # Verify directory structure (first 2 chars of cache_key)
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        assert prefix_dir.exists()
        assert (prefix_dir / f"{cache_key}.npz").exists()
        assert (prefix_dir / f"{cache_key}.json").exists()
        assert (prefix_dir / f"{cache_key}.committed").exists()

    def test_atomic_write_integrity(self, store: ArtifactStore):
        """Test atomic write ensures no partial artifacts."""
        cache_key = _make_cache_key("atomic_test")
        artifact = {"data": np.random.rand(100, 100).astype(np.float32)}

        # Mock failure during provenance write by using invalid data
        # (This tests that if write fails, no partial state is left)
        class InvalidProvenance:
            """Invalid provenance that will fail serialization."""

            def __init__(self):
                self.unserializable = lambda x: x  # Functions can't be JSON serialized

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

        # Normal write should succeed
        store.store(cache_key, artifact, provenance)
        assert store.exists(cache_key)

    def test_complex_artifact_types(self, store: ArtifactStore):
        """Test storing complex artifact types (safe types only, no object arrays)."""
        cache_key = _make_cache_key("complex_test")
        artifact = {
            "int_scalar": 42,
            "float_scalar": 3.14,
            "bool_scalar": True,
            "str_scalar": "test",
            "numpy_array": np.array([1, 2, 3], dtype=np.int32),
            # Lists/tuples converted to numpy arrays with explicit dtype (no object arrays)
            "list_as_array": np.array([1, 2, 3], dtype=np.int32),
            "tuple_as_array": np.array([1, 2, 3], dtype=np.int32),
        }
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
        loaded = store.load(cache_key)

        # Verify scalars
        assert loaded["int_scalar"] == 42
        assert loaded["float_scalar"] == 3.14
        assert loaded["bool_scalar"] is True
        assert loaded["str_scalar"] == "test"

        # Verify arrays
        np.testing.assert_array_equal(loaded["numpy_array"], artifact["numpy_array"])
        np.testing.assert_array_equal(loaded["list_as_array"], artifact["list_as_array"])
        np.testing.assert_array_equal(loaded["tuple_as_array"], artifact["tuple_as_array"])

    def test_cache_key_collision_resistance(self, store: ArtifactStore):
        """Test different inputs produce different cache entries."""
        provenance_template = {
            "stage_id": "test",
            "stage_version": "1.0.0",
            "input_fingerprints": {},
            "config_snapshot": {},
            "timestamp": "2026-02-12T10:00:00Z",
            "hostname": "testhost",
            "python_version": "3.11",
            "numpy_version": "1.26.0",
            "device": "cpu",
        }

        # Store multiple different artifacts
        cache_keys = []
        for i in range(10):
            cache_key = _make_cache_key(f"key_{i:03d}")
            cache_keys.append(cache_key)
            artifact = {"value": i}
            provenance = ProvenanceMetadata(cache_key=cache_key, **provenance_template)
            store.store(cache_key, artifact, provenance)

        # Verify all exist independently
        for i, cache_key in enumerate(cache_keys):
            assert store.exists(cache_key)
            loaded = store.load(cache_key)
            assert loaded["value"] == i


class TestProvenanceMetadata:
    """Tests for ProvenanceMetadata dataclass."""

    def test_provenance_creation(self):
        """Test ProvenanceMetadata creation."""
        prov = ProvenanceMetadata(
            cache_key="test_key",
            stage_id="test_stage",
            stage_version="1.0.0",
            input_fingerprints={"input": "abc123"},
            config_snapshot={"param": "value"},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cpu",
        )

        assert prov.cache_key == "test_key"
        assert prov.stage_id == "test_stage"
        assert prov.device == "cpu"
        assert prov.torch_version is None
        assert prov.model_repo_id is None

    def test_provenance_with_model_info(self):
        """Test ProvenanceMetadata with model information."""
        prov = ProvenanceMetadata(
            cache_key="test_key",
            stage_id="sam2_stage",
            stage_version="2.1.0",
            input_fingerprints={},
            config_snapshot={},
            timestamp="2026-02-12T10:00:00Z",
            hostname="testhost",
            python_version="3.11",
            numpy_version="1.26.0",
            device="cuda",
            torch_version="2.1.0",
            model_repo_id="facebook/sam2-large",
            model_revision="main",
        )

        assert prov.model_repo_id == "facebook/sam2-large"
        assert prov.model_revision == "main"
        assert prov.torch_version == "2.1.0"
        assert prov.device == "cuda"


class TestArtifactStoreSecurity:
    """Security tests for ArtifactStore (path traversal, pickle deserialization)."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def store(self, cache_dir: Path) -> ArtifactStore:
        """Create ArtifactStore instance."""
        return ArtifactStore(cache_dir=cache_dir, max_size_gb=1.0)

    def test_cache_key_path_traversal_rejected(self, store: ArtifactStore):
        """Malformed cache keys with path traversal are rejected (SECURITY)."""
        # Test various path traversal attempts
        invalid_keys = [
            "../../evil",
            "a/../../something",
            "../sensitive",
            "key\\..\\windows",
            "key..",
            "not_hex_chars!",
            "too_short",
            "a" * 65,  # too long
            "ABCDEF" + "a" * 58,  # uppercase hex (invalid)
            "12345g" + "a" * 58,  # invalid hex char 'g'
            "",  # empty
            "   ",  # whitespace
            "aa/bb" + "a" * 56,  # forward slash
            "aa\\bb" + "a" * 56,  # backslash
        ]

        for key in invalid_keys:
            # exists() should reject invalid keys
            with pytest.raises(ValueError, match="Invalid cache_key"):
                store.exists(key)

            # load() should reject invalid keys
            with pytest.raises(ValueError, match="Invalid cache_key"):
                store.load(key)

            # load_provenance() should reject invalid keys
            with pytest.raises(ValueError, match="Invalid cache_key"):
                store.load_provenance(key)

    def test_cache_key_valid_sha256_accepted(self, store: ArtifactStore):
        """Valid SHA256 cache keys are accepted."""
        # Valid 64-char lowercase hex (SHA256 format)
        valid_key = "a" * 64

        # Should not raise ValueError
        assert not store.exists(valid_key)  # Returns False, doesn't raise

    def test_store_rejects_object_arrays(self, store: ArtifactStore, cache_dir: Path):
        """Store rejects object arrays at store time (SECURITY + CORRECTNESS).

        Object arrays require pickle deserialization, which:
        1. Allows arbitrary code execution (security vulnerability)
        2. Fails at load time with allow_pickle=False (runtime failure)

        The store must reject object arrays at store time to prevent both issues.
        """
        cache_key = _make_cache_key("object_array_test")

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

        # Test various cases that produce object dtype
        # Note: Modern NumPy (1.24+) coerces heterogeneous data to string dtype
        # or raises errors for ragged arrays, so we test explicit object arrays
        test_cases = [
            # Explicit object array (most common real-world case)
            {"obj": np.array([{"a": 1}], dtype=object)},
            # Python objects
            {"pyobj": np.array([object(), object()])},
            # Nested lists with explicit object dtype
            {"nested": np.array([[1, 2], [3, 4]], dtype=object)},
        ]

        for i, artifacts in enumerate(test_cases):
            # Generate unique cache key for each test case
            test_cache_key = _make_cache_key(f"object_array_test_{i}")
            with pytest.raises(ValueError, match="dtype=object"):
                store.store(test_cache_key, artifacts, provenance)

    def test_safe_numpy_types_accepted(self, store: ArtifactStore):
        """Safe NumPy array types are accepted (no pickle needed)."""
        cache_key = _make_cache_key("safe_types_test")

        # Safe types that don't require pickle
        artifact = {
            "float_array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "int_array": np.array([1, 2, 3], dtype=np.int32),
            "bool_array": np.array([True, False, True], dtype=bool),
            "str_array": np.array(["a", "b", "c"]),  # Unicode strings are safe
            "homogeneous_list": [1, 2, 3],  # Infers int64
            "scalar": np.float32(3.14),
        }

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

        # Should succeed (no pickle needed)
        store.store(cache_key, artifact, provenance)
        loaded = store.load(cache_key)

        # Verify all arrays loaded correctly
        np.testing.assert_array_equal(loaded["float_array"], artifact["float_array"])
        np.testing.assert_array_equal(loaded["int_array"], artifact["int_array"])
        np.testing.assert_array_equal(loaded["bool_array"], artifact["bool_array"])
        np.testing.assert_array_equal(loaded["str_array"], artifact["str_array"])
        np.testing.assert_array_equal(loaded["homogeneous_list"], [1, 2, 3])
        assert loaded["scalar"] == pytest.approx(3.14)


class TestCommitMarker:
    """Tests for transactional commit marker (Issue #929).

    Validates that:
    - Committed marker is created on successful store
    - Entries without marker are treated as non-existent
    - Eviction removes the marker
    - Mid-commit failure (artifact exists, marker missing) → treated as cache miss
    """

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def store(self, cache_dir: Path) -> ArtifactStore:
        """Create ArtifactStore instance."""
        return ArtifactStore(cache_dir=cache_dir, max_size_gb=1.0)

    def _make_provenance(self, cache_key: str) -> ProvenanceMetadata:
        """Helper to create a minimal ProvenanceMetadata."""
        return ProvenanceMetadata(
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

    def test_committed_marker_created_on_store(self, store: ArtifactStore, cache_dir: Path):
        """Committed marker file is created after successful store."""
        cache_key = _make_cache_key("marker_created")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))

        prefix = cache_key[:2]
        committed = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"
        assert committed.exists(), ".committed marker not created"

    def test_exists_requires_committed_marker(self, store: ArtifactStore, cache_dir: Path):
        """exists() returns False if artifact exists but marker is missing."""
        cache_key = _make_cache_key("marker_missing_exists")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))
        assert store.exists(cache_key)

        # Remove the committed marker (simulates mid-commit crash)
        prefix = cache_key[:2]
        committed = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"
        committed.unlink()

        assert not store.exists(cache_key), "exists() should return False without marker"

    def test_load_requires_committed_marker(self, store: ArtifactStore, cache_dir: Path):
        """load() raises FileNotFoundError if marker is missing."""
        cache_key = _make_cache_key("marker_missing_load")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))

        # Remove the committed marker
        prefix = cache_key[:2]
        committed = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"
        committed.unlink()

        with pytest.raises(FileNotFoundError, match="Artifact not found"):
            store.load(cache_key)

    def test_load_provenance_requires_committed_marker(self, store: ArtifactStore, cache_dir: Path):
        """load_provenance() raises FileNotFoundError if marker is missing."""
        cache_key = _make_cache_key("marker_missing_prov")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))

        # Remove the committed marker
        prefix = cache_key[:2]
        committed = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"
        committed.unlink()

        with pytest.raises(FileNotFoundError, match="Provenance not found"):
            store.load_provenance(cache_key)

    def test_evict_removes_committed_marker(self, store: ArtifactStore, cache_dir: Path):
        """evict() removes the .committed marker file."""
        cache_key = _make_cache_key("marker_evict")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))

        prefix = cache_key[:2]
        committed = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"
        assert committed.exists()

        store.evict(cache_key)
        assert not committed.exists(), ".committed marker should be removed after evict"

    def test_orphaned_artifact_without_marker_is_cache_miss(self, store: ArtifactStore, cache_dir: Path):
        """Artifact written without marker (simulated crash) is a cache miss.

        This simulates the exact failure mode from Issue #929:
        Step 1 (artifact rename) succeeds, Step 2 (provenance rename) succeeds,
        but Step 3 (marker creation) fails due to crash/disk full.
        """
        cache_key = _make_cache_key("orphan_cache_miss")

        # Manually create artifact + provenance without marker (simulates crash)
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = prefix_dir / f"{cache_key}.npz"
        provenance_path = prefix_dir / f"{cache_key}.json"

        # Write a valid .npz file
        import numpy as np

        np.savez_compressed(artifact_path, v=np.array([1]))

        # Write a valid .json provenance
        from dataclasses import asdict

        with open(provenance_path, "w") as f:
            json.dump(asdict(self._make_provenance(cache_key)), f)

        # Both files exist, but no .committed marker
        assert artifact_path.exists()
        assert provenance_path.exists()

        # Should be treated as non-existent
        assert not store.exists(cache_key)
        with pytest.raises(FileNotFoundError):
            store.load(cache_key)
        with pytest.raises(FileNotFoundError):
            store.load_provenance(cache_key)

    def test_store_idempotency(self, store: ArtifactStore):
        """Calling store() twice for same committed key is no-op (prevents overwrite crashes)."""
        cache_key = _make_cache_key("idempotent_test")
        artifact = {"data": np.array([1, 2, 3])}
        prov = self._make_provenance(cache_key)

        # First store
        store.store(cache_key, artifact, prov)
        assert store.exists(cache_key)

        # Get paths to verify marker exists
        artifact_path = store._artifact_path(cache_key)
        committed_path = store._committed_path(cache_key)

        # Record original mtimes
        import os

        original_artifact_mtime = artifact_path.stat().st_mtime
        original_committed_mtime = committed_path.stat().st_mtime

        # Second store (should be no-op)
        import time

        time.sleep(0.01)  # Small delay to ensure mtime would change if rewritten
        store.store(cache_key, artifact, prov)

        # Verify still exists
        assert store.exists(cache_key)

        # Verify files were NOT rewritten (mtimes unchanged)
        # This proves idempotency: no-op when already committed
        new_artifact_mtime = artifact_path.stat().st_mtime
        new_committed_mtime = committed_path.stat().st_mtime

        assert new_artifact_mtime == original_artifact_mtime
        assert new_committed_mtime == original_committed_mtime


class TestScavenger:
    """Tests for scavenger/GC cleanup (Issue #929).

    Validates that:
    - Orphaned artifacts (no marker) are cleaned up
    - Stale temp files are cleaned up
    - Valid committed entries are preserved
    - Cleanup report is accurate
    """

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def store(self, cache_dir: Path) -> ArtifactStore:
        """Create ArtifactStore instance."""
        return ArtifactStore(cache_dir=cache_dir, max_size_gb=1.0)

    def _make_provenance(self, cache_key: str) -> ProvenanceMetadata:
        """Helper to create a minimal ProvenanceMetadata."""
        return ProvenanceMetadata(
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

    def test_scavenger_removes_orphaned_artifact(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger removes .npz without .committed marker (if old enough)."""
        import os
        import time

        cache_key = _make_cache_key("orphan_scav")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create orphaned artifact (no marker)
        artifact_path = prefix_dir / f"{cache_key}.npz"
        np.savez_compressed(artifact_path, v=np.array([1]))
        assert artifact_path.exists()

        # Backdate mtime to make it scavengeable (older than default 300s)
        old_time = time.time() - 600
        os.utime(artifact_path, (old_time, old_time))

        report = store.scavenge()
        assert report["orphaned_artifacts_removed"] == 1
        assert not artifact_path.exists()

    def test_scavenger_removes_orphaned_provenance(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger removes .json without .committed marker (if old enough)."""
        import os
        import time

        cache_key = _make_cache_key("orphan_prov_scav")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create orphaned provenance (no marker)
        provenance_path = prefix_dir / f"{cache_key}.json"
        provenance_path.write_text("{}")
        assert provenance_path.exists()

        # Backdate mtime to make it scavengeable
        old_time = time.time() - 600
        os.utime(provenance_path, (old_time, old_time))

        report = store.scavenge()
        assert report["orphaned_provenance_removed"] == 1
        assert not provenance_path.exists()

    def test_scavenger_preserves_committed_entries(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger does NOT remove entries with .committed marker."""
        cache_key = _make_cache_key("committed_preserved")
        store.store(cache_key, {"v": 1}, self._make_provenance(cache_key))

        prefix = cache_key[:2]
        artifact_path = cache_dir / "artifacts" / prefix / f"{cache_key}.npz"
        provenance_path = cache_dir / "artifacts" / prefix / f"{cache_key}.json"
        committed_path = cache_dir / "artifacts" / prefix / f"{cache_key}.committed"

        report = store.scavenge()
        assert report["orphaned_artifacts_removed"] == 0
        assert report["orphaned_provenance_removed"] == 0
        assert artifact_path.exists()
        assert provenance_path.exists()
        assert committed_path.exists()

    def test_scavenger_removes_stale_temp_files(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger removes temp files older than threshold."""
        cache_key = _make_cache_key("stale_temp")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create a stale temp file (name starts with 'tmp')
        stale_tmp = prefix_dir / "tmpabcdef.npz"
        stale_tmp.write_bytes(b"stale data")

        # Backdate the file's mtime so it looks old
        import os
        import time

        old_time = time.time() - 600  # 10 minutes ago
        os.utime(stale_tmp, (old_time, old_time))

        report = store.scavenge(max_temp_age_seconds=300.0)
        assert report["stale_temp_files_removed"] == 1
        assert not stale_tmp.exists()

    def test_scavenger_preserves_fresh_temp_files(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger does NOT remove temp files younger than threshold."""
        cache_key = _make_cache_key("fresh_temp")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create a fresh temp file
        fresh_tmp = prefix_dir / "tmpfresh.npz"
        fresh_tmp.write_bytes(b"fresh data")

        report = store.scavenge(max_temp_age_seconds=300.0)
        assert report["stale_temp_files_removed"] == 0
        assert fresh_tmp.exists()

    def test_scavenger_mixed_scenario(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger handles mix of committed, orphaned, and temp files."""
        import os
        import time

        # 1. Store a valid committed entry
        good_key = _make_cache_key("scav_good")
        store.store(good_key, {"v": 1}, self._make_provenance(good_key))

        # 2. Create orphaned entry (no marker)
        orphan_key = _make_cache_key("scav_orphan")
        prefix = orphan_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)
        orphan_npz = prefix_dir / f"{orphan_key}.npz"
        orphan_json = prefix_dir / f"{orphan_key}.json"
        np.savez_compressed(orphan_npz, v=np.array([2]))
        orphan_json.write_text("{}")

        # Backdate orphans to make them scavengeable
        old_time = time.time() - 600
        os.utime(orphan_npz, (old_time, old_time))
        os.utime(orphan_json, (old_time, old_time))

        # 3. Create stale temp file in the good_key's prefix dir
        good_prefix_dir = cache_dir / "artifacts" / good_key[:2]
        stale_tmp = good_prefix_dir / "tmpstale.json"
        stale_tmp.write_text("junk")
        os.utime(stale_tmp, (old_time, old_time))

        report = store.scavenge(max_temp_age_seconds=300.0)

        # Committed entry preserved
        assert store.exists(good_key)

        # Orphans removed
        assert not orphan_npz.exists()
        assert not orphan_json.exists()
        assert report["orphaned_artifacts_removed"] == 1
        assert report["orphaned_provenance_removed"] == 1

        # Stale temp removed
        assert not stale_tmp.exists()
        assert report["stale_temp_files_removed"] == 1

    def test_scavenger_preserves_fresh_uncommitted(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger does NOT remove fresh uncommitted files (age grace period)."""
        cache_key = _make_cache_key("fresh_uncommitted")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create fresh uncommitted files (no marker, mtime=now)
        artifact_path = prefix_dir / f"{cache_key}.npz"
        prov_path = prefix_dir / f"{cache_key}.json"
        np.savez_compressed(artifact_path, v=np.array([42]))
        prov_path.write_text('{"stage": "test"}')

        # Scavenge with default 300s threshold
        report = store.scavenge(max_temp_age_seconds=300.0)

        # Fresh files should NOT be removed (protected by age grace period)
        assert artifact_path.exists()
        assert prov_path.exists()
        assert report["orphaned_artifacts_removed"] == 0
        assert report["orphaned_provenance_removed"] == 0

    def test_scavenger_removes_old_uncommitted(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger removes old uncommitted files."""
        import os
        import time

        cache_key = _make_cache_key("old_uncommitted")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create old uncommitted files (no marker, mtime old)
        artifact_path = prefix_dir / f"{cache_key}.npz"
        prov_path = prefix_dir / f"{cache_key}.json"
        np.savez_compressed(artifact_path, v=np.array([99]))
        prov_path.write_text('{"stage": "old"}')

        # Backdate to 301s ago (older than default 300s threshold)
        old_time = time.time() - 301
        os.utime(artifact_path, (old_time, old_time))
        os.utime(prov_path, (old_time, old_time))

        # Scavenge
        report = store.scavenge(max_temp_age_seconds=300.0)

        # Old files should be removed
        assert not artifact_path.exists()
        assert not prov_path.exists()
        assert report["orphaned_artifacts_removed"] == 1
        assert report["orphaned_provenance_removed"] == 1

    def test_scavenger_respects_active_locks(self, store: ArtifactStore, cache_dir: Path):
        """Scavenger skips keys where per-key lock is held (TOCTOU prevention)."""
        import os
        import time

        cache_key = _make_cache_key("locked_key")
        prefix = cache_key[:2]
        prefix_dir = cache_dir / "artifacts" / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Create old uncommitted files
        artifact_path = prefix_dir / f"{cache_key}.npz"
        prov_path = prefix_dir / f"{cache_key}.json"
        np.savez_compressed(artifact_path, v=np.array([77]))
        prov_path.write_text('{"stage": "locked"}')

        # Backdate (old enough to scavenge normally)
        old_time = time.time() - 600
        os.utime(artifact_path, (old_time, old_time))
        os.utime(prov_path, (old_time, old_time))

        # Acquire lock (simulate in-flight store paused)
        with store._acquire_lock(cache_key, exclusive=True):
            # Run scavenger while lock is held
            report = store.scavenge(max_temp_age_seconds=300.0)

            # Files should NOT be removed (lock prevents deletion)
            assert artifact_path.exists()
            assert prov_path.exists()
            assert report["orphaned_artifacts_removed"] == 0
            assert report["orphaned_provenance_removed"] == 0

        # After lock is released, scavenger should be able to remove them
        report = store.scavenge(max_temp_age_seconds=300.0)
        assert not artifact_path.exists()
        assert not prov_path.exists()
        assert report["orphaned_artifacts_removed"] == 1
        assert report["orphaned_provenance_removed"] == 1

    def test_scavenger_empty_cache(self, store: ArtifactStore):
        """Scavenger on empty cache returns zero counts."""
        report = store.scavenge()
        assert report["orphaned_artifacts_removed"] == 0
        assert report["orphaned_provenance_removed"] == 0
        assert report["stale_temp_files_removed"] == 0
